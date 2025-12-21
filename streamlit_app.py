import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
import json
import re
import os
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION ---
WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client_openai = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
    return model, client

EMBEDDING_MODEL, W_CLIENT = load_resources()

# --- 2. PDF PROCESSING FUNCTION ---
def process_uploaded_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Text splitting for better RAG retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    
    coll = W_CLIENT.collections.get("PAAPolicy")
    for chunk in chunks:
        vector = EMBEDDING_MODEL.encode(chunk).tolist()
        coll.data.insert(properties={"content": chunk, "category": "pdf_upload"}, vector=vector)
    return len(chunks)

# --- 3. REFINED SUB-AGENTS ---
def flight_inquiry_agent(query):
    coll = W_CLIENT.collections.get("PAAFlightStatus")
    # Priority keyword search (alpha=0.4)
    res = coll.query.hybrid(query=query, vector=EMBEDDING_MODEL.encode(query).tolist(), limit=3, alpha=0.4)
    return "\n---\n".join([o.properties['content'] for o in res.objects]) if res.objects else "No flight data found."

def policy_agent(query):
    coll = W_CLIENT.collections.get("PAAPolicy")
    res = coll.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=3)
    return "\n---\n".join([o.properties['content'] for o in res.objects]) if res.objects else "No policy info found."

def web_agent(query):
    coll = W_CLIENT.collections.get("PAAWebLink")
    res = coll.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=3)
    return "\n---\n".join([o.properties['content'] for o in res.objects]) if res.objects else "No links found."

# --- 4. SUPERVISOR LOGIC ---
def supervisor_agent(user_input):
    tools = [
        {"type": "function", "function": {"name": "flight_inquiry_agent", "description": "Get detailed flight, gate, counter info.", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}},
        {"type": "function", "function": {"name": "policy_agent", "description": "Search baggage rules and passenger policies.", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}},
        {"type": "function", "function": {"name": "web_agent", "description": "Get official PAA website links.", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}
    ]

    messages = [
        {"role": "system", "content": "You are the PAA Master Agent. Provide comprehensive flight details (Gate, Belt, Counter) and official links. Use bullet points."},
        {"role": "user", "content": user_input}
    ]
    
    response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            q = json.loads(tc.function.arguments).get('q')
            if tc.function.name == "flight_inquiry_agent": res = flight_inquiry_agent(q)
            elif tc.function.name == "policy_agent": res = policy_agent(q)
            else: res = web_agent(q)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": res})
        
        final = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final.choices[0].message.content
    return msg.content

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="PAA Master RAG", layout="wide")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("📂 Knowledge Management")
    uploaded_pdf = st.file_uploader("Upload PAA Policy PDF", type="pdf")
    if uploaded_pdf:
        with st.spinner("Indexing PDF into RAG..."):
            num_chunks = process_uploaded_pdf(uploaded_pdf)
            st.success(f"Indexed {num_chunks} chunks from PDF!")

st.title("✈️ PAA.GOV.PK Operations")

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ex: What is the baggage policy and status of PK841?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        ans = supervisor_agent(prompt)
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
