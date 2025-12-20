# --- Save this code as streamlit_app.py on GitHub (Final Optimized DeepSeek Version) ---

import streamlit as st
from openai import OpenAI 
import json
import re
from typing import List
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject 
from weaviate.classes.query import Filter 
from sentence_transformers import SentenceTransformer 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
import xml.etree.ElementTree as ET
import os
import warnings

# Suppress InsecureRequestWarning
warnings.filterwarnings("ignore", "Unverified HTTPS request is being made")

# --- 1. CONFIGURATION ---
try:
    WEAVIATE_URL_BASE = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud" 
    WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"] 
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"] 
except KeyError as e:
    st.error(f"Missing API key in Streamlit Secrets: {e}. Please add it to your settings.")
    st.stop()

DATA_PATHS = {
    "pdf": "policy_baggage.pdf",
    "xml": "flight_records.xml",
    "web": "https://paa.gov.pk/"
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_MODEL = load_embedding_model()
WEAVIATE_CLIENT = None 

# DeepSeek Client
try:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
except Exception as e:
    st.error(f"DeepSeek Init Failed: {e}")
    st.stop()

# --- 2. VECTOR DB SETUP ---
@st.cache_resource(show_spinner="Connecting to Weaviate...")
def setup_weaviate_client():
    global WEAVIATE_CLIENT
    try:
        client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL_BASE, auth_credentials=Auth.api_key(WEAVIATE_API_KEY))
        
        for name in ["PAAPolicy", "PAAFlightStatus", "PAAWebLink"]:
            if client.collections.exists(name): client.collections.delete(name)
        
        VECTORIZER_CONFIG = Configure.Vectorizer.none()
        client.collections.create(name="PAAPolicy", properties=[Property(name="content", data_type=DataType.TEXT), Property(name="source", data_type=DataType.TEXT)], vectorizer_config=VECTORIZER_CONFIG)
        client.collections.create(name="PAAFlightStatus", properties=[Property(name="content", data_type=DataType.TEXT), Property(name="flight_num", data_type=DataType.TEXT), Property(name="status", data_type=DataType.TEXT)], vectorizer_config=VECTORIZER_CONFIG)
        client.collections.create(name="PAAWebLink", properties=[Property(name="content", data_type=DataType.TEXT), Property(name="url_href", data_type=DataType.TEXT)], vectorizer_config=VECTORIZER_CONFIG)
        
        WEAVIATE_CLIENT = client
        return client
    except Exception as e:
        st.error(f"Weaviate Error: {e}")
        st.stop()

@st.cache_resource(show_spinner="Ingesting data...")
def ingest_all_data(_client):
    def process_pdf(client, path):
        try:
            reader = PdfReader(path)
            text = "".join([p.extract_text() for p in reader.pages])
            chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_text(text)
            objs = [DataObject(properties={"content": c, "source": os.path.basename(path)}, vector=EMBEDDING_MODEL.encode(c).tolist()) for c in chunks]
            client.collections.get("PAAPolicy").data.insert_many(objs)
        except: pass

    def process_xml(client, path):
        try:
            with open(path, 'r') as f: content = f.read()
            # Simplified parsing logic
            root = ET.fromstring(content)
            # Add your specific XML logic here if needed
        except: pass

    def process_web(client, url):
        try:
            res = requests.get(url, verify=False, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            text = soup.body.get_text(separator=' ', strip=True)
            chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_text(text)
            objs = [DataObject(properties={"content": c, "url_href": url}, vector=EMBEDDING_MODEL.encode(c).tolist()) for c in chunks]
            client.collections.get("PAAWebLink").data.insert_many(objs)
        except: pass

    process_pdf(_client, DATA_PATHS["pdf"])
    process_web(_client, DATA_PATHS["web"])
    return True

# --- 3. TOOLS ---
def query_policy_and_baggage(query: str):
    col = WEAVIATE_CLIENT.collections.get("PAAPolicy")
    res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=1)
    return f"Policy: {res.objects[0].properties['content']}" if res.objects else "No policy found."

def query_flight_status(query: str):
    # Dummy logic for example
    return "Flight PK300 is On Time."

def query_web_links_and_forms(query: str):
    col = WEAVIATE_CLIENT.collections.get("PAAWebLink")
    res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=1)
    return f"Web Info: {res.objects[0].properties['content']}" if res.objects else "No web info found."

# --- 4. GENERATION ---
def generate_answer_with_llm(user_query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"Use this context to answer: {context}\n\nQuestion: {user_query}"
    try:
        res = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- 5. ORCHESTRATOR ---
def orchestrator_agent(query_text: str):
    tools_list = [query_policy_and_baggage, query_flight_status, query_web_links_and_forms]
    tool_map = {t.__name__: t for t in tools_list}
    
    messages = [
        {"role": "system", "content": "You are a PAA Agent. Use tools to answer questions."},
        {"role": "user", "content": query_text}
    ]
    
    tool_defs = [{
        "type": "function",
        "function": {
            "name": t.__name__,
            "description": f"Query {t.__name__}",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    } for t in tools_list]

    try:
        res = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tool_defs,
            tool_choice="auto"
        )
        
        msg = res.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            results = []
            used_names = []
            for tc in msg.tool_calls:
                fn = tool_map[tc.function.name]
                out = fn(query_text)
                results.append(out)
                used_names.append(tc.function.name)
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": out})
            
            final_res = deepseek_client.chat.completions.create(model="deepseek-chat", messages=messages)
            return final_res.choices[0].message.content, used_names
        
        return msg.content, ["Direct Response"]
    except Exception as e:
        return f"DeepSeek Error: {e}", ["Error"]

# --- 6. UI ---
st.set_page_config(page_title="PAA RAG", layout="wide")
st.title("🇵🇰 PAA Agentic Chatbot")

try:
    client = setup_weaviate_client()
    ingest_all_data(client)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Ask me something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.spinner("Analyzing..."):
            ans, tools = orchestrator_agent(prompt)
            full_ans = f"{ans}\n\n*(Tools: {', '.join(tools)})*"
            with st.chat_message("assistant"): st.markdown(full_ans)
            st.session_state.messages.append({"role": "assistant", "content": full_ans})
except Exception as e:
    st.error(f"App Error: {e}")
