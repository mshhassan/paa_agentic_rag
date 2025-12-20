import streamlit as st
import google.generativeai as genai
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject 
from sentence_transformers import SentenceTransformer 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
try:
    WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud" 
    WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"] 
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"] 
    
    # Configure Gemini
    genai.configure(api_key=GEMINI_KEY)
except KeyError as e:
    st.error(f"Secrets missing: {e}")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_MODEL = load_embedding_model()

# --- 2. WEAVIATE SETUP ---
@st.cache_resource(show_spinner="Connecting to DB...")
def get_client():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    # Create collection if not exists
    if not client.collections.exists("PAAPolicy"):
        client.collections.create(
            name="PAAPolicy",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[Property(name="content", data_type=DataType.TEXT)]
        )
    return client

@st.cache_resource(show_spinner="Ingesting PDF...")
def ingest_data(_client):
    if os.path.exists("policy_baggage.pdf"):
        reader = PdfReader("policy_baggage.pdf")
        text = "".join([p.extract_text() for p in reader.pages])
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        
        collection = _client.collections.get("PAAPolicy")
        # Check if empty before inserting
        count = collection.aggregate.over_all(total_count=True).total_count
        if count == 0:
            objs = [DataObject(properties={"content": c}, vector=EMBEDDING_MODEL.encode(c).tolist()) for c in chunks]
            collection.data.insert_many(objs)
    return True

# --- 3. RETRIEVAL TOOL ---
def get_baggage_info(query: str):
    """Retrieves baggage policy information from the uploaded PDF."""
    client = get_client()
    col = client.collections.get("PAAPolicy")
    res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=1)
    if res.objects:
        return res.objects[0].properties['content']
    return "No specific baggage information found in the document."

# --- 4. AGENTIC LOGIC (STABLE VERSION) ---
def agent_chat(user_input: str):
    # Model define karein (Flash 1.5 is very stable)
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        tools=[get_baggage_info],
        system_instruction="You are a PAA assistant. Use the 'get_baggage_info' tool to answer questions about baggage policy."
    )
    
    # Chat shuru karein tools ke saath
    chat = model.start_chat(enable_automatic_function_calling=True)
    
    try:
        response = chat.send_message(user_input)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- 5. UI ---
st.set_page_config(page_title="PAA Agent", page_icon="🇵🇰")
st.title("🇵🇰 PAA Agentic Chatbot")

client = get_client()
ingest_data(client)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Asalam-o-Alaikum! How can I help you with PAA services?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ask me about PIA Baggage Policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.spinner("Searching files..."):
        answer = agent_chat(prompt)
        with st.chat_message("assistant"): st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
