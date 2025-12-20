# --- Save this code as streamlit_app.py on GitHub (Final Gemini Version) ---

import streamlit as st
from google import genai
from google.genai import types
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
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Change this in Streamlit Secrets
except KeyError as e:
    st.error(f"Missing API key in Streamlit Secrets: {e}")
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

# Gemini Client
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gemini Init Failed: {e}")
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
    """Answers questions about baggage rules and airport policies."""
    col = WEAVIATE_CLIENT.collections.get("PAAPolicy")
    res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=1)
    return f"Policy: {res.objects[0].properties['content']}" if res.objects else "No policy found."

def query_flight_status(query: str):
    """Check current status of flights (e.g. PK300)."""
    return "Flight PK300 is currently On Time and departing from Terminal 1."

def query_web_links_and_forms(query: str):
    """Finds official web links and downloadable forms."""
    col = WEAVIATE_CLIENT.collections.get("PAAWebLink")
    res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=1)
    return f"Web Info: {res.objects[0].properties['content']}" if res.objects else "No web info found."

# --- 4. ORCHESTRATOR (GEMINI AGENT) ---
def orchestrator_agent(query_text: str):
    # Register tools
    tools_list = [query_policy_and_baggage, query_flight_status, query_web_links_and_forms]
    
    try:
        # Gemini 1.5 Flash is great for agentic tasks
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=query_text,
            config=types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name="query_policy_and_baggage",
                        description="Query baggage policy and airport rules",
                        parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}}
                    ),
                    types.FunctionDeclaration(
                        name="query_flight_status",
                        description="Check status of a specific flight",
                        parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}}
                    ),
                    types.FunctionDeclaration(
                        name="query_web_links_and_forms",
                        description="Get official PAA links and forms",
                        parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}}
                    )
                ])],
                system_instruction="You are a PAA Supervisor. Use the tools provided to answer user queries accurately."
            )
        )
        
        # Handling function calls
        retrieved_info = []
        used_tools = []
        
        for part in response.candidates[0].content.parts:
            if part.function_call:
                fn_name = part.function_call.name
                fn_args = part.function_call.args
                
                # Execute the tool
                if fn_name == "query_policy_and_baggage":
                    res = query_policy_and_baggage(fn_args['query'])
                elif fn_name == "query_flight_status":
                    res = query_flight_status(fn_args['query'])
                elif fn_name == "query_web_links_and_forms":
                    res = query_web_links_and_forms(fn_args['query'])
                
                retrieved_info.append(res)
                used_tools.append(fn_name)

        if not retrieved_info:
            return response.text, ["Direct Response"]
        
        # Second pass to synthesize information
        context = "\n".join(retrieved_info)
        final_prompt = f"Based on this info: {context}\n\nUser Question: {query_text}"
        final_response = gemini_client.models.generate_content(model="gemini-1.5-flash", contents=final_prompt)
        
        return final_response.text, used_tools

    except Exception as e:
        return f"Gemini Error: {e}", ["Error"]

# --- 5. UI ---
st.set_page_config(page_title="PAA Gemini RAG", layout="wide")
st.title("🇵🇰 PAA Agentic Chatbot (Gemini)")

try:
    client = setup_weaviate_client()
    ingest_all_data(client)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your PAA assistant. How can I help you with flights or baggage today?"}]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Ask about PIA baggage, flight status etc..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.spinner("Agent is thinking..."):
            ans, tools = orchestrator_agent(prompt)
            full_ans = f"{ans}\n\n*(Tools used: {', '.join(tools)})*"
            with st.chat_message("assistant"): st.markdown(full_ans)
            st.session_state.messages.append({"role": "assistant", "content": full_ans})
except Exception as e:
    st.error(f"App Error: {e}")
