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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. CONFIGURATION ---
try:
    WEAVIATE_URL_BASE = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud" 
    WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"] 
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except KeyError as e:
    st.error(f"Missing Secret: {e}. Please add GEMINI_API_KEY and WEAVIATE_API_KEY in Streamlit Secrets.")
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

# 🟢 Fixed Gemini Client Init
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    # Latest standard model ID
    MODEL_NAME = "gemini-1.5-flash" 
except Exception as e:
    st.error(f"Gemini Init Failed: {e}")
    st.stop()

# --- 2. VECTOR DB SETUP ---
@st.cache_resource(show_spinner="Connecting to Weaviate...")
def setup_weaviate_client():
    global WEAVIATE_CLIENT
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL_BASE, 
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
        )
        # Create Collections if they don't exist
        for name in ["PAAPolicy", "PAAFlightStatus", "PAAWebLink"]:
            if not client.collections.exists(name):
                client.collections.create(
                    name=name, 
                    properties=[Property(name="content", data_type=DataType.TEXT)],
                    vectorizer_config=Configure.Vectorizer.none()
                )
        WEAVIATE_CLIENT = client
        return client
    except Exception as e:
        st.error(f"Weaviate Error: {e}")
        st.stop()

@st.cache_resource(show_spinner="Ingesting documents...")
def ingest_all_data(_client):
    def process_pdf(client, path):
        if not os.path.exists(path): return
        try:
            reader = PdfReader(path)
            text = "".join([p.extract_text() for p in reader.pages])
            chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_text(text)
            objs = [DataObject(properties={"content": c}, vector=EMBEDDING_MODEL.encode(c).tolist()) for c in chunks]
            client.collections.get("PAAPolicy").data.insert_many(objs)
        except: pass

    process_pdf(_client, DATA_PATHS["pdf"])
    return True

# --- 3. RETRIEVAL TOOLS ---
def query_policy_and_baggage(query: str):
    col = WEAVIATE_CLIENT.collections.get("PAAPolicy")
    res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=1)
    return f"Policy Context: {res.objects[0].properties['content']}" if res.objects else "No matching policy found."

def query_flight_status(query: str):
    return "Status Update: Flight PK301 is On Time. Departure: 14:00 PKT."

def query_web_links_and_forms(query: str):
    return "Official Link: https://paa.gov.pk/forms/baggage-claim"

# --- 4. AGENTIC ORCHESTRATOR (RAG) ---


def orchestrator_agent(query_text: str):
    try:
        # Define Tools for Gemini
        tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name="query_policy_and_baggage",
                    description="Get information about baggage policy and airport rules",
                    parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}
                ),
                types.FunctionDeclaration(
                    name="query_flight_status",
                    description="Get current status of a flight",
                    parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}
                ),
                types.FunctionDeclaration(
                    name="query_web_links_and_forms",
                    description="Find official PAA links and forms",
                    parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}
                )
            ])
        ]

        # Step 1: Tool Selection Call
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=query_text,
            config=types.GenerateContentConfig(
                tools=tools,
                system_instruction="You are a PAA Supervisor. Use tools to fetch accurate data before answering."
            )
        )
        
        used_tools = []
        context_chunks = []

        # Step 2: Execute Tool Calls
        if response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    name = part.function_call.name
                    args = part.function_call.args
                    used_tools.append(name)
                    
                    if name == "query_policy_and_baggage":
                        context_chunks.append(query_policy_and_baggage(args['query']))
                    elif name == "query_flight_status":
                        context_chunks.append(query_flight_status(args['query']))
                    elif name == "query_web_links_and_forms":
                        context_chunks.append(query_web_links_and_forms(args['query']))

        if not context_chunks:
            return response.text if response.text else "Please ask about baggage or flights.", ["Direct"]

        # Step 3: Synthesis Call
        final_prompt = f"Context: {' '.join(context_chunks)}\n\nQuestion: {query_text}\n\nAnswer like a helpful assistant:"
        final_response = gemini_client.models.generate_content(model=MODEL_NAME, contents=final_prompt)
        
        return final_response.text, used_tools

    except Exception as e:
        return f"Gemini Error: {str(e)}", ["Error"]

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="PAA Gemini Agent", page_icon="🇵🇰", layout="wide")
st.title("🇵🇰 PAA Agentic Chatbot (Gemini)")

try:
    client = setup_weaviate_client()
    ingest_all_data(client)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with PAA services today?"}]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Ask about baggage, flights..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            ans, tools = orchestrator_agent(prompt)
            formatted_ans = f"{ans}\n\n*(Used Tools: {', '.join(tools)})*"
            with st.chat_message("assistant"): st.markdown(formatted_ans)
            st.session_state.messages.append({"role": "assistant", "content": formatted_ans})

except Exception as e:
    st.error(f"App Error: {e}")
