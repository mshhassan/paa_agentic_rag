import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject 
from sentence_transformers import SentenceTransformer 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
import xml.etree.ElementTree as ET
import os
import json
import re
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
try:
    WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud" 
    WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"] 
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] 
    
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
except KeyError as e:
    st.error(f"Secret Missing: {e}. Please ensure API keys are in Streamlit Secrets.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_MODEL = load_embedding_model()

# --- 2. WEAVIATE SETUP ---
@st.cache_resource(show_spinner="Connecting to Weaviate...")
def get_weaviate_client():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    for col_name in ["PAAPolicy", "PAAWeb"]:
        if not client.collections.exists(col_name):
            client.collections.create(
                name=col_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[Property(name="content", data_type=DataType.TEXT)]
            )
    return client

@st.cache_resource(show_spinner="Ingesting Knowledge Base...")
def ingest_data(_client):
    # Process PDF
    if os.path.exists("policy_baggage.pdf"):
        col = _client.collections.get("PAAPolicy")
        if col.aggregate.over_all(total_count=True).total_count == 0:
            reader = PdfReader("policy_baggage.pdf")
            text = "".join([p.extract_text() for p in reader.pages])
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
            objs = [DataObject(properties={"content": c}, vector=EMBEDDING_MODEL.encode(c).tolist()) for c in chunks]
            col.data.insert_many(objs)
    return True

# --- 3. TOOLS (PDF, WEB & XML) ---

def query_knowledge_base(query: str):
    """Searches PDF and Web data for PAA rules/policies."""
    client = get_weaviate_client()
    results = []
    for col_name in ["PAAPolicy", "PAAWeb"]:
        col = client.collections.get(col_name)
        res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=2)
        results.extend([o.properties['content'] for o in res.objects])
    return "\n".join(results) if results else "No policy found."

def get_flight_status_from_xml(flight_num: str, travel_date: str = None):
    """Directly parses flight_records.xml for flight status."""
    if not os.path.exists("flight_records.xml"):
        return "Flight records XML file not found in directory."
    
    try:
        tree = ET.parse("flight_records.xml")
        root = tree.getroot()
        flight_num = flight_num.strip().upper()

        for flight in root.findall('flight'):
            xml_num = flight.find('number').text.strip().upper()
            xml_date = flight.find('date').text.strip()
            
            # Match flight number
            if flight_num in xml_num:
                # If date is provided, try to match it as well
                if travel_date:
                    # Basic check if date mentioned in query exists in XML date field
                    clean_travel_date = travel_date.lower().replace(" ", "")
                    clean_xml_date = xml_date.lower().replace(" ", "")
                    if clean_travel_date not in clean_xml_date:
                        continue

                status = flight.find('status').text
                dep = flight.find('departure').text
                arr = flight.find('arrival').text
                return f"XML RECORD: Flight {xml_num} on {xml_date} is {status}. Departure: {dep}, Arrival: {arr}."
        
        return f"Could not find exact match for {flight_num} on {travel_date} in flight_records.xml."
    except Exception as e:
        return f"XML Parsing Error: {str(e)}"

# --- 4. AGENT ORCHESTRATOR ---

def run_agent(user_input):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_knowledge_base",
                "description": "Get baggage policy and general PAA information from PDF/Web.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_flight_status_from_xml",
                "description": "Check real-time flight status from XML records.",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "flight_num": {"type": "string", "description": "Flight number like SV726"},
                        "travel_date": {"type": "string", "description": "Date like 11 Nov 2025"}
                    }, 
                    "required": ["flight_num"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a PAA Supervisor. For flight status queries, ALWAYS extract flight number and date and use 'get_flight_status_from_xml'. For baggage/rules, use 'query_knowledge_base'."},
        {"role": "user", "content": user_input}
    ]

    try:
        response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                if name == "query_knowledge_base":
                    result = query_knowledge_base(args['query'])
                elif name == "get_flight_status_from_xml":
                    result = get_flight_status_from_xml(args.get('flight_num'), args.get('travel_date'))
                
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": name, "content": result})
            
            final_res = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
            return final_res.choices[0].message.content
        return msg.content
    except Exception as e:
        return f"Agent Error: {str(e)}"

# --- 5. UI ---
st.set_page_config(page_title="PAA Unified Agent", page_icon="🇵🇰")
st.title("🇵🇰 PAA Agent (PDF + XML + Web)")

w_client = get_weaviate_client()
ingest_data(w_client)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Asalam-o-Alaikum! How can I help you today?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Check status of SV726 on 11 Nov..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.spinner("Analyzing request..."):
        ans = run_agent(prompt)
        with st.chat_message("assistant"): st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
