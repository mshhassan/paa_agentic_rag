import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject 
from sentence_transformers import SentenceTransformer 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from pypdf import PdfReader
import xml.etree.ElementTree as ET
import os
import json
import re
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
try:
    WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud" 
    WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"] 
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] 
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
except KeyError as e:
    st.error(f"Secret Missing: {e}. Please add WEAVIATE_API_KEY and OPENAI_API_KEY in Secrets.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_MODEL = load_embedding_model()

# --- 2. WEAVIATE SETUP ---
@st.cache_resource
def get_weaviate_client():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    if not client.collections.exists("PAAPolicy"):
        client.collections.create(
            name="PAAPolicy",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[Property(name="content", data_type=DataType.TEXT)]
        )
    return client

@st.cache_resource
def ingest_data(_client):
    if os.path.exists("policy_baggage.pdf"):
        col = _client.collections.get("PAAPolicy")
        if col.aggregate.over_all(total_count=True).total_count == 0:
            reader = PdfReader("policy_baggage.pdf")
            text = "".join([p.extract_text() for p in reader.pages])
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
            objs = [DataObject(properties={"content": c}, vector=EMBEDDING_MODEL.encode(c).tolist()) for c in chunks]
            col.data.insert_many(objs)
    return True

# --- 3. UPDATED ROBUST XML TOOL ---

def get_flight_status_from_xml(flight_num: str, travel_date: str = None):
    """Parses XML with multi-layer matching for flight and date."""
    if not os.path.exists("flight_records.xml"):
        return "System Error: flight_records.xml not found."
    
    try:
        tree = ET.parse("flight_records.xml")
        root = tree.getroot()
        
        # 1. Flight Number Normalization (726 -> SV726)
        target_f = flight_num.strip().upper()
        if not target_f.startswith('SV') and len(target_f) <= 4:
            target_f = f"SV{target_f}"
        
        # 2. Date Normalization (11Nov-25 -> 11nov)
        target_d_clean = ""
        if travel_date:
            target_d_clean = re.sub(r'[^a-zA-Z0-9]', '', travel_date.lower())
            # "11nov25" -> "11nov" (taking only day and month for safety)
            if len(target_d_clean) > 5: target_d_clean = target_d_clean[:5]

        for flight in root.findall('flight'):
            xml_num = flight.find('number').text.strip().upper()
            xml_date_raw = flight.find('date').text.strip()
            xml_date_clean = re.sub(r'[^a-zA-Z0-9]', '', xml_date_raw.lower())
            
            # Match Flight Number
            if target_f == xml_num or target_f in xml_num:
                # Match Date (if provided)
                if not target_d_clean or (target_d_clean in xml_date_clean):
                    status = flight.find('status').text
                    dep = flight.find('departure').text
                    arr = flight.find('arrival').text
                    return (f"✅ **Flight Status Verified:**\n"
                            f"- **Flight:** {xml_num}\n"
                            f"- **Date:** {xml_date_raw}\n"
                            f"- **Status:** {status}\n"
                            f"- **Schedule:** {dep} to {arr}")
        
        return f"❌ No record found in XML for Flight {target_f} on {travel_date}."
    except Exception as e:
        return f"Internal XML Error: {str(e)}"

# --- 4. AGENT LOGIC (Memory + Context) ---

def run_agent(user_input):
    tools = [{
        "type": "function",
        "function": {
            "name": "get_flight_status_from_xml",
            "description": "Check flight status using number (SV726) and date (11 Nov).",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_num": {"type": "string"},
                    "travel_date": {"type": "string"}
                },
                "required": ["flight_num"]
            }
        }
    }]

    # System instruction with strict history focus
    messages = [{"role": "system", "content": "You are a PAA expert. Today is Dec 2025. Use 'get_flight_status_from_xml' for ALL flight status queries. Remember previous flight numbers if the user only provides a date later."}]
    
    # Add history
    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})
    
    messages.append({"role": "user", "content": user_input})

    response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = get_flight_status_from_xml(args.get('flight_num'), args.get('travel_date'))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": "get_flight_status_from_xml", "content": result})
        
        final = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final.choices[0].message.content
    return msg.content

# --- 5. UI ---
st.set_page_config(page_title="PAA Unified Agent", page_icon="✈️")
st.title("✈️ PAA Intelligent Agent")

# Ingest data on load
w_client = get_weaviate_client()
ingest_data(w_client)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("E.g., What is the status of SV726 on 11 Nov?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.spinner("Checking records..."):
        ans = run_agent(prompt)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(ans)
