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
    st.error(f"Secret Missing: {e}")
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

# --- 3. UPDATED ROBUST XML TOOL (NAMESPACE AWARE) ---

def get_flight_status_from_xml(flight_num: str, travel_date: str = None):
    """Parses complex Envelope XML with Namespaces for Flight Data."""
    if not os.path.exists("flight_records.xml"):
        return "Error: flight_records.xml not found."
    
    try:
        tree = ET.parse("flight_records.xml")
        root = tree.getroot()
        
        # XML Namespace (from your data)
        ns = {'ns': 'http://schema.ultra-as.com'}
        
        # Normalize Input
        target_f = flight_num.strip().upper()
        if not target_f.startswith('SV'): target_f = f"SV{target_f}"
        
        # Clean date for comparison (e.g., 2025-11-30)
        target_d = ""
        if travel_date:
            # Matches YYYY-MM-DD or MM-DD pattern
            target_d = re.sub(r'[^0-9-]', '', travel_date)

        # Search for AFDSFlightData within the XML
        for flight_node in root.findall('.//ns:AFDSFlightData', ns):
            ident = flight_node.find('ns:FlightIdentification', ns)
            if ident is not None:
                xml_num = ident.find('ns:FlightIdentity', ns).text.strip().upper()
                xml_date = ident.find('ns:ScheduledDate', ns).text.strip() # 2025-11-30+05:00
                
                # Check Flight Identity
                if target_f == xml_num:
                    # Check Date (Partial match)
                    if not target_d or (target_d in xml_date):
                        # Extract Operational Details
                        ops = flight_node.find('.//ns:OperationalTimes', ns)
                        est_time = ops.find('ns:EstimatedDateTime', ns).text if ops is not None else "Not Available"
                        
                        # Extract Carousel
                        carousel = "N/A"
                        bag_node = flight_node.find('.//ns:BaggageReclaimCarouselID', ns)
                        if bag_node is not None: carousel = bag_node.text

                        # Flight Direction
                        direction = ident.find('ns:FlightDirection', ns).text
                        
                        return (f"✅ **Flight Found (AODB System):**\n"
                                f"- **Flight Identity:** {xml_num}\n"
                                f"- **Direction:** {direction}\n"
                                f"- **Scheduled Date:** {xml_date}\n"
                                f"- **Estimated Time:** {est_time}\n"
                                f"- **Baggage Carousel:** {carousel}")

        return f"❌ No matching record for {target_f} on {travel_date} in the system."
    except Exception as e:
        return f"XML System Error: {str(e)}"

# --- 4. AGENT LOGIC (Memory + Tooling) ---

def run_agent(user_input):
    tools = [{
        "type": "function",
        "function": {
            "name": "get_flight_status_from_xml",
            "description": "Checks AODB XML for flight status, timing, and baggage carousel.",
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

    # System instruction: Essential for context keeping
    messages = [{"role": "system", "content": "You are a PAA Agent. You have access to AODB flight records. If a user provides a flight number first and a date later, remember the flight number from history. Today is late 2025."}]
    
    # Add history from Streamlit session
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
st.set_page_config(page_title="PAA Intelligent Agent", page_icon="✈️")
st.title("✈️ PAA Intelligent Agent")

w_client = get_weaviate_client()
ingest_data(w_client)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Flight status of SV726?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.spinner("Accessing AODB Records..."):
        ans = run_agent(prompt)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(ans)
