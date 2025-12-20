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
    for col_name in ["PAAPolicy", "PAAWeb"]:
        if not client.collections.exists(col_name):
            client.collections.create(
                name=col_name,
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

# --- 3. UPDATED TOOLS (Robust XML Matching) ---

def query_knowledge_base(query: str):
    client = get_weaviate_client()
    results = []
    for col_name in ["PAAPolicy", "PAAWeb"]:
        col = client.collections.get(col_name)
        res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=2)
        results.extend([o.properties['content'] for o in res.objects])
    return "\n".join(results) if results else "No policy found."

def get_flight_status_from_xml(flight_num: str, travel_date: str = None):
    """Parses XML with robust date and flight number matching."""
    if not os.path.exists("flight_records.xml"):
        return "Error: flight_records.xml not found."
    
    try:
        tree = ET.parse("flight_records.xml")
        root = tree.getroot()
        
        target_f = flight_num.strip().upper()
        # Normalizing input date: "11Nov-25" -> "11nov"
        target_d = travel_date.lower().replace(" ", "").replace("-", "").replace("/", "") if travel_date else ""
        if len(target_d) > 5: target_d = target_d[:5] # Focus on Day and Month

        for flight in root.findall('flight'):
            xml_num = flight.find('number').text.strip().upper()
            xml_date = flight.find('date').text.strip().lower()
            clean_xml_date = xml_date.replace(" ", "").replace("-", "").replace("/", "")
            
            if target_f in xml_num:
                # Flexible date match
                if not target_d or (target_d in clean_xml_date):
                    status = flight.find('status').text
                    dep = flight.find('departure').text
                    arr = flight.find('arrival').text
                    full_date = flight.find('date').text
                    return f"✅ **Flight Found:** {xml_num} on {full_date}\n- **Status:** {status}\n- **Dep:** {dep} | **Arr:** {arr}"
        
        return f"❌ No record for {flight_num} on {travel_date} in XML records."
    except Exception as e:
        return f"XML Error: {str(e)}"

# --- 4. AGENT LOGIC ---

def run_agent(user_input):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_knowledge_base",
                "description": "Baggage and PAA policy info.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_flight_status_from_xml",
                "description": "Flight status from XML.",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "flight_num": {"type": "string"},
                        "travel_date": {"type": "string"}
                    }, 
                    "required": ["flight_num"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a PAA Supervisor. For flight status, extract flight number and date for the tool. Use 'get_flight_status_from_xml'."},
        {"role": "user", "content": user_input}
    ]

    response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "query_knowledge_base":
                res = query_knowledge_base(args['query'])
            else:
                res = get_flight_status_from_xml(args.get('flight_num'), args.get('travel_date'))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": res})
        
        final = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final.choices[0].message.content
    return msg.content

# --- 5. UI ---
st.set_page_config(page_title="PAA Unified Agent")
st.title("🇵🇰 PAA Agent (PDF + XML)")

w_client = get_weaviate_client()
ingest_data(w_client)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Asalam-o-Alaikum! How can I help you today?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Flight status of SV726 on 11Nov..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.spinner("Searching records..."):
        ans = run_agent(prompt)
        with st.chat_message("assistant"): st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
