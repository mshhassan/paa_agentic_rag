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
    st.error(f"Secret Missing: {e}. Please add OPENAI_API_KEY in Streamlit Secrets.")
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
    if not client.collections.exists("PAAPolicy"):
        client.collections.create(
            name="PAAPolicy",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[Property(name="content", data_type=DataType.TEXT)]
        )
    return client

@st.cache_resource(show_spinner="Ingesting Policy Data...")
def ingest_data(_client):
    if os.path.exists("policy_baggage.pdf"):
        reader = PdfReader("policy_baggage.pdf")
        text = "".join([p.extract_text() for p in reader.pages])
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        
        collection = _client.collections.get("PAAPolicy")
        count = collection.aggregate.over_all(total_count=True).total_count
        if count == 0:
            objs = [DataObject(properties={"content": c}, vector=EMBEDDING_MODEL.encode(c).tolist()) for c in chunks]
            collection.data.insert_many(objs)
    return True

# --- 3. TOOLS (PDF & XML) ---

def search_baggage_policy(query: str):
    """Searches the PAA baggage policy PDF."""
    w_client = get_weaviate_client()
    col = w_client.collections.get("PAAPolicy")
    res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=2)
    return "\n".join([o.properties['content'] for o in res.objects]) if res.objects else "No policy found."

def get_flight_status(flight_num: str, date: str = None):
    """Parses flight_records.xml to find the status of a specific flight."""
    if not os.path.exists("flight_records.xml"):
        return "Flight records file not found."
    
    try:
        tree = ET.parse("flight_records.xml")
        root = tree.getroot()
        
        # Flight number clean-up (e.g., SV726)
        flight_num = flight_num.strip().upper()
        
        for flight in root.findall('flight'):
            xml_flight_num = flight.find('number').text.strip().upper()
            xml_date = flight.find('date').text.strip()
            
            if xml_flight_num == flight_num:
                # Agar date di gayi hai toh match karo, warna flight number se return karo
                if date and date not in xml_date:
                    continue
                
                status = flight.find('status').text
                dep = flight.find('departure').text
                arr = flight.find('arrival').text
                return f"Flight {flight_num} on {xml_date}: Status is {status}. Departure: {dep}, Arrival: {arr}."
        
        return f"No record found for flight {flight_num} on the requested date."
    except Exception as e:
        return f"Error reading XML: {str(e)}"

# --- 4. OPENAI AGENT ORCHESTRATOR ---



def run_agent(user_input):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_baggage_policy",
                "description": "Get baggage weight limits and airport rules from the PDF.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_flight_status",
                "description": "Check flight status, departure, and arrival from the XML records.",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "flight_num": {"type": "string", "description": "e.g., SV726"},
                        "date": {"type": "string", "description": "Optional date e.g., 11 Nov 2025"}
                    },
                    "required": ["flight_num"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a PAA expert. Use 'search_baggage_policy' for luggage queries and 'get_flight_status' for flight info from XML. Always provide data-driven answers."},
        {"role": "user", "content": user_input}
    ]

    response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            if name == "search_baggage_policy":
                result = search_baggage_policy(args['query'])
            elif name == "get_flight_status":
                result = get_flight_status(args.get('flight_num'), args.get('date'))
            
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": name, "content": result})
        
        final_res = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final_res.choices[0].message.content
    
    return msg.content

# --- 5. UI ---
st.set_page_config(page_title="PAA Multi-Tool Agent", page_icon="✈️")
st.title("✈️ PAA Intelligent Agent (PDF + XML)")

w_client = get_weaviate_client()
ingest_data(w_client)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Asalam-o-Alaikum! Ask me about SV726 status or baggage policies."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Check status of SV726 on 11 Nov..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.spinner("Agent is checking records..."):
        ans = run_agent(prompt)
        with st.chat_message("assistant"): st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
