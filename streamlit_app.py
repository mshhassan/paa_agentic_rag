import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
import json
import re
import warnings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# --- 1. INITIALIZATION & RESOURCE LOADING ---
try:
    WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
    WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
except KeyError as e:
    st.error(f"Secret Missing: {e}")
    st.stop()

@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    return model, client

EMBEDDING_MODEL, W_CLIENT = load_resources()

# --- 2. THE MASTER INGESTOR (Operational Data) ---
def initialize_knowledge_base():
    """Wipes and re-creates all 3 pillars of the PAA Knowledge Base."""
    collections = {
        "PAAFlightStatus": "Flight records, timings, gates, and status.",
        "PAAPolicy": "PDF Documents, baggage rules, and passenger rights.",
        "PAAWebLink": "Official URLs for Lost & Found, NOTAMs, and Services."
    }
    
    for coll_name in collections:
        if W_CLIENT.collections.exists(coll_name):
            W_CLIENT.collections.delete(coll_name)
        W_CLIENT.collections.create(
            name=coll_name,
            properties=[Property(name="content", data_type=DataType.TEXT)],
            vectorizer_config=Configure.Vectorizer.none()
        )

    # A. Ingest Web Links (Operational Links)
    web_data = [
        {"content": "Lost and Found Baggage Procedures", "url": "https://www.paa.gov.pk/lost-found"},
        {"content": "NOTAMs and Aeronautical Information", "url": "https://www.paa.gov.pk/notams"},
        {"content": "Passenger Facilitation & Complaint Cell", "url": "https://www.paa.gov.pk/complaints"}
    ]
    web_coll = W_CLIENT.collections.get("PAAWebLink")
    for item in web_data:
        text = f"{item['content']} URL: {item['url']}"
        web_coll.data.insert(properties={"content": text}, vector=EMBEDDING_MODEL.encode(text).tolist())

    return "✅ Knowledge Base Initialized!"

# --- 3. SUB-AGENTS (The Specialized Workers) ---

def retrieval_agent(query, collection_name):
    """Generic retrieval for any PAA collection."""
    try:
        coll = W_CLIENT.collections.get(collection_name)
        # Using Hybrid search to prioritize keywords like 'Lost' or 'PK841'
        response = coll.query.hybrid(
            query=query, 
            vector=EMBEDDING_MODEL.encode(query).tolist(), 
            limit=3,
            alpha=0.5
        )
        return "\n".join([o.properties.get('content', '') for o in response.objects])
    except:
        return ""

# --- 4. MASTER SUPERVISOR (The Logic Brain) ---

def paa_supervisor(user_input):
    # Tools definition
    tools = [
        {"type": "function", "function": {"name": "get_flight_info", "description": "Search flight AODB data.", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}},
        {"type": "function", "function": {"name": "get_policy_info", "description": "Search passenger policies/PDFs.", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}},
        {"type": "function", "function": {"name": "get_web_links", "description": "Get official PAA website links.", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}
    ]

    system_prompt = """You are the PAA Operational Intelligence Agent.
    - If asked for links (Lost & Found, NOTAMs), use 'get_web_links'.
    - If asked for flights, use 'get_flight_info'. Mention Gates/Counters clearly.
    - For Weather in Islamabad: State it's approx 10-15°C (Dec average) if tools have no data.
    - Provide raw data as clickable links for URLs.
    Today: Dec 21, 2025."""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    
    response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            q = json.loads(tc.function.arguments).get('q')
            if tc.function.name == "get_flight_info": res = retrieval_agent(q, "PAAFlightStatus")
            elif tc.function.name == "get_policy_info": res = retrieval_agent(q, "PAAPolicy")
            else: res = retrieval_agent(q, "PAAWebLink")
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": res or "No data found."})
        
        final = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final.choices[0].message.content
    return msg.content

# --- 5. INTERFACE ---
st.set_page_config(page_title="PAA Operations", layout="wide")

with st.sidebar:
    st.title("⚙️ Admin Panel")
    if st.button("🔄 Sync PAA Data"):
        with st.spinner("Processing XML & Web Links..."):
            status = initialize_knowledge_base()
            st.success(status)

st.title("✈️ PAA.GOV.PK Operational RAG")
st.markdown("Query flight data, passenger policies, and official resources directly.")

if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("How do I contact Lost and Found and what is status of PK841?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        response = paa_supervisor(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
