import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
import json
import re
import warnings
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

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
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    return model, client

EMBEDDING_MODEL, W_CLIENT = load_resources()

# --- 2. LOGGING CONSOLE HELPER ---
def log_agent_activity(agent_name, message):
    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = []
    log_entry = f"[{agent_name.upper()}]: {message}"
    st.session_state.agent_logs.append(log_entry)
    with st.sidebar:
        st.code("\n".join(st.session_state.agent_logs[-15:]), language="bash")

# --- 3. SUB-AGENTS ---

def flight_inquiry_agent(query):
    log_agent_activity("Flight-Agent", f"Searching AODB for Detailed Info: {query}")
    coll = W_CLIENT.collections.get("PAAFlightStatus")
    
    # Hybrid search with 0.4 alpha to prioritize keywords like flight numbers
    response = coll.query.hybrid(
        query=query,
        vector=EMBEDDING_MODEL.encode(query).tolist(),
        limit=3,
        alpha=0.4
    )
    
    if not response.objects:
        log_agent_activity("Flight-Agent", "Result: Empty")
        return "No specific flight record found."
    
    # Context summary containing all resource allocations
    results = [o.properties.get('content', '') for o in response.objects]
    log_agent_activity("Flight-Agent", f"Retrieved context for {len(results)} records.")
    return "\n---\n".join(results)

def policy_documentation_agent(query):
    log_agent_activity("Policy-Agent", f"Reading PDF docs for: {query}")
    coll = W_CLIENT.collections.get("PAAPolicy")
    response = coll.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=3)
    return "\n".join([o.properties.get('content', '') for o in response.objects]) if response.objects else "No documents found."

def web_query_agent(query):
    log_agent_activity("Web-Agent", f"Checking PAA Web Links for: {query}")
    coll = W_CLIENT.collections.get("PAAWebLink")
    response = coll.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=5)
    links = [f"🔗 [{o.properties.get('content', '')}]({o.properties.get('url_href', '#')})" for o in response.objects]
    return "\n".join(links) if links else "No web links found."

# --- 4. MASTER SUPERVISOR ---

def supervisor_agent(user_input):
    log_agent_activity("Supervisor", "Routing query and expecting high-detail results...")
    
    tools = [
        {"type": "function", "function": {
            "name": "flight_inquiry_agent", 
            "description": "Get detailed flight info including Gate, Counters, Belts, and Status.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }},
        {"type": "function", "function": {
            "name": "policy_documentation_agent", 
            "description": "Baggage, NOTAMs, and Lost & Found procedures.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }},
        {"type": "function", "function": {
            "name": "web_query_agent", 
            "description": "Official PAA website links.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }}
    ]

    system_msg = """You are the PAA Master Supervisor.
    Your task is to provide COMPREHENSIVE flight information. 
    If the context provided by 'flight_inquiry_agent' contains:
    - Gate Identity
    - Check-in Counters
    - Baggage Reclaim/Belt
    - Stand or Status
    You MUST include these details in your final answer. If a resource is 'TBD' or 'N/A', state that it is not yet assigned.
    Format your output using bullet points for clarity.
    Today's Date: Dec 21, 2025."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_input}
    ]

    response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        agent_map = {
            "flight_inquiry_agent": flight_inquiry_agent,
            "policy_documentation_agent": policy_documentation_agent,
            "web_query_agent": web_query_agent
        }
        
        for tc in msg.tool_calls:
            arg_query = json.loads(tc.function.arguments).get('query')
            result = agent_map[tc.function.name](arg_query)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result})
        
        log_agent_activity("Supervisor", "Synthesizing detailed final answer...")
        final_res = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final_res.choices[0].message.content
    
    return msg.content

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="PAA Master Agent", layout="wide")

with st.sidebar:
    st.header("🕵️ Agent Logic Trace")
    st.info("AEDB Resource tracking enabled.")
    st.markdown("---")
    if st.button("Clear Console"):
        st.session_state.agent_logs = []
    st.write("Live Data Flow:")

st.title("🏢 PAA Intelligent Master Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ex: What is the gate and counter for flight CZ8069?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        answer = supervisor_agent(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
