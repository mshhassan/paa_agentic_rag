import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
import json
import re
import warnings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# --- 1. CORE CONFIGURATION ---
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

# --- 2. SPECIALIZED SUB-AGENT FUNCTIONS (RAG TOOLS) ---

def flight_inquiry_agent(query):
    """Sub-Agent for Flight Status Records."""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAFlightStatus")
    
    # Extract flight ID for filtering
    match = re.search(r'([A-Z]{2}\d{2,4})', query.upper())
    filters = Filter.by_property("flight_num").equal(match.group(1)) if match else None
    
    response = coll.query.near_vector(near_vector=query_vector, limit=2, filters=filters)
    return "\n".join([o.properties['content'] for o in response.objects]) if response.objects else "No flight data found."

def policy_documentation_agent(query):
    """Sub-Agent for Baggage & PAA Documentation."""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAPolicy")
    response = coll.query.near_vector(near_vector=query_vector, limit=3)
    return "\n".join([o.properties['content'] for o in response.objects]) if response.objects else "No policy data found."

def web_query_agent(query):
    """Sub-Agent for PAA Website Links & General Info."""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAWebLink")
    response = coll.query.near_vector(near_vector=query_vector, limit=2)
    return "\n".join([o.properties['content'] for o in response.objects]) if response.objects else "No web links found."

# --- 3. MASTER SUPERVISOR LOGIC ---

def supervisor_agent(user_input):
    # Tool definitions for the Supervisor
    tools = [
        {"type": "function", "function": {"name": "flight_inquiry_agent", "description": "Get flight timing, status, and carousel info.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "policy_documentation_agent", "description": "Get baggage rules, weight limits, and PAA documents.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "web_query_agent", "description": "Get official PAA website links and contact info.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ]

    # Prepare conversation history
    messages = [{"role": "system", "content": "You are the PAA Master Supervisor. Your job is to delegate tasks to specialized agents. If the user asks something general, reply directly. If it's about flights, baggage, or links, call the appropriate sub-agent(s). You can call multiple agents if needed. Today is Dec 2025."}]
    
    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})
    
    messages.append({"role": "user", "content": user_input})

    # Supervisor decides who to call
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments).get('query')
            
            # Routing to Sub-Agents
            if func_name == "flight_inquiry_agent":
                result = flight_inquiry_agent(args)
            elif func_name == "policy_documentation_agent":
                result = policy_documentation_agent(args)
            elif func_name == "web_query_agent":
                result = web_query_agent(args)
            
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": func_name, "content": result})
        
        # Final synthesis by Supervisor
        final_res = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final_res.choices[0].message.content
    
    return msg.content

# --- 4. STREAMLIT INTERFACE ---

st.set_page_config(page_title="PAA Master Agent", page_icon="🏢", layout="wide")

st.title("🏢 PAA Master Supervisor System")
st.info("The Supervisor Agent will route your query to specialized Flight, Policy, or Web agents.")



if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for History Clear
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# Display Chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("How much baggage is allowed on SV726?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner("Supervisor delegating tasks..."):
        ans = supervisor_agent(prompt)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(ans)
