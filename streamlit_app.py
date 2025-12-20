import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
import json
import re
import warnings
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

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

# --- 2. SUB-AGENTS (The "Workers") ---

def flight_inquiry_agent(query):
    """Sub-Agent: AODB XML Expert - Specialized in re-formatted search"""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAFlightStatus")
    
    # Extracting Flight Number if present
    match = re.search(r'([A-Z]{2}\d{2,4})', query.upper())
    f_filter = Filter.by_property("flight_num").equal(match.group(1)) if match else None
    
    # Using a slightly higher limit for better vector matching
    response = coll.query.near_vector(near_vector=query_vector, limit=5, filters=f_filter)
    
    if not response.objects:
        return f"⚠️ No flight records found in AODB for the search criteria: {query}"
    
    return "\n".join([o.properties.get('content', '') for o in response.objects])

def policy_documentation_agent(query):
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAPolicy")
    response = coll.query.near_vector(near_vector=query_vector, limit=3)
    return "\n".join([o.properties.get('content', '') for o in response.objects]) if response.objects else "No policy found."

def web_query_agent(query):
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAWebLink")
    response = coll.query.near_vector(near_vector=query_vector, limit=3)
    return "\n".join([f"Link: {o.properties.get('url_href')} - {o.properties.get('content')}" for o in response.objects]) if response.objects else "No links found."

# --- 3. MASTER SUPERVISOR & PARALLEL EXECUTION ---

def supervisor_agent(user_input):
    tools = [
        {"type": "function", "function": {
            "name": "flight_inquiry_agent", 
            "description": "Queries flight timing/status. MANDATORY: Reformat the search string to 'Flight [Num] on [YYYY-MM-DD]' to match database format.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }},
        {"type": "function", "function": {
            "name": "policy_documentation_agent", 
            "description": "Queries baggage rules and official PAA PDF docs.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }},
        {"type": "function", "function": {
            "name": "web_query_agent", 
            "description": "Queries PAA official website links.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }}
    ]

    # Enhanced System Prompt for Weather and Reformatting
    system_prompt = """You are the PAA Master Supervisor. 
    1. WEATHER: If a user asks for weather, provide the information from your internal knowledge. Do NOT use tools for weather unless it is specifically about airport weather policies.
    2. FLIGHT REFORMATTING: The database stores dates in ISO format (YYYY-MM-DD). If a user says '11 Nov', you MUST call flight_inquiry_agent with '2025-11-11'.
    3. MULTI-TASKING: Call agents in parallel if the user has multiple requests.
    4. Today is Dec 21, 2025. Be helpful and professional."""

    messages = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    # Step 1: Supervisor Analysis & Tool Routing
    response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto")
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        
        with st.status("🚀 Supervisor Dispatching Specialized Agents...", expanded=True) as status_box:
            agent_map = {
                "flight_inquiry_agent": (flight_inquiry_agent, "✈️ Flight Agent"),
                "policy_documentation_agent": (policy_documentation_agent, "📄 Policy Agent"),
                "web_query_agent": (web_query_agent, "🌐 Web Agent")
            }
            
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(agent_map[tc.function.name][0], json.loads(tc.function.arguments).get('query')): tc for tc in msg.tool_calls}
                
                for future in futures:
                    tc = futures[future]
                    result_data = future.result()
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result_data})
                    st.write(f"✅ {agent_map[tc.function.name][1]} has finished.")

            status_box.update(label="✅ Sub-agents processing complete!", state="complete", expanded=False)

        # Final Synthesis including LLM's own knowledge (for weather)
        final_res = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final_res.choices[0].message.content
    
    return msg.content

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="PAA Multi-Agent System", page_icon="🏢", layout="wide")
st.title("🏢 PAA Master Supervisor AI")



if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("**Status:** System Ready 🟢")

# Display History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input
if prompt := st.chat_input("Ex: Weather in Islamabad and baggage for PIA?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = supervisor_agent(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
