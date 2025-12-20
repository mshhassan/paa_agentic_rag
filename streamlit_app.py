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

# --- 2. SPECIALIZED SUB-AGENTS (The "Workers") ---

def flight_inquiry_agent(query):
    """Sub-Agent: AODB XML Expert"""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAFlightStatus")
    
    # Robust Flight Number Extraction
    match = re.search(r'([A-Z]{2}\d{2,4})', query.upper())
    f_filter = Filter.by_property("flight_num").equal(match.group(1)) if match else None
    
    # Near Vector + Optional Filter
    response = coll.query.near_vector(near_vector=query_vector, limit=3, filters=f_filter)
    
    if not response.objects:
        return "⚠️ No flight records found for the specific flight number/date in AODB."
    
    return "\n".join([o.properties.get('content', '') for o in response.objects])

def policy_documentation_agent(query):
    """Sub-Agent: PDF Policy Expert"""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAPolicy")
    response = coll.query.near_vector(near_vector=query_vector, limit=3)
    
    if not response.objects:
        return "⚠️ No baggage policy details found in the provided documentation."
    
    return "\n".join([o.properties.get('content', '') for o in response.objects])

def web_query_agent(query):
    """Sub-Agent: Website Link Expert"""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAWebLink")
    response = coll.query.near_vector(near_vector=query_vector, limit=3)
    
    if not response.objects:
        return "⚠️ No relevant PAA web links or official resources found."
    
    return "\n".join([f"Link: {o.properties.get('url_href')} - {o.properties.get('content')}" for o in response.objects])

# --- 3. MASTER SUPERVISOR & PARALLEL EXECUTION ---

def supervisor_agent(user_input):
    tools = [
        {"type": "function", "function": {"name": "flight_inquiry_agent", "description": "Queries flight status, timing, and AODB data.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "policy_documentation_agent", "description": "Queries baggage rules and official PAA PDF docs.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "web_query_agent", "description": "Queries PAA official website links and contact info.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ]

    messages = [{"role": "system", "content": "You are the PAA Master Supervisor. For ANY request about flights, baggage, or airport info, you MUST use the sub-agents. You can call them in parallel for multi-part questions. Today is Dec 2025. If info is missing, explain why politely."}]
    
    for m in st.session_state.messages[-6:]: # History Retention (last 6 messages)
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    # Step 1: Supervisor Analysis
    response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        
        # Step 2: Parallel Dispatch
        with st.status("🚀 Supervisor Dispatching Parallel Agents...", expanded=True) as status_box:
            agent_map = {
                "flight_inquiry_agent": (flight_inquiry_agent, "✈️ Flight Agent"),
                "policy_documentation_agent": (policy_documentation_agent, "📄 Policy Agent"),
                "web_query_agent": (web_query_agent, "🌐 Web Agent")
            }
            
            with ThreadPoolExecutor() as executor:
                futures = []
                for tc in msg.tool_calls:
                    func, label = agent_map[tc.function.name]
                    args = json.loads(tc.function.arguments).get('query')
                    st.write(f"{label} is processing: '{args}'")
                    futures.append(executor.submit(func, args))
                
                # Collecting results
                for i, future in enumerate(futures):
                    result_data = future.result()
                    messages.append({
                        "role": "tool",
                        "tool_call_id": msg.tool_calls[i].id,
                        "name": msg.tool_calls[i].function.name,
                        "content": result_data
                    })
            
            status_box.update(label="✅ All Agents Responded!", state="complete", expanded=False)

        # Step 3: Final Response Synthesis
        final_res = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final_res.choices[0].message.content
    
    return msg.content

# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="PAA Multi-Agent AI", page_icon="🏢", layout="wide")
st.title("🏢 PAA Master Supervisor Agent")



if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Agent Settings")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.write("---")
    st.markdown("**Supervisor:** Active 🟢\n\n**Sub-Agents:**\n- Flight Inquiry ✈️\n- Policy Documentation 📄\n- Web Query 🌐")

# Display Chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input
if prompt := st.chat_input("Ask me anything about PAA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = supervisor_agent(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
