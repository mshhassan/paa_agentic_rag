import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
import json
import re
import warnings
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor # For Parallel Execution

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

# --- 2. SUB-AGENT RAG FUNCTIONS ---

def flight_inquiry_agent(query):
    with st.status("✈️ Flight Agent searching AODB...", expanded=False):
        query_vector = EMBEDDING_MODEL.encode(query).tolist()
        coll = W_CLIENT.collections.get("PAAFlightStatus")
        match = re.search(r'([A-Z]{2}\d{2,4})', query.upper())
        filters = Filter.by_property("flight_num").equal(match.group(1)) if match else None
        response = coll.query.near_vector(near_vector=query_vector, limit=2, filters=filters)
        return "\n".join([o.properties['content'] for o in response.objects]) if response.objects else "No flight data found."

def policy_documentation_agent(query):
    with st.status("📄 Policy Agent checking Documentation...", expanded=False):
        query_vector = EMBEDDING_MODEL.encode(query).tolist()
        coll = W_CLIENT.collections.get("PAAPolicy")
        response = coll.query.near_vector(near_vector=query_vector, limit=3)
        return "\n".join([o.properties['content'] for o in response.objects]) if response.objects else "No policy data found."

def web_query_agent(query):
    with st.status("🌐 Web Agent crawling PAA links...", expanded=False):
        query_vector = EMBEDDING_MODEL.encode(query).tolist()
        coll = W_CLIENT.collections.get("PAAWebLink")
        response = coll.query.near_vector(near_vector=query_vector, limit=2)
        return "\n".join([o.properties['content'] for o in response.objects]) if response.objects else "No web links found."

# --- 3. PARALLEL EXECUTION HANDLER ---

def execute_agents_parallel(tool_calls):
    """Executes multiple agent calls at the same time."""
    results = []
    # Map function names to actual functions
    agent_map = {
        "flight_inquiry_agent": flight_inquiry_agent,
        "policy_documentation_agent": policy_documentation_agent,
        "web_query_agent": web_query_agent
    }

    with ThreadPoolExecutor() as executor:
        # Create a list of future tasks
        future_to_tool = {
            executor.submit(agent_map[tc.function.name], json.loads(tc.function.arguments).get('query')): tc 
            for tc in tool_calls
        }
        
        for future in future_to_tool:
            tool_call = future_to_tool[future]
            try:
                data = future.result()
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": data
                })
            except Exception as e:
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": f"Error in agent: {str(e)}"
                })
    return results

# --- 4. MASTER SUPERVISOR LOGIC ---

def supervisor_agent(user_input):
    tools = [
        {"type": "function", "function": {"name": "flight_inquiry_agent", "description": "Flight timing and status.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "policy_documentation_agent", "description": "Baggage rules and PAA policies.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "web_query_agent", "description": "PAA website links and general info.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ]

    messages = [{"role": "system", "content": "You are the PAA Master Supervisor. Delegate tasks to specialized agents. You can call MULTIPLE agents if the query requires different info (e.g., flight status AND baggage rules). Today is Dec 2025."}]
    
    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    # Supervisor identifies required agents
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        
        # Trigger Parallel Execution
        with st.status("🤖 Supervisor delegating tasks in parallel...", expanded=True) as status:
            tool_results = execute_agents_parallel(msg.tool_calls)
            for res in tool_results:
                messages.append(res)
            status.update(label="✅ Sub-agents completed their tasks!", state="complete", expanded=False)
        
        # Final response synthesis
        final_res = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final_res.choices[0].message.content
    
    return msg.content

# --- 5. UI SETUP ---

st.set_page_config(page_title="PAA Multi-Agent System", page_icon="🏢", layout="wide")
st.title("🏢 PAA Intelligent Multi-Agent System")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Agent Controls")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    st.write("---")
    st.write("**Active Agents:**")
    st.write("- 👮 Supervisor Agent")
    st.write("- ✈️ Flight Inquiry Agent")
    st.write("- 📄 Policy Agent")
    st.write("- 🌐 Web Query Agent")

# Display History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input
if prompt := st.chat_input("Ex: Status of SV726 and what are the baggage rules?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = supervisor_agent(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
