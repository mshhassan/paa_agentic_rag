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

# --- 2. SUB-AGENTS (The "Workers") ---

def flight_inquiry_agent(query):
    """Sub-Agent: AODB XML Expert - Using ISO Date Format"""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAFlightStatus")
    
    # Priority search for Flight Number
    match = re.search(r'([A-Z]{2}\d{2,4})', query.upper())
    f_filter = Filter.by_property("flight_num").equal(match.group(1)) if match else None
    
    response = coll.query.near_vector(near_vector=query_vector, limit=5, filters=f_filter)
    return "\n".join([o.properties.get('content', '') for o in response.objects]) if response.objects else "No flight records found."

def policy_documentation_agent(query):
    """Sub-Agent: PDF Policy Expert (Baggage/Claims)"""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAPolicy")
    response = coll.query.near_vector(near_vector=query_vector, limit=3)
    return "\n".join([o.properties.get('content', '') for o in response.objects]) if response.objects else "No policy documents found."

def web_query_agent(query):
    """Sub-Agent: Web Links (Lost & Found, Contact, etc.)"""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAWebLink")
    # Using broader search for links
    response = coll.query.near_vector(near_vector=query_vector, limit=5)
    
    links = []
    for o in response.objects:
        url = o.properties.get('url_href', '#')
        text = o.properties.get('content', '')
        links.append(f"🔗 [{text}]({url})")
    
    return "\n".join(links) if links else "No relevant web links found."

# --- 3. PARALLEL EXECUTION HANDLER ---

def execute_parallel(tool_calls):
    agent_map = {
        "flight_inquiry_agent": (flight_inquiry_agent, "✈️ Flight Agent"),
        "policy_documentation_agent": (policy_documentation_agent, "📄 Policy Agent"),
        "web_query_agent": (web_query_agent, "🌐 Web Agent")
    }
    
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(agent_map[tc.function.name][0], json.loads(tc.function.arguments).get('query')): tc for tc in tool_calls}
        for future in futures:
            tc = futures[future]
            try:
                data = future.result()
                results.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": data})
                st.write(f"✅ {agent_map[tc.function.name][1]} has processed data.")
            except Exception as e:
                results.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": f"Error: {str(e)}"})
    return results

# --- 4. MASTER SUPERVISOR ---

def supervisor_agent(user_input):
    tools = [
        {"type": "function", "function": {
            "name": "flight_inquiry_agent", 
            "description": "Get flight status. Format query as 'Flight [No] on [YYYY-MM-DD]'. Example: SV726 on 2025-11-11.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }},
        {"type": "function", "function": {
            "name": "policy_documentation_agent", 
            "description": "Baggage rules, lost bags, and airport policies.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }},
        {"type": "function", "function": {
            "name": "web_query_agent", 
            "description": "Official PAA links for Lost and Found, Contact Us, and Information.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }}
    ]

    # Critical Instructions for Reformatting and Logic
    system_msg = """You are the PAA Master Supervisor. 
    1. WEATHER: Provide current or typical weather based on your training data immediately. Do not use tools for weather.
    2. REFORMATTING: User dates (e.g., 11 Nov) MUST be changed to ISO (2025-11-11) for the flight tool.
    3. LOST & FOUND: If a user mentions lost items, you MUST call 'web_query_agent' with query 'lost and found' AND 'policy_documentation_agent'.
    4. PARALLEL: Use multiple tools if the user asks multiple things.
    Current Date: Dec 21, 2025."""

    messages = [{"role": "system", "content": system_msg}]
    for m in st.session_state.messages[-8:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        with st.status("🤖 Supervisor Orchestrating Parallel Agents...", expanded=True):
            tool_outputs = execute_parallel(msg.tool_calls)
            messages.extend(tool_outputs)
        
        final_res = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final_res.choices[0].message.content
    
    return msg.content

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="PAA Intelligent Supervisor", layout="wide")
st.title("🏢 PAA Intelligent Master Agent")



if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ex: Status of SV726 and how to report a lost bag?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        answer = supervisor_agent(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
