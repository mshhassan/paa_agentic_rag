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

# --- 2. IMPROVED SUB-AGENTS ---

def flight_inquiry_agent(query):
    """Sub-Agent: AODB Specialist"""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAFlightStatus")
    
    # Exact flight number match is crucial for AODB
    match = re.search(r'([A-Z]{2}\d{2,4})', query.upper())
    f_filter = Filter.by_property("flight_num").equal(match.group(1)) if match else None
    
    response = coll.query.near_vector(near_vector=query_vector, limit=5, filters=f_filter)
    if not response.objects:
        return "DATABASE_EMPTY: No specific flight record found for this query."
    return "\n".join([o.properties.get('content', '') for o in response.objects])

def policy_documentation_agent(query):
    """Sub-Agent: PDF Policy Expert (Baggage, NOTAMs, Lost & Found)"""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAPolicy")
    # Increase limit to capture more context
    response = coll.query.near_vector(near_vector=query_vector, limit=5)
    return "\n".join([o.properties.get('content', '') for o in response.objects]) if response.objects else "No documents found."

def web_query_agent(query):
    """Sub-Agent: Web URL Expert"""
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    coll = W_CLIENT.collections.get("PAAWebLink")
    response = coll.query.near_vector(near_vector=query_vector, limit=8)
    
    links = []
    for o in response.objects:
        url = o.properties.get('url_href', '#')
        text = o.properties.get('content', '')
        links.append(f"🔗 [{text}]({url})")
    
    return "\n".join(links) if links else "No web links found."

# --- 3. MASTER SUPERVISOR ---

def supervisor_agent(user_input):
    tools = [
        {"type": "function", "function": {
            "name": "flight_inquiry_agent", 
            "description": "Get flight timings. Reformat date to YYYY-MM-DD.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }},
        {"type": "function", "function": {
            "name": "policy_documentation_agent", 
            "description": "Baggage rules, NOTAMs, and Lost & Found procedures.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }},
        {"type": "function", "function": {
            "name": "web_query_agent", 
            "description": "Official PAA links for NOTAMs and Lost and Found.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }}
    ]

    system_msg = """You are the PAA Master Supervisor.
    1. WEATHER: Provide typical weather from your own knowledge.
    2. FLIGHTS: Always reformat dates to ISO format.
    3. SEARCH LOGIC: If a user asks for 'NOTAMs' or 'Lost and Found', you MUST call 'web_query_agent' AND 'policy_documentation_agent' with the EXACT keywords 'NOTAM' or 'Lost and Found'.
    4. LINKS: If the web agent returns links, display them as clickable Markdown links.
    Today is Dec 21, 2025."""

    messages = [{"role": "system", "content": system_msg}]
    for m in st.session_state.messages[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    response = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        with st.status("🤖 Supervisor Delegating Parallel Tasks...", expanded=True):
            agent_map = {
                "flight_inquiry_agent": flight_inquiry_agent,
                "policy_documentation_agent": policy_documentation_agent,
                "web_query_agent": web_query_agent
            }
            
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(agent_map[tc.function.name], json.loads(tc.function.arguments).get('query')): tc for tc in msg.tool_calls}
                for future in futures:
                    tc = futures[future]
                    result = future.result()
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result})
        
        final_res = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final_res.choices[0].message.content
    
    return msg.content

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="PAA Master Agent", layout="wide")
st.title("🏢 PAA Intelligent Master Agent")



if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ex: Show me the link for NOTAMs and flight SV726?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        answer = supervisor_agent(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
