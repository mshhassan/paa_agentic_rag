import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
import os
import json
import re
import warnings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
try:
    WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
    WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
except KeyError as e:
    st.error(f"Secret Missing: {e}. Please add them in Streamlit Secrets.")
    st.stop()

@st.cache_resource
def load_models():
    return SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_MODEL = load_models()

@st.cache_resource
def get_weaviate_client():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    return client

# --- 2. THE POWERFUL HYBRID SEARCH TOOL ---

def paa_hybrid_search(query_text: str):
    """Searches Weaviate collections using Hybrid Search (Vector + Filters)."""
    client = get_weaviate_client()
    query_vector = EMBEDDING_MODEL.encode(query_text).tolist()
    
    results_context = ""
    
    # 1. Check for Flight Number in Query
    flight_match = re.search(r'([A-Z]{2}\d{2,4})', query_text.upper())
    
    collections = {
        "PAAPolicy": ["content", "source"],
        "PAAFlightStatus": ["content", "flight_num", "status"],
        "PAAWebLink": ["content", "url_href"]
    }

    for coll_name, props in collections.items():
        try:
            collection = client.collections.get(coll_name)
            
            # Hybrid Logic
            if coll_name == "PAAFlightStatus" and flight_match:
                target_flight = flight_match.group(1)
                # Exact Filter for Flight Status
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=2,
                    filters=Filter.by_property("flight_num").equal(target_flight),
                    return_properties=props
                )
            else:
                # Pure Vector Search for Policy and Web
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=2,
                    return_properties=props
                )

            for obj in response.objects:
                content = obj.properties.get("content", "")
                results_context += f"\n[Source: {coll_name}] {content}\n"
                
        except Exception as e:
            continue

    return results_context if results_context else "No specific records found in database."

# --- 3. AGENT CORE ---

def run_paa_agent(user_input):
    # Defining the tool for OpenAI
    tools = [{
        "type": "function",
        "function": {
            "name": "paa_hybrid_search",
            "description": "Search PAA database for flight status, baggage policies, and airport rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "The search query, e.g., 'status of PK234' or 'baggage limits'"}
                },
                "required": ["query_text"]
            }
        }
    }]

    messages = [
        {"role": "system", "content": "You are the PAA Intelligent Assistant. Use the 'paa_hybrid_search' tool to find accurate information from our database. If a flight status is found, explain it clearly. If baggage rules are found, summarize them. Today is late 2025."}
    ]

    # Add chat history
    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})
    
    messages.append({"role": "user", "content": user_input})

    # First Call
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message

    # Handle Tool Call
    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            search_result = paa_hybrid_search(args.get('query_text'))
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "paa_hybrid_search",
                "content": search_result
            })
        
        # Second Call for final answer
        final_response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return final_response.choices[0].message.content
    
    return msg.content

# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="PAA AI Agent", page_icon="✈️", layout="centered")

st.title("✈️ PAA Intelligent Assistant")
st.markdown("Ask about **Flight Status**, **Baggage Policies**, or **Airport Queries**.")



if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat Input
if prompt := st.chat_input("How much extra baggage is allowed on PK234?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching PAA Database..."):
        answer = run_paa_agent(prompt)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
