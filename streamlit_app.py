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

# --- 2. IMPROVED SEARCH TOOL ---

def paa_hybrid_search(query_text: str):
    """Searches Weaviate with flexible date handling and flight filters."""
    client = get_weaviate_client()
    query_vector = EMBEDDING_MODEL.encode(query_text).tolist()
    
    results_context = ""
    
    # Extract Flight Number (e.g., SV726)
    flight_match = re.search(r'([A-Z]{2}\d{2,4})', query_text.upper())
    
    collections = {
        "PAAFlightStatus": ["content", "flight_num", "status"],
        "PAAPolicy": ["content", "source"],
        "PAAWebLink": ["content", "url_href"]
    }

    for coll_name, props in collections.items():
        try:
            collection = client.collections.get(coll_name)
            
            # Specialized Search for Flights
            if coll_name == "PAAFlightStatus" and flight_match:
                target_flight = flight_match.group(1)
                # Try Filter First
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=3,
                    filters=Filter.by_property("flight_num").equal(target_flight),
                    return_properties=props
                )
                # If filter gives 0 results, fallback to pure Vector Search for that flight
                if not response.objects:
                    response = collection.query.near_vector(
                        near_vector=query_vector,
                        limit=3,
                        return_properties=props
                    )
            else:
                # Standard Search for Policies
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=2,
                    return_properties=props
                )

            for obj in response.objects:
                content = obj.properties.get("content", "")
                results_context += f"\n[Collection: {coll_name}] {content}\n"
                
        except Exception as e:
            continue

    return results_context if results_context else "No specific records found in database for this query."

# --- 3. AGENT LOGIC ---

def run_paa_agent(user_input):
    tools = [{
        "type": "function",
        "function": {
            "name": "paa_hybrid_search",
            "description": "Fetch flight status, baggage rules, and PAA policies from the vector database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "The search term including flight number or policy topic."}
                },
                "required": ["query_text"]
            }
        }
    }]

    messages = [
        {"role": "system", "content": "You are the PAA Intelligent Assistant. Use 'paa_hybrid_search' for ALL queries. If a user asks for a flight like SV726 on a specific date, search for it. If the database returns multiple dates, pick the one closest to the user's request. Today is Dec 2025."}
    ]

    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})
    
    messages.append({"role": "user", "content": user_input})

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )
    
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            search_result = paa_hybrid_search(args.get('query_text'))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": "paa_hybrid_search", "content": search_result})
        
        final = client_openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final.choices[0].message.content
    
    return msg.content

# --- 4. UI ---
st.set_page_config(page_title="PAA AI Agent", page_icon="✈️")
st.title("✈️ PAA Intelligent Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Flight status of SV726?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner("Accessing Records..."):
        answer = run_paa_agent(prompt)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"): st.markdown(answer)
