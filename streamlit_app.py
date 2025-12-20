import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject 
from sentence_transformers import SentenceTransformer 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from pypdf import PdfReader
import os
import json
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
try:
    WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud" 
    WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"] 
    # Lazmi: Streamlit secrets mein 'OPENAI_API_KEY' add karein
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] 
    
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
except KeyError as e:
    st.error(f"Secret Missing: {e}. Please add OPENAI_API_KEY to Streamlit Secrets.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_MODEL = load_embedding_model()

# --- 2. WEAVIATE SETUP ---
@st.cache_resource(show_spinner="Connecting to Weaviate...")
def get_weaviate_client():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    if not client.collections.exists("PAAPolicy"):
        client.collections.create(
            name="PAAPolicy",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[Property(name="content", data_type=DataType.TEXT)]
        )
    return client

@st.cache_resource(show_spinner="Reading PDF...")
def ingest_data(_client):
    if os.path.exists("policy_baggage.pdf"):
        reader = PdfReader("policy_baggage.pdf")
        text = "".join([p.extract_text() for p in reader.pages])
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        
        collection = _client.collections.get("PAAPolicy")
        count = collection.aggregate.over_all(total_count=True).total_count
        if count == 0:
            objs = [DataObject(properties={"content": c}, vector=EMBEDDING_MODEL.encode(c).tolist()) for c in chunks]
            collection.data.insert_many(objs)
    return True

# --- 3. RETRIEVAL TOOL ---
def search_baggage_policy(query: str):
    """Searches the PAA baggage policy PDF for specific rules and weight limits."""
    w_client = get_weaviate_client()
    col = w_client.collections.get("PAAPolicy")
    res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=2)
    if res.objects:
        return "\n".join([o.properties['content'] for o in res.objects])
    return "No relevant baggage policy found in the documents."

# --- 4. OPENAI AGENT LOGIC ---


def run_openai_agent(user_input):
    # Tool definition for OpenAI format
    tools = [{
        "type": "function",
        "function": {
            "name": "search_baggage_policy",
            "description": "Get detailed baggage and luggage rules from the PAA PDF policy.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search term"}},
                "required": ["query"]
            }
        }
    }]

    messages = [
        {"role": "system", "content": "You are a professional PAA Supervisor. Use the provided search tool to get facts from the PDF before answering. Be polite and helpful."},
        {"role": "user", "content": user_input}
    ]

    try:
        # Step 1: Pehli call OpenAI ko tool decide karne ke liye
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini", # Sasta aur behtreen
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Step 2: Agar OpenAI ko data chahiye tool se
        if tool_calls:
            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_args = json.loads(tool_call.function.arguments)
                # Tool ko execute karein
                function_response = search_baggage_policy(function_args.get("query"))
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "search_baggage_policy",
                    "content": function_response,
                })
            
            # Step 3: Final response with data
            second_response = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            return second_response.choices[0].message.content
        
        return response_message.content

    except Exception as e:
        return f"OpenAI Error: {str(e)}"

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="PAA OpenAI Agent", page_icon="✈️")
st.title("✈️ PAA Agentic AI (Powered by OpenAI)")

w_client = get_weaviate_client()
ingest_data(w_client)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Asalam-o-Alaikum! How can I help you with PAA services today?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask about baggage weight or rules..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.spinner("AI is retrieving policy details..."):
        ans = run_openai_agent(prompt)
        with st.chat_message("assistant"): st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
