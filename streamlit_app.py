import streamlit as st
import google.generativeai as genai
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject 
from sentence_transformers import SentenceTransformer 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from pypdf import PdfReader
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
try:
    # Get Keys from Streamlit Secrets
    WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud" 
    WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"] 
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"] 
    
    # Simple configuration
    genai.configure(api_key=GEMINI_KEY)
except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_MODEL = load_embedding_model()

# --- 2. WEAVIATE SETUP ---
@st.cache_resource(show_spinner="Connecting to Database...")
def get_client():
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL, 
            auth_credentials=Auth.api_key(WEAVIATE_KEY)
        )
        # Collection create karein agar nahi hai
        if not client.collections.exists("PAAPolicy"):
            client.collections.create(
                name="PAAPolicy",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[Property(name="content", data_type=DataType.TEXT)]
            )
        return client
    except Exception as e:
        st.error(f"Weaviate Connection Failed: {e}")
        return None

@st.cache_resource(show_spinner="Reading PDF Policy...")
def ingest_data(_client):
    if _client and os.path.exists("policy_baggage.pdf"):
        try:
            reader = PdfReader("policy_baggage.pdf")
            text = "".join([p.extract_text() for p in reader.pages])
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_text(text)
            
            collection = _client.collections.get("PAAPolicy")
            count = collection.aggregate.over_all(total_count=True).total_count
            
            if count == 0:
                objs = [DataObject(properties={"content": c}, vector=EMBEDDING_MODEL.encode(c).tolist()) for c in chunks]
                collection.data.insert_many(objs)
        except Exception as e:
            st.warning(f"Ingestion Warning: {e}")
    return True

# --- 3. RETRIEVAL TOOL ---
def get_baggage_policy(query: str):
    """Answers questions specifically about baggage rules and airline policy."""
    client = get_client()
    if not client: return "Database connection error."
    
    col = client.collections.get("PAAPolicy")
    res = col.query.near_vector(near_vector=EMBEDDING_MODEL.encode(query).tolist(), limit=2)
    
    if res.objects:
        return "\n".join([obj.properties['content'] for obj in res.objects])
    return "No specific baggage policy found in the records."

# --- 4. AGENT LOGIC (STABLE ENDPOINT) ---
def agent_response(user_input: str):
    # 'gemini-1.5-flash' is the correct ID. 
    # If it fails, try 'gemini-1.5-flash-latest'
    try:
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            tools=[get_baggage_policy],
            system_instruction="You are a PAA (Pakistan Aviation Authority) Expert. Always check the baggage policy tool before answering questions about luggage or rules."
        )
        
        # Start chat with automatic function calling enabled
        chat = model.start_chat(enable_automatic_function_calling=True)
        response = chat.send_message(user_input)
        return response.text
    except Exception as e:
        return f"Model Error: {str(e)}"

# --- 5. UI INTERFACE ---
st.set_page_config(page_title="PAA Agent", page_icon="🇵🇰")
st.title("🇵🇰 PAA Agentic AI")

client = get_client()
ingest_data(client)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you with PAA services today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.write(msg["content"])

if prompt := st.chat_input("Pia baggage policy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.write(prompt)
    
    with st.spinner("Processing with Agent..."):
        answer = agent_response(prompt)
        with st.chat_message("assistant"): 
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
