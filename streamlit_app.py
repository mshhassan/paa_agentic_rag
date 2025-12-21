import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import json

# --- 1. CONFIG & INITIALIZATION ---
st.set_page_config(page_title="PAA Enterprise Intelligence", layout="wide")

# Initialize Session States (VVIP for avoiding Blank Page)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trace" not in st.session_state:
    st.session_state.trace = []

client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    return model

EMBED = load_resources()
THRESHOLD = 0.7 

# --- 2. WEAVIATE RETRIEVER ---
def fetch_from_weaviate(query, collection_name):
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=st.secrets["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
        )
        coll = client.collections.get(collection_name)
        
        res = coll.query.near_vector(
            near_vector=EMBED.encode(query).tolist(), 
            limit=5,
            return_properties=["content"]
        )
        client.close()
        
        if not res.objects:
            return "No specific record found."
            
        return "\n".join([o.properties['content'] for o in res.objects])
    except Exception as e:
        return f"Error: {str(e)}"

# --- 3. AGENTIC ENGINE ---
def run_paa_engine(query):
    st.session_state.trace = [] # Reset trace for new query
    st.session_state.trace.append(f"🔍 **Analyzing Query:** {query}")
    
    # Routing Decision
    analysis_prompt = f"""
    Analyze query: "{query}"
    Scores (0-1): XML (Flight info), Web (Links), Docs (Baggage/Policy).
    Return JSON: {{"XML": score, "Web": score, "Docs": score}}
    """
    
    resp = client_openai.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={"type":"json_object"}, 
        messages=[{"role":"system", "content":"You are a PAA Supervisor Agent."}, {"role":"user","content":analysis_prompt}]
    )
    scores = json.loads(resp.choices[0].message.content)

    context = ""
    mapping = {"XML": "PAAWeb", "Web": "PAAWebLink", "Docs": "PAAPolicy"}
    
    for key, score in scores.items():
        if score >= THRESHOLD:
            st.session_state.trace.append(f"📡 **{key} Agent:** Active (Score {score})")
            retrieved_text = fetch_from_weaviate(query, mapping[key])
            context += f"\n--- {key} DATA ---\n{retrieved_text}\n"
        else:
            st.session_state.trace.append(f"⚪ **{key} Agent:** Bypassed")

    # Final Answer
    system_instruction = f"""
    You are the PAA (Pakistan Airports Authority) Official Assistant.
    STRICT RULE: Only use the provided context to answer. 
    If data is missing, say 'Information not found in current records'.
    CONTEXT: {context}
    """
    
    ans_resp = client_openai.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": system_instruction}] + st.session_state.messages[-5:] + [{"role": "user", "content": query}]
    )
    answer = ans_resp.choices[0].message.content
    
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- 4. UI LAYOUT ---
st.title("✈️ PAA AI: Enterprise Intelligence")

# Sidebar for Trace
with st.sidebar:
    st.header("🔍 Agentic Trace")
    for t in st.session_state.trace:
        st.write(t)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How can I help you today?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Consulting PAA Agents..."):
        run_paa_engine(prompt)
    st.rerun()
