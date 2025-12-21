import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import json

# --- 1. CONFIG & INITIALIZATION ---
st.set_page_config(page_title="PAA Enterprise Intelligence", layout="wide")

# Custom CSS for UI Matching (Black Trace Console & Dots)
st.markdown("""
    <style>
    .trace-box {
        background-color: #0e1117;
        color: #00ff00;
        padding: 15px;
        border-radius: 10px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.85rem;
        height: 600px;
        overflow-y: auto;
        border: 1px solid #333;
    }
    .agent-dot {
        height: 20px;
        width: 20px;
        border-radius: 50%;
        display: inline-block;
        margin-bottom: 5px;
    }
    .dot-active { background-color: #00e5ff; box-shadow: 0 0 10px #00e5ff; }
    .dot-inactive { background-color: #333; }
    .dot-supervisor { background-color: #ffcc00; }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "trace" not in st.session_state: st.session_state.trace = []
if "agent_status" not in st.session_state: 
    st.session_state.agent_status = {"XML": False, "Web": False, "Docs": False}

client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_resources():
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

EMBED = load_resources()
THRESHOLD = 0.5 

# --- 2. WEAVIATE RETRIEVER ---
def fetch_from_weaviate(query, collection_name):
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=st.secrets["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
        )
        coll = client.collections.get(collection_name)
        res = coll.query.near_vector(near_vector=EMBED.encode(query).tolist(), limit=3, return_properties=["content"])
        client.close()
        return "\n".join([o.properties['content'] for o in res.objects]) if res.objects else ""
    except: return ""

# --- 3. AGENTIC ENGINE ---
def run_paa_engine(query):
    st.session_state.trace = []
    st.session_state.agent_status = {"XML": False, "Web": False, "Docs": False}
    st.session_state.trace.append(f"> 📥 Query Received: {query}")
    
    # Routing Logic
    analysis_prompt = f"Analyze: '{query}'. Scores (0-1): XML, Web, Docs. Return JSON format."
    try:
        resp = client_openai.chat.completions.create(
            model="gpt-4o-mini", response_format={"type":"json_object"}, 
            messages=[{"role":"system", "content":"You are a PAA Supervisor."}, {"role":"user","content":analysis_prompt}]
        )
        scores = json.loads(resp.choices[0].message.content)
        st.session_state.trace.append(f"> 🤖 Routing Scores: {scores}")

        context_data = ""
        mapping = {"XML": "PAAWeb", "Web": "PAAWebLink", "Docs": "PAAPolicy"}
        
        for key in ["XML", "Web", "Docs"]:
            score = scores.get(key, 0.0)
            if score >= THRESHOLD:
                st.session_state.agent_status[key] = True
                st.session_state.trace.append(f"> 🔮 {key} Agent Activated")
                retrieved = fetch_from_weaviate(query, mapping[key])
                if retrieved and len(retrieved.strip()) > 0:
                    context_data += f"\n[{key} Official Data]: {retrieved}"
                else:
                    st.session_state.trace.append(f"> ⚠️ {key} Agent returned NO data from DB.")

        # --- SMART LLM RESPONSE LOGIC ---
        if context_data.strip():
            # Agar data mil gaya toh official response
            system_message = f"""
            You are the PAA Official Assistant. 
            Answer using the Official Context provided below.
            
            OFFICIAL CONTEXT:
            {context_data}
            """
        else:
            # Agar DB khali hai toh LLM knowledge with Disclaimer
            st.session_state.trace.append("> ℹ️ No DB records found. Switching to LLM Knowledge.")
            system_message = f"""
            You are the PAA Official Assistant. 
            CRITICAL INSTRUCTION: No data was found in the PAA AODB/Database for this query.
            1. You MUST start your response with this EXACT disclaimer: 
               "⚠️ *Disclaimer: Yeh maloomat general internet records se li gayi hain, PAA ke official AODB database mein filhal iska record maujood nahi hai.*"
            2. After the disclaimer, provide the best possible answer using your general knowledge.
            """

        ans_resp = client_openai.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "system", "content": system_message}] + st.session_state.messages[-3:] + [{"role": "user", "content": query}]
        )
        answer = ans_resp.choices[0].message.content
        
        st.session_state.trace.append("> ✅ Process Complete.")
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Engine Error: {e}")

# --- 4. THREE-COLUMN LAYOUT ---
col1, col2, col3 = st.columns([1.2, 1.5, 2], gap="medium")

# COLUMN 1: Trace Console (Black)
with col1:
    st.markdown("### 📁 Trace Console")
    if st.button("Clear All"):
        st.session_state.messages, st.session_state.trace = [], []
        st.rerun()
    
    trace_content = "".join([f"<div style='margin-bottom:5px;'>{t}</div>" for t in st.session_state.trace])
    st.markdown(f"<div class='trace-box'>{trace_content}</div>", unsafe_allow_html=True)

# COLUMN 2: Agentic Flow Visualization (Dots)
with col2:
    st.markdown("### 🛠️ Agentic Flow Visualization")
    st.write("")
    # Supervisor
    st.markdown("<center><div class='agent-dot dot-supervisor'></div><br><b>Supervisor Agent</b><br>↓</center>", unsafe_allow_html=True)
    
    v_col1, v_col2, v_col3 = st.columns(3)
    agents = [("XML", v_col1), ("Web", v_col2), ("Docs", v_col3)]
    
    for name, col in agents:
        status_class = "dot-active" if st.session_state.agent_status[name] else "dot-inactive"
        with col:
            st.markdown(f"<center><div class='agent-dot {status_class}'></div><br><b>{name} Agent</b><br>↓<br><small>RAG_{name}</small></center>", unsafe_allow_html=True)

# COLUMN 3: PAA Smart Chat
with col3:
    st.markdown("### 💬 PAA Smart Chat")
    chat_container = st.container(height=500)
    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Ask about flights, web links, or documents..."):
        run_paa_engine(prompt)
        st.rerun()
