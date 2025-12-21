import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import json
import time

# --- CONFIG ---
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBED = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# --- UI SETTINGS ---
st.set_page_config(page_title="PAA Trace Intelligence", layout="wide")

# Custom CSS for the Console look and Lights
st.markdown("""
    <style>
    .console-box { background-color: #1e1e1e; color: #00ff00; padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace; height: 300px; overflow-y: auto; }
    .status-light { height: 15px; width: 15px; border-radius: 50%; display: inline-block; margin-right: 5px; }
    .light-off { background-color: #333; }
    .light-yellow { background-color: #ffcc00; box-shadow: 0 0 10px #ffcc00; }
    .light-green { background-color: #00ff00; box-shadow: 0 0 10px #00ff00; }
    .light-blue { background-color: #00ccff; box-shadow: 0 0 10px #00ccff; }
    </style>
""", unsafe_allow_html=True)

# --- INTERNAL STATE ---
if "trace" not in st.session_state:
    st.session_state.trace = []
if "final_ans" not in st.session_state:
    st.session_state.final_ans = ""

# --- RAG RETRIEVAL FUNCTION ---
def get_rag_data(q, collection):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=st.secrets["WEAVIATE_URL"],
        auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
    )
    try:
        coll = client.collections.get(collection)
        res = coll.query.near_vector(near_vector=EMBED.encode(q).tolist(), limit=2)
        data = [o.properties['content'] for o in res.objects]
        return " | ".join(data) if data else "No data found."
    finally:
        client.close()

# --- AGENTIC PROCESS ---
def run_agentic_flow(query):
    st.session_state.trace = []
    st.session_state.trace.append(f"🔍 User Query: {query}")
    
    # 1. SUPERVISOR ANALYSIS
    st.session_state.step = "supervisor"
    analysis_prompt = f"Analyze this query: '{query}'. Which RAGs are needed? (RAG1_XML, RAG2_Web, RAG3_Docs). Give scores 0-1. Format: JSON only."
    resp = client_openai.chat.completions.create(model="gpt-4o-mini", response_format={ "type": "json_object" }, messages=[{"role": "user", "content": analysis_prompt}])
    scores = json.loads(resp.choices[0].message.content)
    st.session_state.trace.append(f"🤖 Supervisor Decision: {scores}")

    # 2. PARALLEL SUB-AGENTS
    context_parts = []
    for rag, score in scores.items():
        if score > 0.3:
            st.session_state.trace.append(f"📡 Routing to {rag} (Score: {score})")
            data = get_rag_data(query, rag)
            st.session_state.trace.append(f"📥 {rag} Response: {data[:100]}...")
            context_parts.append(data)
    
    # 3. FINAL LLM REASONING
    final_prompt = f"Based on this context: {' '.join(context_parts)}\nAnswer this: {query}"
    final_resp = client_openai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": final_prompt}])
    st.session_state.final_ans = final_resp.choices[0].message.content
    st.session_state.trace.append("✅ Final Answer Compiled.")

# --- UI LAYOUT ---
col_left, col_right = st.columns([0.6, 0.4])

with col_left:
    st.subheader("🛠️ Agentic Flow Visualization")
    
    # Merging Visual Logic
    s_light = "light-yellow" if "trace" in st.session_state and len(st.session_state.trace) > 1 else "light-off"
    r1_light = "light-blue" if any("RAG1" in t for t in st.session_state.trace) else "light-off"
    r2_light = "light-blue" if any("RAG2" in t for t in st.session_state.trace) else "light-off"
    r3_light = "light-blue" if any("RAG3" in t for t in st.session_state.trace) else "light-off"
    f_light = "light-green" if st.session_state.final_ans else "light-off"

    # Hierarchy Display
    st.markdown(f"**[LLM GPT-4o]**")
    st.markdown(f" ↓ ")
    st.markdown(f"<div class='status-light {s_light}'></div> **Supervisor Agent**", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"↙️ <div class='status-light {r1_light}'></div> **XML Agent**", unsafe_allow_html=True)
    with c2: st.markdown(f"↓ <div class='status-light {r2_light}'></div> **Web Agent**", unsafe_allow_html=True)
    with c3: st.markdown(f"↘️ <div class='status-light {r3_light}'></div> **Doc Agent**", unsafe_allow_html=True)
    
    st.markdown("---")
    if st.session_state.final_ans:
        st.markdown(f"### 💬 Final Response\n<div class='status-light {f_light}'></div> {st.session_state.final_ans}", unsafe_allow_html=True)

with col_right:
    st.subheader("📁 System Trace Console")
    # This acts as your separate terminal window
    console_html = "".join([f"<p>> {t}</p>" for t in st.session_state.trace])
    st.markdown(f"<div class='console-box'>{console_html}</div>", unsafe_allow_html=True)
    
    if st.button("🗑️ Clear History"):
        st.session_state.trace = []
        st.session_state.final_ans = ""
        st.rerun()

# --- INPUT ---
prompt = st.chat_input("Enter your PAA query...")
if prompt:
    with st.spinner("Processing..."):
        run_agentic_flow(prompt)
        st.rerun()
