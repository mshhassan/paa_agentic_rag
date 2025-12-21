import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import json

# --- CONFIG & STYLES ---
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBED = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

st.set_page_config(page_title="PAA Enterprise Intelligence", layout="wide")

st.markdown("""
    <style>
    .center-div { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 10px; }
    .sidebar-console { background-color: #0e1117; color: #00ff00; padding: 10px; border-radius: 5px; font-family: 'Courier New', monospace; font-size: 11px; height: 70vh; overflow-y: auto; border: 1px solid #333; }
    .status-light { height: 16px; width: 16px; border-radius: 50%; display: inline-block; }
    .light-off { background-color: #333; }
    .light-yellow { background-color: #ffcc00; box-shadow: 0 0 8px #ffcc00; }
    .light-green { background-color: #00ff00; box-shadow: 0 0 8px #00ff00; }
    .light-blue { background-color: #00ccff; box-shadow: 0 0 8px #00ccff; }
    .arrow { color: #555; font-size: 20px; margin: 2px 0; }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "trace" not in st.session_state: st.session_state.trace = []
if "scores" not in st.session_state: st.session_state.scores = {"XML":0, "Web":0, "Docs":0}

# --- SIDEBAR CONSOLE ---
with st.sidebar:
    st.header("📁 Trace Console")
    if st.button("🗑️ Clear All"):
        st.session_state.messages = []; st.session_state.trace = []; st.session_state.scores = {"XML":0, "Web":0, "Docs":0}
        st.rerun()
    
    trace_content = "".join([f"<p style='margin:2px;'>> {t}</p>" for t in st.session_state.trace])
    st.markdown(f"<div class='sidebar-console'>{trace_content}</div>", unsafe_allow_html=True)

# --- AGENTIC ENGINE ---
def run_paa_engine(query):
    st.session_state.trace.append(f"<b>Query:</b> {query}")
    
    # 1. Routing Decision
    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-2:]])
    analysis_prompt = f"History: {history}\nQuery: {query}\nProvide JSON scores (0-1) for: XML, Web, Docs."
    
    resp = client_openai.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={"type":"json_object"}, 
        messages=[{"role":"system", "content":"You are a PAA Supervisor Agent."}, {"role":"user","content":analysis_prompt}]
    )
    st.session_state.scores = json.loads(resp.choices[0].message.content)
    st.session_state.trace.append(f"🤖 Supervisor Routing: {st.session_state.scores}")

    # 2. Parallel RAG Retrieval (Logic for LLM context)
    context = ""
    for rag, score in st.session_state.scores.items():
        if score > 0.3:
            st.session_state.trace.append(f"📡 Sub-Agent active: {rag} (Score: {score})")
            context += f"[Simulated Data from {rag} regarding {query}] "

    # 3. Final Response Generation
    final_messages = [{"role": "system", "content": f"You are PAA Expert. Context: {context}"}] + st.session_state.messages + [{"role": "user", "content": query}]
    ans_resp = client_openai.chat.completions.create(model="gpt-4o", messages=final_messages)
    answer = ans_resp.choices[0].message.content
    
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.trace.append("✅ Final Answer Compounded.")

# --- MAIN UI ---
col_vis, col_chat = st.columns([0.45, 0.55])

with col_vis:
    st.subheader("🛠️ Agentic Flow Visualization")
    
    # Symmetrical Hierarchy
    s_active = "light-yellow" if st.session_state.trace else "light-off"
    st.markdown(f"<div class='center-div'><div class='status-light {s_active}'></div><br><b>Supervisor Agent</b><div class='arrow'>↓</div></div>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i, (name, score) in enumerate(st.session_state.scores.items()):
        light = "light-blue" if score > 0.3 else "light-off"
        with cols[i]:
            st.markdown(f"""<div class='center-div'>
                <div class='status-light {light}'></div><b>{name} Agent</b>
                <div class='arrow'>↓</div>
                <div class='status-light {light}'></div><i>RAG_{name}</i>
            </div>""", unsafe_allow_html=True)

with col_chat:
    st.subheader("💬 PAA Smart Chat")
    chat_box = st.container(height=500)
    for m in st.session_state.messages:
        chat_box.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("Ask about flights, web links, or documents..."):
        run_paa_engine(prompt)
        st.rerun()
