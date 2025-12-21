import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import json

# --- SETUP ---
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBED = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

st.set_page_config(page_title="PAA Persistent Intelligence", layout="wide")

# Custom CSS for UI
st.markdown("""
    <style>
    .center-div { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 10px; }
    .console-box { background-color: #1e1e1e; color: #00ff00; padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace; height: 350px; overflow-y: auto; font-size: 12px; border: 1px solid #444; }
    .status-light { height: 18px; width: 18px; border-radius: 50%; display: inline-block; }
    .light-off { background-color: #333; }
    .light-yellow { background-color: #ffcc00; box-shadow: 0 0 10px #ffcc00; }
    .light-green { background-color: #00ff00; box-shadow: 0 0 10px #00ff00; }
    .light-blue { background-color: #00ccff; box-shadow: 0 0 10px #00ccff; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION (Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Chat history for LLM
if "trace_history" not in st.session_state:
    st.session_state.trace_history = [] # Logs for Console
if "current_scores" not in st.session_state:
    st.session_state.current_scores = {"RAG1_XML":0, "RAG2_Web":0, "RAG3_Docs":0}

# --- AGENTIC ENGINE ---
def run_agentic_engine(user_query):
    # 1. Start Trace
    st.session_state.trace_history.append(f"--- New Query: {user_query} ---")
    
    # 2. Supervisor Scoring (LLM Contextual Decision)
    # Hum pichli history bhi bhej rahe hain taaki context yaad rahe
    history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
    
    analysis_prompt = f"History: {history_context}\nCurrent Query: {user_query}\nTask: Provide JSON scores (0-1) for RAG1_XML, RAG2_Web, RAG3_Docs based on relevance."
    
    resp = client_openai.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={"type":"json_object"}, 
        messages=[{"role":"system", "content":"You are a routing supervisor."}, {"role":"user","content":analysis_prompt}]
    )
    
    scores = json.loads(resp.choices[0].message.content)
    st.session_state.current_scores = scores
    st.session_state.trace_history.append(f"🤖 Supervisor routing scores: {scores}")

    # 3. Simulated RAG Fetch (Yahan aapka actual weaviate call ayega)
    context_data = ""
    for rag, score in scores.items():
        if score > 0.3:
            st.session_state.trace_history.append(f"📡 Sub-Agent active: {rag}")
            context_data += f"[Data from {rag} about {user_query}] "

    # 4. Final LLM Response with Memory
    messages_for_llm = [
        {"role": "system", "content": f"You are PAA Expert. Use this RAG context: {context_data}"}
    ] + st.session_state.messages + [{"role": "user", "content": user_query}]
    
    final_resp = client_openai.chat.completions.create(model="gpt-4o", messages=messages_for_llm)
    answer = final_resp.choices[0].message.content
    
    # Update Session State
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.trace_history.append("✅ Final response compiled and added to history.")

# --- UI LAYOUT ---
vis_col, chat_col = st.columns([0.5, 0.5])

with vis_col:
    st.subheader("🛠️ Agentic Flow Visualization")
    
    # Active Lights logic
    s_active = "light-yellow" if st.session_state.trace_history else "light-off"
    
    # Symmetrical Diagram
    st.markdown(f"<div class='center-div'><div class='status-light {s_active}'></div><br><b>Supervisor Agent</b><br>↓</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    for i, (rag, score) in enumerate(st.session_state.current_scores.items()):
        light = "light-blue" if score > 0.3 else "light-off"
        with [c1, c2, c3][i]:
            st.markdown(f"<div class='center-div'><div class='status-light {light}'></div><br><b>{rag.split('_')[1]} Agent</b><br>↓<br><div class='status-light {light}'></div><br><i>RAG_{rag.split('_')[1]}</i></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📁 Trace Console")
    console_text = "".join([f"<p>> {t}</p>" for t in st.session_state.trace_history[-10:]]) # Last 10 logs
    st.markdown(f"<div class='console-box'>{console_text}</div>", unsafe_allow_html=True)

with chat_col:
    st.subheader("💬 PAA Smart Chat")
    
    # Display Chat History (Isse query ghaib nahi hogi)
    chat_container = st.container(height=500)
    for m in st.session_state.messages:
        with chat_container.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input area
    if prompt := st.chat_input("Ask about flights, policies or web info..."):
        run_agentic_engine(prompt)
        st.rerun() # Refresh to show new messages and lights

if st.sidebar.button("🗑️ Clear All History"):
    st.session_state.messages = []
    st.session_state.trace_history = []
    st.session_state.current_scores = {"RAG1_XML":0, "RAG2_Web":0, "RAG3_Docs":0}
    st.rerun()
