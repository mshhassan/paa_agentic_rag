import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import json

# --- CONFIG ---
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBED = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

st.set_page_config(page_title="PAA Trace Intelligence", layout="wide")

# Custom CSS for Symmetrical Alignment and Lights
st.markdown("""
    <style>
    .center-div { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 20px; }
    .console-box { background-color: #1e1e1e; color: #00ff00; padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace; height: 400px; overflow-y: auto; font-size: 13px; }
    .status-light { height: 18px; width: 18px; border-radius: 50%; display: inline-block; margin-bottom: -3px; }
    .light-off { background-color: #333; }
    .light-yellow { background-color: #ffcc00; box-shadow: 0 0 12px #ffcc00; }
    .light-green { background-color: #00ff00; box-shadow: 0 0 12px #00ff00; }
    .light-blue { background-color: #00ccff; box-shadow: 0 0 12px #00ccff; }
    .arrow-container { font-size: 24px; color: #555; line-height: 1; margin: 5px 0; }
    </style>
""", unsafe_allow_html=True)

if "trace" not in st.session_state: st.session_state.trace = []
if "final_ans" not in st.session_state: st.session_state.final_ans = ""
if "scores" not in st.session_state: st.session_state.scores = {"RAG1_XML":0, "RAG2_Web":0, "RAG3_Docs":0}

# --- AGENTIC LOGIC ---
def run_flow(query):
    st.session_state.trace = []
    st.session_state.trace.append(f"🔍 Input: {query}")
    
    # 1. Supervisor Decision
    analysis_prompt = f"Analyze query: '{query}'. Provide JSON scores (0-1) for RAG1_XML, RAG2_Web, RAG3_Docs."
    resp = client_openai.chat.completions.create(model="gpt-4o-mini", response_format={"type":"json_object"}, messages=[{"role":"user","content":analysis_prompt}])
    st.session_state.scores = json.loads(resp.choices[0].message.content)
    st.session_state.trace.append(f"🤖 Supervisor routing: {st.session_state.scores}")
    
    # 2. Parallel Fetch (Simulated for Trace)
    context = ""
    for rag, score in st.session_state.scores.items():
        if score > 0.2:
            st.session_state.trace.append(f"📡 {rag} Active... Fetching embeddings.")
            # Yahan aapka actual 'get_rag_data' function call hoga
            context += f"Data from {rag}..." 
    
    st.session_state.final_ans = "This is a response generated based on PAA verified data."
    st.session_state.trace.append("✅ Final Answer Generated.")

# --- UI LAYOUT ---
col_vis, col_trace = st.columns([0.6, 0.4])

with col_vis:
    st.subheader("🛠️ Agentic Flow Visualization")
    
    # Determination of Lights
    s_light = "light-yellow" if st.session_state.trace else "light-off"
    f_light = "light-green" if st.session_state.final_ans else "light-off"
    
    def get_rag_light(name):
        return "light-blue" if st.session_state.scores.get(name, 0) > 0.2 else "light-off"

    # Hierarchy Visualization
    # 1. Supervisor (Center)
    st.markdown(f"""<div class='center-div'>
        <div class='status-light {s_light}'></div> <b>Supervisor Agent</b>
        <div class='arrow-container'>↓</div>
    </div>""", unsafe_allow_html=True)

    # 2. Sub-Agents (3 Columns)
    a1, a2, a3 = st.columns(3)
    with a1: st.markdown(f"<div class='center-div'><div class='status-light {get_rag_light('RAG1_XML')}'></div> <b>XML Agent</b></div>", unsafe_allow_html=True)
    with a2: st.markdown(f"<div class='center-div'><div class='status-light {get_rag_light('RAG2_Web')}'></div> <b>Web Agent</b></div>", unsafe_allow_html=True)
    with a3: st.markdown(f"<div class='center-div'><div class='status-light {get_rag_light('RAG3_Docs')}'></div> <b>Doc Agent</b></div>", unsafe_allow_html=True)

    # 3. Arrows to RAGs
    st.columns(3) # Space
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("<div class='center-div'><div class='arrow-container'>↓</div></div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='center-div'><div class='arrow-container'>↓</div></div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='center-div'><div class='arrow-container'>↓</div></div>", unsafe_allow_html=True)

    # 4. RAG Nodes
    r1, r2, r3 = st.columns(3)
    with r1: st.markdown(f"<div class='center-div'><div class='status-light {get_rag_light('RAG1_XML')}'></div> <i>RAG: AODB</i></div>", unsafe_allow_html=True)
    with r2: st.markdown(f"<div class='center-div'><div class='status-light {get_rag_light('RAG2_Web')}'></div> <i>RAG: Website</i></div>", unsafe_allow_html=True)
    with r3: st.markdown(f"<div class='center-div'><div class='status-light {get_rag_light('RAG3_Docs')}'></div> <i>RAG: Policies</i></div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.final_ans:
        st.markdown(f"**Final Answer:**\n\n{st.session_state.final_ans}")

with col_trace:
    st.subheader("📁 Trace Console")
    console_text = "".join([f"<p>> {t}</p>" for t in st.session_state.trace])
    st.markdown(f"<div class='console-box'>{console_text}</div>", unsafe_allow_html=True)
    if st.button("Clear Console"):
        st.session_state.trace = []; st.session_state.final_ans = ""; st.rerun()

# --- INPUT ---
u_query = st.chat_input("Ask PAA Assistant...")
if u_query:
    run_flow(u_query)
    st.rerun()
