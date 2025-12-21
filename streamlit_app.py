import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import json

# --- CONFIG ---
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBED = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# Threshold: Isse kam score par agent trigger nahi hoga
THRESHOLD = 0.7 

st.set_page_config(page_title="PAA Enterprise Intelligence", layout="wide")

st.markdown("""
    <style>
    .center-div { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 10px; }
    .sidebar-console { background-color: #0e1117; color: #00ff00; padding: 10px; border-radius: 5px; font-family: 'Courier New', monospace; font-size: 11px; height: 80vh; overflow-y: auto; border: 1px solid #333; }
    .status-light { height: 16px; width: 16px; border-radius: 50%; display: inline-block; transition: all 0.3s ease; }
    .light-off { background-color: #333; }
    .light-yellow { background-color: #ffcc00; box-shadow: 0 0 12px #ffcc00; }
    .light-green { background-color: #00ff00; box-shadow: 0 0 12px #00ff00; }
    .light-blue { background-color: #00ccff; box-shadow: 0 0 12px #00ccff; }
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

# --- WEAVIATE RETRIEVER ---
def fetch_from_weaviate(query, collection_name):
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=st.secrets["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
        )
        coll = client.collections.get(collection_name)
        # Vector search
        res = coll.query.near_vector(near_vector=EMBED.encode(query).tolist(), limit=2)
        client.close()
        return "\n".join([o.properties['content'] for o in res.objects]) if res.objects else ""
    except Exception as e:
        return f"Error: {str(e)}"

# --- AGENTIC ENGINE ---
def run_paa_engine(query):
    st.session_state.trace.append(f"<b>Query Received:</b> {query}")
    
    # 1. Supervisor Routing Logic (With strict Greeting rule)
    analysis_prompt = f"""
    Analyze the user query: "{query}"
    If it is a simple greeting (Hi, Hello, etc.), return all scores as 0.0.
    If it asks for flight status, give XML a high score (>0.8).
    If it asks for links or website info, give Web a high score (>0.8).
    If it asks for baggage or airport policies, give Docs a high score (>0.8).
    
    Return ONLY a JSON object: {{"XML": score, "Web": score, "Docs": score}}
    """
    
    resp = client_openai.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={"type":"json_object"}, 
        messages=[{"role":"system", "content":"You are a PAA Supervisor Agent."}, {"role":"user","content":analysis_prompt}]
    )
    scores = json.loads(resp.choices[0].message.content)
    st.session_state.scores = scores
    st.session_state.trace.append(f"🤖 Routing Scores: {scores}")

    # 2. Parallel RAG Retrieval (Only if score > THRESHOLD)
    context = ""
    # Mapping friendly names to your Weaviate collection names from screenshot
    mapping = {"XML": "PAAWeb", "Web": "PAAWebLink", "Docs": "PAAPolicy"}
    
    for key, score in scores.items():
        if score >= THRESHOLD:
            st.session_state.trace.append(f"📡 {key} Agent Activated (Score {score})")
            retrieved_text = fetch_from_weaviate(query, mapping[key])
            context += f"\nSource {key}: {retrieved_text}"
        else:
            st.session_state.trace.append(f"⚪ {key} Agent Bypassed (Score {score} < {THRESHOLD})")

    # 3. Final Response
    final_messages = [{"role": "system", "content": f"You are PAA Expert. Use context if provided: {context}"}] + st.session_state.messages + [{"role": "user", "content": query}]
    ans_resp = client_openai.chat.completions.create(model="gpt-4o", messages=final_messages)
    answer = ans_resp.choices[0].message.content
    
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.trace.append("✅ Process Complete.")

# --- MAIN UI ---
col_vis, col_chat = st.columns([0.45, 0.55])

with col_vis:
    st.subheader("🛠️ Agentic Flow Visualization")
    
    # Supervisor Light (Yellow on any query)
    s_active = "light-yellow" if st.session_state.trace else "light-off"
    st.markdown(f"<div class='center-div'><div class='status-light {s_active}'></div><br><b>Supervisor Agent</b><div class='arrow'>↓</div></div>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    rag_keys = ["XML", "Web", "Docs"]
    for i, key in enumerate(rag_keys):
        score = st.session_state.scores.get(key, 0)
        # Trigger light only if score meets threshold
        light_style = "light-blue" if score >= THRESHOLD else "light-off"
        
        with cols[i]:
            st.markdown(f"""<div class='center-div'>
                <div class='status-light {light_style}'></div><b>{key} Agent</b>
                <div class='arrow'>↓</div>
                <div class='status-light {light_style}'></div><i>RAG_{key}</i>
            </div>""", unsafe_allow_html=True)

with col_chat:
    st.subheader("💬 PAA Smart Chat")
    chat_box = st.container(height=500)
    for m in st.session_state.messages:
        chat_box.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("Ask about flights, web links, or documents..."):
        run_paa_engine(prompt)
        st.rerun()
