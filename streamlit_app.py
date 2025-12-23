import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import json
import re

# --- 1. CONFIG & INITIALIZATION ---
st.set_page_config(page_title="PAA Enterprise Intelligence", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .trace-box { background-color: #0e1117; color: #00ff00; padding: 15px; border-radius: 10px; 
                font-family: monospace; font-size: 0.85rem; height: 500px; overflow-y: auto; border: 1px solid #333; }
    .agent-dot { height: 20px; width: 20px; border-radius: 50%; display: inline-block; margin-bottom: 5px; }
    .dot-active { background-color: #00e5ff; box-shadow: 0 0 10px #00e5ff; }
    .dot-inactive { background-color: #333; }
    .dot-supervisor { background-color: #ffcc00; }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "trace" not in st.session_state: st.session_state.trace = []
if "agent_status" not in st.session_state: st.session_state.agent_status = {"XML": False, "Web": False, "Docs": False}

client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_resources():
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

EMBED = load_resources()

# --- 2. WEAVIATE RETRIEVER ---
def fetch_from_weaviate(query, collection_name):
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=st.secrets["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
        )
        coll = client.collections.get(collection_name)
        res = coll.query.near_vector(near_vector=EMBED.encode(query).tolist(), limit=3)
        client.close()
        return "\n".join([o.properties['content'] for o in res.objects]) if res.objects else ""
    except: return ""

# --- 3. AGENTIC ENGINE (STRICT ROUTING) ---
def run_paa_engine(query):
    st.session_state.trace = []
    st.session_state.agent_status = {"XML": False, "Web": False, "Docs": False}
    st.session_state.trace.append(f"> 📥 Query Received: {query}")
    
    # --- STEP 1: ROUTER DECISION ---
    router_prompt = f"""
    Classify this query: "{query}"
    
    Categories:
    1. FLIGHT: If query is a flight number (SV726, PK300), flight status, or schedule.
    2. BAGGAGE: If query is about baggage, pets, liquids, or travel rules.
    3. WEB: If query asks for official links, notices, tenders, or paa.gov.pk website content.
    4. GENERAL: If it's a greeting (hi, hello) or unrelated to PAA specific data.

    Return ONLY JSON: {{"route": "CATEGORY_NAME"}}
    """
    
    try:
        # Regex for Flight Numbers (Immediate Override)
        if re.search(r'[a-zA-Z]{2}\d{2,4}', query):
            route = "FLIGHT"
            st.session_state.trace.append("> 🛠️ Regex Triggered: Flight Pattern Found.")
        else:
            resp = client_openai.chat.completions.create(
                model="gpt-4o-mini", response_format={"type":"json_object"}, 
                messages=[{"role":"system", "content":"You are a PAA Query Router."}, {"role":"user","content":router_prompt}]
            )
            route = json.loads(resp.choices[0].message.content).get("route", "GENERAL")

        st.session_state.trace.append(f"> 🤖 Router Decision: {route}")

        context_data = ""
        
        # --- STEP 2: ACTIVATE SUB-AGENT BASED ON ROUTE ---
        if route == "FLIGHT":
            st.session_state.agent_status["XML"] = True
            context_data = fetch_from_weaviate(query, "PAAWeb")
        elif route == "BAGGAGE":
            st.session_state.agent_status["Docs"] = True
            context_data = fetch_from_weaviate(query, "PAAPolicy")
        elif route == "WEB":
            st.session_state.agent_status["Web"] = True
            context_data = fetch_from_weaviate(query, "PAAWebLink")
        
        # --- STEP 3: FINAL RESPONSE ---
        if context_data:
            st.session_state.trace.append(f"> 🔮 {route} Data Found in DB.")
            system_msg = f"You are PAA Assistant. Answer using this official data: {context_data}"
        else:
            if route == "GENERAL":
                st.session_state.trace.append("> 🌍 Direct LLM Handling (No Agent Needed).")
                system_msg = "You are the PAA Official Assistant. Greet the user and help generally."
            else:
                st.session_state.trace.append(f"> ⚠️ {route} Agent called, but DB was empty.")
                system_msg = f"""You are the PAA Official Assistant. 
                Start with: "⚠️ *Disclaimer: Yeh maloomat general internet records se li gayi hain, PAA ke official AODB database mein filhal iska record maujood nahi hai.*"
                Then answer "{query}" using your general knowledge."""

        ans_resp = client_openai.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "system", "content": system_msg}] + st.session_state.messages[-3:] + [{"role": "user", "content": query}]
        )
        answer = ans_resp.choices[0].message.content
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.trace.append("> ✅ Task Finished.")

    except Exception as e:
        st.error(f"Error: {e}")

# --- 4. UI LAYOUT ---
col1, col2, col3 = st.columns([1.2, 1.5, 2])

with col1:
    st.markdown("### 📁 Trace Console")
    if st.button("Clear Chat"):
        st.session_state.messages, st.session_state.trace = [], []
        st.rerun()
    t_box = "".join([f"<div>{t}</div>" for t in st.session_state.trace])
    st.markdown(f"<div class='trace-box'>{t_box}</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 🛠️ Agentic Flow")
    st.markdown("<center><div class='agent-dot dot-supervisor'></div><br><b>Supervisor</b></center>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    for n, c, k in [("XML Agent", v1, "XML"), ("Web Agent", v2, "Web"), ("Docs Agent", v3, "Docs")]:
        dot = "dot-active" if st.session_state.agent_status[k] else "dot-inactive"
        with c: st.markdown(f"<center><div class='agent-dot {dot}'></div><br><small>{n}</small></center>", unsafe_allow_html=True)

with col3:
    st.markdown("### 💬 PAA Smart Chat")
    with st.container(height=450):
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("Ask me..."):
        run_paa_engine(p)
        st.rerun()
