import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import re
import json

# --- 1. FLIGHT NUMBER NORMALIZATION ---
AIRLINE_ALIASES = {
    "AIRBLUE": "PA", "AIR BLUE": "PA", "PIA": "PK", "PAKISTAN INTERNATIONAL": "PK",
    "SERENE": "ER", "AIR SIAL": "PF", "TURKISH": "TK", "ETIHAD": "EY", "EMIRATES": "EK"
}

def extract_canonical_flight(query: str):
    q = query.upper().replace("-", " ").replace(".", " ")
    q = re.sub(r"\s+", " ", q)
    m = re.search(r"\b([A-Z]{2})\s*(\d{2,4})\b", q)
    if m: return m.group(1) + m.group(2)
    m2 = re.search(r"\b(\d{2,4})\b", q)
    if not m2: return None
    num = m2.group(1)
    for name, iata in AIRLINE_ALIASES.items():
        if name in q: return iata + num
    return None

# --- 2. CONFIG & RESOURCES ---
st.set_page_config(page_title="PAA Enterprise Intelligence", layout="wide")
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

EMBED = load_embedder()

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "trace" not in st.session_state: st.session_state.trace = []
if "agent_status" not in st.session_state: 
    st.session_state.agent_status = {"XML_AGENT": False, "DOC_AGENT": False, "WEB_AGENT": False}

# --- 4. WEAVIATE SEARCH (MULTIPLE AGENTS) ---
def weaviate_search(query, collection):
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=st.secrets["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
        )
        coll = client.collections.get(collection)
        
        # Flight logic specific to XML Agent
        if collection == "PAA_XML_FLIGHTS":
            flight_no = extract_canonical_flight(query)
            if flight_no:
                exact = coll.query.fetch_objects(
                    filters=weaviate.classes.query.Filter.by_property("flight_number").equal(flight_no),
                    limit=1
                )
                if exact.objects:
                    client.close()
                    return [o.properties for o in exact.objects]

        # Semantic Fallback for all agents
        res = coll.query.near_vector(near_vector=EMBED.encode(query).tolist(), limit=3)
        client.close()
        return [o.properties for o in res.objects] if res.objects else []
    except Exception as e:
        st.error(f"Search Error in {collection}: {e}")
        return []

# --- 5. SUPERVISOR ROUTER (LLM BASED) ---
# ================= FIXED SUPERVISOR ROUTER =================
def supervisor_router(query):
    q = query.lower()

    # detect flight intent even without airline code
    flight_hint = bool(
        re.search(r"\bflight\b", q)
        or re.search(r"\b\d{2,4}\b", q)   # like 270
        or re.search(r"\bstatus\b", q)
    )

    baggage_hint = bool(re.search(r"\bbaggage|luggage|hand\s?carry|check[- ]?in\b", q))
    web_hint = bool(re.search(r"website|notice|tender|official", q))

    agents = []
    if flight_hint:
        agents.append("XML_AGENT")
    if baggage_hint:
        agents.append("DOC_AGENT")
    if web_hint:
        agents.append("WEB_AGENT")

    if not agents:
        agents = ["NONE"]

    return agents

# --- 6. CORE ENGINE ---
def run_engine(user_query):
    st.session_state.trace = []
    st.session_state.agent_status = {k: False for k in st.session_state.agent_status}
    st.session_state.trace.append(f"📥 Received: {user_query}")

    agents = supervisor_router(user_query)
    st.session_state.trace.append(f"🧠 Routing: {agents}")

    if "NONE" in agents or not agents:
        resp = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Greet professionally as PAA Assistant."}] + st.session_state.messages[-3:] + [{"role": "user", "content": user_query}]
        )
        return resp.choices[0].message.content

    internal_data = []
    missing_agents = []

    for agent in agents:
        if agent == "NONE": continue
        st.session_state.agent_status[agent] = True
        st.session_state.trace.append(f"➡️ Activating {agent}...")

        if agent == "XML_AGENT": data = weaviate_search(user_query, "PAA_XML_FLIGHTS")
        elif agent == "DOC_AGENT": data = weaviate_search(user_query, "PAAPolicy")
        elif agent == "WEB_AGENT": data = weaviate_search(user_query, "RAG2_Web")
        else: data = []

        if data:
            internal_data.append({agent: data})
            st.session_state.trace.append(f"✅ Data found in {agent}")
        else:
            missing_agents.append(agent)
            st.session_state.trace.append(f"⚠️ No internal data in {agent}")

    # FINAL PROMPT GENERATION
    final_prompt = f"""
    You are the PAA Enterprise AI. 
    INTERNAL RECORDS: {internal_data}
    NOT FOUND IN AGENTS: {missing_agents}

    STRICT RULES:
    1. If data is in INTERNAL RECORDS, use it as 'Official Database Record'.
    2. If an agent is in 'NOT FOUND', you MUST say: "Hamare internal database mein iska record mojud nahi hai, lekin aam maloomat ke mutabiq..." then answer using your own knowledge.
    3. If query is a flight (like SV726) and not in XML_AGENT, provide status from internet knowledge but with the internal-not-found disclaimer.
    4. Be professional and concise and reply or use English only.
    """

    ans = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": final_prompt}, {"role": "user", "content": user_query}]
    )
    return ans.choices[0].message.content

# --- 7. UI LAYOUT ---
st.title("✈️ PAA Enterprise Intelligence")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Trace Console")
    trace_html = "".join([f"<div style='margin-bottom:5px;'>{t}</div>" for t in st.session_state.trace])
    st.markdown(f"<div style='background:#0e1117; color:#00ff00; padding:15px; height:450px; overflow-y:auto; font-family:monospace; border-radius:10px; border:1px solid #333;'>{trace_html}</div>", unsafe_allow_html=True)
    
    st.markdown("### 🤖 Active Agents")
    for agent, status in st.session_state.agent_status.items():
        color = "#00e5ff" if status else "#333"
        st.markdown(f"<span style='color:{color}'>●</span> {agent}", unsafe_allow_html=True)

with col2:
    st.subheader("💬 Chat")
    chat_container = st.container(height=400)
    for m in st.session_state.messages:
        with chat_container.chat_message(m["role"]): st.markdown(m["content"])

    if q := st.chat_input("Flight status, baggage rules, or NOTAMs..."):
        with st.chat_message("user"): st.markdown(q)
        answer = run_engine(q)
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
