import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import re
import json

# --- Flight number normalization ---
AIRLINE_ALIASES = {
    "AIRBLUE": "PA", "AIR BLUE": "PA", "PAKISTAN INTERNATIONAL AIRLINE": "PK",
    "PAKISTAN INTERNATIONAL": "PK", "SERENE AIR": "ER", "SERENE": "ER",
    "AIR SIAL": "PF", "AIR SIAL AIRWAYS": "PF", "TURKISH AIRLINES": "TK",
    "TURKISH": "TK", "PIA": "PK", "PK": "PK", "PA": "PA", "PF": "PF", "TK": "TK",
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

# ================= CONFIG =================
st.set_page_config(page_title="PAA Enterprise Intelligence", layout="wide")
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

EMBED = load_embedder()

# ================= SESSION STATE =================
if "messages" not in st.session_state: st.session_state.messages = []
if "trace" not in st.session_state: st.session_state.trace = []
if "agent_status" not in st.session_state: 
    st.session_state.agent_status = {"XML_AGENT": False, "DOC_AGENT": False, "WEB_AGENT": False}

# ================= WEAVIATE SEARCH =================
def weaviate_search(query, collection):
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=st.secrets["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
        )
        coll = client.collections.get(collection)
        flight_no = extract_canonical_flight(query)
        if flight_no and collection == "PAA_XML_FLIGHTS":
            # Correct Filter Syntax for Weaviate v4
            exact = coll.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("flight_number").equal(flight_no),
                limit=1
            )
            if exact.objects:
                client.close()
                return [o.properties for o in exact.objects]
        
        semantic = coll.query.near_vector(near_vector=EMBED.encode(query).tolist(), limit=3)
        client.close()
        return [o.properties for o in semantic.objects] if semantic.objects else []
    except Exception as e:
        st.warning(f"Weaviate search failed: {e}")
        return []

# ================= UPDATED SUPERVISOR (LLM INTEGRATED) =================
def supervisor_router(query):
    # Combined Regex + Intent Logic
    flight_hint = bool(re.search(r"\b[A-Z]{2}\d{2,4}\b|\bflight\b|\b\d{2,4}\b", query, re.I))
    baggage_hint = bool(re.search(r"\bbaggage|luggage|hand\s?carry|check[- ]?in\b", query, re.I))
    web_hint = bool(re.search(r"website|notice|tender|official|notam", query, re.I))

    agents = []
    if flight_hint: agents.append("XML_AGENT")
    if baggage_hint: agents.append("DOC_AGENT")
    if web_hint: agents.append("WEB_AGENT")
    
    # Simple check for pure greetings to avoid agent activation
    if not agents or re.match(r"^(hi|hello|hey|salaam|aoa)\s*$", query, re.I):
        return ["NONE"]
    return agents

# ================= QUERY DECOMPOSITION =================
def decompose_query(query, agents):
    decomposition = {}
    for a in agents:
        if a=="XML_AGENT": decomposition[a] = f"Status for flight: {query}"
        elif a=="DOC_AGENT": decomposition[a] = f"Rules for: {query}"
        elif a=="WEB_AGENT": decomposition[a] = f"Official info for: {query}"
    return decomposition

# ================= MAIN ENGINE =================
def run_engine(user_query):
    st.session_state.trace.clear()
    st.session_state.agent_status = {k:False for k in st.session_state.agent_status}
    st.session_state.trace.append(f"📥 User Query: {user_query}")

    agents = supervisor_router(user_query)
    st.session_state.trace.append(f"🧠 Supervisor Routing: {agents}")

    if agents==["NONE"]:
        answer = client_openai.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role":"system","content":"Greet professionally as PAA Assistant in English/Urdu. Avoid Hindi script."},{"role":"user","content":user_query}]
        ).choices[0].message.content
        return answer

    sub_queries = decompose_query(user_query, agents)
    internal_results = []
    missing_entities = []

    for agent, sub_q in sub_queries.items():
        st.session_state.agent_status[agent] = True
        st.session_state.trace.append(f"➡️ {agent} activated")
        if agent=="XML_AGENT": data = weaviate_search(sub_q, "PAA_XML_FLIGHTS")
        elif agent=="DOC_AGENT": data = weaviate_search(sub_q, "PAAPolicy")
        elif agent=="WEB_AGENT": data = weaviate_search(sub_q, "RAG2_Web")
        else: data = []

        if data:
            internal_results.append({agent: data})
            st.session_state.trace.append(f"✅ {agent} found data")
        else:
            missing_entities.append(agent)
            st.session_state.trace.append(f"⚠️ {agent} returned NOT_FOUND")

    # IMPROVED FINAL PROMPT
    final_prompt = f"""
You are a professional PAA (Pakistan Airports Authority) Virtual Assistant.

Internal Database Info: {internal_results}
Agents with NO Record: {missing_entities}

STRICT RULES:
1. If Internal Data is available, prioritize it as official record.
2. If Internal Data is NOT found for a flight (like {user_query}), you MUST say: "Hamare internal records mein iska live status nahi mila, lekin general information ke mutabiq..." 
3. Provide the flight status/info using your own knowledge ONLY after giving the disclaimer above.
4. Language: Use a mix of English and Roman Urdu. NEVER use Hindi Devanagari script.
5. Be professional. Do not mention "Agents" or "Databases" to the user.

User Query: {user_query}
"""
    answer = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":final_prompt}]
    ).choices[0].message.content
    return answer

# ================= UI =================
st.title("✈️ PAA Enterprise Intelligence")
col1,col2 = st.columns([1.2,2])

with col1:
    st.subheader("🧾 Trace Console")
    st.markdown("<div style='background:#0e1117;padding:10px;height:400px;overflow:auto;font-family:monospace;color:#00ff00;'>"
                + "<br>".join(st.session_state.trace)
                + "</div>", unsafe_allow_html=True)

with col2:
    st.subheader("💬 Chat")
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if q := st.chat_input("Ask about flights, baggage, or PAA info"):
        answer = run_engine(q)
        st.session_state.messages.append({"role":"user","content":q})
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.rerun()
