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
        
        # 1. Flight Status ke liye (XML_AGENT)
        flight_no = extract_canonical_flight(query)
        if flight_no and collection == "PAA_XML_FLIGHTS":
            exact = coll.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("flight_number").equal(flight_no),
                limit=1
            )
            if exact.objects:
                client.close()
                return [o.properties for o in exact.objects]

        # 2. Baggage/Policy ke liye (DOC_AGENT)
        # return_metadata zaroori hai taake pata chale match kitna strong hai
        semantic = coll.query.near_vector(
            near_vector=EMBED.encode(query).tolist(), 
            limit=3,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )
        client.close()

        # 0.45 ko 0.55 ya 0.6 kar dein temporary testing ke liye
        results = [o.properties for o in semantic.objects if o.metadata.distance <= 0.6]
        return results
    except Exception as e:
        st.warning(f"Weaviate search failed: {e}")
        return []
# ================= UPDATED SUPERVISOR (LLM INTEGRATED) =================
def supervisor_router(query):
    q = query.lower()
    
    # Check patterns
    has_flight_no = bool(re.search(r"\b[A-Z]{2}\s?\d{2,4}\b|\b\d{3,4}\b", q, re.I))
    
    # Keywords
    baggage_keywords = ["baggage", "weight", "luggage", "kg", "policy", "liquid", "items", "allowance", "carry on"]
    status_keywords = ["status", "time", "gate", "schedule", "arrival", "departure", "landed", "where is", "detail"]
    web_keywords = ["notam", "tender", "official", "website", "complaint", "career"]

    is_baggage = any(word in q for word in baggage_keywords)
    is_status = any(word in q for word in status_keywords)
    is_web = any(word in q for word in web_keywords)

    agents = []

    # --- REFINED INTENT LOGIC ---
    
    # Case A: If user asks for baggage (with or without flight number)
    if is_baggage:
        agents.append("DOC_AGENT")
        # Only trigger XML if they specifically ask for "status" or "details" ALONG with baggage
        if is_status:
            agents.append("XML_AGENT")
            
    # Case B: Pure Flight Status query
    elif has_flight_no or is_status:
        agents.append("XML_AGENT")

    # Case C: Web related
    if is_web:
        agents.append("WEB_AGENT")

    if not agents:
        if re.match(r"^(hi|hello|hey|salaam|aoa)\s*$", q): return ["NONE"]
        return ["XML_AGENT"]

    return list(set(agents))

# ================= QUERY DECOMPOSITION =================
def decompose_query(query, agents):
    decomposition = {}
    for a in agents:
        if a == "XML_AGENT":
            flight_no = extract_canonical_flight(query)
            decomposition[a] = flight_no if flight_no else query
            
        elif a == "DOC_AGENT":
            # Sirf flight number remove nahi karna, balki Airline identify karni hai
            flight_no = extract_canonical_flight(query)
            clean_q = re.sub(r"\b[A-Z]{2}\s?\d{2,4}\b|\b\d{3,4}\b", "", query, flags=re.I).strip()
            
            airline_context = ""
            if flight_no:
                prefix = flight_no[:2].upper()
                # Reverse mapping taake SV se 'Saudia' mil jaye
                INV_ALIASES = {v: k for k, v in AIRLINE_ALIASES.items()}
                airline_name = INV_ALIASES.get(prefix, "")
                airline_context = f"{airline_name} baggage policy"
            
            # Agar airline mili toh wo use karein, warna original clean query
            decomposition[a] = airline_context if airline_context else clean_q
            
    return decomposition
# ================= COMPLETE CLEAN RUN_ENGINE =================
def run_engine(user_query):
    st.session_state.trace.clear()
    # Resetting agent status for the new query
    st.session_state.agent_status = {k: False for k in st.session_state.agent_status}
    st.session_state.trace.append(f"📥 User Query: {user_query}")

    # 1. Routing
    agents = supervisor_router(user_query)
    st.session_state.trace.append(f"🧠 Supervisor Routing: {agents}")

    # 2. Handle simple greetings or off-topic queries
    if agents == ["NONE"]:
        response = client_openai.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "Respond professionally as a PAA Virtual Assistant in English only. Keep it brief."},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content

    # 3. Decompose query and fetch data from agents
    sub_queries = decompose_query(user_query, agents)
    internal_results = []
    data_was_found = False

    for agent, sub_q in sub_queries.items():
        st.session_state.agent_status[agent] = True
        st.session_state.trace.append(f"➡️ {agent} activated")
        
        if agent == "XML_AGENT": 
            data = weaviate_search(sub_q, "PAA_XML_FLIGHTS")
        elif agent == "DOC_AGENT": 
            data = weaviate_search(sub_q, "PAAPolicy")
        elif agent == "WEB_AGENT": 
            data = weaviate_search(sub_q, "RAG2_Web")
        else: 
            data = []

        if data:
            internal_results.append({ "source_agent": agent, "content": data })
            data_was_found = True
            st.session_state.trace.append(f"✅ {agent} found data")
        else:
            st.session_state.trace.append(f"⚠️ {agent} returned NOT_FOUND")

    # 4. Final Reasoning and Response Construction (UPDATED LOGIC)
    final_prompt = f"""
You are a professional PAA (Pakistan Airports Authority) Virtual Assistant.
RESPOND ONLY IN ENGLISH.

INTERNAL DATABASE CONTEXT: {internal_results if data_was_found else "NONE"}

STRICT RESPONSE RULES:
1. SPECIFICITY: 
   - If user asks about FLIGHT STATUS (XML_AGENT): Provide details in professional bullet points.
   - If user asks about BAGGAGE/POLICIES (DOC_AGENT): Summarize the rules clearly in bullets.
   - If a specific detail (like 'gate' or 'liquid limit') is asked, provide ONLY that.

2. DATA INTEGRITY: Only mention fields that have actual, valid values in the context. 
3. HIDE EMPTY FIELDS: If a field is missing, null, or says 'Not Specified' in the database, DO NOT mention it.
4. FALLBACK: If 'INTERNAL DATABASE CONTEXT' is NONE, say: "The specific details were not found in our records. However, based on general knowledge..." and answer in English.
5. NO DISCLAIMER ON SUCCESS: If data is found from any agent, do NOT say "Internal records not found".
6. FORMATTING: Use clean bullet points. Strictly no Urdu/Hindi script.

User Query: {user_query}
"""
    
    # 5. Call LLM for the final answer
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": final_prompt}, {"role": "user", "content": user_query}],
            temperature=0.1
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"I apologize, but I encountered an error: {e}"
        st.session_state.trace.append(f"❌ LLM Error: {e}")

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
