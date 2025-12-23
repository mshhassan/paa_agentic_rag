import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import re
import json

# --- Flight number normalization --- Updated Comprehensive Airline Aliases for Pakistan Operations ---
AIRLINE_ALIASES = {
    # Domestic & Major Local
    "PIA": "PK", "PAKISTAN INTERNATIONAL": "PK", "PAKISTAN INTERNATIONAL AIRLINE": "PK",
    "AIRBLUE": "PA", "AIR BLUE": "PA",
    "AIR SIAL": "PF", "AIRSIAL": "PF", "AIR SIAL AIRWAYS": "PF",
    "SERENE AIR": "ER", "SERENE": "ER",
    "FLY JINNAH": "9P", "JINNAH": "9P",

    # Middle Eastern Carriers
    "EMIRATES": "EK",
    "ETIHAD": "EY", "ETIHAD AIRWAYS": "EY",
    "QATAR": "QR", "QATAR AIRWAYS": "QR",
    "SAUDIA": "SV", "SAUDI AIR": "SV", "SAUDI ARABIAN AIRLINES": "SV",
    "GULF AIR": "GF", "GULF": "GF",
    "KUWAIT AIRWAYS": "KU", "KUWAIT": "KU",
    "AIR ARABIA": "G9",
    "FLYNAS": "XY", "FLY NAS": "XY", "NAS AIR": "XY",
    "SALAM AIR": "OV", "SALAM": "OV",
    "FLY DUBAI": "FZ", "FLYDUBAI": "FZ",
    "JAZEERA AIRWAYS": "J9", "JAZEERA": "J9",

    # European & Western
    "BRITISH AIRWAYS": "BA", "BRITISH": "BA",
    "TURKISH AIRLINES": "TK", "TURKISH": "TK", "TURKISH AIR": "TK",

    # Asian & Others
    "THAI AIR": "TG", "THAI AIRWAYS": "TG", "THAI": "TG",
    "CHINA SOUTHERN": "CZ",
    "AIR CHINA": "CA",
    "ARYANA AFGHAN": "FG", "ARYANA": "FG", "AFGHAN AIR": "FG",
    "KAM AIR": "RQ", "KAM": "RQ",
    "AZERBAIJAN AIRLINES": "J2", "AZERBAIJAN": "J2", "AZAL": "J2",
    "IRAQI AIRWAYS": "IA", "IRAQI": "IA",
    "SOMON AIR": "SZ", "SOMON": "SZ",
    "UZBEKISTAN AIRWAYS": "HY", "UZBEKISTAN": "HY",
    "FLYADEAL": "F3", "FLY ADEAL": "F3",
    
    # Aviation/Cargo/Private
    "FLY SKY": "FS", "FLY SKY AVIATION": "FS",
    
    # Direct IATA Mappings (Safety catch)
    "PK": "PK", "PA": "PA", "PF": "PF", "ER": "ER", "9P": "9P",
    "EK": "EK", "EY": "EY", "QR": "QR", "SV": "SV", "TK": "TK",
    "BA": "BA", "G9": "G9", "FZ": "FZ", "XY": "XY"
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
        
        # --- A. FLIGHT XML AGENT LOGIC ---
        if collection == "PAA_XML_FLIGHTS":
            flight_no = extract_canonical_flight(query)
            # Case 1: Specific Flight Number
            if flight_no:
                exact = coll.query.fetch_objects(
                    filters=weaviate.classes.query.Filter.by_property("flight_number").equal(flight_no),
                    limit=1
                )
                if exact.objects:
                    client.close()
                    return [o.properties for o in exact.objects]
            
            # Case 2: Airline Filtering (If no flight number, check for Airline Name)
            q_upper = query.upper()
            matched_airline = next((name for name in AIRLINE_ALIASES.keys() if name in q_upper), None)
            if matched_airline:
                airline_results = coll.query.fetch_objects(
                    filters=weaviate.classes.query.Filter.by_property("airline_name").like(f"*{matched_airline}*"),
                    limit=15
                )
                client.close()
                return [o.properties for o in airline_results.objects]

        # --- B. SEMANTIC SEARCH (DOC_AGENT & WEB_AGENT) ---
        # Web Agent ke liye hum limit thori barhate hain (more context)
        limit_val = 5 if collection == "RAG2_Web" else 3
        
        semantic = coll.query.near_vector(
            near_vector=EMBED.encode(query).tolist(), 
            limit=limit_val,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )
        client.close()

        # Web Agent ke liye threshold naram (0.7) rakha hai taake general queries match ho sakein
        threshold = 0.7 if collection == "RAG2_Web" else 0.6
        results = [o.properties for o in semantic.objects if o.metadata.distance <= threshold]
        return results
        
    except Exception as e:
        st.warning(f"Weaviate search failed: {e}")
        return []
# ================= UPDATED SUPERVISOR (LLM INTEGRATED) =================
def supervisor_router(query):
    q = query.lower()
    has_flight_no = bool(re.search(r"\b[A-Z]{2}\s?\d{2,4}\b|\b\d{3,4}\b", q, re.I))
    
    baggage_keywords = ["baggage", "weight", "luggage", "kg", "policy", "liquid", "items", "allowance", "carry on"]
    status_keywords = ["status", "time", "gate", "schedule", "arrival", "departure", "landed", "where is", "detail"]
    # Added "paa" and "intro" keywords
    web_keywords = ["notam", "tender", "official", "website", "complaint", "career", "paa", "about", "introduction", "who is"]

    is_baggage = any(word in q for word in baggage_keywords)
    is_status = any(word in q for word in status_keywords)
    is_web = any(word in q for word in web_keywords)

    agents = []
    if is_baggage:
        agents.append("DOC_AGENT")
        if is_status: agents.append("XML_AGENT")
    elif has_flight_no or is_status:
        agents.append("XML_AGENT")

    if is_web:
        agents.append("WEB_AGENT")

    if not agents:
        if re.match(r"^(hi|hello|hey|salaam|aoa)\s*$", q): return ["NONE"]
        # Defaulting to both if unsure, to maximize data retrieval
        return ["XML_AGENT", "WEB_AGENT"]

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

    # 4. Final Reasoning and Response Construction
    final_prompt = f"""
You are a professional PAA (Pakistan Airports Authority) Virtual Assistant.
RESPOND ONLY IN ENGLISH.

INTERNAL DATABASE CONTEXT: {internal_results if data_was_found else "NONE"}

STRICT RESPONSE RULES:
1. IDENTITY RECOGNITION: 'PAA' and 'Pakistan Airports Authority' are the same entity. Always use context about 'Pakistan Airports Authority' to answer 'PAA' queries.
2. CITATION (MANDATORY): If information is retrieved from WEB_AGENT context, you MUST include the 'source' URL at the very end of your response as: "Source: [URL]".
3. SPECIFICITY: 
   - FLIGHT STATUS: Professional bullet points.
   - BAGGAGE: Clear summary.
   - WEB INFO (NOTAMs/Tenders/About PAA): Directly use the text found in WEB_AGENT context.
4. DATA INTEGRITY: Only mention fields with valid values. 
5. FALLBACK: If context is NONE, provide a helpful general answer based on common airport knowledge but state it's general info. Never say you don't have info if PAA-related text exists in context.
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
