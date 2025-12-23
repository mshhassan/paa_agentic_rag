import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import json
import re

# ================= CONFIG =================
st.set_page_config(page_title="PAA Enterprise Intelligence", layout="wide")

client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

EMBED = load_embedder()

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "trace" not in st.session_state:
    st.session_state.trace = []

if "agent_status" not in st.session_state:
    st.session_state.agent_status = {
        "XML_AGENT": False,
        "DOC_AGENT": False,
        "WEB_AGENT": False
    }

# ================= WEAVIATE SEARCH =================
def weaviate_search(query, collection):
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=st.secrets["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
        )
        coll = client.collections.get(collection)
        res = coll.query.near_vector(
            near_vector=EMBED.encode(query).tolist(),
            limit=3
        )
        client.close()
        return [o.properties["content"] for o in res.objects]
    except:
        return []

#====================

def fetch_flight_exact_or_semantic(flight_no):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=st.secrets["WEAVIATE_URL"],
        auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
    )

    coll = client.collections.get("PAA_XML_FLIGHTS")

    # 1️⃣ EXACT MATCH FIRST
    exact = coll.query.fetch_objects(
        filters={
            "path": ["flight_number"],
            "operator": "Equal",
            "valueText": flight_no
        },
        limit=1
    )

    if exact.objects:
        client.close()
        return exact.objects[0].properties

    # 2️⃣ SEMANTIC FALLBACK
    semantic = coll.query.near_vector(
        near_vector=EMBED.encode(flight_no).tolist(),
        limit=1
    )

    client.close()
    if semantic.objects:
        return semantic.objects[0].properties

    return None




# ================= SUPERVISOR =================
def supervisor_router(query):
    routing_prompt = f"""
You are an intent and scope analyzer.

Decide which agents are required for the user query.

AGENTS:
- XML_AGENT → flight status, arrival, departure, gate, check-in
- DOC_AGENT → baggage, hand carry, check-in baggage rules
- WEB_AGENT → paa.gov.pk notices, tenders, official info
- NONE → greetings or casual talk

Rules:
- A query may require more than one agent.
- Include XML_AGENT if a flight number is mentioned.
- Include DOC_AGENT if baggage, luggage, hand carry, or check-in baggage is mentioned.
- Include WEB_AGENT only for website or official notices.
- Return NONE only if no agent applies.

Return JSON only.

Example:
{{ "agents": ["XML_AGENT", "DOC_AGENT"] }}

Query:
"{query}"
"""

    # 🔹 Soft signal detection (NOT override)
    flight_hint = bool(re.search(r"\b[A-Z]{2}\d{2,4}\b", query, re.I))
    baggage_hint = bool(re.search(r"\bbaggage|luggage|hand\s?carry|check[- ]?in\b", query, re.I))

    resp = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": routing_prompt}]
    )

    agents = json.loads(resp.choices[0].message.content).get("agents", [])

    # 🔹 Safety net (merge hints, don't replace)
    if flight_hint and "XML_AGENT" not in agents:
        agents.append("XML_AGENT")

    if baggage_hint and "DOC_AGENT" not in agents:
        agents.append("DOC_AGENT")

    if not agents:
        agents = ["NONE"]

    return agents

# ================= QUERY DECOMPOSITION =================
def decompose_query(query, agents):
    decomposition = {}
    for a in agents:
        if a == "XML_AGENT":
            decomposition[a] = f"Flight status information for: {query}"
        elif a == "DOC_AGENT":
            decomposition[a] = f"Baggage and check-in rules related to: {query}"
        elif a == "WEB_AGENT":
            decomposition[a] = f"Official PAA website information about: {query}"
    return decomposition

# ================= MAIN ENGINE =================
def run_engine(user_query):
    st.session_state.trace.clear()
    st.session_state.agent_status = {k: False for k in st.session_state.agent_status}

    st.session_state.trace.append(f"📥 User Query: {user_query}")

    agents = supervisor_router(user_query)
    st.session_state.trace.append(f"🧠 Supervisor Routing: {agents}")

    if agents == ["NONE"]:
        answer = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_query}]
        ).choices[0].message.content

        return answer

    sub_queries = decompose_query(user_query, agents)
    internal_results = []
    missing_entities = []

    for agent, sub_q in sub_queries.items():
        st.session_state.agent_status[agent] = True
        st.session_state.trace.append(f"➡️ {agent} activated")

        if agent == "XML_AGENT":
            data = weaviate_search(sub_q, "PAAWeb")
        elif agent == "DOC_AGENT":
            data = weaviate_search(sub_q, "PAAPolicy")
        elif agent == "WEB_AGENT":
            data = weaviate_search(sub_q, "RAG2_Web")
        else:
            data = []

        if data:
            internal_results.extend(data)
            st.session_state.trace.append(f"✅ {agent} found data")
        else:
            missing_entities.append(agent)
            st.session_state.trace.append(f"⚠️ {agent} returned NOT_FOUND")

    # ================= FINAL RESPONSE =================
    final_prompt = f"""
You are a professional PAA virtual assistant.

Internal Data:
{internal_results}

Missing Agents:
{missing_entities}

Rules:
- Use internal data if available.
- If an agent returned NOT_FOUND, acknowledge professionally.
- Use external knowledge ONLY for missing parts.
- Do not mention agents, RAG, or databases.

User Query:
{user_query}
"""

    answer = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": final_prompt}]
    ).choices[0].message.content

    return answer

# ================= UI =================
st.title("✈️ PAA Enterprise Intelligence")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("🧾 Trace Console")
    st.markdown(
        "<div style='background:#0e1117;padding:10px;height:400px;overflow:auto;font-family:monospace;'>"
        + "<br>".join(st.session_state.trace)
        + "</div>",
        unsafe_allow_html=True
    )

with col2:
    st.subheader("💬 Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if q := st.chat_input("Ask about flights, baggage, or PAA info"):
        answer = run_engine(q)
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
