import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import json

# --- CONFIG ---
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBED = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
THRESHOLD = 0.7 

st.set_page_config(page_title="PAA Enterprise Intelligence", layout="wide")

# (CSS styles remain same as previous version)

# --- WEAVIATE RETRIEVER (Fixed for deep search) ---
def fetch_from_weaviate(query, collection_name):
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=st.secrets["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
        )
        coll = client.collections.get(collection_name)
        
        # Hummingbird/Keyword search match for flight numbers like SV726
        res = coll.query.near_vector(
            near_vector=EMBED.encode(query).tolist(), 
            limit=5, # Zyada records fetch kar rahe hain for safety
            return_properties=["content"]
        )
        client.close()
        
        if not res.objects:
            return "No specific flight record found in AODB."
            
        return "\n".join([o.properties['content'] for o in res.objects])
    except Exception as e:
        return f"Error connecting to Weaviate: {str(e)}"

# --- AGENTIC ENGINE ---
def run_paa_engine(query):
    st.session_state.trace.append(f"<b>Query:</b> {query}")
    
    # 1. Routing Decision
    analysis_prompt = f"""
    Analyze query: "{query}"
    Scores (0-1): XML (Flight info), Web (Links), Docs (Baggage/Policy).
    Return JSON: {{"XML": score, "Web": score, "Docs": score}}
    """
    
    resp = client_openai.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={"type":"json_object"}, 
        messages=[{"role":"system", "content":"You are a PAA Supervisor Agent."}, {"role":"user","content":analysis_prompt}]
    )
    scores = json.loads(resp.choices[0].message.content)
    st.session_state.scores = scores

    # 2. RAG Retrieval
    context = ""
    mapping = {"XML": "PAAWeb", "Web": "PAAWebLink", "Docs": "PAAPolicy"}
    
    for key, score in scores.items():
        if score >= THRESHOLD:
            st.session_state.trace.append(f"📡 {key} Agent Active (Score {score})")
            retrieved_text = fetch_from_weaviate(query, mapping[key])
            context += f"\n--- {key} DATA ---\n{retrieved_text}\n"
        else:
            st.session_state.trace.append(f"⚪ {key} Bypassed")

    # 3. Final Answer (STRICT PROMPT)
    # Yahan hum LLM ko force kar rahe hain ke hallucinate na kare
    system_instruction = f"""
    You are the PAA (Pakistan Airports Authority) Official Assistant.
    STRICT RULE: Only use the provided context to answer. 
    If context says nothing about the flight, say 'Flight data not found in current records'.
    DO NOT use your own knowledge about airlines (like MJets or Thai charters).
    
    CONTEXT DATA:
    {context}
    """
    
    final_messages = [
        {"role": "system", "content": system_instruction}
    ] + st.session_state.messages + [{"role": "user", "content": query}]
    
    ans_resp = client_openai.chat.completions.create(model="gpt-4o", messages=final_messages)
    answer = ans_resp.choices[0].message.content
    
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.trace.append("✅ Final Compounded Answer Generated.")

# (Rest of UI Layout remains same as previous version)
