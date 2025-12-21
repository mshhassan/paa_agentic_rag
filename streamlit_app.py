import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer

client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBED = SentenceTransformer('all-MiniLM-L6-v2')

def get_answer(q):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url="04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud",
        auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
    )
    try:
        context = ""
        for c_name in ["RAG1_XML", "RAG2_Web", "RAG3_Docs"]:
            if client.collections.exists(c_name):
                res = client.collections.get(c_name).query.near_vector(near_vector=EMBED.encode(q).tolist(), limit=2)
                context += "\n".join([o.properties['content'] for o in res.objects])
        
        resp = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"You are PAA Assistant. Use context:\n{context}"}, {"role": "user", "content": q}]
        )
        return resp.choices[0].message.content
    finally: client.close()

st.title("🏛️ PAA Unified Intelligence")
prompt = st.chat_input("Ask anything...")
if prompt: st.write(get_answer(prompt))
