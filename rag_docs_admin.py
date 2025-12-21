import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import os
from pypdf import PdfReader

WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

st.title("📂 RAG 3: Internal Docs Manager")

# Filter files
ignore = [".py", "requirement.txt", "readme.txt", ".git", ".xml", ".png", ".jpg"]
files = [f for f in os.listdir('.') if os.path.isfile(f) and not any(x in f.lower() for x in ignore)]

selected = []
st.write("### 📁 Available Documents")
for f in files:
    if st.checkbox(f): selected.append(f)

if st.button("🏗️ Train from Selected Docs"):
    client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
    try:
        if client.collections.exists("RAG3_Docs"): client.collections.delete("RAG3_Docs")
        coll = client.collections.create(name="RAG3_Docs", vectorizer_config=Configure.Vectorizer.none(), properties=[Property(name="content", data_type=DataType.TEXT)])
        for file in selected:
            content = ""
            if file.endswith('.pdf'):
                reader = PdfReader(file)
                content = " ".join([p.extract_text() for p in reader.pages])
            else:
                with open(file, 'r') as f: content = f.read()
            
            # Simple chunking
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            for c in chunks:
                coll.data.insert(properties={"content": f"File: {file} | {c}"}, vector=MODEL.encode(c).tolist())
            st.write(f"Indexed: {file}")
        st.success("✅ Internal Docs RAG Ready.")
    finally: client.close()
