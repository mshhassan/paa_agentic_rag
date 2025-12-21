import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🌐 RAG 2: Website Crawler")
urls = st.text_area("Enter URLs (one per line):", "https://www.paa.gov.pk/lost-found\nhttps://www.paa.gov.pk/complaints\nhttps://www.paa.gov.pk/notams")

if st.button("🕷️ Crawl & Index Website"):
    client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
    try:
        if client.collections.exists("RAG2_Web"): client.collections.delete("RAG2_Web")
        coll = client.collections.create(name="RAG2_Web", vectorizer_config=Configure.Vectorizer.none(), properties=[Property(name="content", data_type=DataType.TEXT)])
        for url in urls.split('\n'):
            res = requests.get(url.strip())
            soup = BeautifulSoup(res.text, 'html.parser')
            text = ' '.join([p.text for p in soup.find_all(['p', 'h1', 'h2', 'li'])])
            chunks = [text[i:i+800] for i in range(0, len(text), 800)]
            for c in chunks:
                coll.data.insert(properties={"content": f"Source: {url} | {c}"}, vector=MODEL.encode(c).tolist())
            st.write(f"Done: {url}")
        st.success("✅ Website RAG Updated.")
    finally: client.close()
