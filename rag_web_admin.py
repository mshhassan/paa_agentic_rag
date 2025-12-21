import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import torch # Error fix ke liye zaroori hai
from urllib.parse import urljoin, urlparse

# --- ERROR FIX: Force CPU for Torch ---
device = "cpu"
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)

WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]

st.set_page_config(page_title="RAG 2: Auto-Web Crawler", layout="wide")
st.title("🌐 RAG 2: PAA Web Crawler (Auto-Discovery)")

# --- LINK DISCOVERY FUNCTION ---
def discover_paa_links(base_url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        for a in soup.find_all('a', href=True):
            full_url = urljoin(base_url, a['href'])
            # Sirf paa.gov.pk ke internal links rakhein
            if urlparse(base_url).netloc in urlparse(full_url).netloc:
                links.add(full_url)
        return sorted(list(links))
    except Exception as e:
        st.error(f"Discovery Error: {e}")
        return []

# --- AUTO-LOAD LINKS ---
if "found_links" not in st.session_state:
    with st.spinner("Discovering links from paa.gov.pk..."):
        st.session_state.found_links = discover_paa_links("https://www.paa.gov.pk")

st.subheader("🔗 Managed Links")
st.info("Niche PAA ke discovered links hain. Aap mazeed links (jaise caa.gov.pk) manually add kar sakte hain.")

# Text area with discovered links as default
all_links_text = st.text_area(
    "Edit Links (One per line):", 
    value="\n".join(st.session_state.found_links),
    height=300
)

if st.button("🕷️ Start Production Indexing"):
    links_to_crawl = [l.strip() for l in all_links_text.split('\n') if l.strip()]
    
    client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
    try:
        # Collection Cleanup
        if client.collections.exists("RAG2_Web"):
            client.collections.delete("RAG2_Web")
        
        coll = client.collections.create(
            name="RAG2_Web",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT)
            ]
        )

        progress_bar = st.progress(0)
        for i, url in enumerate(links_to_crawl):
            try:
                res = requests.get(url, timeout=5)
                soup = BeautifulSoup(res.text, 'html.parser')
                # Content cleaning
                for tag in soup(["script", "style", "nav", "footer"]): tag.decompose()
                text = soup.get_text(separator=' ', strip=True)
                
                # Production Chunking (800 chars)
                chunks = [text[j:j+800] for j in range(0, len(text), 800)]
                for c in chunks:
                    if len(c) > 100:
                        coll.data.insert(
                            properties={"content": c, "source": url},
                            vector=MODEL.encode(c).tolist()
                        )
                st.write(f"✅ Indexed: {url}")
            except:
                st.write(f"❌ Failed: {url}")
            
            progress_bar.progress((i + 1) / len(links_to_crawl))
            
        st.success(f"Production RAG updated with {len(links_to_crawl)} pages!")
        st.balloons()
    finally:
        client.close()
