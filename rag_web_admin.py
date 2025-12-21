import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import torch
from urllib.parse import urljoin, urlparse
import urllib3

# SSL Warnings ko band karne ke liye
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- ERROR FIX: Force CPU for Torch ---
device = "cpu"
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)

WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]

st.set_page_config(page_title="RAG 2: Auto-Web Crawler", layout="wide")
st.title("🌐 RAG 2: PAA Web Crawler (SSL Bypass)")

# --- LINK DISCOVERY FUNCTION (Fixed SSL) ---
def discover_paa_links(base_url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        # verify=False yahan SSL error khatam karega
        response = requests.get(base_url, headers=headers, timeout=15, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        for a in soup.find_all('a', href=True):
            full_url = urljoin(base_url, a['href'])
            if urlparse(base_url).netloc in urlparse(full_url).netloc:
                links.add(full_url)
        return sorted(list(links))
    except Exception as e:
        st.error(f"Discovery Error: {e}")
        return []

# --- AUTO-LOAD LINKS ---
if "found_links" not in st.session_state:
    with st.spinner("Discovering links from paa.gov.pk (SSL Secure Mode)..."):
        links = discover_paa_links("https://www.paa.gov.pk")
        st.session_state.found_links = links

st.subheader("🔗 Managed Links")
all_links_text = st.text_area(
    "Edit Links (One per line):", 
    value="\n".join(st.session_state.found_links) if st.session_state.found_links else "https://www.paa.gov.pk",
    height=300
)

if st.button("🕷️ Start Production Indexing"):
    links_to_crawl = [l.strip() for l in all_links_text.split('\n') if l.strip()]
    
    client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
    try:
        # Collection Cleanup (Sirf RAG2 delete hoga)
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
                # verify=False yahan bhi scraping ke liye
                res = requests.get(url, timeout=10, verify=False, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(res.text, 'html.parser')
                for tag in soup(["script", "style", "nav", "footer"]): tag.decompose()
                text = soup.get_text(separator=' ', strip=True)
                
                chunks = [text[j:j+1000] for j in range(0, len(text), 1000)]
                for c in chunks:
                    if len(c) > 100:
                        coll.data.insert(
                            properties={"content": c, "source": url},
                            vector=MODEL.encode(c).tolist()
                        )
                st.write(f"✅ Indexed: {url}")
            except Exception as e:
                st.write(f"❌ Failed: {url} | Reason: {e}")
            
            progress_bar.progress((i + 1) / len(links_to_crawl))
            
        st.success(f"Production RAG 2 updated!")
        st.balloons()
    finally:
        client.close()
