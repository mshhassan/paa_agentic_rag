import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib3
import re
import time

# SSL/Security Warnings bypass (Necessary for gov sites)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 1. MODEL & CONFIG ---
@st.cache_resource
def load_resources():
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

MODEL = load_resources()
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]

st.set_page_config(page_title="PAA Intelligence Admin", layout="wide", page_icon="🌐")
st.title("🌐 PAA Enterprise Web Indexer")
st.markdown("---")

# --- 2. ADVANCED DISCOVERY (Deep Crawl) ---
def discover_paa_links(base_url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
        response = requests.get(base_url, headers=headers, timeout=20, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        
        # PAA ki website ke patterns ko target karna
        for a in soup.find_all('a', href=True):
            full_url = urljoin(base_url, a['href']).split('#')[0].rstrip('/')
            
            # Domain check aur unwanted files filtering
            if urlparse(base_url).netloc in urlparse(full_url).netloc:
                excluded = ['.pdf', '.jpg', '.png', '.zip', '.docx', '.xlsx', 'logout', 'login']
                if not any(ext in full_url.lower() for ext in excluded):
                    links.add(full_url)
        
        return sorted(list(links))
    except Exception as e:
        st.error(f"Discovery Error: {e}")
        return ["https://www.paa.gov.pk"]

# Session state to manage discovered links
if "discovered_links" not in st.session_state:
    with st.spinner("🔍 Mapping PAA Website Structure..."):
        st.session_state.discovered_links = discover_paa_links("https://www.paa.gov.pk")

# --- 3. UI LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Settings")
    chunk_size = st.slider("Chunk Size", 500, 1500, 800)
    overlap = st.slider("Overlap", 50, 300, 150)
    st.info("Large overlap preserves context between chunks.")

with col2:
    st.subheader("URLs to Index")
    selected_links = st.text_area("URLs List (Editable):", 
                                  value="\n".join(st.session_state.discovered_links), 
                                  height=300)

# --- 4. CORE INDEXING ENGINE (Smart Batching) ---
if st.button("🚀 Start Carrier-Grade Indexing"):
    target_urls = [l.strip() for l in selected_links.split('\n') if l.strip()]
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    try:
        # Re-create collection for fresh data
        if client.collections.exists("RAG2_Web"):
            client.collections.delete("RAG2_Web")
        
        coll = client.collections.create(
            name="RAG2_Web",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="page_title", data_type=DataType.TEXT)
            ]
        )

        progress_bar = st.progress(0)
        log_box = st.empty()

        for idx, url in enumerate(target_urls):
            try:
                # Scraping with timeout & retries
                res = requests.get(url, timeout=15, verify=False, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(res.text, 'html.parser')
                
                # Metadata extraction
                title = soup.title.string if soup.title else "PAA Page"
                
                # Heavy Noise Cleaning (Removing headers, footers, scripts)
                for tag in soup(["script", "style", "nav", "footer", "header", "form"]): 
                    tag.decompose()
                
                # Content Cleaning
                raw_text = soup.get_text(separator=' ', strip=True)
                clean_text = re.sub(r"\s+", " ", raw_text)

                # --- SMART OVERLAP CHUNKING ---
                chunks = [clean_text[j:j+chunk_size] for j in range(0, len(clean_text), chunk_size - overlap)]

                with coll.batch.dynamic() as batch:
                    for c in chunks:
                        if len(c.strip()) > 150:
                            # Embedding each chunk
                            vector = MODEL.encode(c).tolist()
                            
                            # Injecting Source and Context into Content (Critical for LLM)
                            enriched_content = f"WEBSITE: {title} | URL: {url} | DATA: {c}"
                            
                            batch.add_object(
                                properties={
                                    "content": enriched_content,
                                    "source": url,
                                    "page_title": title
                                },
                                vector=vector
                            )
                log_box.success(f"Successfully Indexed: {url}")
            except Exception as e:
                log_box.error(f"Failed to process {url}: {e}")
            
            progress_bar.progress((idx + 1) / len(target_urls))
            time.sleep(0.1) # Prevent rate limiting

        st.success("🎯 Carrier-Grade Knowledge Base is Ready!")
        st.balloons()

    finally:
        client.close()
