import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import requests
import time
import re

# --- 1. CONFIG & SESSION STATE ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

MODEL = load_model()
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]

if "processed_links" not in st.session_state:
    st.session_state.processed_links = set()

st.set_page_config(page_title="PAA Web Control Center", layout="wide")
st.title("🌐 PAA Web Knowledge Management")

# --- 2. DATA CLEANING FUNCTION ---
def clean_web_text(raw_text):
    if not raw_text: return ""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', raw_text) # Remove Images
    text = re.sub(r'blob:http://localhost/\S+', '', text) # Remove Blobs
    noise = ["Main Menu", "Follow Us", "Share", "Email Portals", "textLarge", "textSmall", "increment", "decrement"]
    for word in noise: text = text.replace(word, "")
    text = re.sub(r'#+', '', text)
    text = re.sub(r'={2,}', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

# --- 3. LINK GROUPS ---
LINK_GROUPS = {
    "📌 Core & Feedback": ["https://paa.gov.pk/", "https://paa.gov.pk/e-complains", "https://paa.gov.pk/about-us/introduction"],
    "✈️ Passenger Info": ["https://paa.gov.pk/passenger-information/passenger-guide", "https://paa.gov.pk/passenger-information/lost-and-found"],
    "💼 Business & Tenders": ["https://paa.gov.pk/allTender", "https://paa.gov.pk/business-opportunities/advertisement"],
    "🏗️ Projects & Jobs": ["https://paa.gov.pk/media/job-opportunities", "https://paa.gov.pk/projects/ongoing-projects"],
}

# --- 4. UI CONTROLS ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Settings")
    delete_existing = st.checkbox("🔥 Delete ALL existing Web Data?", value=False)
    wait_time = st.slider("Wait time (seconds)", 1, 10, 2)
    
    # --- NAYA BOX: CUSTOM URL ENTRY ---
    st.markdown("---")
    st.subheader("➕ Add Custom Links")
    custom_links_input = st.text_area("Paste URLs here (one per line or comma separated)", placeholder="https://example.com/page1\nhttps://example.com/page2")

with col2:
    st.subheader("🔗 Select Links to Index")
    selected_urls = []
    
    # Grouped Checkboxes
    for group_name, urls in LINK_GROUPS.items():
        with st.expander(group_name, expanded=True):
            for url in urls:
                is_done = url in st.session_state.processed_links
                label = f"✅ {url}" if is_done else url
                if st.checkbox(label, key=url):
                    selected_urls.append(url)

# --- 5. MERGE CUSTOM LINKS ---
if custom_links_input:
    # Split by newline or comma and clean whitespace
    extra_links = [link.strip() for link in re.split(r'[\n,]', custom_links_input) if link.strip().startswith("http")]
    if extra_links:
        st.info(f"➕ {len(extra_links)} custom links added to queue.")
        selected_urls.extend(extra_links)

# --- 6. PROCESSING LOGIC ---
if st.button("🚀 Start Group Indexing"):
    if not selected_urls:
        st.warning("Pehle koi link toh select ya enter karein!")
        st.stop()

    client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
    
    try:
        if delete_existing and client.collections.exists("RAG2_Web"):
            client.collections.delete("RAG2_Web")
            st.session_state.processed_links.clear()
        
        if not client.collections.exists("RAG2_Web"):
            coll = client.collections.create(
                name="RAG2_Web",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[Property(name="content", data_type=DataType.TEXT), Property(name="source", data_type=DataType.TEXT)]
            )
        else:
            coll = client.collections.get("RAG2_Web")

        progress_bar = st.progress(0)
        status = st.empty()

        for i, url in enumerate(selected_urls):
            status.info(f"🔍 Scraping ({i+1}/{len(selected_urls)}): {url}")
            try:
                res = requests.get(f"https://r.jina.ai/{url}", timeout=30)
                if res.status_code == 200:
                    clean_text = clean_web_text(res.text)
                    if len(clean_text) > 100:
                        chunks = [clean_text[j:j+800] for j in range(0, len(clean_text), 650)]
                        with coll.batch.dynamic() as batch:
                            for chunk in chunks:
                                vec = MODEL.encode(chunk).tolist()
                                batch.add_object(properties={"content": chunk, "source": url}, vector=vec)
                        st.session_state.processed_links.add(url)
                        st.success(f"🟢 Indexed: {url}")
                    else: st.warning(f"⚠️ Low content: {url}")
                else: st.error(f"❌ Error {res.status_code} on {url}")
            except Exception as e: st.error(f"⚠️ Failed {url}: {e}")

            progress_bar.progress((i + 1) / len(selected_urls))
            if i < len(selected_urls) - 1: time.sleep(wait_time)

        st.success("🎯 Indexing Complete!")
        st.balloons()
        time.sleep(2)
        st.rerun()

    finally:
        client.close()
