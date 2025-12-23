import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import requests
import time

# --- 1. CONFIG & SESSION STATE ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

MODEL = load_model()
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]

# Indexing history track karne ke liye
if "processed_links" not in st.session_state:
    st.session_state.processed_links = set()

st.set_page_config(page_title="PAA Web Control Center", layout="wide")
st.title("🌐 PAA Web Knowledge Management")

# --- 2. LINK GROUPS (Organized) ---
LINK_GROUPS = {
    "📌 Core & Feedback": [
        "https://paa.gov.pk/",
        "https://paa.gov.pk/e-complains",
        "https://paa.gov.pk/about-us/introduction",
        "https://paa.gov.pk/about-us"
    ],
    "✈️ Passenger Info": [
        "https://paa.gov.pk/passenger-information/passenger-guide",
        "https://paa.gov.pk/passenger-information/lost-and-found",
        "https://paa.gov.pk/e-services/e-flight-inquiry"
    ],
    "💼 Business & Tenders": [
        "https://paa.gov.pk/allTender",
        "https://paa.gov.pk/business-opportunities/advertisement",
        "https://paa.gov.pk/business-opportunities/land-lease",
        "https://paa.gov.pk/business-opportunities/cargo-services"
    ],
    "🏗️ Projects & Jobs": [
        "https://paa.gov.pk/media/job-opportunities",
        "https://paa.gov.pk/projects/ongoing-projects",
        "https://paa.gov.pk/projects/completed-projects"
    ],
    "🏢 Major Airports": [
        "https://paa.gov.pk/airports/islamabad-international-airport",
        "https://paa.gov.pk/airports/jinnah-international-airport",
        "https://paa.gov.pk/airports/allama-iqbal-international-airport",
        "https://paa.gov.pk/airports/bacha-khan-international-airport"
    ],
    "🛡️ Safety & NOTAMs": [
        "https://paa.gov.pk/e-services/e-notam",
        "https://paa.gov.pk/aeronautical-information/aeronautical-information-publication-supplement",
        "https://paa.gov.pk/safety-and-security/safety-management-system"
    ]
}

# --- 3. UI CONTROLS ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Indexing Settings")
    delete_existing = st.checkbox("Delete existing Web Data before starting?", value=False)
    batch_size = st.slider("Links per Batch", 1, 15, 5)
    wait_time = st.slider("Wait time between links (seconds)", 1, 10, 2)
    
    st.markdown("""
    **Color Legend:**
    * 🟢 **Green:** Already Processed (in this session)
    * ⚪ **White:** New / Pending
    """)

with col2:
    st.subheader("🔗 Organized Link Groups")
    selected_urls = []
    
    for group_name, urls in LINK_GROUPS.items():
        with st.expander(group_name, expanded=True):
            for url in urls:
                # Color logic based on history
                is_done = url in st.session_state.processed_links
                label = f"✅ {url}" if is_done else url
                
                # Checkbox with custom styling hint
                if st.checkbox(label, key=url, value=False):
                    selected_urls.append(url)

# --- 4. PROCESSING LOGIC ---
if st.button("🚀 Start Group Indexing"):
    if not selected_urls:
        st.warning("Please select at least one link.")
        st.stop()

    client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
    
    try:
        # Collection Cleanup logic
        if delete_existing and client.collections.exists("RAG2_Web"):
            client.collections.delete("RAG2_Web")
            st.info("🗑️ Existing data cleared.")
        
        # Create collection if it doesn't exist
        if not client.collections.exists("RAG2_Web"):
            coll = client.collections.create(
                name="RAG2_Web",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT)
                ]
            )
        else:
            coll = client.collections.get("RAG2_Web")

        # Process in batches
        progress_bar = st.progress(0)
        status = st.empty()

        for i, url in enumerate(selected_urls):
            status.markdown(f"⏳ **Processing ({i+1}/{len(selected_urls)}):** {url}")
            
            # Jina Reader Call
            try:
                jina_url = f"https://r.jina.ai/{url}"
                res = requests.get(jina_url, timeout=30)
                
                if res.status_code == 200 and len(res.text) > 100:
                    text_content = res.text
                    chunks = [text_content[j:j+800] for j in range(0, len(text_content), 650)]
                    
                    with coll.batch.dynamic() as batch:
                        for chunk in chunks:
                            vec = MODEL.encode(chunk).tolist()
                            batch.add_object(properties={"content": chunk, "source": url}, vector=vec)
                    
                    st.session_state.processed_links.add(url)
                    st.write(f"🟢 Successfully indexed: {url}")
                else:
                    st.write(f"❌ Failed or Empty: {url}")
            except Exception as e:
                st.write(f"⚠️ Error with {url}: {e}")

            # Batching & Waiting
            progress_bar.progress((i + 1) / len(selected_urls))
            if i < len(selected_urls) - 1:
                time.sleep(wait_time)

        st.success("🎯 Batch Indexing Complete!")
        st.balloons()
        st.rerun() # Refresh to update colors

    finally:
        client.close()
