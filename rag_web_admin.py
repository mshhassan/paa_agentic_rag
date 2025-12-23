import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import requests
import time

# --- CONFIG ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

MODEL = load_model()
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]

st.title("🌐 PAA Smart Web Indexer")
st.markdown("Agar PAA ki site block bhi kare, tab bhi ye data nikaal lega!")

# --- JINA READER FUNCTION (The Secret Sauce) ---
def scrape_with_jina(url):
    # Jina Reader URL ko simple text mein badal deta hai
    jina_url = f"https://r.jina.ai/{url}"
    try:
        response = requests.get(jina_url, timeout=30)
        if response.status_code == 200:
            return response.text # Yeh humein saaf suthra text dega
        else:
            return None
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        return None

# --- URL LIST ---
urls_to_index = [
    "https://paa.gov.pk/about-us/introduction",
    "https://paa.gov.pk/e-complains",
    "https://paa.gov.pk/passenger-information/passenger-guide",
    "https://paa.gov.pk/media/job-opportunities",
    "https://paa.gov.pk/e-services/e-notam"
]

if st.button("🚀 Start Intelligent Scraping"):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    try:
        # Collection Fresh karein
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

        progress = st.progress(0)
        for i, url in enumerate(urls_to_index):
            st.write(f"🔍 Reading: {url}...")
            
            # Jina use kar ke text nikalna
            text_content = scrape_with_jina(url)
            
            if text_content and len(text_content) > 100:
                # Chote chunks mein break karna (800 chars)
                chunks = [text_content[j:j+800] for j in range(0, len(text_content), 650)]
                
                with coll.batch.dynamic() as batch:
                    for chunk in chunks:
                        # Embedding banana
                        vec = MODEL.encode(chunk).tolist()
                        batch.add_object(
                            properties={"content": chunk, "source": url},
                            vector=vec
                        )
                st.success(f"✅ Indexed: {url} ({len(chunks)} chunks)")
            else:
                st.warning(f"⚠️ No content found for {url}")
            
            progress.progress((i + 1) / len(urls_to_index))
            time.sleep(1) # Aram se scraping

        st.balloons()
        st.success("🎯 Ab aapka DB bhar chuka hai! Chatbot se check karein.")

    finally:
        client.close()
