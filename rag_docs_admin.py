import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import os
from pypdf import PdfReader
import torch

# --- CONFIG ---
device = "cpu"
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)

MODEL = load_model()
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]

# Folder path jahan aap documents rakhenge
DOCS_DIR = "rag_docs_data"

# Agar folder nahi bana hua to bana dein
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

st.title("📂 RAG 3: Internal Docs Manager")
st.info(f"Put your PDF or Text files in the `{DOCS_DIR}` folder in your Git repo.")

# --- FILE SCANNING ---
allowed_ext = [".pdf", ".txt", ".docx", ".md"]
files = [f for f in os.listdir(DOCS_DIR) if any(f.lower().endswith(ext) for ext in allowed_ext)]

if not files:
    st.warning(f"No documents found in `{DOCS_DIR}`. Please upload files to Git first.")
else:
    # --- SESSION STATE FOR SELECTION ---
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []

    st.write(f"### 📁 Found {len(files)} Documents")
    
    col1, col2 = st.columns(2)
    
    if col1.button("✅ Select All"):
        st.session_state.selected_files = files
        st.rerun()

    if col2.button("❌ Deselect All"):
        st.session_state.selected_files = []
        st.rerun()

    # Multiselect provides the most stable "Select All" behavior in Streamlit
    selected = st.multiselect(
        "Select files to train:", 
        options=files, 
        default=st.session_state.selected_files,
        key="file_selector"
    )

    # --- TRAINING LOGIC ---
    if st.button("🏗️ Train / Update Vector Database"):
        if not selected:
            st.error("Please select at least one file to train.")
        else:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL, 
                auth_credentials=Auth.api_key(WEAVIATE_KEY)
            )
            try:
                collection_name = "PAAPolicy" 
                
                # Naya collection banane se pehle purana delete karein
                if client.collections.exists(collection_name):
                    client.collections.delete(collection_name)
                
                coll = client.collections.create(
                    name=collection_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[Property(name="content", data_type=DataType.TEXT)]
                )
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file_name in enumerate(selected):
                    file_path = os.path.join(DOCS_DIR, file_name)
                    content = ""
                    
                    status_text.text(f"Processing: {file_name}...")
                    
                    try:
                        if file_name.lower().endswith('.pdf'):
                            reader = PdfReader(file_path)
                            content = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
                        else:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        
                        # Overlap chunking (1000 chars with 200 overlap)
                        chunk_size = 1000
                        overlap = 200
                        chunks = [content[j:j+chunk_size] for j in range(0, len(content), chunk_size - overlap)]
                        
                        # Batch insert chunks
                        with coll.batch.dynamic() as batch:
                            for c in chunks:
                                if len(c.strip()) > 50:
                                    meta_content = f"Source: {file_name} | {c}"
                                    batch.add_object(
                                        properties={"content": meta_content},
                                        vector=MODEL.encode(c).tolist()
                                    )
                        
                    except Exception as e:
                        st.error(f"Failed to process {file_name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(selected))

                status_text.empty()
                st.success(f"🚀 RAG 3 is now updated with {len(selected)} documents!")
                st.balloons()
            finally:
                client.close()
