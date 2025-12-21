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
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]

# Folder path jahan aap documents rakhenge
DOCS_DIR = "rag_docs_data"

# Agar folder nahi bana hua to bana dein
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

st.title("📂 RAG 3: Internal Docs Manager")
st.info(f"Put your PDF or Text files in the `{DOCS_DIR}` folder in your Git repo.")

# --- FILE SCANNING ---
# Hum sirf relevant extensions scan karenge
allowed_ext = [".pdf", ".txt", ".docx", ".md"]
files = [f for f in os.listdir(DOCS_DIR) if any(f.lower().endswith(ext) for ext in allowed_ext)]

if not files:
    st.warning(f"No documents found in `{DOCS_DIR}`. Please upload files to Git first.")
else:
    st.write(f"### 📁 Found {len(files)} Documents")
    
    # Select All / Deselect All logic
    col1, col2 = st.columns(2)
    select_all = col1.button("✅ Select All")
    deselect_all = col2.button("❌ Deselect All")

    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []

    if select_all:
        st.session_state.selected_files = files
    if deselect_all:
        st.session_state.selected_files = []

    # Display Checkboxes
    selected = []
    for f in files:
        is_checked = f in st.session_state.selected_files
        if st.checkbox(f, value=is_checked, key=f):
            selected.append(f)

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
                # Collection naming as per your previous requirement
                collection_name = "PAAPolicy" 
                
                if client.collections.exists(collection_name):
                    client.collections.delete(collection_name)
                
                coll = client.collections.create(
                    name=collection_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[Property(name="content", data_type=DataType.TEXT)]
                )
                
                progress_bar = st.progress(0)
                for i, file_name in enumerate(selected):
                    file_path = os.path.join(DOCS_DIR, file_name)
                    content = ""
                    
                    # Extraction logic
                    try:
                        if file_name.lower().endswith('.pdf'):
                            reader = PdfReader(file_path)
                            content = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
                        else:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        
                        # Overlap chunking for better context (1000 chars with 200 overlap)
                        chunk_size = 1000
                        overlap = 200
                        chunks = [content[j:j+chunk_size] for j in range(0, len(content), chunk_size - overlap)]
                        
                        for c in chunks:
                            if len(c.strip()) > 50: # Avoid empty chunks
                                meta_content = f"Source: {file_name} | {c}"
                                coll.data.insert(
                                    properties={"content": meta_content},
                                    vector=MODEL.encode(c).tolist()
                                )
                        st.write(f"✅ Indexed: {file_name}")
                    except Exception as e:
                        st.error(f"Failed to process {file_name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(selected))

                st.success(f"🚀 RAG 3 is now updated with {len(selected)} documents!")
                st.balloons()
            finally:
                client.close()
