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

# Docs Directory
DOCS_DIR = "rag_docs_data"
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

st.title("📂 PAA Policy Manager (DOC_AGENT Admin)")
st.info(f"Upload your PDFs/Docs to `{DOCS_DIR}` to train the Baggage & Policy Agent.")

# --- FILE SCANNING ---
allowed_ext = [".pdf", ".txt", ".docx", ".md"]
files = [f for f in os.listdir(DOCS_DIR) if any(f.lower().endswith(ext) for ext in allowed_ext)]

if not files:
    st.warning(f"No documents found. Please add files to the `{DOCS_DIR}` folder.")
else:
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

    selected = st.multiselect(
        "Select files to index:", 
        options=files, 
        default=st.session_state.selected_files,
        key="file_selector"
    )

    # --- TRAINING LOGIC ---
    if st.button("🏗️ Index Knowledge Base"):
        if not selected:
            st.error("Please select at least one file.")
        else:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL, 
                auth_credentials=Auth.api_key(WEAVIATE_KEY)
            )
            try:
                collection_name = "PAAPolicy" 
                
                # Delete old collection to refresh data
                if client.collections.exists(collection_name):
                    client.collections.delete(collection_name)
                
                # Added properties for better filtering later
                coll = client.collections.create(
                    name=collection_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="page", data_type=DataType.INT)
                    ]
                )
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file_name in enumerate(selected):
                    file_path = os.path.join(DOCS_DIR, file_name)
                    status_text.text(f"Processing: {file_name}...")
                    
                    try:
                        if file_name.lower().endswith('.pdf'):
                            reader = PdfReader(file_path)
                            # Process page by page for better accuracy
                            for page_num, page in enumerate(reader.pages):
                                text = page.extract_text()
                                if not text or len(text.strip()) < 50:
                                    continue
                                
                                # Chunking within the page
                                chunk_size = 800 # Smaller chunks for higher precision
                                overlap = 150
                                chunks = [text[j:j+chunk_size] for j in range(0, len(text), chunk_size - overlap)]
                                
                                with coll.batch.dynamic() as batch:
                                    for c in chunks:
                                        # Combining metadata into content for better search retrieval
                                        meta_content = f"FILE: {file_name} (Page {page_num+1}) | {c}"
                                        batch.add_object(
                                            properties={
                                                "content": meta_content,
                                                "source": file_name,
                                                "page": page_num + 1
                                            },
                                            vector=MODEL.encode(c).tolist()
                                        )
                        else:
                            # Handling text files
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                chunks = [content[j:j+800] for j in range(0, len(content), 800 - 150)]
                                with coll.batch.dynamic() as batch:
                                    for c in chunks:
                                        batch.add_object(
                                            properties={"content": f"Source: {file_name} | {c}", "source": file_name, "page": 0},
                                            vector=MODEL.encode(c).tolist()
                                        )
                                        
                    except Exception as e:
                        st.error(f"Error in {file_name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(selected))

                st.success(f"🚀 DOC_AGENT is now trained with {len(selected)} documents!")
                st.balloons()
            finally:
                client.close()
