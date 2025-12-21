import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import re

# Configuration
WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🛡️ PAA Admin - Data Training")

if st.button("🚀 Start Training (Clean & Ingest)"):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    try:
        # Clear existing collections
        for name in ["PAAFlightStatus", "PAAPolicy", "PAAWebLink"]:
            if client.collections.exists(name): client.collections.delete(name)
        
        # Create Flight Collection
        client.collections.create(
            name="PAAFlightStatus",
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="flight_num", data_type=DataType.TEXT)
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
        
        # Process XML (Direct Regex to avoid namespace issues)
        with open("flight_records.xml", "r") as f:
            xml_data = f.read()
        
        flights = re.findall(r'<FlightIdentity>(.*?)</FlightIdentity>.*?<ScheduledDate>(.*?)</ScheduledDate>', xml_data, re.DOTALL)
        
        coll = client.collections.get("PAAFlightStatus")
        for f_id, f_date in flights:
            content = f"Flight {f_id} status for scheduled date {f_date}."
            vector = MODEL.encode(content).tolist()
            coll.data.insert(properties={"content": content, "flight_num": f_id}, vector=vector)
            st.write(f"✅ Ingested: {f_id}")
            
        st.success("Training complete! Data is now searchable.")
    finally:
        client.close()
