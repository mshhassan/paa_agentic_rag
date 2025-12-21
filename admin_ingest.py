import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET
import re

# Config
WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = "TTFrTjlobjJDTTJXZkdSaF92bEVBOW56T1BrY093SGVGTys4dG5nNG56UTdMM1JnOEw1OEJxVkpvajhRPV92MjAw"
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🛡️ PAA Admin - Data Training")

if st.button("🚀 Start Ingestion / Training"):
    client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
    
    # 1. Clear Old Data
    for name in ["PAAFlightStatus", "PAAPolicy", "PAAWebLink"]:
        if client.collections.exists(name): client.collections.delete(name)
    
    # 2. Create Schema
    client.collections.create(
        name="PAAFlightStatus",
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="flight_num", data_type=DataType.TEXT)
        ],
        vectorizer_config=Configure.Vectorizer.none()
    )
    
    # 3. Enhanced XML Parsing (Fixing the 'Empty Array' bug)
    try:
        with open("/content/flight_records.xml", 'r') as f:
            xml_data = f.read()
        
        # Regex to find all Flight records regardless of tags
        flights = re.findall(r'<FlightIdentity>(.*?)</FlightIdentity>.*?<ScheduledDate>(.*?)</ScheduledDate>', xml_data, re.DOTALL)
        
        collection = client.collections.get("PAAFlightStatus")
        for f_id, f_date in flights:
            content = f"Flight {f_id} status for date {f_date}."
            vector = MODEL.encode(content).tolist()
            collection.data.insert(properties={"content": content, "flight_num": f_id}, vector=vector)
            st.write(f"✅ Ingested: {f_id}")
            
        st.success("Training Complete! Database is now populated.")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        client.close()
