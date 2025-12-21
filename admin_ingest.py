import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import re
import os

WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🛡️ PAA Admin - Database Repair")

if st.button("🔧 Re-Ingest Flight Data (Detailed)"):
    if not os.path.exists("flight_records.xml"):
        st.error("❌ flight_records.xml file nahi mili! Pehle file upload karein.")
        st.stop()

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    try:
        # Step 1: Read and Parse XML carefully
        with open("flight_records.xml", "r") as f:
            xml_data = f.read()

        # Regex to find each flight block
        records = re.findall(r'<AFDSFlightData>(.*?)</AFDSFlightData>', xml_data, re.DOTALL)
        
        if not records:
            st.error("❌ XML parse nahi ho saki. Tags check karein.")
            st.stop()

        # Step 2: Delete and Recreate only if records found
        if client.collections.exists("PAAFlightStatus"):
            client.collections.delete("PAAFlightStatus")
        
        coll = client.collections.create(
            name="PAAFlightStatus",
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="flight_num", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
        
        # Step 3: Ingesting with details
        for rec in records:
            # Clean extraction of tags (stripping namespaces if any)
            def get_tag(tag_name):
                match = re.search(rf'<{tag_name}>(.*?)</{tag_name}>', rec)
                return match.group(1) if match else "N/A"

            f_id = get_tag("FlightIdentity")
            f_date = get_tag("ScheduledDate")
            f_status = get_tag("FlightStatusCode")
            f_gate = get_tag("GateIdentity")
            f_belt = get_tag("BaggageReclaimIdentity")
            f_counter = get_tag("CheckInDesks")
            f_direction = "Arrival" if "BaggageReclaimIdentity" in rec else "Departure"

            full_summary = (
                f"Flight: {f_id} | Date: {f_date} | Type: {f_direction} | "
                f"Status: {f_status} | Gate: {f_gate} | Belt: {f_belt} | Counters: {f_counter}"
            )
            
            vector = MODEL.encode(full_summary).tolist()
            coll.data.insert(
                properties={"content": full_summary, "flight_num": f_id},
                vector=vector
            )
            st.write(f"✅ Re-filled: {f_id} ({f_direction})")
            
        st.success(f"Database successfully filled with {len(records)} records!")
    finally:
        client.close()
