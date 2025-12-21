import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import re

WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🛡️ PAA Admin - High-Detail Ingestion")

if st.button("🚀 Re-Train with Full Details"):
    client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
    
    try:
        # Recreating collection to add more properties if needed
        if client.collections.exists("PAAFlightStatus"): client.collections.delete("PAAFlightStatus")
        
        client.collections.create(
            name="PAAFlightStatus",
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="flight_num", data_type=DataType.TEXT),
                Property(name="status", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
        
        with open("flight_records.xml", "r") as f:
            xml_data = f.read()

        # Regex to capture all specific tags
        # Note: We use .get() style defaults to avoid errors if a tag is missing
        records = re.findall(r'<AFDSFlightData>(.*?)</AFDSFlightData>', xml_data, re.DOTALL)
        
        coll = client.collections.get("PAAFlightStatus")
        
        for rec in records:
            f_id = re.search(r'<FlightIdentity>(.*?)</FlightIdentity>', rec).group(1) if re.search(r'<FlightIdentity>(.*?)</FlightIdentity>', rec) else "N/A"
            f_date = re.search(r'<ScheduledDate>(.*?)</ScheduledDate>', rec).group(1) if re.search(r'<ScheduledDate>(.*?)</ScheduledDate>', rec) else "N/A"
            f_status = re.search(r'<FlightStatusCode>(.*?)</FlightStatusCode>', rec).group(1) if re.search(r'<FlightStatusCode>(.*?)</FlightStatusCode>', rec) else "SCH"
            f_gate = re.search(r'<GateIdentity>(.*?)</GateIdentity>', rec).group(1) if re.search(r'<GateIdentity>(.*?)</GateIdentity>', rec) else "TBD"
            f_belt = re.search(r'<BaggageReclaimIdentity>(.*?)</BaggageReclaimIdentity>', rec).group(1) if re.search(r'<BaggageReclaimIdentity>(.*?)</BaggageReclaimIdentity>', rec) else "TBD"
            f_counter = re.search(r'<CheckInDesks>(.*?)</CheckInDesks>', rec).group(1) if re.search(r'<CheckInDesks>(.*?)</CheckInDesks>', rec) else "N/A"
            
            # Creating a rich content string for the RAG
            full_summary = (
                f"Flight {f_id} on {f_date}. Status: {f_status}. "
                f"Gate: {f_gate}, Baggage Belt: {f_belt}, Check-in Counters: {f_counter}. "
                f"Full XML Fragment: {rec[:200]}..."
            )
            
            vector = MODEL.encode(full_summary).tolist()
            coll.data.insert(
                properties={"content": full_summary, "flight_num": f_id, "status": f_status},
                vector=vector
            )
            st.write(f"✅ Ingested: **{f_id}** | Status: {f_status} | Gate: {f_gate}")
            
        st.success("Detailed Training Complete!")
    finally:
        client.close()
