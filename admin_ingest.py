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

st.title("🔧 PAA Database Repair (Deep Parse)")

if st.button("🚀 Force Re-Ingest Flight Data"):
    if not os.path.exists("flight_records.xml"):
        st.error("❌ flight_records.xml file nahi mili!")
        st.stop()

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    try:
        with open("flight_records.xml", "r", encoding='utf-8') as f:
            xml_data = f.read()

        # Step 1: Find all flight blocks using a more flexible regex
        # Yeh har us cheez ko pakray ga jo FlightData ya AFDSFlightData ke darmian hai
        records = re.findall(r'<(?:[a-zA-Z0-9]+:)?AFDSFlightData>(.*?)</(?:[a-zA-Z0-9]+:)?AFDSFlightData>', xml_data, re.DOTALL)
        
        if not records:
            # Agar AFDSFlightData nahi mila, toh generic FlightData check karte hain
            records = re.findall(r'<(?:[a-zA-Z0-9]+:)?FlightData>(.*?)</(?:[a-zA-Z0-9]+:)?FlightData>', xml_data, re.DOTALL)

        if not records:
            st.error("❌ XML Blocks nahi milay. File ka content empty hai ya tags mukhtalif hain.")
            st.stop()

        # Step 2: Collection Refresh
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
        
        # Step 3: Deep Tag Extraction
        success_count = 0
        for rec in records:
            def extract(tag):
                # Yeh regex namespaces (like ns0:Tag) ko ignore kar ke data nikalta hai
                m = re.search(rf'<(?:[a-zA-Z0-9]+:)?{tag}[^>]*>(.*?)</(?:[a-zA-Z0-9]+:)?{tag}>', rec, re.DOTALL)
                return m.group(1).strip() if m else "N/A"

            f_id = extract("FlightIdentity")
            # Kuch files mein FlightIdentity ki jagah FlightNumber hota hai
            if f_id == "N/A": f_id = extract("FlightNumber")
            
            f_date = extract("ScheduledDate")
            f_time = extract("ScheduledDateTime")
            f_status = extract("FlightStatusCode")
            f_gate = extract("GateIdentity")
            f_belt = extract("BaggageReclaimIdentity")
            f_counter = extract("CheckInDesks")
            
            # Create a Very Detailed Summary for RAG
            full_summary = (
                f"Flight Information for {f_id}:\n"
                f"- Scheduled Date/Time: {f_date} {f_time}\n"
                f"- Operational Status: {f_status}\n"
                f"- Assigned Gate: {f_gate}\n"
                f"- Baggage Reclaim Belt: {f_belt}\n"
                f"- Check-in Counters: {f_counter}\n"
                f"Note: This is a live AODB record."
            )
            
            vector = MODEL.encode(full_summary).tolist()
            coll.data.insert(
                properties={"content": full_summary, "flight_num": f_id},
                vector=vector
            )
            success_count += 1
            st.write(f"✅ Loaded: **{f_id}** (Status: {f_status})")
            
        st.success(f"Success! {success_count} records ingested into Weaviate.")
    
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
    finally:
        client.close()
