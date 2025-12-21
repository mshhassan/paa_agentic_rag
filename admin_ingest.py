import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import re
import os

# --- CONFIG ---
WEAVIATE_URL = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="PAA Admin Ingestor", layout="centered")
st.title("🛡️ PAA RAG - Clean & Rebuild")
st.warning("Yeh process purana saara database delete kar ke naya data load karega.")

if st.button("🚀 Wipe Old Data & Sync New AODB"):
    if not os.path.exists("flight_records.xml"):
        st.error("❌ flight_records.xml nahi mili! Pehle file upload karein.")
        st.stop()

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    try:
        # 1. WIPE OLD COLLECTIONS
        st.info("🗑️ Purana data saaf kiya ja raha hai...")
        target_colls = ["PAAFlightStatus", "PAAWebLink", "PAAPolicy"]
        for name in target_colls:
            if client.collections.exists(name):
                client.collections.delete(name)
            client.collections.create(
                name=name,
                properties=[Property(name="content", data_type=DataType.TEXT)],
                vectorizer_config=Configure.Vectorizer.none()
            )

        # 2. READ & PARSE XML
        with open("flight_records.xml", "r", encoding='utf-8') as f:
            xml_content = f.read()

        # Flexible extraction for AFDSFlightData or FlightData
        blocks = re.findall(r'<(?:[a-zA-Z0-9]+:)?(?:AFDS)?FlightData>(.*?)</(?:[a-zA-Z0-9]+:)?(?:AFDS)?FlightData>', xml_content, re.DOTALL)
        
        if not blocks:
            st.error("❌ XML Blocks nahi milay! Tags check karein.")
            st.stop()

        coll_flight = client.collections.get("PAAFlightStatus")
        success_count = 0

        for block in blocks:
            def get_val(tag):
                m = re.search(rf'<(?:[a-zA-Z0-9]+:)?{tag}[^>]*>(.*?)</(?:[a-zA-Z0-9]+:)?{tag}>', block, re.DOTALL)
                return m.group(1).strip() if m and m.group(1) else "TBD"

            f_id = get_val("FlightIdentity") or get_val("FlightNumber")
            # Creating a rich, operational summary
            summary = (
                f"Flight: {f_id} | Date: {get_val('ScheduledDate')} | Time: {get_val('ScheduledTime')}\n"
                f"Status: {get_val('FlightStatusCode')} | Gate: {get_val('GateIdentity')} | "
                f"Belt: {get_val('BaggageReclaimIdentity')} | Counters: {get_val('CheckInDesks')}\n"
                f"Direction: {'Arrival' if 'BaggageReclaimIdentity' in block else 'Departure'}"
            )
            
            vector = MODEL.encode(summary).tolist()
            coll_flight.data.insert(properties={"content": summary}, vector=vector)
            success_count += 1

        # 3. LOAD DEFAULT WEB LINKS (So system doesn't crash)
        coll_web = client.collections.get("PAAWebLink")
        links = [
            "Lost & Found Baggage: https://www.paa.gov.pk/lost-found",
            "Passenger Complaints: https://www.paa.gov.pk/complaints",
            "Flight Inquiry Live: https://www.paa.gov.pk/flight-inquiry"
        ]
        for l in links:
            coll_web.data.insert(properties={"content": l}, vector=MODEL.encode(l).tolist())

        st.success(f"✅ Clean Sync Successful! {success_count} flights and operational links loaded.")
        st.balloons()

    except Exception as e:
        st.error(f"Unexpected Error: {e}")
    finally:
        client.close()

# --- VALIDATOR ---
st.markdown("---")
if st.button("📊 Verify Database Count"):
    client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
    try:
        coll = client.collections.get("PAAFlightStatus")
        count = len(coll.query.fetch_objects(limit=1).objects) # Quick check
        st.write(f"Flights in Database: **{success_count if 'success_count' in locals() else 'Check Log'}**")
    finally:
        client.close()
