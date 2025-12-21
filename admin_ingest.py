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

st.title("🛡️ PAA Knowledge Base Ingestor")
st.markdown("Yeh script Flights, Web Links, aur Policy data load karegi.")

if st.button("🚀 Build Full Database"):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    try:
        # 1. Sab collections ko refresh karein
        collections_to_init = ["PAAFlightStatus", "PAAWebLink", "PAAPolicy"]
        for name in collections_to_init:
            if client.collections.exists(name):
                client.collections.delete(name)
            client.collections.create(
                name=name,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT), # Optional filtering ke liye
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )

        # 2. FLIGHT INGESTION (From XML)
        if os.path.exists("flight_records.xml"):
            with open("flight_records.xml", "r", encoding='utf-8') as f:
                xml_data = f.read()
            
            records = re.findall(r'<(?:[a-zA-Z0-9]+:)?AFDSFlightData>(.*?)</(?:[a-zA-Z0-9]+:)?AFDSFlightData>', xml_data, re.DOTALL)
            coll_flight = client.collections.get("PAAFlightStatus")
            
            for rec in records:
                def extract(tag):
                    m = re.search(rf'<(?:[a-zA-Z0-9]+:)?{tag}[^>]*>(.*?)</(?:[a-zA-Z0-9]+:)?{tag}>', rec, re.DOTALL)
                    return m.group(1).strip() if m else "N/A"

                f_id = extract("FlightIdentity") or extract("FlightNumber")
                summary = f"Flight {f_id}: Date {extract('ScheduledDate')}, Status {extract('FlightStatusCode')}, Gate {extract('GateIdentity')}, Belt {extract('BaggageReclaimIdentity')}, Counters {extract('CheckInDesks')}."
                
                coll_flight.data.insert(
                    properties={"content": summary, "category": "flight"},
                    vector=MODEL.encode(summary).tolist()
                )
            st.success(f"✅ {len(records)} Flights Ingested.")

        # 3. WEB LINKS INGESTION (Operational Links)
        coll_web = client.collections.get("PAAWebLink")
        links = [
            {"t": "Lost and Found Baggage", "u": "https://www.paa.gov.pk/lost-found"},
            {"t": "Aeronautical Information (NOTAMs)", "u": "https://www.paa.gov.pk/notams"},
            {"t": "Passenger Complaint Cell", "u": "https://www.paa.gov.pk/complaints"},
            {"t": "Flight Inquiry PAA", "u": "https://www.paa.gov.pk/flight-inquiry"}
        ]
        for l in links:
            txt = f"{l['t']} Link: {l['u']}"
            coll_web.data.insert(properties={"content": txt, "category": "web"}, vector=MODEL.encode(txt).tolist())
        st.success("✅ Operational Web Links Ingested.")

        # 4. POLICY PLACEHOLDER (Wait for PDF)
        coll_policy = client.collections.get("PAAPolicy")
        policy_txt = "Standard PAA Baggage Policy: Economy 20kg, Business 30kg. Claims for lost items must be made within 24 hours."
        coll_policy.data.insert(properties={"content": policy_txt, "category": "policy"}, vector=MODEL.encode(policy_txt).tolist())
        
        st.balloons()
        st.success("🎉 PAA Knowledge Base is now Ready!")

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        client.close()
