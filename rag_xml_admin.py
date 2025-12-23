import streamlit as st
import xml.etree.ElementTree as ET
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import os
import re
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(page_title="PAA XML Flight Agent", layout="wide")
st.title("✈️ PAA XML Flight Intelligence Agent")

# ================= CONFIG =================
XML_PATH = "rag_xml_data/flight_snapshot.xml"
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]

COLLECTION_NAME = "PAA_XML_FLIGHTS"

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

EMBED = load_embedder()

# ================= HELPERS =================
def clean_xml(raw):
    raw = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", raw)
    return raw.strip()

def text(el, path):
    found = el.find(path)
    return found.text if found is not None else None

# ================= XML PARSER =================
def parse_xml_flights(xml_file):
    with open(xml_file, "r", encoding="utf-8", errors="ignore") as f:
        raw = clean_xml(f.read())

    root = ET.fromstring(raw)

    flights = []

    for env in root.findall(".//{http://schema.ultra-as.com}Envelope"):
        body = env.find(".//{http://schema.ultra-as.com}Body")
        if body is None:
            continue

        flight_data = body.find(".//{http://schema.ultra-as.com}AFDSFlightData")
        if flight_data is None:
            continue

        # Ignore junk / status-only messages
        flight_ident = flight_data.find(".//{http://schema.ultra-as.com}FlightIdentification")
        if flight_ident is None:
            continue

        flight_id = text(flight_ident, "{http://schema.ultra-as.com}FlightIdentity")
        direction = text(flight_ident, "{http://schema.ultra-as.com}FlightDirection")
        sched_date = text(flight_ident, "{http://schema.ultra-as.com}ScheduledDate")

        if not flight_id:
            continue

        fd = flight_data.find(".//{http://schema.ultra-as.com}FlightData")
        if fd is None:
            continue

        airport = fd.find(".//{http://schema.ultra-as.com}Airport")
        flight = fd.find(".//{http://schema.ultra-as.com}Flight")
        ops = fd.find(".//{http://schema.ultra-as.com}OperationalTimes")

        record = {
            "flight_number": flight_id,
            "direction": direction,
            "scheduled_date": sched_date,
            "airport": text(airport, "{http://schema.ultra-as.com}AirportIATACode") if airport is not None else None,
            "terminal": text(airport, "{http://schema.ultra-as.com}PassengerTerminalCode") if airport is not None else None,
            "gate": text(airport, "{http://schema.ultra-as.com}Gate/{http://schema.ultra-as.com}GateNumber") if airport is not None else None,
            "checkin_open": text(airport, "{http://schema.ultra-as.com}Checkin/{http://schema.ultra-as.com}CheckinOpenDateTime") if airport is not None else None,
            "checkin_close": text(airport, "{http://schema.ultra-as.com}Checkin/{http://schema.ultra-as.com}CheckinCloseDateTime") if airport is not None else None,
            "status_code": text(flight, "{http://schema.ultra-as.com}FlightStatusCode") if flight is not None else None,
            "scheduled_time": text(ops, "{http://schema.ultra-as.com}ScheduledDateTime") if ops is not None else None,
            "actual_time": text(ops, "{http://schema.ultra-as.com}LatestKnownDateTime") if ops is not None else None
        }

        flights.append(record)

    return flights

# ================= WEAVIATE INGEST =================
def ingest_flights(flights):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )

    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)

    coll = client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="flight_number", data_type=DataType.TEXT),
            Property(name="direction", data_type=DataType.TEXT),
            Property(name="airport", data_type=DataType.TEXT),
            Property(name="terminal", data_type=DataType.TEXT),
            Property(name="gate", data_type=DataType.TEXT),
            Property(name="status_code", data_type=DataType.TEXT),
            Property(name="summary", data_type=DataType.TEXT),
        ]
    )

    with coll.batch.dynamic() as batch:
        for f in flights:
            summary = (
                f"Flight {f['flight_number']} is a {f['direction']} flight at airport {f['airport']}. "
                f"Terminal {f['terminal']}, Gate {f['gate']}. "
                f"Status code {f['status_code']}. "
                f"Scheduled {f['scheduled_time']}, Latest {f['actual_time']}."
            )

            batch.add_object(
                properties={
                    "flight_number": f["flight_number"],
                    "direction": f["direction"],
                    "airport": f["airport"],
                    "terminal": f["terminal"],
                    "gate": f["gate"],
                    "status_code": f["status_code"],
                    "summary": summary
                },
                vector=EMBED.encode(summary).tolist()
            )

    client.close()

# ================= UI =================
st.markdown("### 📄 XML Source")
st.code(XML_PATH)

if not os.path.exists(XML_PATH):
    st.error("❌ flight_snapshot.xml not found.")
    st.stop()

if st.button("🏗️ Parse XML & Update Flight Index"):
    with st.spinner("Processing XML and updating flight intelligence index..."):
        flights = parse_xml_flights(XML_PATH)

        if not flights:
            st.warning("No valid flight records found.")
        else:
            ingest_flights(flights)
            st.success(f"✅ Indexed {len(flights)} flight records successfully!")
            st.balloons()
