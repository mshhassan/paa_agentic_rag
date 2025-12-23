import os
import re
import csv
import json
import glob
import xml.etree.ElementTree as ET
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import streamlit as st

# ================= PAGE CONFIG =================
st.set_page_config(page_title="PAA RAG Flight Admin", layout="wide")
st.title("✈️ PAA RAG Flight Admin - CSV & XML Ingestion")

# ================= CONFIG =================
DATA_FOLDER = "rag_xml_data"
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
COLLECTION_NAME = "PAA_XML_FLIGHTS"

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

EMBED = load_embedder()

# ================= HELPERS =================
def clean_text(raw):
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", raw).strip()

def parse_csv_file(path):
    records = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize keys and remove empty columns
            clean_row = {k.strip(): v.strip() for k, v in row.items() if v.strip()}
            if clean_row:
                records.append(clean_row)
    return records

def parse_xml_file(path):
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = clean_text(f.read())
        envelopes = re.findall(r'(<Envelope[\s\S]*?</Envelope>)', raw)
        ns = {"ns": "http://schema.ultra-as.com"}

        for env_xml in envelopes:
            try:
                root = ET.fromstring(env_xml)
            except ET.ParseError:
                continue
            flight_data = root.find(".//ns:AFDSFlightData", ns)
            if not flight_data:
                continue
            flight_ident = flight_data.find(".//ns:FlightIdentification", ns)
            if not flight_ident:
                continue
            flight_id = flight_ident.findtext("ns:FlightIdentity", default=None, namespaces=ns)
            direction = flight_ident.findtext("ns:FlightDirection", default=None, namespaces=ns)
            sched_date = flight_ident.findtext("ns:ScheduledDate", default=None, namespaces=ns)
            fd = flight_data.find(".//ns:FlightData", ns)
            airport = fd.find(".//ns:Airport", ns) if fd is not None else None
            flight = fd.find(".//ns:Flight", ns) if fd is not None else None
            ops = fd.find(".//ns:OperationalTimes", ns) if fd is not None else None

            record = {
                "flight_number": flight_id,
                "direction": direction,
                "scheduled_date": sched_date,
                "airport": airport.findtext("ns:AirportIATACode", default=None, namespaces=ns) if airport else None,
                "terminal": airport.findtext("ns:PassengerTerminalCode", default=None, namespaces=ns) if airport else None,
                "gate": airport.findtext(".//ns:GateNumber", default=None, namespaces=ns) if airport else None,
                "status_code": flight.findtext("ns:FlightStatusCode", default=None, namespaces=ns) if flight else None,
                "scheduled_time": ops.findtext("ns:ScheduledDateTime", default=None, namespaces=ns) if ops else None,
                "actual_time": ops.findtext("ns:LatestKnownDateTime", default=None, namespaces=ns) if ops else None,
            }
            records.append(record)
    return records

def ingest_to_weaviate(records):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )

    # Delete if exists
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
        for f in records:
            summary = (
                f"Flight {f.get('flight_number')} is a {f.get('direction')} flight at airport {f.get('airport')}. "
                f"Terminal {f.get('terminal')}, Gate {f.get('gate')}. "
                f"Status code {f.get('status_code')}. "
                f"Scheduled {f.get('scheduled_time')}, Latest {f.get('actual_time')}."
            )
            batch.add_object(
                properties={
                    "flight_number": f.get("flight_number"),
                    "direction": f.get("direction"),
                    "airport": f.get("airport"),
                    "terminal": f.get("terminal"),
                    "gate": f.get("gate"),
                    "status_code": f.get("status_code"),
                    "summary": summary
                },
                vector=EMBED.encode(summary).tolist()
            )
    client.close()

# ================= UI =================
st.markdown(f"### 📂 Scanning folder: {DATA_FOLDER}")
files = glob.glob(os.path.join(DATA_FOLDER, "*.*"))

if not files:
    st.error("No CSV or XML files found in the data folder.")
    st.stop()

st.write(files)

if st.button("🏗️ Parse & Index All Files"):
    all_records = []
    for f in files:
        if f.lower().endswith(".csv"):
            all_records.extend(parse_csv_file(f))
        elif f.lower().endswith(".xml"):
            all_records.extend(parse_xml_file(f))

    if all_records:
        ingest_to_weaviate(all_records)
        st.success(f"✅ Indexed {len(all_records)} flight records successfully!")
        st.balloons()
    else:
        st.warning("No valid flight records found.")
