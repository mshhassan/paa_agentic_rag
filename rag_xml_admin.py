import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import os
import re

# ================= PAGE CONFIG =================
st.set_page_config(page_title="PAA XML Flight Intelligence Agent", layout="wide")
st.title("✈️ PAA XML Flight Intelligence Agent")

# ================= CONFIG =================
XML_PATH = "rag_xml_data/flight_snapshot.xml"
CSV_FOLDER = "rag_xml_data"
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
COLLECTION_NAME = "PAAWeb"

# ================= MODEL =================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

EMBED = load_embedder()

# ================= HELPERS =================
def clean_xml(raw):
    raw = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", raw)
    return raw.strip()

def build_dynamic_mappings(csv_folder=CSV_FOLDER):
    """
    Scan all CSV files and build a dict of dicts automatically.
    Each CSV: first two columns are treated as key -> value mapping.
    """
    mappings = {}
    for fname in os.listdir(csv_folder):
        if fname.lower().endswith(".csv"):
            fpath = os.path.join(csv_folder, fname)
            try:
                df = pd.read_csv(fpath)
                cols = df.columns[:2].tolist()
                if len(cols) < 2:
                    continue
                key_col, value_col = cols
                map_name = os.path.splitext(fname)[0]  # filename without extension
                mappings[map_name] = dict(zip(df[key_col].astype(str), df[value_col].astype(str)))
            except Exception as e:
                st.warning(f"⚠️ Could not process {fname}: {e}")
    return mappings

# ================= XML PARSER =================
def parse_xml_flights(xml_file, mappings):
    with open(xml_file, "r", encoding="utf-8", errors="ignore") as f:
        raw = clean_xml(f.read())

    envelopes = re.findall(r'(<Envelope[\s\S]*?</Envelope>)', raw)
    flights = []

    for env_xml in envelopes:
        try:
            root = ET.fromstring(env_xml)
        except ET.ParseError:
            continue

        ns = {"ns": "http://schema.ultra-as.com"}
        flight_data = root.find(".//ns:AFDSFlightData", ns)
        if flight_data is None:
            continue

        flight_ident = flight_data.find(".//ns:FlightIdentification", ns)
        if flight_ident is None:
            continue

        flight_id = flight_ident.findtext("ns:FlightIdentity", default=None, namespaces=ns)
        direction = flight_ident.findtext("ns:FlightDirection", default=None, namespaces=ns)

        fd = flight_data.find(".//ns:FlightData", ns)
        airport_el = fd.find(".//ns:Airport", ns) if fd is not None else None
        flight_el = fd.find(".//ns:Flight", ns) if fd is not None else None
        ops_el = fd.find(".//ns:OperationalTimes", ns) if fd is not None else None

        if not flight_id:
            continue

        # --- Use dynamic mappings ---
        airline_map = mappings.get("airlines", {})
        airport_map = mappings.get("airports", {})
        status_map = mappings.get("status_codes", {})
        aircraft_map = mappings.get("aircraft_types", {})

        carrier_code = flight_el.findtext("ns:CarrierIATACode", default=None, namespaces=ns) if flight_el is not None else None
        flight_status_code = flight_el.findtext("ns:FlightStatusCode", default=None, namespaces=ns) if flight_el is not None else None
        airport_code = airport_el.findtext("ns:AirportIATACode", default=None, namespaces=ns) if airport_el is not None else None
        terminal = airport_el.findtext("ns:PassengerTerminalCode", default=None, namespaces=ns) if airport_el is not None else None
        gate = airport_el.findtext("ns:GateNumber", default=None, namespaces=ns) if airport_el is not None else None
        aircraft_code = flight_el.findtext("ns:AircraftSubtypeIATACode", default=None, namespaces=ns) if flight_el is not None else None

        sched_time = ops_el.findtext("ns:ScheduledDateTime", default=None, namespaces=ns) if ops_el is not None else None
        actual_time = ops_el.findtext("ns:LatestKnownDateTime", default=None, namespaces=ns) if ops_el is not None else None

        record = {
            "flight_number": flight_id,
            "direction": direction,
            "airport": airport_map.get(airport_code, airport_code),
            "terminal": terminal,
            "gate": gate,
            "status_code": status_map.get(flight_status_code, flight_status_code),
            "airline": airline_map.get(carrier_code, carrier_code),
            "aircraft": aircraft_map.get(aircraft_code, aircraft_code),
            "scheduled_time": sched_time,
            "actual_time": actual_time,
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
            Property(name="airline", data_type=DataType.TEXT),
            Property(name="aircraft", data_type=DataType.TEXT),
            Property(name="summary", data_type=DataType.TEXT),
        ]
    )

    with coll.batch.dynamic() as batch:
        for f in flights:
            summary = (
                f"Flight {f['flight_number']} ({f['airline']}) is a {f['direction']} flight at airport {f['airport']}. "
                f"Terminal {f['terminal']}, Gate {f['gate']}. "
                f"Aircraft: {f['aircraft']}, Status: {f['status_code']}. "
                f"Scheduled {f['scheduled_time']}, Latest {f['actual_time']}."
            )

            batch.add_object(
                properties={
                    **f,
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
        mappings = build_dynamic_mappings(CSV_FOLDER)
        flights = parse_xml_flights(XML_PATH, mappings)

        if not flights:
            st.warning("No valid flight records found.")
        else:
            ingest_flights(flights)
            st.success(f"✅ Indexed {} flight records successfully!".format(len(flights)))
            st.balloons()
