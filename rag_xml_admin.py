import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="PAA XML Processor", layout="wide")

# --- CONFIG & SECRETS ---
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

MODEL = load_model()

# Mapping logic function
def get_mapping(file_path, code_col, name_col):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return pd.Series(df[name_col].values, index=df[code_col]).to_dict()
    return {}

def process_and_train():
    st.info("🔄 Processing Files...")
    
    # 1. Load CSV Mappings
    try:
        airline_map = get_mapping('rag_xml_data/airlines.csv', 'IATACode', 'AirlineName')
        airport_map = get_mapping('rag_xml_data/airports.csv', 'IATACode', 'CityName')
        status_map = get_mapping('rag_xml_data/status_codes.csv', 'StatusCode', 'Description')
        aircraft_map = get_mapping('rag_xml_data/aircraft_types.csv', 'SubtypeCode', 'ModelName')
    except Exception as e:
        st.error(f"CSV Loading Error: {e}")
        return

    # 2. Parse XML
    xml_file = 'rag_xml_data/flight_snapshot.xml'
    if not os.path.exists(xml_file):
        st.error(f"XML File not found at {xml_file}")
        return

    tree = ET.parse(xml_file)
    root = tree.getroot()
    enriched_narratives = []
    
    for flight in root.findall('.//FlightData'):
        def get_t(tag): return flight.findtext(tag) or "N/A"
        
        airline = airline_map.get(get_t('CarrierIATACode'), get_t('CarrierIATACode'))
        origin_dest = airport_map.get(get_t('PortOfCallIATACode'), get_t('PortOfCallIATACode'))
        status = status_map.get(get_t('FlightStatusCode'), get_t('FlightStatusCode'))
        aircraft = aircraft_map.get(get_t('AircraftSubtypeIATACode'), get_t('AircraftSubtypeIATACode'))
        
        narrative = f"""
        Flight {get_t('FlightIdentity')} ({get_t('ICAOFlightIdentifier')}) is a {get_t('FlightClassificationCode')} 
        {'Arrival' if get_t('FlightDirection') == 'A' else 'Departure'} operated by {airline}.
        Route: {origin_dest}, Sector: {'International' if get_t('FlightSectorCode') == 'I' else 'Domestic'}.
        Status: {status}, Aircraft: {aircraft} (Reg: {get_t('AircraftRegistration')}).
        Timings: Scheduled at {get_t('ScheduledDate')}, Gate {get_t('GateNumber')}.
        """
        enriched_narratives.append(narrative.strip())

    # 3. Batch Upload to Weaviate
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, 
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    try:
        collection_name = "PAAWeb"
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            
        coll = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[Property(name="content", data_type=DataType.TEXT)]
        )

        with coll.batch.dynamic() as batch:
            for text in enriched_narratives:
                batch.add_object(
                    properties={"content": text},
                    vector=MODEL.encode(text).tolist()
                )
        st.success(f"🚀 Successfully enriched and indexed {len(enriched_narratives)} flights into PAAWeb!")
        st.balloons()
    finally:
        client.close()

# --- MAIN UI ---
st.title("✈️ PAA Flight Data Manager (XML)")
st.write("Is app ke zariye aap XML snapshot ko process karke Weaviate Database update kar sakte hain.")

if st.button("🏗️ Process XML & Update Vector DB"):
    process_and_train()
