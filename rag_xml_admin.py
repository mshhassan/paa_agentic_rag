import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import os
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="PAA XML Processor", layout="wide")

# --- CONFIG & SECRETS ---
try:
    WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
    WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
except KeyError:
    st.error("❌ Secrets missing! Please add WEAVIATE_URL and WEAVIATE_API_KEY in Streamlit Cloud.")
    st.stop()

@st.cache_resource
def load_model():
    # Cache model to avoid reloading on every click
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

MODEL = load_model()

def get_mapping(file_path, code_col, name_col):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            return pd.Series(df[name_col].values, index=df[code_col]).to_dict()
        except Exception as e:
            st.warning(f"Could not load {file_path}: {e}")
            return {}
    return {}

def clean_xml_string(xml_str):
    """XML se ajeeb characters hatane ke liye function"""
    # Sirf valid XML characters rehne dein
    xml_str = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', '', xml_str)
    return xml_str.strip()

def process_and_train():
    st.info("🔄 Step 1: Loading CSV Mappings...")
    
    # 1. Load CSV Mappings
    airline_map = get_mapping('rag_xml_data/airlines.csv', 'IATACode', 'AirlineName')
    airport_map = get_mapping('rag_xml_data/airports.csv', 'IATACode', 'CityName')
    status_map = get_mapping('rag_xml_data/status_codes.csv', 'StatusCode', 'Description')
    aircraft_map = get_mapping('rag_xml_data/aircraft_types.csv', 'SubtypeCode', 'ModelName')

    # 2. Parse XML (Robust Method)
    xml_file = 'rag_xml_data/flight_snapshot.xml'
    if not os.path.exists(xml_file):
        st.error(f"❌ XML File not found at {xml_file}")
        return

    st.info("🔄 Step 2: Parsing XML File...")
    try:
        with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
            raw_xml = f.read()
        
        clean_xml = clean_xml_string(raw_xml)
        root = ET.fromstring(clean_xml)
        
        enriched_narratives = []
        flights = root.findall('.//FlightData')
        
        if not flights:
            st.warning("⚠️ No <FlightData> tags found in XML.")
            return

        for flight in flights:
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

    except ET.ParseError as e:
        st.error(f"❌ XML Parse Error: {e}. Check if your XML file is valid.")
        return
    except Exception as e:
        st.error(f"❌ Error during parsing: {e}")
        return

    # 3. Batch Upload to Weaviate
    st.info(f"🔄 Step 3: Uploading {len(enriched_narratives)} flights to Weaviate...")
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL, 
            auth_credentials=Auth.api_key(WEAVIATE_KEY)
        )
        
        collection_name = "PAAWeb"
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            
        coll = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[Property(name="content", data_type=DataType.TEXT)]
        )

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        with coll.batch.dynamic() as batch:
            for i, text in enumerate(enriched_narratives):
                batch.add_object(
                    properties={"content": text},
                    vector=MODEL.encode(text).tolist()
                )
                if i % 10 == 0:
                    my_bar.progress(i / len(enriched_narratives), text=progress_text)
        
        my_bar.empty()
        st.success(f"🚀 Successfully enriched and indexed {len(enriched_narratives)} flights into PAAWeb!")
        st.balloons()
        client.close()
        
    except Exception as e:
        st.error(f"❌ Weaviate Error: {e}")

# --- MAIN UI ---
st.title("✈️ PAA Flight Data Manager (XML)")
st.write("Is app ke zariye aap XML snapshot ko process karke Weaviate Database update kar sakte hain.")

if st.button("🏗️ Process XML & Update Vector DB"):
    process_and_train()
