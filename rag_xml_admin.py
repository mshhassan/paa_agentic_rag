import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer

# Mapping logic function
def get_mapping(df, code_col, name_col):
    return pd.Series(df[name_col].values, index=df[code_col]).to_dict()

def process_and_train():
    # 1. Load CSV Mappings
    try:
        airline_map = get_mapping(pd.read_csv('rag_xml_data/airlines.csv'), 'IATACode', 'AirlineName')
        airport_map = get_mapping(pd.read_csv('rag_xml_data/airports.csv'), 'IATACode', 'CityName')
        status_map = get_mapping(pd.read_csv('rag_xml_data/status_codes.csv'), 'StatusCode', 'Description')
        aircraft_map = get_mapping(pd.read_csv('rag_xml_data/aircraft_types.csv'), 'SubtypeCode', 'ModelName')
    except Exception as e:
        st.error(f"CSV Loading Error: {e}")
        return

    # 2. Parse XML
    tree = ET.parse('rag_xml_data/flight_snapshot.xml')
    root = tree.getroot()
    
    enriched_data = []
    
    # XML tags parsing (Simplified for logic)
    for flight in root.findall('.//FlightData'):
        def get_t(tag): return flight.findtext(tag) or "N/A"
        
        # Mapping apply karein
        airline = airline_map.get(get_t('CarrierIATACode'), get_t('CarrierIATACode'))
        origin_dest = airport_map.get(get_t('PortOfCallIATACode'), get_t('PortOfCallIATACode'))
        status = status_map.get(get_t('FlightStatusCode'), get_t('FlightStatusCode'))
        aircraft = aircraft_map.get(get_t('AircraftSubtypeIATACode'), get_t('AircraftSubtypeIATACode'))
        
        # Enriched Narrative (RAG ke liye best format)
        narrative = f"""
        Flight {get_t('FlightIdentity')} ({get_t('ICAOFlightIdentifier')}) is a {get_t('FlightClassificationCode')} 
        {'Arrival' if get_t('FlightDirection') == 'A' else 'Departure'} operated by {airline}.
        Route: {origin_dest}, Sector: {'International' if get_t('FlightSectorCode') == 'I' else 'Domestic'}.
        Status: {status}, Aircraft: {aircraft} (Reg: {get_t('AircraftRegistration')}), Capacity: {get_t('AircraftPassengerCapacity')}.
        Timings: Scheduled at {get_t('ScheduledDate')}, Gate {get_t('GateNumber')} (Opens: {get_t('GateOpenDateTime')}, Closes: {get_t('GateCloseDateTime')}).
        Check-in: Counters {get_t('CheckinDeskRange')} (Open: {get_t('CheckinOpenDateTime')}, Close: {get_t('CheckinCloseDateTime')}).
        Arrival Info: Belt/Carousel {get_t('BaggageReclaimCarouselID')}, Stand {get_t('StandPosition')}.
        Handling: {get_t('HandlingAgentService')} by {get_t('HandlingAgentIATACode')}.
        """
        enriched_data.append(narrative)

    # 3. Batch Upload to Weaviate
    # (Yahan purana Weaviate upload code use karein jo 'PAAWeb' collection mein save kare)
    st.success(f"Enriched {len(enriched_data)} flights with full descriptions!")
