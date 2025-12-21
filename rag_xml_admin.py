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

st.title("📑 RAG 1: XML AODB Manager")

if st.button("🔄 Sync Flight Records"):
    if not os.path.exists("flight_records.xml"):
        st.error("flight_records.xml nahi mili!")
    else:
        client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
        try:
            if client.collections.exists("RAG1_XML"): client.collections.delete("RAG1_XML")
            coll = client.collections.create(
                name="RAG1_XML",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[Property(name="content", data_type=DataType.TEXT)]
            )
            with open("flight_records.xml", "r", encoding='utf-8') as f: xml_data = f.read()
            blocks = re.findall(r'<(?:[a-zA-Z0-9]+:)?(?:AFDS)?FlightData>(.*?)</(?:[a-zA-Z0-9]+:)?(?:AFDS)?FlightData>', xml_data, re.DOTALL)
            for b in blocks:
                def get_v(tag):
                    m = re.search(rf'<(?:[a-zA-Z0-9]+:)?{tag}[^>]*>(.*?)</(?:[a-zA-Z0-9]+:)?{tag}>', b, re.DOTALL)
                    return m.group(1).strip() if m else "N/A"
                f_id = get_v("FlightIdentity") or get_v("FlightNumber")
                txt = f"Flight {f_id}: Date {get_v('ScheduledDate')}, Status {get_v('FlightStatusCode')}, Gate {get_v('GateIdentity')}, Belt {get_v('BaggageReclaimIdentity')}, Counter {get_v('CheckInDesks')}."
                coll.data.insert(properties={"content": txt}, vector=MODEL.encode(txt).tolist())
            st.success(f"✅ {len(blocks)} Flights Processed.")
        finally: client.close()
