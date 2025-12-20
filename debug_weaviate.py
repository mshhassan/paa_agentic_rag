import weaviate
from weaviate.classes.init import Auth
import streamlit as st

# Connection setup
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud",
    auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"])
)

def check_data():
    coll = client.collections.get("PAAFlightStatus")
    # Hum 'J2143' ke liye keyword search kar rahe hain directly
    response = coll.query.bm25(
        query="J2143",
        limit=5
    )
    
    for obj in response.objects:
        print(f"ID: {obj.uuid}")
        print(f"Content: {obj.properties.get('content')}")
        print("-" * 30)

check_data()
client.close()
