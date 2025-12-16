# --- Save this code as streamlit_app.py on GitHub (Final Version) ---

# --- IMPORTS (MUST BE AT THE TOP) ---
import streamlit as st
from openai import OpenAI
import json
import re
from typing import List
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject 
from weaviate.classes.query import Filter 
from sentence_transformers import SentenceTransformer 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
import xml.etree.ElementTree as ET
import os
import warnings

# Suppress InsecureRequestWarning for PAA site
warnings.filterwarnings("ignore", "Unverified HTTPS request is being made")


# --- 1. CONFIGURATION (READING SECRETS) ---
# NOTE: Keys are read from Streamlit Secrets (secrets.toml)
try:
    WEAVIATE_URL_BASE = "04xfvperaudv4jaql4uq.c0.asia-southeast1.gcp.weaviate.cloud" 
    WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"] 
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] 
except KeyError as e:
    st.error(f"Missing API key in Streamlit Secrets: {e}. Please add it to your secrets.toml file.")
    st.stop()


DATA_PATHS = {
    "pdf": "policy_baggage.pdf", # Path on GitHub/Streamlit
    "xml": "flight_records.xml", # Path on GitHub/Streamlit
    "web": "https://paa.gov.pk/"
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# GLOBAL CLIENTS
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_MODEL = load_embedding_model()
openai_client = OpenAI(api_key=OPENAI_API_KEY) 
WEAVIATE_CLIENT = None 


# --- 2. CORE RAG / AGENTIC FUNCTIONS ---

@st.cache_resource(show_spinner="Connecting to Weaviate and setting up schema...")
def setup_weaviate_client():
    global WEAVIATE_CLIENT
    try:
        client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL_BASE, auth_credentials=Auth.api_key(WEAVIATE_API_KEY))
    except Exception as e:
        st.error(f"Weaviate Connection Error. Check API key. Error: {e}")
        st.stop()
    if not client.is_connected():
        st.error("Weaviate is reachable but connection failed.")
        st.stop()
    
    # Schema setup (Deleting/Recreating for a fresh start)
    for name in ["PAAPolicy", "PAAFlightStatus", "PAAWebLink"]:
        try:
            if client.collections.exists(name):
                client.collections.delete(name)
        except Exception:
            pass
    VECTORIZER_CONFIG = Configure.Vectorizer.none()
    client.collections.create(name="PAAPolicy", properties=[Property(name="content", data_type=DataType.TEXT), Property(name="source", data_type=DataType.TEXT),], vectorizer_config=VECTORIZER_CONFIG,)
    client.collections.create(name="PAAFlightStatus", properties=[Property(name="content", data_type=DataType.TEXT), Property(name="flight_num", data_type=DataType.TEXT), Property(name="status", data_type=DataType.TEXT),], vectorizer_config=VECTORIZER_CONFIG,)
    client.collections.create(name="PAAWebLink", properties=[Property(name="content", data_type=DataType.TEXT), Property(name="url_text", data_type=DataType.TEXT), Property(name="url_href", data_type=DataType.TEXT),], vectorizer_config=VECTORIZER_CONFIG,)
    
    WEAVIATE_CLIENT = client
    return client

@st.cache_resource(show_spinner="Ingesting data into Vector Store...")
# 🚩 CORRECTION: 'client' argument changed to '_client' to prevent hashing error
def ingest_all_data(_client):
    # Data ingestion logic (simplified for Streamlit cache context)
    client = _client # Assigning it back for consistency within the function
    
    def process_pdf(client, file_path):
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages: text += page.extract_text() + "\n"
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""])
            chunks = text_splitter.split_text(text)
            policy_collection = client.collections.get("PAAPolicy")
            chunk_embeddings = EMBEDDING_MODEL.encode(chunks)
            final_objects = []
            for chunk, vector in zip(chunks, chunk_embeddings):
                final_objects.append(DataObject(properties={"content": chunk, "source": os.path.basename(file_path),}, vector=vector.tolist()))
            policy_collection.data.insert_many(final_objects)
        except Exception: pass

    def process_xml(client, file_path):
        data_to_embed = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f: xml_content = f.read()
            xml_messages = ['<?xml version="1.0" encoding="UTF-8"?>' + part.strip() for part in xml_content.split('<?xml version="1.0" encoding="UTF-8"?>') if part.strip()]
            for message in xml_messages:
                try:
                    root = ET.fromstring(message)
                    flight_data_element = root.find('.//{http://schema.ultra-as.com}AFDSFlightData')
                    if flight_data_element is not None:
                        flight_num = root.find('.//FlightIdentity').text if root.find('.//FlightIdentity') is not None else "N/A"
                        scheduled_time = root.find('.//ScheduledDateTime').text if root.find('.//ScheduledDateTime') is not None else "N/A"
                        status_code = root.find('.//FlightStatusCode').text if root.find('.//FlightStatusCode') is not None else "N/A"
                        destination = root.find('.//PortOfCallIATACode').text if root.find('.//PortOfCallIATACode') is not None else "N/A"
                        natural_content = (f"Flight number {flight_num} from {destination} has a Scheduled Time of "f"{scheduled_time}. The operational status code is {status_code}.")
                        data_to_embed.append({"content": natural_content, "flight_num": flight_num, "status": status_code,})
                except Exception: pass
            if data_to_embed:
                contents = [d["content"] for d in data_to_embed]
                content_embeddings = EMBEDDING_MODEL.encode(contents)
                final_objects = []
                for data_dict, vector in zip(data_to_embed, content_embeddings):
                    final_objects.append(DataObject(properties=data_dict, vector=vector.tolist()))
                client.collections.get("PAAFlightStatus").data.insert_many(final_objects)
        except Exception: pass

    def process_web_data(client, url_to_scrape):
        data_to_embed = []
        try:
            response = requests.get(url_to_scrape, timeout=15, verify=False) 
            soup = BeautifulSoup(response.text, 'html.parser')
            main_content = soup.find('body')
            clean_text = main_content.get_text(separator=' ', strip=True) if main_content else ""
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
            chunks = text_splitter.split_text(clean_text)
            for chunk in chunks: data_to_embed.append({"content": chunk, "url_href": url_to_scrape, "url_text": "General Web Content"})
            if data_to_embed:
                contents = [d["content"] for d in data_to_embed]
                content_embeddings = EMBEDDING_MODEL.encode(contents)
                final_objects = []
                for data_dict, vector in zip(data_to_embed, content_embeddings):
                    final_objects.append(DataObject(properties=data_dict, vector=vector.tolist()))
                client.collections.get("PAAWebLink").data.insert_many(final_objects)
        except Exception: pass

    process_pdf(client, DATA_PATHS["pdf"])
    process_xml(client, DATA_PATHS["xml"])
    process_web_data(client, DATA_PATHS["web"])
    
    return True

# --- 3. RAG RETRIEVAL TOOLS ---
def query_policy_and_baggage(query: str) -> str:
    client = WEAVIATE_CLIENT
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    collection = client.collections.get("PAAPolicy")
    response = collection.query.near_vector(near_vector=query_vector, limit=1, return_properties=["content", "source"])
    if response.objects:
        chunk = response.objects[0].properties.get("content")
        source = response.objects[0].properties.get("source")
        return f"Policy Context (Source: {source}): {chunk}"
    return "Policy Context: No relevant policy found."

def query_flight_status(query: str) -> str:
    client = WEAVIATE_CLIENT
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    flight_num_match = re.search(r'([A-Z]{2}\d{2,4})', query.upper())
    search_kwargs = {"near_vector": query_vector, "limit": 1, "return_properties": ["content", "flight_num", "status"],}
    if flight_num_match:
        target_flight_num = flight_num_match.group(1) 
        flight_filter = Filter.by_property("flight_num").equal(target_flight_num)
        search_kwargs["filters"] = flight_filter
    collection = client.collections.get("PAAFlightStatus")
    response = collection.query.near_vector(**search_kwargs)
    if response.objects:
        chunk = response.objects[0].properties.get("content")
        flight = response.objects[0].properties.get("flight_num")
        status = response.objects[0].properties.get("status")
        return f"Flight Context (Flight: {flight}, Status: {status}): {chunk}"
    return "Flight Context: No relevant flight status found."

def query_web_links_and_forms(query: str) -> str:
    client = WEAVIATE_CLIENT
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    collection = client.collections.get("PAAWebLink")
    response = collection.query.near_vector(near_vector=query_vector, limit=1, return_properties=["content", "url_href"])
    if response.objects:
        chunk = response.objects[0].properties.get("content")
        url = response.objects[0].properties.get("url_href")
        return f"Web Context (URL: {url}): {chunk}"
    return "Web Context: No relevant form or link found."
# --- 4. LLM GENERATION FUNCTION (Final Corrected Version) ---
def generate_answer_with_llm(user_query, retrieved_chunks: List[str]):
    context_text = "\n---\n".join([chunk for chunk in retrieved_chunks if not chunk.endswith("found.")])
    if not context_text.strip():
        return "I am sorry, I could not find any relevant information in the available documents (Policy, Flight Status, or Website) to answer your question."
    
    prompt = f"""You are an AI assistant for Pakistan Airport Authority (PAA). Your goal is to answer a user's question by combining information from the provided contexts, which are separated by source tags (e.g., 'Policy Context:', 'Flight Context:'). 1. **Strictly adhere** to the information in the CONTEXT section. 2. Synthesize all relevant points into one concise, helpful response. 3. If any requested piece of information is missing from the context, state it clearly. --- USER QUESTION: {user_query} --- CONTEXT (Synthesize these sources): {context_text} --- Final Answer (In English, based ONLY on the context):"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful and concise RAG assistant."}, {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        
        # 🟢 CRITICAL FIX: Checking for None before calling .strip()
        llm_content = response.choices[0].message.content
        if llm_content is not None:
            return llm_content.strip()
        else:
            # Fallback if OpenAI returns None content (even if the request succeeded)
            return "The language model returned an empty or invalid response. Please try again."
            
    except Exception as e:
        # Fallback for API/network errors
        return f"An error occurred during LLM generation: {e}"
        

# --- 5. AGENTIC ORCHESTRATOR (FINAL CORRECTED VERSION) ---
def orchestrator_agent(query_text: str):
    tools = [query_policy_and_baggage, query_flight_status, query_web_links_and_forms]
    tool_map = {t.__name__: t for t in tools}
    
    messages = [{"role": "system", "content": "You are a highly analytical PAA supervisor agent. Analyze the user's query and decide which RAG functions (tools) are necessary to fully answer it. You can call multiple tools in parallel if needed."}, {"role": "user", "content": query_text}]
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-1106", 
            messages=messages,
            tools=[{"type": "function", "function": {"name": t.__name__, "description": t.__doc__.strip(), "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The specific part of the user query relevant to this tool's function."}}, "required": ["query"],}}} for t in tools],
            tool_choice="auto", 
            temperature=0.0
        )
    except Exception as e:
        # Catch API call failure right here
        return f"OpenAI API call failed during orchestration: {e}", ["API Error"]

    # 🚩 CRITICAL FIX: Ensure response and tool_calls are not None
    if not response or not response.choices or not response.choices[0].message:
        return "The Orchestration model returned an empty or invalid primary response.", ["Invalid Response"]

    tool_calls = response.choices[0].message.tool_calls
    retrieved_chunks = []
    tools_used = []

    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = tool_map.get(function_name)
            if function_to_call:
                # Assuming query_text is always a string, so tool_output should be safe.
                tool_output = function_to_call(query_text) 
                retrieved_chunks.append(tool_output)
                tools_used.append(function_name.replace("query_", "").replace("_", " ").title())
    else:
        # If no tool calls are made, use the default policy tool
        retrieved_chunks.append(query_policy_and_baggage(query_text))
        tools_used.append("Policy and Baggage (Default)")

    final_answer = generate_answer_with_llm(query_text, retrieved_chunks)
    return final_answer, tools_used

# --- 6. STREAMLIT UI DEFINITION ---
st.set_page_config(page_title="PAA Agentic RAG System", layout="wide")
st.title("🇵🇰 PAA Agentic RAG Chatbot")
st.markdown("This agent can answer complex questions by automatically routing queries to Policy (PDF), Flight Status (XML), or Web Link RAG sources.")
st.markdown("---")

try:
    # 1. Initialize Clients and Ingest Data (Cached to run only once)
    client = setup_weaviate_client()
    # 🚩 CORRECTION: 'client' passed as '_client' to avoid hashing error in ingest_all_data
    ingestion_status = ingest_all_data(client) 
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I am the PAA Supervisor Agent. I can answer questions about Baggage Policy, Flight Status, and PAA Web Forms. How can I help you today?"})

    # 2. Display Chat Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Handle User Input
    if prompt := st.chat_input("Ask about Flight PK234, Baggage Limits, or Lost Claims..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.spinner(f"Agent is analyzing your request for '{prompt}'..."):
            try:
                final_answer, tools_used = orchestrator_agent(prompt)
                
                metadata = f"*(Used Tools: {', '.join(tools_used)})*"
                full_response = f"{final_answer}\n\n{metadata}"
                
            except Exception as e:
                full_response = f"An unexpected error occurred during orchestration. Error: {e}"
        
        with st.chat_message("assistant"):
            st.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

except Exception as e:
    st.error(f"A critical error occurred during setup or execution: {e}. Check your logs.")
