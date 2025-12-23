PAA Enterprise Intelligence: Multi-Agent RAG System

This project is a Proof-of-Concept (PoC) developed for the Pakistan Airports Authority (PAA). It utilizes a Specialized Multi-Agent Orchestration framework to unify fragmented aviation data into a single conversational interface.
______________
Live Demo URLs
The system is deployed as a modular suite of Streamlit applications. Please access the components via the following links:
•	Primary Chat Interface: PAA Enterprise Intelligence
	o	https://paaagenticrag-jx4quubb7vtvnuswrcsxsh.streamlit.app/
	o	The main passenger/staff-facing chatbot for unified queries.
•	Operational (XML) Admin: PAA XML RAG Admin
	o	https://paaagenticrag-jhpbwguwx53lpcryihxzju.streamlit.app/
	o	Control center for processing and training real-time flight snapshots.
•	Policy (DOC) Admin: PAA Document RAG Admin
	o	https://paaagenticrag-vvv2m4kejux89vku4hbbwq.streamlit.app/
	o	Interface for ingesting and vectorizing regulatory PDF manuals.
•	Web Control Center: PAA Web Control Center
	o	https://paaagenticrag-rjzmzt63bnhyplezvphw4w.streamlit.app/
	o	Management portal for live web scraping and public notice retrieval.
________________________________________
Project Structure & Data Ingestion
For the system to train and retrieve information correctly, data files must be placed in their respective directories within the project root:
1.	rag_docs_data/: Place all PDF files here (e.g., Baggage Policies, Security Protocols). The DOC_AGENT uses this folder to build the policy vector store.
2.	rag_xml_data/: Place the XML flight snapshots (AFDS data) here. The XML_AGENT parses files in this directory to provide real-time flight status updates.

________________________________________
Key Features
•	Supervisor Router: A custom heuristic-based intelligence layer that identifies user intent and delegates tasks to specialized agents.
•	Multi-Source Synthesis: Ability to combine answers from structured XML, unstructured PDFs, and live Web data in a single response.
•	Dual-Threshold Retrieval: Optimized semantic search using a 0.6 threshold for high-precision documents and a 0.7 threshold for broader web context.
•	Identity Recognition: Unified entity resolution for "PAA" and "Pakistan Airports Authority" to ensure consistent retrieval.

________________________________________
Tech Stack
•	Language: Python
•	Frontend: Streamlit
•	Vector Database: Weaviate Cloud
•	Embeddings: all-MiniLM-L6-v2
•	LLM Core: GPT-4o / Gemini 1.5 Flash

