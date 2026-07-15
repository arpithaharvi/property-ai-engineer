# Property AI Engineer - Document Intelligence System

AI-powered property document intelligence for banks and NBFCs.
Reads any property document - digital or scanned, English or Kannada.

## What it does
- Reads scanned Kannada and English property documents automatically
- Extracts owner details, survey numbers, boundaries, market value
- Validates for bank submission readiness
- Scores investment risk with BUY/HOLD/SELL recommendation
- Saves all results to database - persists between sessions

## Tech Stack
- LLM: Groq LLaMA 3.3 70B (free)
- Document AI: PyMuPDF + Groq Vision (scanned pages)
- RAG: LangChain + ChromaDB + HuggingFace Embeddings
- Validation: Rule-based Python engine
- Database: SQLite
- UI: Streamlit
- Cost: 100% free stack

## Setup
1. Clone this repository
2. Create virtual environment: python -m venv venv
3. Activate: venv\Scripts\activate
4. Install: pip install -r requirements.txt
5. Copy .env.example to .env and add your GROQ_API_KEY
6. Run: streamlit run app2.py

## Get free Groq API key
https://console.groq.com

## Property types supported
Residential, Flat, Commercial, Industrial, IT Park, Vacant Site

## Built by
Property valuation professional (2 years, Karnataka banks)
combining domain expertise with Generative AI engineering.
