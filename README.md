# 🏘️ Property AI Engineer — Month 1: Document Intelligence App

> AI-powered property document intelligence system for banks and NBFCs.
> Reads any property document — digital or scanned, English or Kannada —
> extracts structured data, validates for bank submission, and scores investment risk.

---

## 🎯 Problem this solves

Property valuers in India spend **3-4 hours per report** manually:

- Reading scanned sale deeds (often in Kannada)
- Extracting owner details, survey numbers, boundaries
- Checking compliance for bank submission
- Scoring investment risk

**This system reduces that to 15-20 minutes** — automatically.

---

## ✅ What it does

| Feature              | Details                                                         |
| -------------------- | --------------------------------------------------------------- |
| 📄 PDF Intelligence  | Reads digital + scanned PDFs automatically                      |
| 🌐 Language Support  | English + Kannada documents                                     |
| 🏗️ Property Types  | Residential, Flat, Commercial, Industrial, IT Park, Vacant Site |
| 🔍 Document Search   | Ask questions across multiple uploaded reports                  |
| ✅ Validation Engine | Flags missing fields before bank submission                     |
| 📊 Risk Analysis     | Investment score + BUY/HOLD/SELL recommendation                 |
| 💾 Persistence       | SQLite database — results saved between sessions               |
| 📥 Reports           | One-click Excel download                                        |

---

## 🏦 Real domain use cases

- **NBFC loan processing** — validate property documents before approval
- **Chartered valuers** — automate report generation from scanned deeds
- **Co-operative banks** — batch risk scoring of property portfolios
- **Property consultants** — search across hundreds of past valuation reports

---

## 🛠️ Tech stack

```
Document AI:    PyMuPDF + Groq Vision LLaMA 4 Scout (scanned pages)
LLM:            Groq LLaMA 3.3 70B (free API)
RAG Pipeline:   LangChain + ChromaDB + HuggingFace Embeddings
Validation:     Rule-based engine (deterministic, auditable)
Database:       SQLite (Python's built-in sqlite3 module)
UI:             Streamlit (multi-page)
Cost:           ₹0 — 100% free stack
```

---

## 📁 Project structure

```
Month_1_AI_RAG_Project/
├── app.py                  — main Streamlit application
├── config.py               — central configuration
├── requirements.txt        — pinned dependencies
├── .env.example            — environment variables template
├── modules/
│   ├── pdf_loader.py       — PDF reading + Groq Vision OCR
│   ├── extractor.py        — Pydantic structured extraction
│   ├── rag.py              — RAG pipeline (LangChain + ChromaDB)
│   ├── llm.py              — LLM + embeddings setup
│   ├── validator.py        — bank submission validation rules
│   └── database.py         — SQLite data layer
├── data/
│   ├── uploads/            — uploaded PDFs (gitignored)
│   └── vector_db/          — ChromaDB (gitignored)
└── database/
    └── properties.db       — SQLite database (gitignored)
```

**Note:** `data/` and `database/` folders are created automatically on first run — they won't exist right after cloning.

## 🚀 How to run locally

```bash
# 1. Clone the repository
git clone https://github.com/arpithaharvi/property-ai-engineer.git
cd property-ai-engineer/Month_1_AI_RAG_Project

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 5. Run the app
streamlit run app.py
```

---

## 🔑 Environment variables

Create a `.env` file with:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at: https://console.groq.com

---

## 📊 App pages

```
🏠 Home              — overview and instructions
📤 Upload Documents  — process any property PDF
🔍 Document Search   — ask questions across all documents
📊 Risk Analysis     — PDF valuation or Excel batch scoring
📋 Reports           — full history from database
```

---

## 👩‍💼 About

Built by a property valuation professional with 4 years of bank
valuation experience, combining domain expertise with AI engineering.

- **Domain:** Property valuation for banks and NBFCs (Karnataka)
- **Stack:** Gen AI, RAG, Document AI, Streamlit, SQLite
- **Languages handled:** English + Kannada property documents

---

## ⚠️ Important notes

- `.env` file is gitignored — never commit your API key
- `data/` folder is gitignored — uploaded PDFs stay local
- `database/properties.db` is gitignored — contains client data
