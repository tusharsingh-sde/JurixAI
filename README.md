<div align="center">

# ⚖️ JurixAI

### Your AI-Powered Legal Saathi for the Indian Legal System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**JurixAI** is a high-performance, **RAG-based** (Retrieval-Augmented Generation) legal assistant purpose-built for the **Indian Legal System**. It goes beyond cold legal facts — acting as a *Legal Saathi* that understands your stress and provides authoritative, law-backed guidance with empathy.

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [API Reference](#-api-reference) · [Contributing](#-contributing)

</div>

---

## 🎯 The Problem

Indian law is vast, complex, and deeply intimidating for the average citizen. From understanding FIR procedures to navigating tenant rights, most people feel overwhelmed by legal jargon and don't know where to start.

**JurixAI bridges this gap by:**

- 📖 **Simplifying complex legal sections** — IPC, BNS, CrPC, POSH Act, RTI Act, Cyber Laws, Domestic Violence Act, Consumer Protection, and more.
- 🚀 **Providing immediate, actionable steps** — Not just theory; clear, practical next steps tailored to your situation.
- 🤝 **Maintaining an empathetic, human-centric tone** — Detects distress and adapts its response from informational to supportive.
- 🌐 **Bilingual intelligence** — Automatically detects and responds in **English**, **Hindi**, or **Hinglish** based on user input.
- 📄 **Document analysis** — Upload FIRs, legal notices, summons, or agreements and get an instant AI-powered risk assessment.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                  │
│                     Port 5173 · Chat Interface                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │  HTTP (REST API)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                           │
│                        Port 8000                                │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │  /chat        │   │  /upload      │   │  Session Memory     │  │
│  │  RAG Pipeline │   │  PDF + OCR    │   │  (In-Memory Store)  │  │
│  └──────┬───────┘   └──────┬───────┘   └─────────────────────┘  │
│         │                  │                                     │
│         ▼                  ▼                                     │
│  ┌──────────────────────────────────────────┐                   │
│  │           LangChain Orchestration         │                   │
│  │  ┌────────────┐  ┌─────────────────────┐ │                   │
│  │  │ ChromaDB   │  │ HuggingFace Embed.  │ │                   │
│  │  │ (Vector DB)│  │ (all-MiniLM-L6-v2)  │ │                   │
│  │  └────────────┘  └─────────────────────┘ │                   │
│  └──────────────────────────┬───────────────┘                   │
│                             │                                    │
│                             ▼                                    │
│                   ┌──────────────────┐                           │
│                   │   Groq Cloud API  │                           │
│                   │  Llama-3.3-70B   │                           │
│                   └──────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### The RAG Pipeline (How it Works)

| Step | Phase | Description |
|------|-------|-------------|
| 1 | **Data Ingestion** | 8 Indian Law PDFs are loaded and split into semantic chunks (1000 chars, 200 overlap) using `RecursiveCharacterTextSplitter` |
| 2 | **Vectorization** | Chunks are embedded using HuggingFace `all-MiniLM-L6-v2` and stored in ChromaDB (`Jurixai_db`) |
| 3 | **Retrieval** | User query triggers a top-5 similarity search against the vector store |
| 4 | **Augmentation** | Retrieved legal context + conversation history (last 4 messages) are injected into a custom "Empathy-First" prompt |
| 5 | **Generation** | Llama 3.3 70B (via Groq) generates a structured, law-backed, human-readable response |

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core runtime |
| **FastAPI** | High-performance async API server |
| **LangChain** | RAG pipeline orchestration |
| **ChromaDB** | Persistent vector database for legal embeddings |
| **HuggingFace Embeddings** | `all-MiniLM-L6-v2` for local, fast semantic search |
| **Groq Cloud** | Lightning-fast LLM inference (Llama-3.3-70b-versatile) |
| **pdfplumber** | PDF text extraction |
| **Tesseract OCR** | Image-to-text for scanned legal documents |
| **Pydantic** | Request/response data validation |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **React 19** | Modern UI framework |
| **Vite 8** | Next-gen build tool with HMR |
| **ESLint** | Code quality enforcement |

---

## 📚 Legal Knowledge Base

JurixAI is trained on **8 authoritative Indian law documents**:

| Document | Coverage |
|----------|----------|
| `IPC.pdf` | Indian Penal Code — Criminal offences & punishments |
| `cyber_law.pdf` | IT Act — Cyber crimes, online fraud, digital evidence |
| `POSH_act.pdf` | Prevention of Sexual Harassment at Workplace |
| `RTI-Act_English.pdf` | Right to Information Act |
| `Consumer_Handbook.pdf` | Consumer Protection Act & dispute resolution |
| `Model-Tenancy-Act_2021.pdf` | Tenancy rights, rent agreements, eviction laws |
| `First_Information_Report.pdf` | FIR filing procedures & police complaints |
| `protection_of_women_from_domestic_violence_act,.pdf` | Domestic Violence Act — Protection orders & rights |

---

## 📁 Project Structure

```
JurixAI_core/
├── backend/
│   ├── backend.py            # FastAPI server — chat & upload endpoints
│   ├── build_database.py     # One-time script to ingest PDFs into ChromaDB
│   ├── query_db.py           # Standalone script to test vector DB queries
│   ├── legal_data/           # 8 Indian Law PDFs (knowledge base)
│   ├── Jurixai_db/           # ChromaDB persistent storage (auto-generated)
│   └── .env                  # API keys (not committed)
│
├── frontend/
│   ├── src/
│   │   ├── main.jsx          # React entry point
│   │   ├── App.jsx           # Main application component
│   │   ├── App.css           # Component styles
│   │   ├── index.css         # Global styles & design tokens
│   │   └── assets/           # Static assets (logos, images)
│   ├── public/               # Favicon & SVG icons
│   ├── index.html            # HTML entry point
│   ├── vite.config.js        # Vite configuration
│   ├── eslint.config.js      # ESLint configuration
│   └── package.json          # Node.js dependencies
│
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10 or higher
- **Node.js** 18+ and **npm**
- **Tesseract OCR** ([Download here](https://github.com/UB-Mannheim/tesseract/wiki))
- A **Groq API Key** ([Get one free](https://console.groq.com))

### 1. Clone the Repository

```bash
git clone https://github.com/tusharsingh-sde/JurixAI.git
cd JurixAI
```

### 2. Backend Setup

```bash
cd backend

# Create & activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install Python dependencies
pip install fastapi uvicorn python-dotenv langchain langchain-community langchain-huggingface langchain-groq chromadb pdfplumber pytesseract Pillow pydantic
```

### 3. Configure Environment Variables

Create a `.env` file inside the `backend/` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Build the Vector Database

> ⚠️ **Run this only once** (or when you update the legal PDFs).

```bash
python build_database.py
```

This will:
- Load all PDFs from `legal_data/`
- Split them into 1000-character chunks with 200-character overlap
- Generate vector embeddings using `all-MiniLM-L6-v2`
- Persist the database to `Jurixai_db/`

### 5. Start the Backend Server

```bash
uvicorn backend:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 6. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## 📡 API Reference

### `POST /chat`

Send a message to JurixAI and receive a law-grounded, empathetic response.

**Request Body:**
```json
{
  "session_id": "user-123",
  "message": "Someone stole my phone, what should I do?"
}
```

**Response:**
```json
{
  "reply": "🛡️ I understand this is really stressful. Here's what you need to do immediately..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `string` | Unique session identifier for conversation memory |
| `message` | `string` | User's legal question or message |

---

### `POST /upload`

Upload a legal document (PDF or image) for AI-powered text extraction and analysis.

**Request:** `multipart/form-data` with a `file` field.

**Supported Formats:** `.pdf`, `.jpg`, `.jpeg`, `.png`

**Response:**
```json
{
  "status": "success",
  "filename": "FIR_copy.pdf",
  "extracted_text": "The extracted and cleaned text content..."
}
```

---

## 🛡️ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Context-Aware RAG** | Never hallucinate — responses grounded only in verified Indian law documents |
| 🌍 **Trilingual Intelligence** | Auto-detects English, Hindi (Devanagari), or Hinglish and responds accordingly |
| 💬 **Conversation Memory** | Maintains session-based chat history for contextual follow-ups |
| 📄 **Document Scanner** | Upload FIRs, legal notices, or agreements — get instant risk assessment |
| 🔍 **OCR Capability** | Extract text from scanned images of legal documents via Tesseract |
| ⚡ **Blazing Fast Inference** | Powered by Groq Cloud for near-instant Llama 3.3 70B responses |
| 🎭 **Empathy-First Responses** | Detects user distress and adapts tone from informational to supportive |
| 📋 **Structured Output** | Every response follows: Empathy → Action → Legal Basis → Next Steps |

---

## 🧪 Testing the Vector DB

You can independently test the database without running the full server:

```bash
cd backend
python query_db.py
```

This runs a sample query (`"What is the punishment for online fraud or cyber crime?"`) against the vector store and returns the top 3 matching legal passages.

---

## ⚖️ Disclaimer

> **This project is for educational and research purposes only.**
> JurixAI is an AI-powered tool and **not a substitute** for professional legal advice from a qualified lawyer. Always verify legal information from official government sources before taking any action.

---

## 🗺️ Roadmap

- [ ] Build full chat UI with message history & document upload
- [ ] Add user authentication & persistent session storage
- [ ] Expand knowledge base with more Indian statutes (CrPC, BNSS, Evidence Act)
- [ ] Implement streaming responses for real-time output
- [ ] Add legal document drafting (complaint letters, RTI applications)
- [ ] Deploy to cloud (Railway / Render / AWS)
- [ ] Mobile-responsive progressive web app

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'feat: add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

Please make sure your code follows the existing project conventions and includes appropriate documentation.

---

## 👨‍💻 Author

**Tushar Singh**

[![GitHub](https://img.shields.io/badge/GitHub-@tusharsingh--sde-181717?style=flat-square&logo=github)](https://github.com/tusharsingh-sde)

*Software Engineering · AI/ML · Full-Stack Development*

---

<div align="center">

**Built with ❤️ for making Indian Law accessible to everyone.**

⭐ Star this repo if JurixAI helped you!

</div>
