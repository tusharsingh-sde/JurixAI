<div align="center">

# ⚖️ JurixAI

### Your AI-Powered Legal Saathi for the Indian Legal System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![LangChain](https://img.shields.io/badge/LangChain-Chroma-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Supabase](https://img.shields.io/badge/Supabase-Auth_%26_DB-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![Pytest](https://img.shields.io/badge/Pytest-15_Passed-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**JurixAI** is a production-hardened, **RAG-based** (Retrieval-Augmented Generation) legal intelligence assistant purpose-built for the **Indian Legal System**. It goes beyond cold legal summaries — acting as a trusted *Legal Saathi* that understands distress and delivers authoritative, law-backed guidance with empathy and clarity.

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [API Reference](#-api-reference) · [Docker Deployment](#-docker-deployment) · [Contributing](#-contributing)

</div>

---

## 🎯 The Problem

Indian law is vast, nuanced, and deeply intimidating for citizens and small businesses. From understanding complex FIR procedures to navigating tenant eviction rights, most people feel overwhelmed by legal jargon and uncertain about procedural remedies.

**JurixAI bridges this gap by:**

- 📖 **Demystifying complex legal sections** — IPC, BNS, CrPC, POSH Act, RTI Act, Cyber Laws (IT Act 2000), Domestic Violence Act, Consumer Protection, and Model Tenancy laws.
- 🚀 **Providing actionable next steps** — Clear, structured procedural guidance tailored to your specific situation.
- 🤝 **Maintaining an empathetic, human-centric tone** — Detects emotional distress and balances supportive reassurance with disciplined legal analysis.
- 🌐 **Trilingual intelligence** — Seamlessly understands and responds in **English**, **Hindi** (Devanagari), and **Hinglish**.
- 📄 **Multi-page document analysis** — Upload FIRs, legal notices, summons, or agreements with automatic OCR fallback for scanned non-text PDFs and images.
- 🛡️ **Enterprise-grade production hardening** — Dual Supabase auth scoping, SlowAPI abuse protection, non-root Docker builds, and threadpool offloading.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React 19 + Vite)                      │
│                  Port 5173 · Cobalt Law Glass Interface                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │  HTTP / REST API (CORS + PNA Allowed)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI Hardened Service)                  │
│                     Port 8000 · Single Worker Non-Root                 │
│                                                                        │
│  ┌───────────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │   Auth & Abuse Guard  │  │   Upload Safety  │  │  Health Probes  │ │
│  │ SlowAPI Rate Limiting │  │ Chunked 413 Abort│  │ /health (Live)  │ │
│  │ Dual Supabase Clients │  │ Magic Byte Check │  │ /ready (Chroma) │ │
│  └───────────┬───────────┘  └────────┬─────────┘  └─────────────────┘ │
│              │                       │                                 │
│              ▼                       ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                     Async Lifespan Engine                        │ │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────┐  │ │
│  │  │ ChromaDB Vector Store   │    │ SentenceTransformer Embeds  │  │ │
│  │  │ (2,158 Embedded Chunks) │    │ (all-MiniLM-L6-v2 Cached)   │  │ │
│  │  └─────────────────────────┘    └─────────────────────────────┘  │ │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────┐  │ │
│  │  │ Tesseract OCR (Eng+Hin) │    │ Per-Session Concurrency Lock│  │ │
│  │  └─────────────────────────┘    └─────────────────────────────┘  │ │
│  └───────────────────────────────────┬──────────────────────────────┘ │
│                                      │                                 │
│                                      ▼                                 │
│                       ┌──────────────────────────────┐                │
│                       │    Google Gemini API SDK     │                │
│                       │  (gemini-3.5/3.7-flash with  │                │
│                       │   automatic multi-fallback)  │                │
│                       └──────────────────────────────┘                │
└────────────────────────────────────────────────────────────────────────┘
```

### The Hardened RAG Pipeline

| Step | Phase | Description |
|------|-------|-------------|
| 1 | **Data Ingestion** | 8 authoritative Indian Law PDFs split into semantic chunks (1,000 chars, 200 overlap) with `RecursiveCharacterTextSplitter`. |
| 2 | **Local Vectorization** | Chunks embedded using `all-MiniLM-L6-v2` and stored in persistent ChromaDB (`backend/Jurixai_db`) with 2,158 pre-indexed legal vectors. |
| 3 | **Threadpool Retrieval** | Non-blocking `similarity_search` offloaded to `run_in_threadpool` to prevent event-loop starvation. |
| 4 | **Context Augmentation** | Sanitized query tags (`<user_message>`, `<conversation_history>`, `<retrieved_context>`, `<uploaded_document>`) injected into the persona prompt with persistent document memory across follow-ups. |
| 5 | **Resilient Generation** | Google Gemini SDK invocation wrapped in threadpools with an automated multi-model fallback cascade (`gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.7-flash`). |

---

## 🛠️ Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11 / 3.14 | Core execution runtime |
| **FastAPI** | 0.115+ | High-performance asynchronous REST API |
| **Uvicorn** | 0.34+ | ASGI web server with proxy header support |
| **SlowAPI** | 0.1.9 | IP & token rate-limiting abuse protection |
| **LangChain Chroma** | 1.1.0 | Vector database interface for ChromaDB |
| **Sentence-Transformers** | 5.2.0 | Offline embedding generation (`all-MiniLM-L6-v2`) |
| **Google GenAI** | 1.63.0 | Gemini multi-model generation with automatic fallback |
| **Supabase** | 2.14.0 | Dual-client authentication and relational PostgreSQL persistence |
| **pdfplumber / pypdfium2** | 0.11.5 / 5.5.0 | High-speed PDF text parsing and raster rendering |
| **PyTesseract** | 0.3.13 | Tesseract OCR engine with English + Hindi support |
| **Pillow** | 11.1.0 | Image manipulation with DecompressionBomb security caps |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 19.0.0 | Component-based reactive UI |
| **Vite** | 8.3.2 | Lightning-fast build tool with HMR |
| **Lucide React** | 0.468.0 | Minimalist iconography |
| **CSS3** | Modern | Custom cobalt theme with Lady Justice backdrop art |

---

## 📚 Legal Knowledge Base

JurixAI is pre-indexed with **8 foundational Indian statutory acts and guides**:

| Document | Coverage |
|----------|----------|
| `IPC.pdf` | Indian Penal Code — Criminal offences, definitions & punishments |
| `cyber_law.pdf` | Information Technology Act 2000 — Cyber fraud, identity theft, electronic evidence |
| `POSH_act.pdf` | Prevention of Sexual Harassment at Workplace Act 2013 |
| `RTI-Act_English.pdf` | Right to Information Act 2005 — Citizen transparency & appeals |
| `Consumer_Handbook.pdf` | Consumer Protection Act 2019 — Defective goods, services, e-commerce grievances |
| `Model-Tenancy-Act_2021.pdf` | Tenancy rights, rent agreements, security deposits, eviction grounds |
| `First_Information_Report.pdf` | FIR registration protocol, cognizable offences & police obligations |
| `protection_of_women_from_domestic_violence_act,.pdf` | Protection of Women from Domestic Violence Act 2005 — Orders & shelter rights |

---

## 📁 Project Structure

```
JurixAI_core/
├── backend/
│   ├── backend.py            # Hardened FastAPI application & lifecycle
│   ├── build_database.py     # Legal PDF ingestion script for ChromaDB
│   ├── query_db.py           # CLI vector database query tool
│   ├── requirements.txt      # Pinned production dependencies
│   ├── requirements-dev.txt  # Pinned development & ingestion tools
│   ├── Dockerfile            # Multi-stage, non-root hardened container
│   ├── .dockerignore         # Docker context exclusions
│   ├── .env.example          # Template for backend environment variables
│   ├── legal_data/           # 8 core Indian Law PDFs
│   ├── Jurixai_db/           # ChromaDB database (tracked in Git, 2,158 vectors)
│   └── tests/                # Automated pytest unit test suite (15 tests)
│       ├── test_auth_and_abuse.py
│       ├── test_data_layer.py
│       └── test_upload_safety.py
│
├── frontend/
│   ├── index.html            # Single page app entry
│   ├── vite.config.js        # Vite dev server configuration
│   ├── package.json          # Node dependencies
│   ├── public/               # Static assets (lady-justice.png)
│   └── src/
│       ├── main.jsx          # React DOM mounting
│       ├── App.jsx           # Full chat UI & session management
│       └── styles.css        # Custom cobalt theme styling
│
├── migrations/
│   └── 001_jurix.sql         # Idempotent Supabase PostgreSQL schema & RLS policies
│
├── .gitignore                # Comprehensive exclusions (secrets, venv, caches)
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python** 3.10+ (Python 3.11 recommended for Docker parity)
- **Node.js** 18+ and **npm**
- **Docker** (optional, for containerized run)
- **Google Gemini API Key** ([Google AI Studio](https://aistudio.google.com/))
- **Supabase Account** ([Supabase](https://supabase.com/))

### 2. Database Migration (Supabase)
1. Open your **Supabase Project Dashboard** $\rightarrow$ **SQL Editor**.
2. Run the SQL script from [`migrations/001_jurix.sql`](migrations/001_jurix.sql).
3. This creates the `sessions` and `messages` tables with persistent document columns, sequential ordering, performance indexes, and Row Level Security (RLS).

### 3. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create `backend/.env` (refer to `backend/.env.example`):

```env
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SECRET_KEY=your_supabase_service_role_key
SUPABASE_ANON_KEY=your_supabase_anon_public_key
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

Start the server:

```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
```

- API Base: `http://localhost:8000`
- Interactive Swagger Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Readiness check: `http://localhost:8000/ready`

### 4. Frontend Setup

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 🐳 Docker Deployment

The backend is packaged as an optimized, non-root, production container with pre-warmed models and health checks:

```bash
# Build the container
docker build -t jurixai-backend ./backend

# Run with environment variables
docker run -p 8000:8000 --env-file backend/.env jurixai-backend
```

---

## 📡 API Reference

### Health & Liveness
- **`GET /health`**: Fast dependency-free liveness probe for load balancers. Returns `{"status": "ok"}`.
- **`GET /ready`**: Readiness check verifying ChromaDB collection count. Returns `{"status": "ready", "documents_loaded": 2158}`.

### Authentication (SlowAPI Rate Limited)
- **`POST /signup`**: Creates an auto-confirmed user account via Supabase Admin (Limit: `5/hour/IP`, min password 8 chars).
- **`POST /login`**: Verifies user credentials via Supabase Anon client and issues JWT token (Limit: `10/minute/IP`).
- **`POST /logout`**: Invalidates user token with `scope="local"`.

### Sessions & Conversations
- **`GET /sessions`**: List conversations for authenticated user.
- **`GET /sessions/{id}/messages`**: Retrieve message history for a session (paginated to 100 messages).
- **`DELETE /sessions/{id}`**: Delete a session and its associated message history.

### Core RAG Chat
- **`POST /chat`** (Limit: `30/minute/user`):
  ```json
  {
    "message": "What is the penalty for cyber fraud?",
    "session_id": "optional-uuid-here",
    "document_text": "optional extracted legal document text"
  }
  ```
  Returns law-grounded response with empathetic guidance, legal sections, and action steps. Automatically persists document text on the session across follow-up queries.

### Document Upload & OCR
- **`POST /upload`** (Limit: `10/minute/user`):
  - Enforces `MAX_UPLOAD_MB` (default 8MB) with early chunked 413 abort.
  - Verifies magic byte headers (`%PDF`, PNG, JPEG).
  - Decompression bomb protection (50M pixel limit).
  - 30-page parsing cap.
  - Automatic OCR fallback via `pytesseract` for scanned non-text PDFs.
  - Returns `{"status": "success", "filename": "...", "extracted_text": "...", "truncated": false}`.

---

## 🧪 Testing & Verification

The backend includes a comprehensive pytest suite with 15 unit tests covering auth, upload security, rate limiting, and database operations with mocked services:

```bash
cd backend
python -m pytest tests/ -v
```

```text
tests/test_auth_and_abuse.py::test_signup_rate_limit PASSED
tests/test_auth_and_abuse.py::test_short_password_rejected PASSED
tests/test_data_layer.py::test_document_persistence_across_followups PASSED
tests/test_upload_safety.py::test_magic_byte_validation PASSED
...
====================== 15 passed in 7.08s ======================
```

---

## ⚖️ Legal Disclaimer

> **JurixAI is an artificial intelligence assistance tool built for informational and educational purposes only.**  
> It does not constitute formal legal advice or establish an attorney-client relationship. Always consult a licensed advocate for critical legal proceedings, court litigation, or official filings.

---

## 👨‍💻 Author & Contributions

**Tushar Singh** — [@tusharsingh-sde](https://github.com/tusharsingh-sde)  
*Full-Stack Engineer & AI/ML Practitioner*

Contributions, bug reports, and suggestions are welcome! Please open an issue or submit a PR.

---

<div align="center">

**Built with ❤️ for accessible, transparent, and empathetic Indian legal support.**

⭐ Star this repository if JurixAI helped you!

</div>
