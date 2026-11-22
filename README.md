<div align="center">

# ⚖️ JurixAI

### Your AI-Powered Legal Saathi for the Indian Legal System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![LangChain](https://img.shields.io/badge/LangChain-Chroma-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Supabase](https://img.shields.io/badge/Supabase-Auth_%26_DB-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![Pytest](https://img.shields.io/badge/Pytest-37_Passed-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**JurixAI** is a production-hardened, **RAG-based** (Retrieval-Augmented Generation) legal intelligence assistant purpose-built for the **Indian Legal System**. It goes beyond cold legal summaries — acting as a trusted *Legal Saathi* that understands distress, navigates new and legacy criminal codes (BNS/IPC, BNSS/CrPC, BSA), and delivers authoritative, statute-backed guidance with empathy, procedural clarity, and robust quota protection.

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [Knowledge Base](#-legal-knowledge-base) · [API Reference](#-api-reference) · [Docker Deployment](#-docker-deployment) · [Testing](#-testing--verification)

</div>

---

## 🎯 The Problem

Indian law is vast, nuanced, and deeply intimidating for citizens and small businesses. From understanding FIR procedures under the newly enacted criminal codes to navigating cheque bounce notices, tenant eviction rights, or cyber fraud, most people feel overwhelmed by legal jargon and uncertain about procedural remedies.

**JurixAI bridges this gap by:**

- 📖 **Demystifying both Modern & Legacy Indian Statutes** — Full coverage of **Bharatiya Nyaya Sanhita (BNS 2023)**, **Bharatiya Nagarik Suraksha Sanhita (BNSS 2023)**, **Bharatiya Sakshya Adhiniyam (BSA 2023)**, alongside legacy **IPC 1860**, **CrPC 1973**, **Constitution of India**, **NI Act 1881**, **POSH Act**, **RTI Act**, **Cyber Law (IT Act)**, **Domestic Violence Act**, **Consumer Protection**, and **Tenancy Laws**.
- 🚀 **Providing actionable next steps** — Clear, structured procedural guidance, emergency helplines (112, 1930, 181, 15100), and draft notices tailored to your specific situation.
- 🤝 **Maintaining an empathetic, human-centric tone** — A "Calm Anchor" persona that detects distress, de-escalates panic, and delivers structured advice (Fastest Exit $\rightarrow$ Statutory Grounding $\rightarrow$ Smart Alternatives).
- 🌐 **Trilingual intelligence** — Seamlessly understands and responds in **English**, **Hindi** (Devanagari), and **Hinglish** without script mixing.
- 📄 **Multi-page document analysis** — Upload FIRs, legal notices, summons, or agreements with automatic OCR fallback via `pytesseract` for scanned documents and images.
- ⚡ **Input-Token & Cost Optimization** — Smart small-talk RAG bypass, strict context budgets (`MAX_RAG_CONTEXT_CHARS=3500`, `MAX_DOC_CONTEXT_CHARS=6000`), bounded history windows (`MAX_HISTORY_MESSAGES=4`, `MAX_HISTORY_MSG_CHARS=700`), and `GEMINI_MAX_OUTPUT_TOKENS=1024`.
- 🛡️ **Enterprise-grade production guardrails** — Global daily kill-switch (`MAX_GLOBAL_DAILY_REQUESTS=250`), strict unauthenticated guest rate-limits (5 req/10min, 15 req/day), message length caps (1500 chars for guests), and dual-scoped Supabase auth.
- 🎨 **Flagship Gemini-Inspired Dark Theme** — Deep midnight canvas (`#090D16`), subtle ambient cobalt glow, Lady Justice artwork, borderless AI replies, and responsive frosted docks.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React 19 + Vite)                      │
│            Flagship Gemini Dark Theme · Responsive Frosted Dock        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │  HTTP / REST API (CORS + PNA Allowed)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI Hardened Service)                  │
│                     Port 8000 · Single Worker Non-Root                 │
│                                                                        │
│  ┌────────────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │   Auth & Abuse Guard   │  │   Upload Safety  │  │  Health Probes  │ │
│  │ SlowAPI Rate Limiting  │  │ Chunked 413 Abort│  │ /health (Live)  │ │
│  │ Global Daily Limit=250 │  │ Magic Byte Check │  │ /ready (Chroma) │ │
│  │ Dual Supabase Scopes   │  │ OCR Image Defense│  │                 │ │
│  └───────────┬────────────┘  └────────┬─────────┘  └─────────────────┘ │
│              │                        │                                │
│              ▼                        ▼                                │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                     Async Lifespan Engine                        │ │
│  │  ┌──────────────────────────┐    ┌─────────────────────────────┐ │ │
│  │  │ Canonical ChromaDB Store │    │ SentenceTransformer Embeds  │ │ │
│  │  │ (6,994 Embedded Chunks)  │    │ (all-MiniLM-L6-v2 Offline)  │ │ │
│  │  └──────────────────────────┘    └─────────────────────────────┘ │ │
│  │  ┌──────────────────────────┐    ┌─────────────────────────────┐ │ │
│  │  │ Tesseract OCR (Eng+Hin)  │    │ Token & Context Optimizer   │ │ │
│  │  │ Auto-Fallback Engine     │    │ Small-Talk Bypass & Budgets │ │ │
│  │  └──────────────────────────┘    └─────────────────────────────┘ │ │
│  └───────────────────────────────────┬──────────────────────────────┘ │
│                                      │                                 │
│                                      ▼                                 │
│                       ┌──────────────────────────────┐                │
│                       │    Google Gemini API SDK     │                │
│                       │  (gemini-2.5-flash cascade   │                │
│                       │   with multi-model fallback) │                │
│                       └──────────────────────────────┘                │
└────────────────────────────────────────────────────────────────────────┘
```

### The Hardened RAG Pipeline

| Step | Phase | Description |
|------|-------|-------------|
| 1 | **Data Ingestion & Chunking** | 19 authoritative Indian statutes and bare acts audited and parsed with `RecursiveCharacterTextSplitter` calibrated to `chunk_size=750` chars and `chunk_overlap=120` chars to fit safely within `all-MiniLM-L6-v2`'s 256-token hard window. Scanned/empty PDFs are automatically purged. |
| 2 | **Canonical Vectorization** | Chunks embedded offline using `all-MiniLM-L6-v2` and indexed into the canonical persistent ChromaDB directory (`backend/Jurixai_db`) containing **6,994 legal vectors** (42.3 MB on disk). |
| 3 | **Smart RAG Retrieval & Small-Talk Bypass** | If the user sends greetings or acknowledgements ("hi", "namaste", "thanks", "ok", "bye" under 5 words with zero legal terms) without an uploaded document, ChromaDB retrieval is bypassed completely for a lightweight conversational prompt (<500 chars). For legal queries, similarity search fetches `k=3` chunks capped at `MAX_RAG_CONTEXT_CHARS=3500`. |
| 4 | **Context & History Augmentation** | Input context is tightly bounded: historical turns capped to last 4 messages (`MAX_HISTORY_MESSAGES=4`), 700 chars per message (`MAX_HISTORY_MSG_CHARS=700`), and uploaded document text capped to 6,000 chars (`MAX_DOC_CONTEXT_CHARS=6000`). Prompt length is logged at `INFO` level. |
| 5 | **Resilient Generation & Quota Protection** | Before invocation, the global daily counter (`MAX_GLOBAL_DAILY_REQUESTS=250`) is evaluated with automatic UTC date rollover. Generation runs with fallback across models (`gemini-2.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.8-flash`) and output tokens capped at 1024. |

---

## 🛠️ Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11 / 3.14 | Core execution runtime |
| **FastAPI** | 0.115+ | High-performance asynchronous REST API |
| **Uvicorn** | 0.34+ | ASGI web server with proxy header support |
| **SlowAPI** | 0.1.9 | IP-based & token rate-limiting abuse protection |
| **LangChain Chroma** | 1.1.0 | Vector database interface for ChromaDB |
| **Sentence-Transformers** | 5.2.0 | Offline embedding generation (`all-MiniLM-L6-v2`) |
| **Google GenAI** | 1.63.0 | Gemini multi-model generation with automatic fallback |
| **Supabase** | 2.14.0 | Dual-client authentication and relational PostgreSQL persistence |
| **pypdf / pdfplumber** | Latest | High-speed PDF text parsing and extraction |
| **PyTesseract** | 0.3.13 | Tesseract OCR engine with English + Hindi support |
| **Pillow** | 11.1.0 | Image manipulation with DecompressionBomb security caps |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 19.0.0 | Component-based reactive UI |
| **Vite** | 8.3.8 | Lightning-fast build tool with HMR |
| **Lucide React** | 0.468.0 | Minimalist iconography |
| **CSS3** | Modern | Gemini-inspired Deep Midnight dark theme (`#090D16`) |

---

## 📚 Legal Knowledge Base

JurixAI is pre-indexed with **19 verified Indian legislative acts, bare acts, and comparative tables** totaling **6,994 semantic chunks**:

| Statute / Document | Chunks | Coverage & Core Provisions |
|:---|:---:|:---|
| `constitution-of-india.pdf` | **1,859** | Fundamental Rights (Arts. 14, 19, 21, 22), Writs (Art. 32/226), Directive Principles |
| `Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf` | **1,495** | BNSS 2023 — Arrest procedure, FIR registration (Sec. 173), bail safeguards, trials |
| `IPC.pdf` | **858** | Indian Penal Code 1860 — Criminal offences, definitions, penalties |
| `BNS2023.pdf` | **811** | Bharatiya Nyaya Sanhita 2023 — Modern criminal law replacing IPC |
| `BSA2023.pdf` | **343** | Bharatiya Sakshya Adhiniyam 2023 — Digital evidence, admissibility, witness rules |
| `Motor Vehicles (Amendment) Act, 2019.pdf` | **234** | Traffic penalties, challans, compensation, licensing regulations |
| `Consumer_Handbook.pdf` | **204** | Consumer Protection Act 2019 — Defective goods, unfair trade practices, e-commerce |
| `negotiable-instrument-act.pdf` | **185** | NI Act 1881 — Section 138 cheque bounce notices, 15-day timelines, jurisdiction |
| `payment of wages.pdf` | **158** | Payment of Wages Act 1936 — Unlawful salary deductions, delay remedies |
| `COMPARATIVE-TABLE-OF-IPC-1860-BNS-2023...pdf` | **129** | Direct mapping table: IPC Section $\leftrightarrow$ Equivalent BNS Section |
| `RTI-Act_English.pdf` | **124** | Right to Information Act 2005 — 30-day filing timeline, PIO duties, appeals |
| `258.pdf` *(Special Marriage Act, 1954)* | **118** | Civil marriage registration, 30-day notice, divorce procedure |
| `29 - The Hindu Marriage Act, 1955.pdf` | **117** | Restitution of conjugal rights, mutual divorce, maintenance under Hindu law |
| `COMPARATIVE-TABLE-OF-CRPC-1973-BNSS-2023...pdf` | **111** | Direct mapping table: CrPC Section $\leftrightarrow$ Equivalent BNSS Section |
| `Model-Tenancy-Act_2021.pdf` | **108** | Tenancy rights, rent agreements, 2-month security deposit caps, eviction bounds |
| `POSH_act.pdf` | **69** | Prevention of Sexual Harassment at Workplace Act 2013 — Internal Complaints Committee |
| `protection_of_women_from_domestic_violence_act,.pdf` | **64** | Domestic Violence Act 2005 — Protection orders, residence orders, Sec. 12 petitions |
| `cyber_law.pdf` | **6** | IT Act 2000 — Identity theft (Sec. 66C), cheating by personation (Sec. 66D) |
| `Section_105_in_The_Transfer_Of_Property_Act_1882.PDF` | **1** | Leases of immovable property, rights and liabilities of lessor/lessee |
| **Total Chunks** | **6,994** | **42.32 MB Persistent Vector Store** |

---

## 📁 Project Structure

```
JurixAI_core/
├── backend/
│   ├── backend.py            # Hardened FastAPI service, guardrails & endpoints
│   ├── build_database.py     # Legal PDF audit & ChromaDB indexing pipeline
│   ├── query_db.py           # CLI vector database query tool
│   ├── requirements.txt      # Pinned production runtime dependencies
│   ├── Dockerfile            # Multi-stage, non-root hardened container
│   ├── .dockerignore         # Container build context exclusions
│   ├── .env.example          # Environment variable template with documentation
│   ├── legal_data/           # 19 audited Indian Law PDFs (text-verified)
│   ├── Jurixai_db/           # Canonical ChromaDB vector store (6,994 vectors)
│   └── tests/                # Automated pytest suite (37 tests)
│       ├── test_auth_and_abuse.py   # Rate limits, daily caps, small talk, tokens
│       ├── test_data_layer.py       # Supabase persistence, session locks, history
│       └── test_upload_safety.py    # Magic bytes, decompression bombs, OCR caps
│
├── frontend/
│   ├── index.html            # SPA entry with dark canvas metadata
│   ├── vite.config.js        # Vite dev server configuration
│   ├── package.json          # Node dependencies
│   ├── public/               # Static assets (lady-justice.png)
│   └── src/
│       ├── main.jsx          # React DOM mounting
│       ├── App.jsx           # Full chat UI, guest history & document handling
│       └── styles.css        # Flagship Gemini Dark Theme styling
│
├── migrations/
│   └── 001_jurix.sql         # Idempotent Supabase PostgreSQL schema & RLS policies
│
├── .dockerignore             # Root docker build exclusions
├── .gitignore                # Comprehensive exclusions (secrets, venv, caches)
├── docker-compose.yml        # Docker compose orchestrator for backend
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python** 3.10+ (Python 3.11 recommended for Docker parity)
- **Node.js** 18+ and **pnpm** (or npm)
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

# Optional Quota & Token Budgets (defaults shown)
MAX_GLOBAL_DAILY_REQUESTS=250
GUEST_CHAT_PER_10MIN=5
GUEST_CHAT_PER_DAY=15
GUEST_UPLOAD_PER_HOUR=3
MAX_GUEST_MESSAGE_CHARS=1500
GEMINI_MAX_OUTPUT_TOKENS=1024
RAG_TOP_K=3
MAX_RAG_CONTEXT_CHARS=3500
MAX_HISTORY_MESSAGES=4
MAX_HISTORY_MSG_CHARS=700
MAX_DOC_CONTEXT_CHARS=6000
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
pnpm install
pnpm dev
```

Open `http://localhost:5173` in your browser.

---

## 🐳 Docker Deployment

The backend is packaged as an optimized, non-root, production container with pre-warmed models and health checks:

```bash
# Using docker-compose
docker compose up -d

# Or manual build:
docker build -t jurixai-backend ./backend
docker run -p 8000:8000 --env-file backend/.env jurixai-backend
```

---

## 📡 API Reference

### Health & Liveness
- **`GET /health`**: Fast dependency-free liveness probe for load balancers. Returns `{"status": "ok"}`.
- **`GET /ready`**: Readiness check verifying ChromaDB collection state. Returns `{"status": "ready", "documents_loaded": 6994}`.

### Authentication (SlowAPI Rate Limited)
- **`POST /signup`**: Creates an unconfirmed user account requiring email verification (Limit: `5/hour/IP`, min password 8 chars, anti-spoofing).
- **`POST /login`**: Verifies user credentials via Supabase Anon client and issues JWT token (Limit: `10/minute/IP`).
- **`POST /logout`**: Invalidates user token with `scope="local"`.

### Sessions & Conversations
- **`GET /sessions`**: List conversations for authenticated user (strictly forbidden for unauthenticated guests).
- **`GET /sessions/{id}/messages`**: Retrieve message history for a session (paginated to 100 messages ordered by sequence).
- **`DELETE /sessions/{id}`**: Delete a session and its associated message history.

### Core RAG Chat
- **`POST /chat`**:
  - **Global Daily Cap**: Enforces `MAX_GLOBAL_DAILY_REQUESTS` (default: 250). Returns HTTP `429` when reached. Auto-resets at 00:00 UTC.
  - **Rate Limits**:
    - Authenticated: `30/minute/user`.
    - Guest Mode: `5/10min/IP` and `15/day/IP`.
  - **Message Length**: Up to 4,000 chars for authenticated users; capped at 1,500 chars for guests.
  - **Small-Talk Optimization**: Short greetings/acknowledgements under 5 words without legal keywords skip ChromaDB retrieval completely.
  - **Payload**:
    ```json
    {
      "message": "What is the timeline for Section 138 notice?",
      "session_id": "optional-uuid-here",
      "document_text": "optional extracted legal document text (capped at 6000 chars)",
      "document_name": "optional-filename.pdf",
      "guest_history": [
        {"role": "User", "content": "Previous question"},
        {"role": "JurixAI", "content": "Previous answer"}
      ]
    }
    ```
  - Returns law-grounded response with empathetic guidance, legal sections, emergency contacts, and action steps.

### Document Upload & OCR
- **`POST /upload`** (Limit: `10/minute/user` for authed users, `3/hour/IP` for unauthenticated guests):
  - Enforces `MAX_UPLOAD_MB` (default 8MB) with early chunked 413 abort.
  - Verifies magic byte headers (`%PDF`, PNG, JPEG).
  - Decompression bomb protection (25M pixel limit).
  - 30-page parsing cap.
  - Automatic OCR fallback via `pytesseract` for scanned non-text PDFs and images.
  - Returns `{"status": "success", "filename": "...", "extracted_text": "...", "truncated": false}`.

---

## 🧪 Testing & Verification

The test suite contains **37 comprehensive unit tests** covering all auth, rate-limit, abuse prevention, RAG optimization, and data layer behaviors:

```bash
backend\.venv\Scripts\pytest backend/tests
```

```text
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\tstus\Desktop\Projects\JurixAI_core
plugins: anyio-4.13.0, langsmith-0.7.30
collected 37 items

backend\tests\test_auth_and_abuse.py ......................              [ 59%]
backend\tests\test_data_layer.py .........                               [ 83%]
backend\tests\test_upload_safety.py ......                               [100%]

======================= 37 passed, 7 warnings in 11.48s =======================
```

### Key Verified Invariants:
1. **Global Daily Kill-Switch**: Exact 429 response when daily budget is reached; automatic UTC date rollover.
2. **Small-Talk RAG Bypass**: Greetings and acknowledgements bypass ChromaDB retrieval; minimal prompt (<500 chars).
3. **Budget Capping**: History capped to 4 turns (700 chars/turn); document text capped to 6,000 chars; RAG context capped to 3,500 chars.
4. **Guest Shield**: Rate-limited to 5/10min and 15/day; message length capped to 1500 chars; zero database writes.
5. **Anti-Spoofing & Pre-Hijacking**: Standard unconfirmed signups prevent account hijacking; spoofed `X-Forwarded-For` headers cannot bypass IP rate limits.
6. **Upload Safety**: Magic bytes verified, chunked early abort, OCR page caps, and image decompression protection.

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
