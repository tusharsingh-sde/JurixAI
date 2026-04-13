# JurixAI ⚖️ 
### *Empowering Legal Clarity with AI & Empathy*

JurixAI is a high-performance, RAG-based (Retrieval-Augmented Generation) legal assistant designed specifically for the Indian Legal System. It doesn't just give cold legal facts; it acts as a "Legal Saathi" that understands the user's stress and provides authoritative, law-backed guidance.

---

## 🚀 The Problem it Solves
Indian law is vast and complex. Most people feel intimidated by legal jargon. **JurixAI** bridges this gap by:
1. Simplifying complex legal sections (IPC, POSH, RTI, etc.).
2. Providing immediate, actionable steps for common legal issues.
3. Maintaining a human-centric, empathetic tone.

---

## 🛠️ Tech Stack (The Muscle)
- **LLM:** Llama-3.3-70b-versatile (via **Groq Cloud** for lightning-fast inference).
- **Orchestration:** **LangChain** (for RAG pipeline).
- **Vector Database:** **ChromaDB** (to store and retrieve legal document embeddings).
- **Embeddings:** **HuggingFace** (`all-MiniLM-L6-v2`) for local, fast semantic search.
- **Environment:** Python 3.10+

---

## 🧠 How it Works (The RAG Pipeline)

1. **Data Ingestion:** PDFs of Indian Laws are loaded and split into semantic chunks.
2. **Vectorization:** Chunks are converted into vector embeddings and stored in `Jurixai_db`.
3. **Retrieval:** When a user asks a question, the system finds the top 8 most relevant legal contexts.
4. **Augmentation:** The context is fed into Llama 3.3 with a custom "Empathy-First" prompt.
5. **Generation:** The AI generates a structured, easy-to-read response in the user's language (English/Hinglish).

## ⚖️ Disclaimer
This project is for **educational and research purposes only**. JurixAI is an AI-powered tool and not a substitute for professional legal advice from a qualified lawyer. Always verify legal information from official government sources before taking any action.

---

## ⚙️ Installation & Setup

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/tusharsingh-sde/JurixAI_core.git
   cd JurixAI_core

  
 2. **Setup Virtual Environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    .\venv\Scripts\activate
    # On Mac/Linux:
    source .venv/bin/activate

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

4. **API Configuration:**

   Create a ```.env``` file in the root directory and add your Groq API Key:

       GROQ_API_KEY=your_key_here

5. **Initialize Database:**

   ```bash

   python build_database.py

6. **Run JurixAI:**

   ```bash

   python chatbot.py

   
## 🛡️ Key Features
- **Context-Aware: Doesn't hallucinate; only answers based on provided law books.**

- **Bilingual: Detects and responds in the user's preferred language (Hinglish/English).**

- **Structured Output: Gives "The #1 Ultimate Action" followed by legal sections and follow-up steps.**


## 👨‍💻 Author
Tushar Singh

- **GitHub: @tusharsingh-sde**

- **Focus: Software Engineering | AI/ML | Software Development**
