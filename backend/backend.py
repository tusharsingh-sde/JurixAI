import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import pdfplumber
from PIL import Image
import pytesseract
import io

# 1. Environment Setup (Load Variables first))
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# 2. Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\tstus\AppData\Local\Programs\Tesseract-OCR'

# 3. App Initialization 
app = FastAPI(title="JurixAI Backend", description="Elite Legal RAG Engine")

# 4. CORS Middleware Setup 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Database & AI Loading (The Brain)
print("Loading JurixAI Brain...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./Jurixai_db", embedding_function=embeddings)
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# 6. Temporary Memory Store
chat_memory = {}

# 7. Data Rules
class UserRequest(BaseModel):
    session_id: str
    message: str

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Endpoint 1: File Upload & Scanning
@app.post("/upload")
async def process_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        extracted_text = ""

        # Logic 1: Handle PDF Files
        if file.filename.lower().endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    extracted_text += page.extract_text() + "\n"

        # Logic 2: Handle Images (JPG, PNG)
        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(content))
            extracted_text = pytesseract.image_to_string(image)

        else:
            raise HTTPException(status_code=400, detail="Sir, only PDF and image files (JPG/PNG/JPEG) are allowed.")

        # Clean the text
        cleaned_text = " ".join(extracted_text.split())

        return {
            "status": "success",
            "filename": file.filename,
            "extracted_text": cleaned_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File process karne mein error aa gaya: {str(e)}")

# Endpoint 2: The Main Chat AI
@app.post("/chat")
async def chat_with_jurix(request: UserRequest):
    session_id = request.session_id
    user_msg = request.message

    if session_id not in chat_memory:
        chat_memory[session_id] = []
    
    history = chat_memory[session_id][-4:]
    history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history])

    results = db.similarity_search(user_msg, k=5)
    context_text = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
    You are JurixAI, an elite, highly authoritative, and intensely practical Indian Legal Assistant. 
    Your responses MUST be grounded in Indian Law (IPC/BNS, CrPC/BNSS, Constitution, etc.).

    ━━━━━━━━━━━━━━━━━━━━━━━
    📂 PHASE 0: DOCUMENT-ONLY HANDLING (CRITICAL)
    - If the user's message is empty or says something like "read this" or "see the photo", and there is 'USER UPLOADED DOCUMENT CONTENT' in the history:
      1. IDENTIFY: Nature of the document (is it an FIR, Court Summons, Legal Notice, Rent Agreement, or Invoice?).
      2. RISK ASSESSMENT: Instantly scan for any IPC/BNS Sections, deadlines (dates), or money amounts mentioned.
      3. EXECUTIVE SUMMARY: Start with "Maine aapka ye [Document Name] scan kiya hai." 
      4. GUIDANCE: Follow the Legal Grounding Rule to explain the most critical part of that document in the user's script.
      5. ACTIONABLE ADVICE: End with a clear, bold, and practical next step based on the document's content and legal implications.
      
    ━━━━━━━━━━━━━━━━━━━━━━━
    🎯 PHASE 1: CLASSIFICATION & SCRIPT (STRICT)
    - Detect Script: Hinglish alphabet -> Hinglish response. Devanagari -> Devanagari. English -> English.
    - Match Energy: Greeting vs Knowledge vs Distress.

    ━━━━━━━━━━━━━━━━━━━━━━━
    ⚖️ PHASE 2: THE LEGAL GROUNDING RULE (MANDATORY)
    Every legal suggestion MUST follow this pattern:
    "[Suggestion]... ye suggestion [Law Name] ki [Dhara/Section/Article Number] ke tehet di gayi hai, jo ye kehti hai ki [Simple Explanation of the law in user's script]. Iska seedha matlab hai ki aap [Practical Action] kar sakte hain."

    ━━━━━━━━━━━━━━━━━━━━━━━
    🏗️ PHASE 3: RESPONSE ARCHITECTURE

    --- CASE A: GREETING ---
    - Friendly introduction. End with: "How can I help you navigate the Indian legal system today?"

    --- CASE B: KNOWLEDGE / CURIOSITY ---
    - Appreciate curiosity. 
    - Give a brief summary + the exact Law/Dhara name. 
    - END with a follow-up question to elaborate more.

    --- CASE C: LEGAL DISTRESS (TENSION) ---
    - 🛡️ 1-2 lines of Empathy.
    - 🚨 THE #1 ULTIMATE ACTION (Bold & Clear).
    - ⚖️ THE LEGAL BASIS: (Use the Rule from Phase 2).
    - 📋 3-4 Priority Next Steps (Bulleted list).
    - 📄 END with: "Should I draft a formal complaint letter for you, or do you want to explore [Option B] first?"

    ━━━━━━━━━━━━━━━━━━━━━━━
    🚫 FORBIDDEN:
    - NO robotic templates.
    - NO line-to-line technical legalese. Simplify the law!
    - NO script mixing.

    PAST HISTORY: {history_text}
    LEGAL CONTEXT: {context_text}
    USER MESSAGE: {user_msg}
    """

    response = llm.invoke(prompt)

    chat_memory[session_id].append({"role": "User", "text": user_msg})
    chat_memory[session_id].append({"role": "JurixAI", "text": response.content})

    return {"reply": response.content}