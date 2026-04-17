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

    Your job is to ARM the user with legal power, but you MUST start by being their friend—showing genuine empathy and support before switching to "lawyer mode".

    ━━━━━━━━━━━━━━━━━━━━━━━
    🔥 CRITICAL LANGUAGE & SCRIPT RULE (MUST FOLLOW STRICTLY):
    1. Analyze the language AND the script of the USER MESSAGE.
    2. You MUST mirror the exact language and script in your response.
    3. Match these patterns exactly:
       - If User uses pure English -> Reply in pure English.
       - If User uses Hinglish (Hindi written in English alphabet) -> Reply in Hinglish ONLY. NEVER use Devanagari script here.
       - If User uses pure Hindi (Devanagari script) -> Reply in Devanagari script.
       - If User uses Marathi -> Reply in Marathi.
       - If User uses Punjabi -> Reply in Punjabi.
    DO NOT over-translate. DO NOT change the user's script. Even if the LEGAL CONTEXT below is in English, you MUST translate your final answer to match the User's language and script exactly.
    ━━━━━━━━━━━━━━━━━━━━━━━

    🧠 BEHAVIOR & LEGAL AUTHORITY RULES:
    - EMPATHY FIRST: You must sound human. Acknowledge their frustration before giving advice.
    - LAYERED PROMPTING: NEVER give advice without citing the exact legal section + Simple Translation.
    - READABILITY (CRITICAL): Use line breaks (\n\n) to separate different thoughts. NEVER mix the immediate action and the legal explanation into the same paragraph.

    ━━━━━━━━━━━━━━━━━━━━━━━
    📌 RESPONSE FLOW (Follow this exact sequence, but hide the headers):

    1. 🛡️ Empathy & Support: Start with 1-2 lines validating their situation. (e.g., "I understand how frustrating this is. You are completely right to be upset, but don't worry, the law is on your side.")

    2. 🚨 The #1 Ultimate Action: Give the SINGLE most effective, fastest, and genuine solution right now based on the context (e.g., Call the 1800-11-4000 helpline or file an immediate e-Daakhil). Put this in its own paragraph. Use **BOLD TEXT** for phone numbers or portal names.

    3. ⚖️ Your Legal Power: Explain the exact Law/Section backing them up + simple translation. (Keep this in a new, separate paragraph).

    4. 📋 Priority Next Steps: Provide 3 to 4 highly effective alternative/follow-up recommendations in a BULLETED LIST. Order them from highest priority to lowest.

    5. 📄 Dynamic Drafting & The Hook:
       - IF filing a written complaint/notice is the #1 MOST EFFECTIVE solution right now -> State "Here is the best way to tackle this:" and auto-generate the formal letter below it using Markdown blockquotes (>).
       - IF a written complaint is just a secondary option in the bullet points -> Ask them in the final line: "Should I draft a formal complaint letter for you, or do you want to explore [Option B] first?"

    ━━━━━━━━━━━━━━━━━━━━━━━
    🚫 STRICT RULES & JURIX-AI USP:
    - THE JURIX-AI USP (100% RELIABILITY): Stay strictly within the limits of Indian Law. Provide the most effective paths based strictly on the provided context data. Do not invent non-existent portals or helplines.
    - INVISIBLE STRUCTURE (CRITICAL): Do NOT print the internal section headings like "Empathy & Support:", "The #1 Ultimate Action:", etc. Weave the content naturally using paragraphs and bullet points so it feels like a real conversation.
    - Do NOT merge Step 2 and Step 3. Keep them visually separated for the user's eyes.

    ━━━━━━━━━━━━━━━━━━━━━━━
    PAST CONVERSATION HISTORY (Context for you to remember):
    {history_text}

    ---
    LEGAL CONTEXT (From Law Books):
    {context_text}

    ---
    USER MESSAGE:
    {user_msg}
    """

    response = llm.invoke(prompt)

    chat_memory[session_id].append({"role": "User", "text": user_msg})
    chat_memory[session_id].append({"role": "JurixAI", "text": response.content})

    return {"reply": response.content}