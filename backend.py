from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # CORS
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

app = FastAPI(title="JurixAI Backend", description="Elite Legal RAG Engine")

# CORS Middleware Setup 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # this will allow all origins, but in production, you should specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Environment & API Setup
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# FastAPI Initialization
app = FastAPI(title="JurixAI Backend", description="Elite Legal RAG Engine")

# 2. Database & AI Loading (this will be our "brain" that we load once when the server starts)
print("Loading JurixAI Brain...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./Jurixai_db", embedding_function=embeddings)
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# 3. Temporary Memory Store (Session ID -> List of messages)
chat_memory = {}

# 4. Data Rule (Pydantic Bouncer - this ensures we get the right data format from the frontend)
class UserRequest(BaseModel):
    session_id: str  # User's unique ID so that it will mix with the memory
    message: str     # User's legal query or message

# 5. The Main API Endpoint (Yahan se UI connect hoga)
@app.post("/chat")
async def chat_with_jurix(request: UserRequest):
    session_id = request.session_id
    user_msg = request.message

    # Agar naya user hai, toh uska memory khaata kholo
    if session_id not in chat_memory:
        chat_memory[session_id] = []
    
    # Pichli 4 messages nikalo (taaki bot bhool na jaye)
    history = chat_memory[session_id][-4:]
    history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history])

    # RAG: Database se law nikalo
    results = db.similarity_search(user_msg, k=5)
    context_text = "\n\n".join([doc.page_content for doc in results])

    # 6. Advanced Prompt (Elite Lawyer UX + STRICT Script Match + RAG Context)
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

    # AI se jawab maango
    response = llm.invoke(prompt)

    # 7. Nayi baatein Memory mein Save karo
    chat_memory[session_id].append({"role": "User", "text": user_msg})
    chat_memory[session_id].append({"role": "JurixAI", "text": response.content})

    # Frontend (UI) ko jawab wapas bhejo
    return {"reply": response.content}