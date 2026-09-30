import os
import uuid
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client, Client
import pdfplumber
from PIL import Image
import pytesseract
import io

#environment setup
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

#tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\tstus\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

#supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:
    raise RuntimeError(
        "❌ Missing Supabase credentials! Please set SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_KEY) in backend/.env"
    )

supabase: Client = create_client(supabase_url, supabase_key)

#FastAPI Initialization
app = FastAPI(title="JurixAI Backend", description="Elite Legal RAG Engine")


#CORS Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#AI Brain Loading

print("Loading JurixAI")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./Jurixai_db", embedding_function=embeddings)
def invoke_gemini(prompt: str) -> str:
    """Invokes Gemini with automatic model fallback to prevent 503 high demand errors."""
    models = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.7-flash"]
    last_err = None
    for model_name in models:
        try:
            model = ChatGoogleGenerativeAI(model=model_name, max_retries=2)
            res = model.invoke(prompt)
            if isinstance(res.content, list):
                return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in res.content)
            return str(res.content)
        except Exception as e:
            print(f"Model {model_name} failed: {e}. Trying fallback model...")
    if last_err:
        raise last_err
    raise RuntimeError("All Gemini models failed to respond.")


#Data Models
class UserRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class AuthRequest(BaseModel):
    email: str
    password: str

class NewSessionRequest(BaseModel):
    title: Optional[str] = "New Conversation"


#Auth Helper — JWT verification via HTTPBearer
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract and verify user from Bearer token sent by the frontend."""
    token = credentials.credentials
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user_response.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please log in again.")

#History Helper — rolling summary for long conversations
MAX_VERBATIM = 10       # Last N messages sent as-is
SUMMARY_THRESHOLD = 20  # Summarize when total messages exceed this

def get_history_for_prompt(session_id: str) -> str:
    """
    Fetch messages from Supabase.
    For long conversations, old messages are summarized to save tokens.
    """
    all_msgs = supabase.table("messages") \
        .select("role, content") \
        .eq("session_id", session_id) \
        .order("created_at", desc=False) \
        .execute().data

    if not all_msgs:
        return "No previous conversation."

    if len(all_msgs) <= MAX_VERBATIM:
        return "\n".join(f"{m['role']}: {m['content']}" for m in all_msgs)

    # Rolling summary: compress old messages, keep recent ones verbatim
    old_msgs = all_msgs[:-MAX_VERBATIM]
    recent_msgs = all_msgs[-MAX_VERBATIM:]

    old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_msgs)
    summary_prompt = f"""Summarize this legal conversation in 3-4 sentences.
Preserve: the user's legal situation, key laws or sections mentioned, and any advice given.

{old_text}"""
    summary = invoke_gemini(summary_prompt)

    recent_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent_msgs)
    return f"[EARLIER SUMMARY]:\n{summary}\n\n[RECENT MESSAGES]:\n{recent_text}"


# API & AUTH ENDPOINTS

@app.post("/signup")
async def signup(request: AuthRequest):
    """Create a new JurixAI user account (automatically confirmed)."""
    try:
        response = supabase.auth.admin.create_user({
            "email": request.email,
            "password": request.password,
            "email_confirm": True
        })
        if response.user:
            return {
                "status": "success",
                "message": "Account created successfully! You can now log in immediately.",
                "user_id": str(response.user.id)
            }
        raise HTTPException(status_code=400, detail="Signup failed. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/login")
async def login(request: AuthRequest):
    """Log in and receive an access token for authenticated requests."""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        if response.session:
            return {
                "status": "success",
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
                "user_id": str(response.user.id),
                "email": response.user.email
            }
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/logout")
async def logout(user=Depends(get_current_user)):
    """Log out the current user."""
    supabase.auth.sign_out()
    return {"status": "success", "message": "Logged out successfully."}

#SESSION ENDPOINTS
@app.post("/sessions")
async def create_session(request: NewSessionRequest, user=Depends(get_current_user)):
    """Create a new chat session for the authenticated user."""
    try:
        response = supabase.table("sessions").insert({
            "user_id": str(user.id),
            "title": request.title
        }).execute()
        return {"status": "success", "session": response.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions(user=Depends(get_current_user)):
    """List all sessions for the authenticated user, newest first."""
    try:
        response = supabase.table("sessions") \
            .select("*") \
            .eq("user_id", str(user.id)) \
            .order("updated_at", desc=True) \
            .execute()
        return {"sessions": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user=Depends(get_current_user)):
    """Fetch all messages for a specific session (must be owned by the user)."""
    try:
        # Verify the user owns this session
        session = supabase.table("sessions") \
            .select("id") \
            .eq("id", session_id) \
            .eq("user_id", str(user.id)) \
            .execute()
        if not session.data:
            raise HTTPException(status_code=404, detail="Session not found.")

        messages = supabase.table("messages") \
            .select("id, role, content, created_at") \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .execute()
        return {"session_id": session_id, "messages": messages.data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user=Depends(get_current_user)):
    """Delete a session and all its messages (cascade handled by DB)."""
    try:
        supabase.table("sessions") \
            .delete() \
            .eq("id", session_id) \
            .eq("user_id", str(user.id)) \
            .execute()
        return {"status": "success", "message": "Session deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#DOCUMENT UPLOAD ENDPOINT

@app.post("/upload")
async def process_document(file: UploadFile = File(...), user=Depends(get_current_user)):
    """Upload a PDF or image document and extract its text content."""
    try:
        content = await file.read()
        extracted_text = ""

        if file.filename.lower().endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"

        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(content))
            extracted_text = pytesseract.image_to_string(image)

        else:
            raise HTTPException(status_code=400, detail="Only PDF and image files (JPG/PNG/JPEG) are allowed.")

        cleaned_text = " ".join(extracted_text.split())
        return {
            "status": "success",
            "filename": file.filename,
            "extracted_text": cleaned_text
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File process karne mein error aa gaya: {str(e)}")


#MAIN CHAT ENDPOINT

@app.post("/chat")
async def chat_with_jurix(request: UserRequest, user=Depends(get_current_user)):
    """
    Main chat endpoint. Requires an authenticated user and a valid session_id.
    Saves all messages to Supabase with full history context.
    """
    try:
        session_id = request.session_id
        user_msg = request.message

        # Check if session_id is a valid UUID
        is_valid_uuid = False
        if session_id and session_id != "string":
            try:
                uuid.UUID(str(session_id))
                is_valid_uuid = True
            except ValueError:
                is_valid_uuid = False

        session_data = None
        if is_valid_uuid:
            res = supabase.table("sessions") \
                .select("id, title") \
                .eq("id", session_id) \
                .eq("user_id", str(user.id)) \
                .execute()
            if res.data:
                session_data = res.data[0]

        # If no valid existing session, auto-create one for this user
        if not session_data:
            short_title = user_msg[:50] + ("..." if len(user_msg) > 50 else "")
            new_session = supabase.table("sessions").insert({
                "user_id": str(user.id),
                "title": short_title
            }).execute()
            session_id = new_session.data[0]["id"]
        elif session_data.get("title") == "New Conversation":
            short_title = user_msg[:50] + ("..." if len(user_msg) > 50 else "")
            supabase.table("sessions") \
                .update({"title": short_title}) \
                .eq("id", session_id) \
                .execute()

        # Build conversation history context
        history_text = get_history_for_prompt(session_id)

        # RAG: fetch relevant legal knowledge from ChromaDB
        results = db.similarity_search(user_msg, k=5)
        context_text = "\n\n".join([doc.page_content for doc in results])
        #promt to gemini
        prompt = f"""
You are JurixAI — a calm, deeply knowledgeable, and compassionate Indian Legal Saathi.
You are NOT a cold legal encyclopedia. You are the user's trusted friend who happens to know Indian Law inside out.
Your responses MUST be grounded in Indian Law (IPC/BNS, CrPC/BNSS, Constitution, Consumer Protection, POSH, RTI, IT Act, Domestic Violence Act, Tenancy Act, etc.).

━━━━━━━━━━━━━━━━━━━━━━━
🧘 CORE PRINCIPLE: CALM ANCHOR (ALWAYS ACTIVE)
- NEVER use alarming or panic-inducing language. No matter how serious the situation is, your FIRST job is to make the user feel safe and heard.
- NEVER say things like "This is very serious!", "You are in danger!", "Act immediately or else!".
- INSTEAD say things like "I completely understand your concern — let's sort this out step by step.", "You're doing the right thing by looking into this.", "Don't worry, you have strong legal options here."
- Frame every situation as solvable. The user came to you stressed — they should leave feeling empowered.
- Use warm, confident language: "Aapke paas clear legal protection hai" > "Ye bahut bada issue hai"

━━━━━━━━━━━━━━━━━━━━━━━
📂 PHASE 0: DOCUMENT-ONLY HANDLING
- If the user's message is empty or says something like "read this", "check this", "see the photo", and there is 'USER UPLOADED DOCUMENT CONTENT' in the history:
  1. IDENTIFY: Nature of the document (FIR, Court Summons, Legal Notice, Rent Agreement, Invoice, Complaint, etc.).
  2. RISK SCAN: Flag any IPC/BNS Sections, deadlines (dates), monetary amounts, or named parties.
  3. CALM SUMMARY: Start with "Maine aapka ye [Document Name] dhyan se padha hai — let me break it down for you."
  4. KEY FINDINGS: Highlight the 2-3 most important things the user NEEDS to know, using the Legal Grounding Rule.
  5. WHAT TO DO NEXT: End with one clear, practical next step and ask if they want you to explain any section in more detail.

━━━━━━━━━━━━━━━━━━━━━━━
🎯 PHASE 1: CLASSIFICATION & SCRIPT DETECTION (STRICT)
- Detect Script: If the user writes in Hinglish (Latin script with Hindi words) → respond in Hinglish. If Devanagari (हिंदी) → respond in Devanagari. If English → respond in English.
- NEVER mix scripts within a single response.
- Classify Intent: Greeting / General Knowledge / Legal Distress or Urgency.

━━━━━━━━━━━━━━━━━━━━━━━
⚖️ PHASE 2: THE LEGAL GROUNDING RULE (MANDATORY FOR EVERY LEGAL CLAIM)
Every legal suggestion you give MUST follow this citation pattern:
"[Your suggestion]... ye [Law Name] ki [Section/Dhara/Article Number] ke tehet aata hai, jo kehti hai ki [Simple one-line explanation in user's script]. Iska matlab hai ki aap [Practical action the user can take]."
- If no matching law exists in the provided context, say so honestly: "Is specific area mein mere paas abhi verified information nahi hai — I'd recommend consulting a lawyer for this."

━━━━━━━━━━━━━━━━━━━━━━━
🏗️ PHASE 3: RESPONSE ARCHITECTURE

--- CASE A: GREETING or CASUAL ---
- Respond warmly and naturally. No legal dump.
- End with a soft conversation opener: "Koi legal sawaal hai toh batao — I'm here to help, no judgement."

--- CASE B: KNOWLEDGE / CURIOSITY ---
- Appreciate the question: "Great question!" or "Ye jaanna bahut important hai."
- Give a clear, concise explanation + cite the exact Law/Dhara/Section.
- END with a hook — a related follow-up the user might not have thought of: "By the way, kya aapko ye bhi pata hai ki [related legal fact]? Want me to explain that too?"

--- CASE C: LEGAL DISTRESS / URGENT SITUATION ---
Structure your response in THIS EXACT priority order (fastest relief first):

🤝 STEP 1 — CALM & VALIDATE (2-3 lines)
Acknowledge their stress. Make them feel heard. Reassure them that solutions exist.
Example: "Pehle ek deep breath lo — aap sahi jagah aaye ho. Jo hua hai uska legal solution hai, aur hum milke sort karenge."

📞 STEP 2 — FASTEST EXIT: EMERGENCY CONTACTS & HELPLINES
Provide the single most relevant helpline or emergency number for their situation FIRST:
- Women's safety: Women Helpline 181, NCW 7827-170-170
- Cyber crime: National Cyber Crime Helpline 1930, cybercrime.gov.in
- Police: Dial 112 (Emergency), or nearest police station for FIR
- Consumer complaints: National Consumer Helpline 1800-11-4000
- Domestic violence: 181 (Women Helpline) or Protection Officer in their district
- Child abuse: Childline 1098
- General legal aid: NALSA 15100 (free legal help for eligible citizens)
Format: "📞 Sabse pehle ye karo: [Helpline Name] pe call karo — [Number]. Ye [24x7 / working hours info]. Agar ye busy ho, toh neeche alternatives hain."

⚖️ STEP 3 — LEGAL GROUNDING (cite the law using Phase 2 rule)
Explain what law protects them and what it means in simple language.

📋 STEP 4 — SMART ALTERNATIVES (3-4 steps, ranked by speed/ease)
If the helpline doesn't work or the user wants to take further action, provide a bulleted list ordered from quickest to most involved:
1. [Fastest — e.g., online complaint portal, SMS-based FIR]
2. [Next best — e.g., visit nearest police station, file written complaint]
3. [Formal route — e.g., approach district court, consumer forum]
4. [Long-term — e.g., hire a lawyer, file PIL, RTI application]

💬 STEP 5 — CONVERSATION HOOK (keep user engaged)
End with a specific, helpful offer — NOT a generic "let me know if you need help":
- "Kya aap chahte ho ki main aapke liye ek complaint letter draft kar doon?"
- "Agar aap batao ki ye kab hua, toh main aapko exact deadline bata sakta hoon filing ki."
- "Want me to explain what will happen after you file the FIR? I can walk you through the whole process."

━━━━━━━━━━━━━━━━━━━━━━━
🚫 STRICTLY FORBIDDEN:
- NO panic language, fear-mongering, or worst-case scenarios upfront.
- NO robotic templates or copy-paste legal walls of text.
- NO raw legalese — always simplify the law into human language.
- NO script mixing (don't write Hindi words in English response or vice versa).
- NO hallucinating laws or sections. If it's not in the context, say you don't know.
- NO generic sign-offs like "Hope this helps!" — always end with a SPECIFIC next offer.

━━━━━━━━━━━━━━━━━━━━━━━

PAST CONVERSATION:
{history_text}

LEGAL KNOWLEDGE BASE:
{context_text}

USER'S MESSAGE: {user_msg}
        """

        reply = invoke_gemini(prompt)

        # Save both turns to Supabase (single batch insert)
        supabase.table("messages").insert([
            {"session_id": session_id, "role": "User",    "content": user_msg},
            {"session_id": session_id, "role": "JurixAI", "content": reply}
        ]).execute()

        return {
            "session_id": session_id,
            "reply": reply
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

