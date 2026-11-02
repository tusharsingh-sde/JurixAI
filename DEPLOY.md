# JurixAI — Production Deployment Guide

This guide details the step-by-step production deployment process for the JurixAI backend service.

---

## 1. System Requirements & Hardware Sizing

- **Minimum Memory (RAM)**: **$\ge$ 1 GB RAM**
  - Container baseline consumption: ~750 MB to 850 MB.
  - Sentence-transformers embedding model (`all-MiniLM-L6-v2`) in-memory: ~350 MB.
  - ChromaDB vector store index: ~120 MB.
  - Fast-API & Uvicorn baseline: ~80 MB.
  - Single-worker design (`--workers 1`) bounds total process memory under 900 MB.
- **Compute (CPU)**: **$\ge$ 1 vCPU** (2 vCPUs recommended for concurrent requests and OCR processing).
- **Disk Storage**: ~1.2 GB container image.

---

## 2. Vector Database (`Jurixai_db`) & Repository Sizing

- **Disk Footprint**: The `backend/Jurixai_db` directory occupies **~24.3 MB** total on disk (the largest single file is `chroma.sqlite3` at **20.46 MB**).
- **GitHub Limit Compliance**: Well within GitHub's 100 MB hard file limit (and below GitHub's 50 MB soft warning threshold). It is safely tracked directly in Git and baked into the Docker image at build time via `COPY`.
- **ChromaDB Version**: Pinned to `chromadb==1.5.7`.
  > [!IMPORTANT]
  > The database was constructed and indexed with ChromaDB 1.5.x. If the ChromaDB version is ever upgraded, rebuild the vector store using `build_database.py`.

---

## 3. Required Environment Variables

Configure these variables in your production environment / platform dashboard (e.g. Cloud Run, Railway, Render, AWS ECS, Fly.io):

| Variable Name | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `GEMINI_API_KEY` | **Yes** | — | Google Gemini API Key (or `GOOGLE_API_KEY`). |
| `SUPABASE_URL` | **Yes** | — | Supabase project URL (`https://<project-ref>.supabase.co`). |
| `SUPABASE_SECRET_KEY` | **Yes** | — | Supabase **Service Role** Key (used for backend database queries and admin actions). |
| `SUPABASE_ANON_KEY` | **Yes** | — | Supabase **Anon** Key (used strictly for credential verification). |
| `ALLOWED_ORIGINS` | **Yes** | `localhost` | Comma-separated CORS allowed origins (e.g. `https://jurixai.com,https://app.jurixai.com`). Never use `*` in production. |
| `FORWARDED_ALLOW_IPS` | **Yes** | `127.0.0.1` | Trusted proxy IP/CIDR for Uvicorn `--forwarded-allow-ips`. Set to your platform's load balancer or reverse proxy CIDR (e.g., `*` on isolated ingress proxies like Cloud Run or Railway). |
| `GEMINI_MODELS` | No | `gemini-3.5-flash,...` | Comma-separated fallback model sequence for Gemini invocations. |
| `MAX_UPLOAD_MB` | No | `8` | Maximum allowable document upload size in megabytes. |
| `MAX_MESSAGE_CHARS` | No | `4000` | Maximum character length for chat input messages. |
| `PORT` | No | `8000` | Port on which the container listens. |

---

## 4. Deployment Order of Operations

Follow this strict 5-step sequence to ensure zero-downtime deployment:

```
[1. Back up Supabase] ──> [2. Apply SQL Migration] ──> [3. Set Env Vars] ──> [4. Deploy Docker] ──> [5. Post-Deploy Checks]
```

### Step 1: Back up Supabase Database
In your Supabase Dashboard:
- Navigate to **Database** $\rightarrow$ **Backups**.
- Click **Create Backup** or verify your latest automated daily backup is successful.

### Step 2: Apply SQL Migration
- Open the Supabase Dashboard $\rightarrow$ **SQL Editor**.
- Open the contents of [`migrations/001_jurix.sql`](file:///c:/Users/tstus/Desktop/Projects/JurixAI_core/migrations/001_jurix.sql).
- Run the script.
- *Verification*: Confirm `messages` contains column `seq` (bigint), `sessions` contains columns `summary`, `summary_upto`, and `document_text`, and RLS is enabled.

### Step 3: Configure Environment Variables
Set all mandatory environment variables listed in Section 3 in your hosting platform.

### Step 4: Build and Deploy Container
Build and push the Docker image:

```bash
# Build the Docker image from root
docker build -t jurixai-backend:latest .

# Run the container locally or on your server
docker run -d \
  --name jurixai-app \
  -p 8000:8000 \
  --env-file .env \
  jurixai-backend:latest
```

### Step 5: Post-Deployment Verification Checks

1. **Liveness Probe**:
   ```bash
   curl -i http://localhost:8000/health
   # Expected: HTTP/1.1 200 OK {"status":"ok"}
   ```

2. **Readiness Probe**:
   ```bash
   curl -i http://localhost:8000/ready
   # Expected: HTTP/1.1 200 OK {"status":"ready","documents_loaded":2158}
   ```

3. **Anti-Spoofing & Abuse Protection Test**:
   Verify that a spoofed `X-Forwarded-For` header cannot bypass the signup rate limiter:
   ```bash
   # Run 6 rapid signup requests with spoofed X-Forwarded-For
   for i in {1..6}; do
     curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/signup \
       -H "Content-Type: application/json" \
       -H "X-Forwarded-For: 203.0.113.$i" \
       -d '{"email":"test@example.com","password":"InvalidPassword123"}'
   done
   # The 6th request MUST return HTTP 429 Too Many Requests.
   ```

4. **Live Chat Flow Verification**:
   Send an authenticated chat query and confirm response grounding and session update:
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Authorization: Bearer <VALID_JWT>" \
     -H "Content-Type: application/json" \
     -d '{"message":"What are my rights as a tenant in India?"}'
   # Expected: HTTP 200 with grounded Indian law response.
   ```

---

## 5. Rollback Plan

If you must revert the release:
1. Re-deploy the previous container tag.
2. In the Supabase SQL Editor, execute the commented rollback block at the bottom of [`migrations/001_jurix.sql`](file:///c:/Users/tstus/Desktop/Projects/JurixAI_core/migrations/001_jurix.sql).
