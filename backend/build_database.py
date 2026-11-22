import os
import re
import shutil
import time
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGAL_DATA_DIR = os.path.join(BASE_DIR, "legal_data")
BACKEND_DB_DIR = os.path.join(BASE_DIR, "Jurixai_db")

def clean_pdf_text(text: str) -> str:
    """Normalize messy whitespace and repeated blank lines from PDF extraction."""
    if not text:
        return ""
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    # Collapse multiple consecutive blank lines to a single blank line
    cleaned_lines = []
    empty_streak = 0
    for line in lines:
        if not line:
            empty_streak += 1
            if empty_streak <= 1:
                cleaned_lines.append("")
        else:
            empty_streak = 0
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    # Collapse multiple spaces and horizontal tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def get_directory_size_mb(path: str) -> float:
    """Calculate total size of directory in MB."""
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 * 1024)

def main():
    print("=" * 80)
    print("JurixAI Legal PDF Audit & ChromaDB Re-Ingestion")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. PDF Text-Quality Audit
    # -------------------------------------------------------------
    print(f"\n[Step 1] Auditing PDF files in: {LEGAL_DATA_DIR}")
    pdf_files = sorted([f for f in os.listdir(LEGAL_DATA_DIR) if f.lower().endswith(".pdf")])
    print(f"Total PDF files found: {len(pdf_files)}\n")

    header = f"{'Filename':<52} | {'Size (KB)':>10} | {'Pages':>5} | {'Chars':>9} | {'Avg Ch/Pg':>10} | {'Status'}"
    print(header)
    print("-" * len(header))

    valid_files = []
    flagged_files = []
    file_stats = {}

    for fname in pdf_files:
        fpath = os.path.join(LEGAL_DATA_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        try:
            reader = pypdf.PdfReader(fpath)
            num_pages = len(reader.pages)
            total_chars = 0
            for page in reader.pages:
                extracted = page.extract_text() or ""
                total_chars += len(extracted.strip())
            avg_chars = total_chars / max(1, num_pages)

            if avg_chars < 100:
                status = "FLAGGED (Scanned/Empty)"
                flagged_files.append((fname, size_kb, num_pages, total_chars, avg_chars))
            else:
                status = "OK"
                valid_files.append(fname)
                file_stats[fname] = {
                    "size_kb": size_kb,
                    "pages": num_pages,
                    "chars": total_chars,
                    "avg_chars": avg_chars,
                }

            print(f"{fname[:52]:<52} | {size_kb:>10.1f} | {num_pages:>5} | {total_chars:>9} | {avg_chars:>10.1f} | {status}")
        except Exception as e:
            print(f"{fname[:52]:<52} | {size_kb:>10.1f} | ERROR: {e}")
            flagged_files.append((fname, size_kb, 0, 0, 0))

    print("\n" + "=" * 80)
    print(f"Audit Summary: {len(valid_files)} valid text PDFs, {len(flagged_files)} flagged scanned/empty PDFs.")
    if flagged_files:
        print("Flagged files excluded from ingestion:")
        for fname, size_kb, pages, chars, avg_c in flagged_files:
            print(f"  - {fname} ({size_kb:.1f} KB, {pages} pages, {chars} chars extracted)")
    print("=" * 80)

    # -------------------------------------------------------------
    # 2. Extract & Chunk Valid PDFs
    # -------------------------------------------------------------
    print("\n[Step 2] Extracting text and creating optimized chunks (750 chars, 120 overlap)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    file_chunk_counts = {}

    for fname in valid_files:
        fpath = os.path.join(LEGAL_DATA_DIR, fname)
        reader = pypdf.PdfReader(fpath)
        file_docs = []

        for page_idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = clean_pdf_text(raw_text)
            if len(cleaned) >= 25:  # skip trivial whitespace
                doc = Document(
                    page_content=cleaned,
                    metadata={"source": fname, "page": page_idx + 1},
                )
                file_docs.append(doc)

        file_chunks = text_splitter.split_documents(file_docs)
        # Filter out any tiny fragment chunks
        file_chunks = [c for c in file_chunks if len(c.page_content.strip()) >= 30]

        file_chunk_counts[fname] = len(file_chunks)
        all_chunks.extend(file_chunks)
        print(f"  Processed {fname[:45]:<45} -> {len(file_chunks):>5} chunks")

    print(f"\nTotal chunks prepared across {len(valid_files)} files: {len(all_chunks):,}")

    # -------------------------------------------------------------
    # 3. Clean Rebuild (No Duplicates)
    # -------------------------------------------------------------
    print(f"\n[Step 3] Wiping existing canonical ChromaDB directory ({BACKEND_DB_DIR}) to guarantee zero duplicate vectors...")
    if os.path.exists(BACKEND_DB_DIR):
        print(f"  Deleting old database directory: {BACKEND_DB_DIR}")
        shutil.rmtree(BACKEND_DB_DIR, ignore_errors=True)

    # Resolve local embedding model
    container_path = "/app/local_model_cache/all-MiniLM-L6-v2"
    local_path = os.path.join(BASE_DIR, "local_model_cache", "all-MiniLM-L6-v2")
    if os.path.exists(container_path):
        model_path = container_path
    elif os.path.exists(local_path):
        model_path = local_path
    else:
        model_path = "all-MiniLM-L6-v2"

    print(f"Loading embedding model from: {model_path}...")
    t0_embed = time.time()
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    print(f"Model loaded in {time.time() - t0_embed:.2f}s")

    print(f"Building persistent ChromaDB in: {BACKEND_DB_DIR}...")
    t0_index = time.time()
    db = Chroma(persist_directory=BACKEND_DB_DIR, embedding_function=embeddings)

    BATCH_SIZE = 500
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        db.add_documents(batch)
        print(f"  Indexed chunks {i + 1} to {min(i + BATCH_SIZE, len(all_chunks))} of {len(all_chunks)}...", flush=True)

    index_time = time.time() - t0_index
    print(f"ChromaDB indexing completed in {index_time:.2f}s ({len(all_chunks) / max(1, index_time):.1f} chunks/sec)!")

    # Canonical database stored exclusively in backend/Jurixai_db
    print(f"Canonical ChromaDB persistent at: {BACKEND_DB_DIR}")

    # -------------------------------------------------------------
    # 4. Post-Ingestion Report
    # -------------------------------------------------------------
    backend_db_size_mb = get_directory_size_mb(BACKEND_DB_DIR)
    indexed_doc_count = db._collection.count()

    print("\n" + "=" * 80)
    print("POST-INGESTION REPORT")
    print("=" * 80)
    print(f"{'Filename':<55} | {'Chunks':>8}")
    print("-" * 66)
    for fname in valid_files:
        print(f"{fname[:55]:<55} | {file_chunk_counts[fname]:>8}")
    print("-" * 66)
    print(f"{'TOTAL CHUNKS INDEXED':<55} | {indexed_doc_count:>8,}")
    print(f"{'ChromaDB Directory Size on Disk':<55} | {backend_db_size_mb:>7.2f} MB")
    print("=" * 80)

    # -------------------------------------------------------------
    # 5. Retrieval Sanity Tests (k=3)
    # -------------------------------------------------------------
    print("\n[Step 5] Running Retrieval Sanity Tests (k=3)...")
    test_queries = [
        "IPC 420 equivalent section in BNS",
        "Police refusing to register FIR under BNSS",
        "Cheque bounce legal notice Section 138 timeline",
        "Fundamental right against illegal arrest Constitution Article 22",
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print("\n" + "-" * 80)
        print(f"TEST QUERY {q_idx}: \"{query}\"")
        print("-" * 80)
        results = db.similarity_search(query, k=3)
        for r_idx, doc in enumerate(results, 1):
            src = doc.metadata.get("source", "Unknown")
            pg = doc.metadata.get("page", "Unknown")
            snippet = " ".join(doc.page_content.split())
            if len(snippet) > 180:
                snippet = snippet[:180] + "..."
            print(f"  [Match {r_idx}] Source: {src} (Page {pg})")
            print(f"            Snippet: {snippet}")

    print("\n" + "=" * 80)
    print("Re-ingestion & retrieval verification complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
