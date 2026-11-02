from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import os

# 1. Load ALL PDFs from the folder
print("Loading all Legal PDFs from folder...")
loader = PyPDFDirectoryLoader("./legal_data")
pages = loader.load()
print(f"Total pages loaded from all PDFs: {len(pages)}")

# 2. Chunking
print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)
print(f"Total chunks created: {len(chunks)}")

# 3. Embeddings (Local AI brain)
print("Loading Embedding Model...")
try:
    model = SentenceTransformer('/app/local_model_cache/all-MiniLM-L6-v2')
    embeddings = HuggingFaceEmbeddings(model_name='/app/local_model_cache/all-MiniLM-L6-v2')
except Exception:
    local_path = os.path.join(os.path.dirname(__file__), "local_model_cache", "all-MiniLM-L6-v2")
    model_path = local_path if os.path.exists(local_path) else "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_path)
    embeddings = HuggingFaceEmbeddings(model_name=model_path)

# 4. Save to ChromaDB permanently
print("Creating Vector Database... (Isme thoda time lag sakta hai)")
db = Chroma.from_documents(chunks, embeddings, persist_directory="./Jurixai_db")
print("Vector Database created successfully!")