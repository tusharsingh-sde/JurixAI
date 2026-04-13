from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Save to ChromaDB permanently
print("Creating Vector Database... (Isme thoda time lag sakta hai)")
db = Chroma.from_documents(chunks, embeddings, persist_directory="./lexai_db")
py