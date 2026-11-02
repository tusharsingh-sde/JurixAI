import os
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load same model (ao that Ai can understand the language of the chunks and the query)
try:
    model = SentenceTransformer('/app/local_model_cache/all-MiniLM-L6-v2')
    embeddings = HuggingFaceEmbeddings(model_name='/app/local_model_cache/all-MiniLM-L6-v2')
except Exception:
    local_path = os.path.join(os.path.dirname(__file__), "local_model_cache", "all-MiniLM-L6-v2")
    model_path = local_path if os.path.exists(local_path) else "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_path)
    embeddings = HuggingFaceEmbeddings(model_name=model_path)

# 2. Connect to the same ChromaDB where we have saved all the chunks (Legal knowledge)
print("Connecting to JurixAI Database...\n")
db_dir = os.path.join(os.path.dirname(__file__), "Jurixai_db") if os.path.exists(os.path.join(os.path.dirname(__file__), "Jurixai_db")) else "./Jurixai_db"
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

# 3. Write your question (Query) here (For your demo, you can later change this to input("Enter your legal query: "))
query = "What is the punishment for online fraud or cyber crime?" 
print(f"User Query: {query}\n")

# 4. Database se top 3 sabse relevant answers nikal ke lao
print("Searching for answers in the law books...\n")
results = db.similarity_search(query, k=3)

# 5. Results print karo
for i, doc in enumerate(results):
    print(f"--- MATCH {i+1} ---")
    print(doc.page_content)
    print("-" * 50 + "\n")