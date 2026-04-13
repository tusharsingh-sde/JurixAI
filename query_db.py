from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Wahi same model wapas load karo (taaki AI ko bhasha samajh aaye)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Apna banaya hua database connect karo
print("Connecting to LexAI Database...\n")
db = Chroma(persist_directory="./lexai_db", embedding_function=embeddings)

# 3. Apna Sawaal (Query) likho - Tu isko change karke kuch bhi puch sakta hai
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