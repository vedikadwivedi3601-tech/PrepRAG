import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="ml_knowledge")

# 1. Pull ALL chunks and find the one with the classification definition
all_data = collection.get(include=["documents", "metadatas"])

print("Searching for the target sentence in stored chunks...\n")
found = False
for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
    if "discrete category" in doc.lower():
        found = True
        print(f"FOUND on page {meta['page']}:")
        print(doc)
        print("-" * 60)

if not found:
    print("NOT FOUND — the sentence doesn't exist in any stored chunk at all.")