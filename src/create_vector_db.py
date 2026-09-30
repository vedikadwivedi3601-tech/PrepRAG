import os
import re
import glob
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_folder = "data/raw"
pdf_paths = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))

# 1. Load embedding model (must match search.py)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Text cleaning + splitting
def clean_text(text):
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'[ \t]+', ' ', text)      # collapse spaces, keep newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# 3. Build chunks WITH source and page
chunks = []
for pdf_path in pdf_paths:
    reader = PdfReader(pdf_path)
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue
        for piece in splitter.split_text(text):
            chunks.append({
                "source": os.path.basename(pdf_path),
                "page": page_number,
                "text": piece,
            })

print("Total chunks:", len(chunks))

# 4. Embed
print("Creating embeddings...")
embeddings = model.encode(
    [c["text"] for c in chunks],
    show_progress_bar=True,
)

# 5. Rebuild the collection from scratch
client = chromadb.PersistentClient(path="chroma_db")

try:
    client.delete_collection("ml_knowledge")
except Exception:
    pass

collection = client.create_collection(
    name="ml_knowledge",
    metadata={"hnsw:space": "cosine"},
)

# Add in batches (Chroma limits how many items one call can take)
batch_size = 500
for start in range(0, len(chunks), batch_size):
    end = start + batch_size
    batch = chunks[start:end]
    collection.add(
        ids=[f"chunk_{i}" for i in range(start, start + len(batch))],
        documents=[c["text"] for c in batch],
        embeddings=embeddings[start:end].tolist(),
        metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
    )

print(f"\nStored {collection.count()} chunks in ChromaDB.")
