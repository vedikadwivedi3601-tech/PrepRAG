import os
import glob
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

pdf_folder = "data/raw"
pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))

print("PDFs found:", pdf_paths)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)

# Created ONCE, outside the loop
all_chunks = []

# 1 & 2. Load each PDF and split into chunks
for pdf_path in pdf_paths:
    file_name = os.path.basename(pdf_path)
    print(f"Processing {file_name}...")

    reader = PdfReader(pdf_path)
    count = 0

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        for chunk in text_splitter.split_text(text):
            all_chunks.append({
                "id": f"{file_name}_{page_number + 1}_{count}",
                "page": page_number + 1,
                "source": file_name,
                "text": chunk,
            })
            count += 1

    print(f"  {file_name}: {count} chunks")

print("Total chunks (all PDFs):", len(all_chunks))

# 3. Embed all chunks once
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)
print("Embedding shape:", embeddings.shape)

# 4. Store in ChromaDB (fresh collection)
client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("ml_knowledge")
except Exception:
    pass
collection = client.get_or_create_collection(name="ml_knowledge")

collection.add(
    ids=[c["id"] for c in all_chunks],
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[{"source": c["source"], "page": c["page"]} for c in all_chunks],
)

print(f"Stored {len(all_chunks)} chunks in ChromaDB.")
