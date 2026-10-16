import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
import time
import os
import glob
from pypdf import PdfReader

# ============================================================
# SETTINGS (tune these)
# ============================================================

TOP_K = 3                  # fewer chunks = shorter prompt = faster generation
MAX_DISTANCE = 1.1         # if the best match is farther than this, treat as out of scope
SNIPPET_CHARS = 300        # how much of each retrieved chunk to show in Sources
MAX_NEW_TOKENS = 150

st.set_page_config(page_title="RAG AI Interviewer", page_icon="🤖")


# ============================================================
# 1. LOAD EMBEDDING MODEL ONLY ONCE
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ============================================================
# 2. LOAD QWEN ONLY ONCE
# ============================================================

@st.cache_resource
def load_qwen():
<<<<<<< HEAD
    llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
=======

    print("Loading Qwen model...")

    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
>>>>>>> 8af866b5af9e8149345e07955bd8aef3e086912e

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype="auto"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return tokenizer, model


tokenizer, llm = load_qwen()


# ============================================================
# 3. CONNECT TO CHROMADB
# ============================================================
@st.cache_resource
def load_chromadb():
<<<<<<< HEAD
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection(name="ml_knowledge")
=======

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(name="ml_knowledge")

    # Build the database on first run (Streamlit Cloud starts empty)
    if collection.count() == 0:

        print("Building ChromaDB from data/raw ...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        raw_dir = os.path.join(base_dir, "data", "raw")

        documents, metadatas, ids = [], [], []
        chunk_size, overlap = 800, 100

        def add_chunks(text, source, page):
            start = 0
            while start < len(text):
                chunk = text[start:start + chunk_size].strip()
                if chunk:
                    documents.append(chunk)
                    metadatas.append({"source": source, "page": page})
                    ids.append(f"{source}_{page}_{len(ids)}")
                if start + chunk_size >= len(text):
                    break
                start += chunk_size - overlap

        for path in sorted(glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)):
            name = os.path.basename(path)

            if path.lower().endswith(".pdf"):
                reader = PdfReader(path)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    add_chunks(text, name, page_num)

            elif path.lower().endswith((".txt", ".md")):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    add_chunks(f.read(), name, 1)

        if not documents:
            st.error("No documents found in data/raw. Check that your files are in the GitHub repo.")
            st.stop()

        # Embed and store in batches
        batch = 500
        for i in range(0, len(documents), batch):
            embeddings = embedding_model.encode(
                documents[i:i + batch], batch_size=32
            ).tolist()
            collection.add(
                documents=documents[i:i + batch],
                embeddings=embeddings,
                metadatas=metadatas[i:i + batch],
                ids=ids[i:i + batch],
            )

        print(f"Added {len(documents)} chunks")

    return collection
>>>>>>> 8af866b5af9e8149345e07955bd8aef3e086912e


collection = load_chromadb()



# ============================================================
# 4. STREAMLIT UI
# ============================================================

st.title("🤖 RAG AI Interviewer")
st.caption(
    "Ask machine learning and deep learning interview questions. "
    "Answers are generated from source pdfs."
)

query = st.text_input("Ask your AI/ML question:")


# ============================================================
# 5. PROCESS QUESTION
# ============================================================

if query:

    # --------------------------------------------------------
    # EMBEDDING + SEARCH (no output yet, just store timings)
    # --------------------------------------------------------

    start = time.time()
    query_embedding = embedding_model.encode([query])
    embedding_time = time.time() - start

    start = time.time()
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distance"
            8af866b5af9e8149345e07955bd8aef3e086912e
    )
    chroma_time = time.time() - start

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # --------------------------------------------------------
    # OUT-OF-SCOPE CHECK (skips the slow LLM call)
    # --------------------------------------------------------

    generation_time = 0.0
    answer = None

    if not documents or min(distances) > MAX_DISTANCE:
        answer = (
            "I could not find this information in the provided knowledge base. "
            "Try asking about a specific ML or deep learning topic, such as "
            "overfitting, the bias-variance tradeoff, or gradient descent."
        )

    else:
        # ----------------------------------------------------
        # CREATE CONTEXT + PROMPT
        # ----------------------------------------------------

        context = "\n\n".join(documents)

        prompt = f"""
You are an AI/ML knowledge assistant.

Answer the user's question using ONLY the information
provided in the context below.

Do not use outside knowledge.

If the answer cannot be found in the context, say:

"I could not find this information in the provided knowledge base."

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

User Question:

{query}

Answer:
"""

        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([text], return_tensors="pt").to(llm.device)

        # ----------------------------------------------------
        # QWEN GENERATION (with spinner)
        # ----------------------------------------------------

        with st.spinner("Generating answer..."):
            start = time.time()
            outputs = llm.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )
            generation_time = time.time() - start

        response = outputs[0][inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(response, skip_special_tokens=True)

    # --------------------------------------------------------
    # 1) ANSWER FIRST
    # --------------------------------------------------------

    st.subheader("🤖 AI Answer")
    st.write(answer)

    # --------------------------------------------------------
    # 2) SOURCES IN A COLLAPSED SECTION
    # --------------------------------------------------------

    with st.expander("📚 Sources used"):
        for i, document in enumerate(documents):
            source = metadatas[i].get("source", "Unknown")
            page = metadatas[i].get("page", "Unknown")

            snippet = document.strip()
            if len(snippet) > SNIPPET_CHARS:
                snippet = snippet[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."

            st.markdown(f"**{i+1}. {source}, page {page}**")
            st.caption(snippet)

    # --------------------------------------------------------
    # 3) DEBUG INFO HIDDEN BY DEFAULT
    # --------------------------------------------------------

    with st.expander("⚙️ Debug info"):
        st.write(f"Embedding time: {embedding_time:.2f} s")
        st.write(f"ChromaDB search time: {chroma_time:.2f} s")
        st.write(f"Generation time: {generation_time:.2f} s")
        st.write(
            "Distances: " + ", ".join(f"{d:.3f}" for d in distances)
        )
