import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
import time


# ============================================================
# 1. LOAD EMBEDDING MODEL ONLY ONCE
# ============================================================

@st.cache_resource
def load_embedding_model():

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


embedding_model = load_embedding_model()


# ============================================================
# 2. LOAD QWEN ONLY ONCE
# ============================================================

@st.cache_resource
def load_qwen():

    print("Loading Qwen model...")

    llm_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype="auto"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    print("Qwen loaded successfully")
    print("Using device:", device)

    return tokenizer, model


tokenizer, llm = load_qwen()


# ============================================================
# 3. CONNECT TO CHROMADB
# ============================================================

@st.cache_resource
def load_chromadb():

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = client.get_collection(
        name="ml_knowledge"
    )

    return collection


collection = load_chromadb()


# ============================================================
# 4. STREAMLIT UI
# ============================================================

st.title("🤖 RAG AI Interviewer")

query = st.text_input(
    "Ask your AI/ML question:"
)


# ============================================================
# 5. PROCESS QUESTION
# ============================================================

if query:

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    start = time.time()

    query_embedding = embedding_model.encode(
        [query]
    )

    embedding_time = time.time() - start

    st.write(
        f"Embedding time: {embedding_time:.2f} seconds"
    )


    # --------------------------------------------------------
    # CHROMADB SEARCH
    # --------------------------------------------------------

    start = time.time()

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    chroma_time = time.time() - start

    st.write(
        f"ChromaDB search time: {chroma_time:.2f} seconds"
    )


    # --------------------------------------------------------
    # DISPLAY RETRIEVED DOCUMENTS
    # --------------------------------------------------------

    st.subheader("📚 Retrieved Knowledge")

    for i, document in enumerate(
        results["documents"][0]
    ):

        distance = results["distances"][0][i]

        metadata = results["metadatas"][0][i]

        st.write(
            f"**Result {i+1} | Distance: {distance:.3f}**"
        )

        st.write(
            f"Source: {metadata.get('source', 'Unknown')}"
        )

        st.write(
            f"Page: {metadata.get('page', 'Unknown')}"
        )

        st.write(document)


    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
        results["documents"][0]
    )


    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # QWEN GENERATION
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(llm.device)


    start = time.time()

    outputs = llm.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )

    generation_time = time.time() - start


    response = outputs[
        0
    ][
        inputs.input_ids.shape[1]:
    ]


    answer = tokenizer.decode(
        response,
        skip_special_tokens=True
    )


    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    st.subheader("🤖 AI Answer")

    st.write(answer)

    st.write(
        f"Generation time: {generation_time:.2f} seconds"
    )
