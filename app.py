import time

import chromadb
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ============================================================
# SETTINGS (tune these)
# ============================================================

TOP_K = 5                  # fewer chunks = shorter prompt = faster, cheaper answers
MAX_DISTANCE = 1.1         # if the best match is farther than this, treat as out of scope
SNIPPET_CHARS = 600        # how much of each retrieved chunk to show in Sources
MAX_NEW_TOKENS = 300
MODEL = "llama-3.3-70b-versatile"   # or "llama-3.1-8b-instant" for faster, lighter answers

st.set_page_config(page_title="RAG AI Interviewer", page_icon="🤖")


# ============================================================
# 1. LOAD EMBEDDING MODEL ONLY ONCE
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ============================================================
# 2. GROQ CLIENT (key comes from Streamlit secrets, never from code)
# ============================================================

@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets[GROQ_API_KEY])


client = get_client()


# ============================================================
# 3. CONNECT TO CHROMADB
# ============================================================

@st.cache_resource
def load_chromadb():
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    return chroma_client.get_collection(name="ml_knowledge")


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

SYSTEM_PROMPT = (
    "You are an AI/ML knowledge assistant. Answer the user's question using ONLY "
    "the information in the context provided. Do not use outside knowledge. "
    "If the answer cannot be found in the context, say exactly: "
    "\"I could not find this information in the provided knowledge base.\""
)

if query:

    # --------------------------------------------------------
    # EMBEDDING + SEARCH
    # --------------------------------------------------------

    start = time.time()
    query_embedding = embedding_model.encode([query])
    embedding_time = time.time() - start

    start = time.time()
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chroma_time = time.time() - start

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    generation_time = 0.0

    # --------------------------------------------------------
    # OUT-OF-SCOPE CHECK (skips the LLM call completely)
    # --------------------------------------------------------

    if not documents or min(distances) > MAX_DISTANCE:
        st.subheader("🤖 AI Answer")
        st.write(
            "I could not find this information in the provided knowledge base. "
            "Try asking about a specific ML or deep learning topic, such as "
            "overfitting, the bias-variance tradeoff, or gradient descent."
        )

    else:
        # ----------------------------------------------------
        # BUILD PROMPT + STREAM THE ANSWER FROM GROQ
        # ----------------------------------------------------

        context = "\n\n".join(documents)
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"

        def token_stream():
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=MAX_NEW_TOKENS,
                temperature=0,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        st.subheader("🤖 AI Answer")
        start = time.time()
        try:
            st.write_stream(token_stream())
        except Exception as e:
            st.error(
                "The answer service is busy or unavailable right now "
                "(you may have hit the free rate limit). Please try again in a minute."
            )
            st.caption(f"Details: {e}")
        generation_time = time.time() - start

    # --------------------------------------------------------
    # SOURCES IN A COLLAPSED SECTION
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
    # DEBUG INFO HIDDEN BY DEFAULT
    # --------------------------------------------------------

    with st.expander("⚙️ Debug info"):
        st.write(f"Embedding time: {embedding_time:.2f} s")
        st.write(f"ChromaDB search time: {chroma_time:.2f} s")
        st.write(f"Generation time: {generation_time:.2f} s")
        st.write("Distances: " + ", ".join(f"{d:.3f}" for d in distances))