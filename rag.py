import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# PATH CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "vector.index")
DOC_PATH = os.path.join(BASE_DIR, "documents.npy")

# =========================
# SAFETY CHECKS
# =========================
if not os.path.exists(INDEX_PATH):
    raise RuntimeError("FAISS index not found. Run ingest.py first.")

if not os.path.exists(DOC_PATH):
    raise RuntimeError("documents.npy not found. Run ingest.py first.")

# =========================
# LOAD EMBEDDING MODEL
# =========================
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.environ.get("HF_HOME", None)
)

# =========================
# LOAD VECTOR STORE
# =========================
index = faiss.read_index(INDEX_PATH)
documents = np.load(DOC_PATH, allow_pickle=True)

# =========================
# RAG ANSWER FUNCTION
# =========================
def generate_answer(query: str) -> str:
    if not query or not query.strip():
        return "Please ask a valid health-related question."

    # Embed query
    query_vec = embedder.encode([query]).astype("float32")

    # Retrieve
    top_k = 6
    distances, indices = index.search(query_vec, top_k)

    # -------------------------
    # STEP 1: Deduplicate
    # -------------------------
    seen = set()
    unique_chunks = []

    for idx in indices[0]:
        if idx < len(documents):
            chunk = documents[idx].strip()
            if chunk and chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)

    if not unique_chunks:
        return (
            "I could not find relevant information in the knowledge base. "
            "Please consult a medical professional."
        )

    # -------------------------
    # STEP 2: Focused synthesis
    # -------------------------
    answer = (
        f"Based on healthcare information related to your query:\n\n"
    )

    # Only keep top 3 most relevant chunks
    for chunk in unique_chunks[:3]:
        answer += f"- {chunk}\n\n"

    # -------------------------
    # STEP 3: Clean closing
    # -------------------------
    answer += (
        "\nGeneral guidance:\n"
        "- Stay hydrated\n"
        "- Take adequate rest\n"
        "- Monitor symptoms closely\n"
        "- Seek medical attention if symptoms worsen\n\n"
        "⚠️ This information is educational and not a medical diagnosis."
    )

    return answer