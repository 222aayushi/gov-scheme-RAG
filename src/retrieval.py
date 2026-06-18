import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "scheme_index.faiss"
CHUNKS_PATH = PROJECT_ROOT / "chunks.pkl"

_MODEL = None
_INDEX = None
_CHUNKS = None


# -----------------------------
# Load Model
# -----------------------------
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


# -----------------------------
# Load FAISS Index
# -----------------------------
def _get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = faiss.read_index(str(INDEX_PATH))
    return _INDEX


# -----------------------------
# Load Chunks
# -----------------------------
def _get_chunks():
    global _CHUNKS
    if _CHUNKS is None:
        with open(CHUNKS_PATH, "rb") as f:
            _CHUNKS = pickle.load(f)
    return _CHUNKS


# -----------------------------
# MAIN RETRIEVAL FUNCTION
# -----------------------------
def retrieve(query: str, k: int = 5):

    model = _get_model()
    index = _get_index()
    chunks = _get_chunks()

    # Encode + normalize (cosine similarity ready)
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype="float32")

    # FAISS search
    distances, indices = index.search(query_embedding, k)

    # Get top chunks safely
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return results, distances

    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])   # ✅ ONLY Document

    return results