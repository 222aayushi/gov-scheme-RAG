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


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def _get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = faiss.read_index(str(INDEX_PATH))
    return _INDEX


def _get_chunks():
    global _CHUNKS
    if _CHUNKS is None:
        with open(CHUNKS_PATH, "rb") as f:
            _CHUNKS = pickle.load(f)
    return _CHUNKS


def retrieve(query: str, k: int = 3):
    """Return the top matching chunks and distances for a query."""
    model = _get_model()
    index = _get_index()
    chunks = _get_chunks()

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), k)

    retrieved_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    return retrieved_chunks, distances
        