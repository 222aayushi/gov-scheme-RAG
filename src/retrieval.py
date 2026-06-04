import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "scheme_index.faiss"
CHUNKS_PATH = PROJECT_ROOT / "chunks.pkl"

# Load model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
index = faiss.read_index(str(INDEX_PATH))

# Load chunks
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

while True:

    query = input("\nAsk a question (or type exit): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        3
    )

    print("\nTop Retrieved Chunks:\n")

    for i, idx in enumerate(indices[0]):
        print(f"\nChunk {i+1}:")
        print("-" * 50)
        print(chunks[idx].page_content[:1000])
        