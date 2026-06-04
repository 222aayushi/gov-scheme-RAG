import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
index = faiss.read_index("scheme_index.faiss")

# Load chunks
with open("chunks.pkl", "rb") as f:
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
        