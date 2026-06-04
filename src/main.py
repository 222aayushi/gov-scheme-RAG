from document_loader import all_docs
from chunking import chunk_documents
from embeddings import create_embeddings
from vector_store import create_faiss_index
from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "chunks.pkl"

print("Creating chunks...")

chunks = chunk_documents(all_docs)

if not chunks:
    print("No chunks created. Add PDFs under data doc/ and rerun.")
    raise SystemExit(0)

print(f"Total Chunks: {len(chunks)}")

# Save chunks for retrieval later
with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("Generating embeddings...")

embeddings = create_embeddings(chunks)

print("Creating FAISS index...")

create_faiss_index(embeddings)

print("FAISS Index Created Successfully!")
print("chunks.pkl Saved Successfully!")
