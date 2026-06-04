from document_loader import all_docs
from chunking import chunk_documents
from embeddings import create_embeddings
from vector_store import create_faiss_index
import pickle

print("Creating chunks...")

chunks = chunk_documents(all_docs)

print(f"Total Chunks: {len(chunks)}")

# Save chunks for retrieval later
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Generating embeddings...")

embeddings = create_embeddings(chunks)

print("Creating FAISS index...")

create_faiss_index(embeddings)

print("FAISS Index Created Successfully!")
print("chunks.pkl Saved Successfully!")
