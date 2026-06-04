import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai

# Gemini API
client = genai.Client(
    api_key="AQ.Ab8RN6LRDRfmZwBSCze7DfLm5HDjXGA9mPb8rYJyiOKKfknVgA"
)

# Embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
index = faiss.read_index("scheme_index.faiss")

# Load chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    query_embedding = model.encode([question])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        3
    )

    context = ""

    for idx in indices[0]:
        context += chunks[idx].page_content + "\n\n"

    prompt = f"""
    Answer ONLY from the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nAnswer:")
    print(response.text)