import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
import ollama   # Local LLaMA (Ollama)

# =========================
# LOAD FAISS + CHUNKS
# =========================
index = faiss.read_index("scheme_index.faiss")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# =========================
# EMBEDDING MODEL
# =========================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("\n🌾 Farmer RAG Chatbot (Profile-Aware System)")
print("Type 'exit' to stop\n")

# =========================
# FARMER PROFILE INPUT
# =========================
print("👉 Enter Farmer Profile Details:\n")

profile = {
    "name": input("Name: "),
    "state": input("State: "),
    "crop": input("Crop: "),
    "land_size": input("Land Size (in acres): "),
    "soil_type": input("Soil Type (black/red/alluvial/etc): "),
    "income": input("Annual Income (₹): ")
}

print("\n✅ Profile Saved Successfully!\n")

# =========================
# CHAT LOOP
# =========================
while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        print("\n👋 Exiting chatbot...")
        break

    # =========================
    # STEP 1: EMBEDDING
    # =========================
    query_embedding = model.encode([question])

    # =========================
    # STEP 2: FAISS SEARCH
    # =========================
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        5
    )

    # =========================
    # STEP 3: BUILD CONTEXT
    # =========================
    context = ""
    for idx in indices[0]:
        context += chunks[idx].page_content + "\n\n"

    # =========================
    # STEP 4: PERSONALIZED PROMPT
    # =========================
    prompt = f"""
You are an expert agricultural government scheme assistant.

Use ONLY:
1. Government scheme context below
2. Farmer profile details

RULES:
- Suggest only eligible schemes
- Consider soil type, income, land size
- If not eligible, clearly say so
- Keep answer simple for farmers

Farmer Profile:
Name: {profile['name']}
State: {profile['state']}
Crop: {profile['crop']}
Land Size: {profile['land_size']} acres
Soil Type: {profile['soil_type']}
Annual Income: ₹{profile['income']}

Context (Government Schemes):
{context}

Question:
{question}

Answer:
"""

    # =========================
    # STEP 5: LLaMA RESPONSE
    # =========================
    try:
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        print("\n📌 Personalized Answer:\n")
        print(response["message"]["content"])

    except Exception as e:
        print("\n❌ Error:")
        print(str(e))