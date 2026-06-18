from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# SBERT Model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_answer(query, retrieved_chunks, top_k_sentences=3):

    if isinstance(retrieved_chunks, str):
        retrieved_chunks = [retrieved_chunks]

    # ==================================================
    # HANDLE CSV SCHEME RECORDS DIRECTLY
    # ==================================================

    for chunk in retrieved_chunks:

        if "scheme_name:" in chunk.lower():

            scheme_name = ""
            details = ""

            for line in chunk.split("\n"):

                line = line.strip()

                if line.lower().startswith("scheme_name:"):
                    scheme_name = line.split(":", 1)[1].strip()

                elif line.lower().startswith("details:"):
                    details = line.split(":", 1)[1].strip()

            if scheme_name and details:

                return (
                    f"✅ Scheme: {scheme_name}<br><br>"
                    f"✅ {details}"
                )

    # ==================================================
    # NORMAL TEXT PROCESSING
    # ==================================================

    all_sentences = []

    for chunk in retrieved_chunks:

        if not chunk:
            continue

        chunk = str(chunk)

        chunk = chunk.replace("scheme_name:", "Scheme Name: ")
        chunk = chunk.replace("details:", "Details: ")
        chunk = chunk.replace("slug:", "")

        sentences = sent_tokenize(chunk)

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) > 20:
                all_sentences.append(sentence)

    if not all_sentences:
        return "No relevant information found in the scheme database."

    # ==================================================
    # EMBEDDINGS (FIXED)
    # ==================================================

    query_embedding = np.array(
        model.encode([query], normalize_embeddings=True),
        dtype="float32"
    )

    sentence_embeddings = np.array(
        model.encode(all_sentences, normalize_embeddings=True),
        dtype="float32"
    )

    similarity_scores = cosine_similarity(
        query_embedding,
        sentence_embeddings
    )[0]

    best_score = max(similarity_scores)

    print(f"\nQuery: {query}")
    print(f"Best Similarity Score: {best_score}")

    # ==================================================
    # EXTRA SAFETY CHECK (RECOMMENDED)
    # ==================================================

    best_sentence = all_sentences[np.argmax(similarity_scores)]

    query_words = set(query.lower().split())
    sentence_words = set(best_sentence.lower().split())

    overlap = len(query_words & sentence_words)

    # ==================================================
    # REJECT IRRELEVANT QUESTIONS
    # ==================================================

    if best_score < 0.55 or overlap == 0:
        return "❌ No relevant information found in the scheme database."

    # ==================================================
    # GET TOP MATCHING SENTENCES
    # ==================================================

    top_indices = np.argsort(similarity_scores)[::-1][:top_k_sentences]

    selected_sentences = []

    for idx in sorted(top_indices):

        sentence = all_sentences[idx]

        if sentence not in selected_sentences:
            selected_sentences.append(sentence)

    return " ".join(selected_sentences)


# ==================================================
# RECOMMENDATION SYSTEM (UNCHANGED)
# ==================================================

def recommend(profile: dict):

    crop = profile.get("crop", "").strip().lower()

    if crop == "cotton":
        return ["PMFBY", "PM-KISAN", "KCC"]

    elif crop == "wheat":
        return ["PM-KISAN", "RKVY", "KCC"]

    elif crop == "sugarcane":
        return ["PM-KISAN", "PMFBY", "KCC"]

    else:
        return ["PM-KISAN", "KCC"]