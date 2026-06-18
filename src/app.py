from pathlib import Path
import math
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from retrieval import retrieve
from rag_chatbot import generate_answer, recommend

app = Flask(__name__)
CORS(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------
# Query classifier
# -----------------------------
def _is_scheme_recommendation_query(query: str) -> bool:
    lowered = (query or "").lower()

    return any(
        phrase in lowered
        for phrase in (
            "what schemes",
            "which schemes",
            "available for",
            "schemes for",
            "recommend",
            "suggest",
        )
    )


# -----------------------------
# Home route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return send_from_directory(PROJECT_ROOT, "index.html")


# -----------------------------
# MAIN API
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    try:
        payload = request.get_json(force=True)

        query = payload.get("query")
        profile = payload.get("profile", {})

        if not query:
            return jsonify({"error": "Missing 'query' in request"}), 400

        # -----------------------------
        # FIX 1: correct retrieval return
        # -----------------------------
        chunks, distances = retrieve(query, k=5)

        print("\n" + "=" * 70)
        print("QUESTION:", query)

        # -----------------------------
        # Safe chunk printing
        # -----------------------------
        for i, chunk in enumerate(chunks):
            print(f"\nCHUNK {i + 1}")
            print("-" * 40)

            if isinstance(chunk, dict):
                text = chunk.get("chunk", "")
            elif hasattr(chunk, "page_content"):
                text = chunk.page_content
            else:
                text = str(chunk)

            print(text[:500])

        # -----------------------------
        # Build context safely
        # -----------------------------
        context = "\n\n".join(
            [
                (
                    chunk.get("chunk") if isinstance(chunk, dict)
                    else chunk.page_content if hasattr(chunk, "page_content")
                    else str(chunk)
                ) or ""
                for chunk in chunks
            ]
        )

        # -----------------------------
        # Recommendations
        # -----------------------------
        recommendations = recommend(profile) if profile else []

        # -----------------------------
        # Answer generation
        # -----------------------------
        if profile and _is_scheme_recommendation_query(query):

            crop = (profile.get("crop") or "your crop").lower()

            answer = (
                f"Recommended schemes for {crop} farmers:<br>"
                + "<br>".join(f"✅ {s}" for s in recommendations)
            )

        else:
            answer = generate_answer(query, context)

        print("\nANSWER:")
        print(answer)
        print("=" * 70)

        # -----------------------------
        # FIX 2: proper confidence from FAISS
        # -----------------------------
        confidence = None

        try:
            if distances is not None and len(distances) > 0:
                best_score = float(distances[0][0])

                # cosine similarity style conversion
                confidence = round(((best_score + 1) / 2) * 100, 2)

        except Exception as e:
            print("Confidence error:", e)

        # -----------------------------
        # Response
        # -----------------------------
        return jsonify({
            "answer": answer,
            "confidence": confidence,
            "sources": [
                (
                    chunk.get("chunk") if isinstance(chunk, dict)
                    else chunk.page_content if hasattr(chunk, "page_content")
                    else str(chunk)
                )
                for chunk in chunks
            ],
            "recommendations": recommendations
        })

    except Exception as exc:
        print("ERROR:", exc)

        return jsonify({
            "error": str(exc),
            "answer": "Temporary backend issue. Please retry.",
            "confidence": None,
            "sources": [],
            "recommendations": []
        }), 502


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )