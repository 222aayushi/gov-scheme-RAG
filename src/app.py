from pathlib import Path
import math

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from retrieval import retrieve
from rag_chatbot import generate_answer, recommend

app = Flask(__name__)
CORS(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(PROJECT_ROOT, "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        payload = request.get_json(force=True)
        query = payload.get("query")
        profile = payload.get("profile", {})

        if not query:
            return jsonify({"error": "Missing 'query' in request"}), 400

        chunks, distances = retrieve(query, k=5)
        context = "\n\n".join(chunk.page_content for chunk in chunks)

        recommendations = recommend(profile) if profile else []
        if profile and _is_scheme_recommendation_query(query):
            crop = (profile.get("crop") or "your crop").lower()
            answer = (
                f"Recommended schemes for {crop} farmers:<br>"
                + "<br>".join(f"✅ {scheme}" for scheme in recommendations)
            )
        else:
            answer = generate_answer(query, context)

        distance = None
        try:
            distance = float(distances[0][0])
        except Exception:
            distance = None

        if distance is None:
            confidence = None
        else:
            confidence = round(math.exp(-distance) * 100, 2)

        return jsonify({
            "answer": answer,
            "confidence": confidence,
            "sources": [chunk.page_content for chunk in chunks],
            "recommendations": recommendations,
        })
    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "answer": "Temporary backend issue. Please retry.",
            "confidence": None,
            "sources": [],
            "recommendations": [],
        }), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)