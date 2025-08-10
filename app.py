"""Flask web application providing API access to the RAG backend."""
from __future__ import annotations

import json
import time
from flask import Flask, request, jsonify, Response, render_template

from rag_backend import RAGBackend

app = Flask(__name__, static_folder="static", template_folder="templates")
backend: RAGBackend | None = None


def get_backend() -> RAGBackend:
    global backend
    if backend is None:
        backend = RAGBackend()
    return backend


@app.route("/api/health")
def health() -> Response:
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/api/stats")
def stats() -> Response:
    """Return static knowledge graph statistics."""
    return jsonify({"relationships": 700, "entities": 954})


@app.route("/api/query", methods=["POST"])
def query() -> Response:
    data = request.get_json(force=True)
    user_query = data.get("query", "")

    def generate():
        result = get_backend().query(user_query)["answer"]
        # Stream word by word to emulate ChatGPT typing
        for token in result.split():
            payload = json.dumps({"content": token + " "})
            yield f"data: {payload}\n\n"
            time.sleep(0.05)
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/")
def index() -> str:
    return render_template("index.html", version=int(time.time()))


if __name__ == "__main__":  # pragma: no cover - manual execution
    app.run(host="0.0.0.0", port=8000, debug=False)
