"""Flask web application providing API access to the RAG backend."""
from __future__ import annotations

import json
import time
import re
from flask import Flask, request, jsonify, Response, render_template

from rag_backend import RAGBackend


import warnings
import os

# Suppress protobuf warnings
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')
warnings.filterwarnings('ignore', message='.*MessageFactory.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

app = Flask(__name__, static_folder="static", template_folder="templates")
backend: RAGBackend | None = None


def get_backend() -> RAGBackend:
    global backend
    if backend is None:
        print("🔧 Initializing RAGBackend...")
        backend = RAGBackend()
        print("✅ RAGBackend initialized successfully")
    return backend


@app.route("/api/health")
def health() -> Response:
    """Health check endpoint with environment information."""
    try:
        health_info = get_backend().get_health()
        return jsonify(health_info)
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500


@app.route("/api/stats")
def stats() -> Response:
    """Return knowledge graph statistics with environment info."""
    try:
        stats_info = get_backend().get_stats()
        return jsonify(stats_info)
    except Exception as e:
        return jsonify({
            "entities": 0,
            "relationships": 0,
            "error": str(e)
        }), 500


@app.route("/api/query", methods=["POST"])
def query() -> Response:
    """Process query and return streaming response."""
    try:
        data = request.get_json(force=True)
        user_query = data.get("query", "")
        
        if not user_query.strip():
            return jsonify({"error": "Empty query"}), 400

        def generate():
            try:
                result = get_backend().query(user_query)
                answer = result.get("answer", "I don't know.")
                
                # Send metadata first
                metadata = {
                    "vector_results": result.get("vector_results", 0),
                    "graph_results": result.get("graph_results", 0),
                    "environment": result.get("environment", "unknown"),
                    "neo4j_uri": result.get("neo4j_uri", "unknown")
                }
                yield f"data: {json.dumps({'metadata': metadata})}\n\n"
                
                # Stream the answer token by token
                # Preserve whitespace and newlines while streaming
                for token in re.findall(r"\S+\s*", answer):
                    payload = json.dumps({"token": token})
                    yield f"data: {payload}\n\n"
                    time.sleep(0.02)
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_payload = json.dumps({"error": str(e)})
                yield f"data: {error_payload}\n\n"
                yield "data: [DONE]\n\n"

        return Response(generate(), mimetype="text/event-stream")
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/environment")
def get_environment() -> Response:
    """Get current environment information."""
    try:
        backend_instance = get_backend()
        return jsonify({
            "environment": backend_instance.config.environment,
            "neo4j_uri": backend_instance._get_current_neo4j_uri(),
            "weaviate_url": backend_instance.config.weaviate.url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index() -> str:
    return render_template("index.html", version=int(time.time()))


if __name__ == "__main__":  # pragma: no cover - manual execution
    print("🚀 Starting Flask app with environment-based Neo4j configuration...")
    app.run(host="0.0.0.0", port=8000, debug=False)