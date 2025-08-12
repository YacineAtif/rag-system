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


    
def ensure_weaviate_connected():
    """Ensure Weaviate client is connected, reconnect if needed"""
    global weaviate_client
    try:
        # Test if client is responsive
        weaviate_client.schema.get()
        return True
    except Exception as e:
        print(f"🔄 Weaviate disconnected, attempting reconnection: {e}")
        try:
            # Reconnect
            weaviate_client.connect()
            print("✅ Weaviate reconnected successfully")
            return True
        except Exception as reconnect_error:
            print(f"❌ Weaviate reconnection failed: {reconnect_error}")
            return False
        
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


@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        # Ensure Weaviate is connected before processing
        if not ensure_weaviate_connected():
            return jsonify({"error": "Weaviate connection unavailable"}), 500
        
        # Your existing query processing logic here...
        # (the rest of your query handler code)
        
    except Exception as e:
        print(f"❌ Query error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index() -> str:
    return render_template("index.html", version=int(time.time()))


if __name__ == "__main__":  # pragma: no cover - manual execution
    print("🚀 Starting Flask app with environment-based Neo4j configuration...")
    app.run(host="0.0.0.0", port=8000, debug=False)