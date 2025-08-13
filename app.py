"""Flask web application providing API access to the RAG backend."""

from __future__ import annotations

import warnings
import os

# Suppress all protobuf warnings
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# Additional warning suppressions
warnings.filterwarnings('ignore', message='.*MessageFactory.*')
warnings.filterwarnings('ignore', category=ResourceWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='neo4j')

# Environment variables to reduce noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'true'  # Suppress HuggingFace progress
os.environ['PROTOBUF_PYTHON_IMPLEMENTATION'] = 'python'  # Force Python protobuf


import json
import time
import re
from flask import Flask, request, jsonify, Response, render_template

# Add error handling for missing modules
try:
    from rag_backend import RAGBackend
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available")
    exit(1)

import warnings
import os

# Suppress protobuf warnings
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')
warnings.filterwarnings('ignore', message='.*MessageFactory.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

app = Flask(__name__, static_folder="static", template_folder="templates")
backend: RAGBackend | None = None

        
def get_backend() -> RAGBackend:
    """Get or initialize the RAG backend singleton."""
    global backend
    if backend is None:
        print("🔧 Initializing RAGBackend...")
        try:
            backend = RAGBackend()
            print("✅ RAGBackend initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize RAGBackend: {e}")
            raise
    return backend


@app.route("/api/health")
def health() -> Response:
    """Health check endpoint with environment information."""
    try:
        backend_instance = get_backend()
        health_info = backend_instance.get_health()
        return jsonify(health_info)
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return jsonify({
            "status": "error", 
            "message": str(e),
            "environment": "unknown"
        }), 500


@app.route("/api/stats")
def stats() -> Response:
    """Return knowledge graph statistics with environment info."""
    try:
        backend_instance = get_backend()
        stats_info = backend_instance.get_stats()
        return jsonify(stats_info)
    except Exception as e:
        print(f"❌ Stats retrieval failed: {e}")
        return jsonify({
            "entities": 0,
            "relationships": 0,
            "environment": "unknown",
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
                # Get backend and ensure it's connected
                backend_instance = get_backend()
                
                # Try to query with error handling
                try:
                    result = backend_instance.query(user_query)
                except Exception as connection_error:
                    print(f"🔄 Backend connection issue, reinitializing: {connection_error}")
                    # Force reinitialization
                    global backend
                    backend = None
                    backend_instance = get_backend()
                    result = backend_instance.query(user_query)
                
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
                print(f"❌ Query processing error: {e}")
                error_payload = json.dumps({"error": str(e)})
                yield f"data: {error_payload}\n\n"
                yield "data: [DONE]\n\n"

        return Response(generate(), mimetype="text/event-stream")
        
    except Exception as e:
        print(f"❌ Query endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index() -> str:
    return render_template("index.html", version=int(time.time()))


if __name__ == "__main__":  # pragma: no cover - manual execution
    print("🚀 Starting Flask app with environment-based Neo4j configuration...")
    
    # Pre-initialize backend to catch errors early
    try:
        get_backend()
        print("🎉 Backend pre-initialization successful")
    except Exception as e:
        print(f"❌ Backend pre-initialization failed: {e}")
        print("Please check your configuration and dependencies")
        exit(1)
    
    app.run(host="0.0.0.0", port=8000, debug=False)