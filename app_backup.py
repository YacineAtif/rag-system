"""Flask web application providing API access to the RAG backend."""

from __future__ import annotations

import warnings
import os

# Suppress all protobuf warnings
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

from query_router import QueryRouter

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
    print(f"⚠ Import error: {e}")
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
query_router: QueryRouter | None = None  # NEW: Add query router global

        
def get_backend() -> RAGBackend:
    """Get or initialize the RAG backend singleton."""
    global backend, query_router  # NEW: Include query_router in global
    if backend is None:
        print("🔧 Initializing RAGBackend...")
        try:
            backend = RAGBackend()
            query_router = QueryRouter(backend)  # NEW: Initialize query router
            print("✅ RAGBackend and QueryRouter initialized successfully")
        except Exception as e:
            print(f"⚠ Failed to initialize RAGBackend: {e}")
            raise
    return backend


def get_query_router() -> QueryRouter:
    """Get the query router, initializing backend if needed."""
    global query_router
    if query_router is None:
        get_backend()  # This will initialize both backend and query_router
    return query_router


@app.route("/api/health")
def health() -> Response:
    """Health check endpoint with environment information."""
    try:
        backend_instance = get_backend()
        health_info = backend_instance.get_health()
        return jsonify(health_info)
    except Exception as e:
        print(f"⚠ Health check failed: {e}")
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
        print(f"⚠ Stats retrieval failed: {e}")
        return jsonify({
            "entities": 0,
            "relationships": 0,
            "environment": "unknown",
            "error": str(e)
        }), 500


@app.route("/api/query", methods=["POST"])
def query() -> Response:
    """Process query with intelligent routing and return streaming response."""
    try:
        data = request.get_json(force=True)
        user_query = data.get("query", "")
        
        if not user_query.strip():
            return jsonify({"error": "Empty query"}), 400

        def generate():
            try:
                # Get query router and ensure it's connected
                router = get_query_router()
                
                # Try to query with routing and error handling
                try:
                    # NEW: Use routed query instead of direct backend query
                    routed_results = router.route_query(user_query)
                    result = routed_results['results']
                    
                    # Extract routing metadata for enhanced response
                    routing_info = {
                        'domain': routed_results.get('classification', {}).get('domain', 'unknown'),
                        'confidence': routed_results.get('classification', {}).get('confidence', 0.0),
                        'enhanced_query': routed_results.get('enhanced_query', user_query),
                        'keywords': routed_results.get('classification', {}).get('keywords', [])
                    }
                    
                except Exception as connection_error:
                    print(f"🔄 Backend connection issue, reinitializing: {connection_error}")
                    # Force reinitialization
                    global backend, query_router
                    backend = None
                    query_router = None
                    router = get_query_router()
                    
                    # Retry with routing
                    routed_results = router.route_query(user_query)
                    result = routed_results['results']
                    routing_info = {
                        'domain': routed_results.get('classification', {}).get('domain', 'unknown'),
                        'confidence': routed_results.get('classification', {}).get('confidence', 0.0),
                        'enhanced_query': routed_results.get('enhanced_query', user_query)
                    }
                
                answer = result.get("answer", "I don't know.")
                
                # Send enhanced metadata first (including routing info)
                metadata = {
                    "vector_results": result.get("vector_results", 0),
                    "graph_results": result.get("graph_results", 0),
                    "environment": result.get("environment", "unknown"),
                    "neo4j_uri": result.get("neo4j_uri", "unknown"),
                    # NEW: Add routing information to metadata
                    "routing": routing_info
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
                print(f"⚠ Query processing error: {e}")
                error_payload = json.dumps({"error": str(e)})
                yield f"data: {error_payload}\n\n"
                yield "data: [DONE]\n\n"

        return Response(generate(), mimetype="text/event-stream")
        
    except Exception as e:
        print(f"⚠ Query endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/explain", methods=["POST"])
def explain_routing() -> Response:
    """NEW: Debug endpoint to explain how a query would be routed."""
    try:
        data = request.get_json(force=True)
        user_query = data.get("query", "")
        
        if not user_query.strip():
            return jsonify({"error": "Empty query"}), 400
        
        router = get_query_router()
        explanation = router.explain_routing(user_query)
        
        return jsonify(explanation)
        
    except Exception as e:
        print(f"⚠ Explain routing error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index() -> str:
    return render_template("index.html", version=int(time.time()))


if __name__ == "__main__":  # pragma: no cover - manual execution
    print("🚀 Starting Flask app with intelligent query routing...")
    
    # Pre-initialize backend and router to catch errors early
    try:
        get_backend()
        get_query_router()
        print("🎉 Backend and QueryRouter pre-initialization successful")
    except Exception as e:
        print(f"⚠ Pre-initialization failed: {e}")
        print("Please check your configuration and dependencies")
        exit(1)
    
    app.run(host="0.0.0.0", port=8000, debug=False)