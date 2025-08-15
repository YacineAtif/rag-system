"""Flask web application providing API access to the RAG backend with enhanced OOD detection."""

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
    print(f"⚠️ Import error: {e}")
    print("Please ensure all required modules are available")
    exit(1)

app = Flask(__name__, static_folder="static", template_folder="templates")
backend: RAGBackend | None = None

        
def get_backend() -> RAGBackend:
    """Get or initialize the RAG backend singleton."""
    global backend
    if backend is None:
        print("🔧 Initializing Enhanced RAGBackend...")
        try:
            backend = RAGBackend()
            print("✅ Enhanced RAGBackend initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize RAGBackend: {e}")
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
        print(f"⚠️ Health check failed: {e}")
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
        print(f"⚠️ Stats retrieval failed: {e}")
        return jsonify({
            "entities": 0,
            "relationships": 0,
            "environment": "unknown",
            "error": str(e)
        }), 500


@app.route("/api/query", methods=["POST"])
def query() -> Response:
    """Process query with enhanced OOD detection and return streaming response."""
    try:
        data = request.get_json(force=True)
        user_query = data.get("query", "")
        
        if not user_query.strip():
            return jsonify({"error": "Empty query"}), 400

        def generate():
            try:
                backend_instance = get_backend()
                
                # Try to query with enhanced OOD detection
                try:
                    result = backend_instance.query(user_query, enable_enhanced_ood=True)
                    
                except Exception as connection_error:
                    print(f"🔄 Backend connection issue, reinitializing: {connection_error}")
                    # Force reinitialization
                    global backend
                    backend = None
                    backend_instance = get_backend()
                    
                    # Retry with enhanced OOD
                    result = backend_instance.query(user_query, enable_enhanced_ood=True)
                
                answer = result.get("answer", "I don't know.")
                
                # Enhanced metadata including OOD diagnostics
                metadata = {
                    "vector_results": result.get("vector_results", 0),
                    "graph_results": result.get("graph_results", 0),
                    "context_sources": result.get("context_sources", 0),
                    "environment": result.get("environment", "unknown"),
                    "neo4j_uri": result.get("neo4j_uri", "unknown"),
                    "weaviate_url": result.get("weaviate_url", "unknown"),
                    # Enhanced OOD information
                    "is_ood": result.get("is_ood", False),
                    "confidence": result.get("confidence", 0.0),
                    "ood_diagnostics": result.get("ood_diagnostics", {}),
                    "rejection_reason": result.get("rejection_reason", None)
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
                print(f"⚠️ Query processing error: {e}")
                error_payload = json.dumps({"error": str(e)})
                yield f"data: {error_payload}\n\n"
                yield "data: [DONE]\n\n"

        return Response(generate(), mimetype="text/event-stream")
        
    except Exception as e:
        print(f"⚠️ Query endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_query() -> Response:
    """NEW: Debug endpoint to analyze how a query is processed by enhanced OOD."""
    try:
        data = request.get_json(force=True)
        user_query = data.get("query", "")
        
        if not user_query.strip():
            return jsonify({"error": "Empty query"}), 400
        
        backend_instance = get_backend()
        
        # Get detailed analysis without processing the full query
        query_is_domain_relevant = backend_instance._is_query_domain_relevant(user_query)
        
        # Get some sample context to test context analysis
        try:
            query_embedding = backend_instance.pipeline.text_embedder.run(text=user_query)["embedding"]
            vector_docs = backend_instance.pipeline.retriever.run(query_embedding=query_embedding)
            vector_results = vector_docs.get("documents", [])
            
            # Analyze context if available
            if vector_results:
                sample_context = [doc.content for doc in vector_results[:3]]
                context_analysis = backend_instance.ood_detector.analyze_context_quality(sample_context)
            else:
                context_analysis = {"overall_assessment": "NO_CONTEXT", "diagnostics": {}}
                
        except Exception as e:
            context_analysis = {"error": str(e), "overall_assessment": "ERROR"}
        
        analysis = {
            "query": user_query,
            "query_domain_relevant": query_is_domain_relevant,
            "context_analysis": context_analysis,
            "would_be_processed": query_is_domain_relevant,
            "recommendation": "ACCEPT" if query_is_domain_relevant else "REJECT"
        }
        
        return jsonify(analysis)
        
    except Exception as e:
        print(f"⚠️ Query analysis error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/test", methods=["POST"])
def test_ood() -> Response:
    """NEW: Test endpoint to run OOD detection tests."""
    try:
        backend_instance = get_backend()
        
        test_cases = [
            # Should be IN-DOMAIN
            ("What is evidence theory in I2Connect?", "IN-DOMAIN"),
            ("How does Dempster-Shafer work for traffic safety?", "IN-DOMAIN"), 
            ("Tell me about Smart Eye gaze tracking", "IN-DOMAIN"),
            ("What are the safety concepts?", "IN-DOMAIN"),
            ("Who are the partners in I2Connect?", "IN-DOMAIN"),
            
            # Should be OUT-OF-DOMAIN
            ("What's the weather today?", "OUT-OF-DOMAIN"),
            ("How do I cook pasta?", "OUT-OF-DOMAIN"),
            ("Provide a sorting algorithm", "OUT-OF-DOMAIN"),
            ("What is paella?", "OUT-OF-DOMAIN"),
        ]
        
        results = []
        passed = 0
        
        for query, expected in test_cases:
            try:
                result = backend_instance.query(query, enable_enhanced_ood=True)
                actual = "OUT-OF-DOMAIN" if result.get('is_ood') else "IN-DOMAIN"
                success = actual == expected
                
                if success:
                    passed += 1
                
                test_result = {
                    "query": query,
                    "expected": expected,
                    "actual": actual,
                    "success": success,
                    "confidence": result.get('confidence', 0),
                    "ood_diagnostics": result.get('ood_diagnostics', {})
                }
                results.append(test_result)
                
            except Exception as e:
                results.append({
                    "query": query,
                    "expected": expected,
                    "actual": "ERROR",
                    "success": False,
                    "error": str(e)
                })
        
        summary = {
            "total_tests": len(test_cases),
            "passed": passed,
            "success_rate": (passed / len(test_cases)) * 100,
            "results": results
        }
        
        return jsonify(summary)
        
    except Exception as e:
        print(f"⚠️ OOD test error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index() -> str:
    return render_template("index.html", version=int(time.time()))


if __name__ == "__main__":  # pragma: no cover - manual execution
    print("🚀 Starting Flask app with Enhanced OOD Detection...")
    
    # Pre-initialize backend to catch errors early
    try:
        get_backend()
        print("🎉 Enhanced RAGBackend pre-initialization successful")
    except Exception as e:
        print(f"⚠️ Pre-initialization failed: {e}")
        print("Please check your configuration and dependencies")
        exit(1)
    
    app.run(host="0.0.0.0", port=8000, debug=False)