#!/usr/bin/env python3
"""
Traffic Safety Knowledge Base with Haystack + Weaviate
Fixed imports for Haystack 2.15.2
"""
import os
import time
import requests
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever

def wait_for_weaviate(url="http://localhost:8080", max_retries=10):
    """Wait for Weaviate to be ready"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/v1/meta")
            if response.status_code == 200:
                print("✅ Weaviate is ready!")
                return True
        except Exception as e:
            pass
        print(f"⏳ Waiting for Weaviate... ({i+1}/{max_retries})")
        time.sleep(2)
    return False

def main():
    print("🚦 Traffic Safety Knowledge Base Test")
    print("=" * 50)
    
    # Wait for Weaviate
    if not wait_for_weaviate():
        print("❌ Weaviate not responding")
        return
    
    try:
        # Initialize Weaviate Document Store with transformers
        print("🔧 Initializing document store...")
        document_store = WeaviateDocumentStore(
            url="http://localhost:8080"
        )
        
        # Sample traffic safety documents
        documents = [
            Document(
                content="Traffic safety assessment involves systematic evaluation of road infrastructure, vehicle conditions, and driver behavior to identify potential hazards.",
                meta={"topic": "assessment", "category": "methodology"}
            ),
            Document(
                content="Interconnected traffic systems use IoT sensors, real-time analytics, and machine learning for improved safety outcomes and incident prevention.",
                meta={"topic": "technology", "category": "smart_systems"}
            ),
            Document(
                content="Risk assessment matrices consider probability and severity of accidents, incorporating traffic volume, weather conditions, and historical incident data.",
                meta={"topic": "risk_analysis", "category": "methodology"}
            ),
            Document(
                content="Vehicle-to-Everything (V2X) communication enables real-time information exchange between vehicles, infrastructure, and traffic management centers.",
                meta={"topic": "v2x", "category": "communication"}
            ),
            Document(
                content="Traffic flow optimization algorithms adjust signal timing and route recommendations based on real-time conditions and safety metrics.",
                meta={"topic": "optimization", "category": "algorithms"}
            )
        ]
        
        # Create indexing pipeline with embedder
        print("📚 Indexing documents...")
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder
        
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
        indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        indexing_pipeline.connect("embedder", "writer")
        
        # Index the documents
        indexing_pipeline.run({"embedder": {"documents": documents}})
        print(f"✅ Successfully indexed {len(documents)} documents!")
        
        # Create search pipeline with text embedder
        print("🔍 Setting up search pipeline...")
        from haystack.components.embedders import SentenceTransformersTextEmbedder
        
        search_pipeline = Pipeline()
        search_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
        search_pipeline.add_component("retriever", WeaviateEmbeddingRetriever(document_store=document_store))
        search_pipeline.connect("text_embedder", "retriever")
        
        # Test searches
        test_queries = [
            "How do risk assessment matrices work?",
            "What is V2X communication?",
            "How can we optimize traffic flow?",
            "What sensors are used in smart traffic systems?"
        ]
        
        print("\n🔍 Testing search functionality:")
        print("=" * 50)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 30)
            
            result = search_pipeline.run({
                "text_embedder": {"text": query},
                "retriever": {"top_k": 2}
            })
            
            documents_found = result["retriever"]["documents"]
            if documents_found:
                for i, doc in enumerate(documents_found, 1):
                    print(f"{i}. Score: {doc.score:.4f}")
                    print(f"   Content: {doc.content[:100]}...")
                    print(f"   Topic: {doc.meta.get('topic', 'N/A')}")
            else:
                print("   No documents found")
        
        print("\n✅ Traffic Safety Knowledge Base is working!")
        print("🎉 Your Haystack + Weaviate setup is ready for your domain documents!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This might be due to missing packages or configuration issues.")
        print("Try running: pip install weaviate-haystack")

if __name__ == "__main__":
    main()