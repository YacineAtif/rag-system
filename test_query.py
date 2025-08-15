#!/usr/bin/env python3
from rag_backend import RAGBackend

def test_queries_without_ood():
    backend = RAGBackend()
    
    queries = [
        "what is risk in i2connect?",
        "what is evidence theory",
        "explain dempster-shafer theory",
        "how does evidence theory work in risk assessment?"
    ]
    
    for query in queries:
        print(f"\n🔍 Testing: '{query}'")
        print("=" * 50)
        
        # Get raw retrieval results
        query_embedding = backend.pipeline.text_embedder.run(text=query)["embedding"]
        vector_docs = backend.pipeline.retriever.run(query_embedding=query_embedding)
        vector_results = vector_docs.get("documents", [])
        
        graph_results = backend.pipeline.graph_builder.graph_search(query)
        
        # Prepare context
        context_parts = []
        for doc in vector_results[:5]:
            if len(doc.content.strip()) > 50:
                context_parts.append(doc.content)
        
        for result in graph_results[:3]:
            context_parts.append(f"{result['source']} {result['relationship']} {result['target']}")
        
        print(f"📊 Found: {len(vector_results)} vector + {len(graph_results)} graph results")
        print(f"📚 Context sources: {len(context_parts)}")
        
        # Generate answer directly (bypass OOD)
        answer = backend.bypass_ood_detection(query, context_parts)
        print(f"\n💬 Answer:")
        print(answer[:400] + "..." if len(answer) > 400 else answer)

if __name__ == "__main__":
    test_queries_without_ood()
