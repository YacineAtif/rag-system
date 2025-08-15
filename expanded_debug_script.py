#!/usr/bin/env python3
"""
Debug script to test relationship extraction
"""

def test_imports():
    """Test if all imports work correctly"""
    print("🔍 Testing imports...")
    
    try:
        from rag_backend import RAGBackend
        print("✅ Successfully imported RAGBackend")
        return True
    except ImportError as e:
        print(f"❌ Failed to import RAGBackend: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing RAGBackend: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_file_structure():
    """Check if required files exist"""
    import os
    print("🔍 Checking file structure...")
    
    required_files = [
        'rag_backend.py',
        'weaviate_rag_pipeline_transformers.py',
        'config.yaml'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_backend_initialization():
    """Test if RAGBackend can be initialized"""
    print("\n🔧 Testing RAGBackend initialization...")
    
    try:
        from rag_backend import RAGBackend
        print("📡 Initializing RAGBackend (this may take a moment)...")
        backend = RAGBackend()
        print("✅ RAGBackend initialized successfully")
        return backend
    except Exception as e:
        print(f"❌ Failed to initialize RAGBackend: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_entity_extraction(backend):
    """Test entity and relationship extraction"""
    if not backend:
        print("❌ Cannot test extraction without backend")
        return None
        
    print("\n🧪 Testing entity and relationship extraction...")
    
    # Test sample text with clear relationships
    test_text = """
    University of Skövde partners with Scania to develop Evidence Theory applications.
    Smart Eye provides driver monitoring technology for the I2Connect project.
    Safety Concept 1 addresses intersection risk through advanced sensor systems.
    Viscando contributes traffic analytics technology to support risk assessment.
    The I2Connect project uses ADAS technology for enhanced safety.
    """
    
    print("📝 Test text:")
    print(test_text.strip())
    
    try:
        print("\n🔬 Running extraction...")
        extracted = backend.pipeline.graph_builder.extract_entities_and_relationships(
            test_text, "test_document"
        )
        
        print(f"\n📊 Extraction Results:")
        entities = extracted.get('entities', [])
        relationships = extracted.get('relationships', [])
        
        print(f"📦 Entities: {len(entities)}")
        for i, entity in enumerate(entities):
            name = entity.get('name', 'Unknown')
            entity_type = entity.get('type', 'Unknown')
            print(f"  {i+1}. {name} ({entity_type})")
        
        print(f"\n🔗 Relationships: {len(relationships)}")
        for i, rel in enumerate(relationships):
            source = rel.get('source', 'Unknown')
            relationship = rel.get('relationship', 'Unknown') 
            target = rel.get('target', 'Unknown')
            print(f"  {i+1}. {source} -> {relationship} -> {target}")
        
        return extracted
        
    except Exception as e:
        print(f"❌ Entity extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_neo4j_database(backend):
    """Check what's in the Neo4j database"""
    if not backend:
        print("❌ Cannot check database without backend")
        return
        
    print("\n🗄️ Checking Neo4j database contents...")
    
    try:
        with backend.pipeline.graph_builder.driver.session() as session:
            # Count entities
            entity_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            entity_count = entity_result.single()["count"]
            
            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            
            print(f"📦 Total entities in database: {entity_count}")
            print(f"🔗 Total relationships in database: {rel_count}")
            
            # Show sample entities
            if entity_count > 0:
                print(f"\n📋 Sample entities:")
                sample_entities = session.run("""
                    MATCH (n:Entity) 
                    RETURN n.name as name, n.type as type 
                    ORDER BY n.name 
                    LIMIT 10
                """)
                for i, record in enumerate(sample_entities, 1):
                    print(f"  {i}. {record['name']} ({record['type']})")
            
            # Show sample relationships
            if rel_count > 0:
                print(f"\n🔗 Sample relationships:")
                sample_rels = session.run("""
                    MATCH (a:Entity)-[r]->(b:Entity) 
                    RETURN a.name as source, type(r) as relationship, b.name as target 
                    ORDER BY a.name
                    LIMIT 10
                """)
                for i, record in enumerate(sample_rels, 1):
                    print(f"  {i}. {record['source']} -> {record['relationship']} -> {record['target']}")
            else:
                print("❌ No relationships found in database!")
                print("💡 This suggests relationship extraction is not working properly")
                
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

def test_query_processing(backend):
    """Test a simple query"""
    if not backend:
        print("❌ Cannot test query without backend")
        return None
        
    print("\n❓ Testing query processing...")
    
    try:
        query = "what is risk in i2connect?"
        print(f"🔍 Testing query: '{query}'")
        
        result = backend.query(query)
        
        vector_count = result.get('vector_results', 0)
        graph_count = result.get('graph_results', 0)
        context_sources = result.get('context_sources', 0)
        
        print(f"📊 Query Results:")
        print(f"  📄 Vector results: {vector_count}")
        print(f"  🕸️  Graph results: {graph_count}")
        print(f"  📚 Context sources: {context_sources}")
        
        answer = result.get('answer', 'No answer')
        print(f"\n💬 Answer:")
        print(f"  {answer[:300]}{'...' if len(answer) > 300 else ''}")
        
        return result
        
    except Exception as e:
        print(f"❌ Query processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main debug function"""
    import os
    print("🧪 Starting RAG System Debug Session")
    print("=" * 50)
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Check file structure first
    if not check_file_structure():
        print("❌ Required files are missing")
        return
    
    # Test imports
    if not test_imports():
        print("❌ Cannot proceed without successful imports")
        return
    
    # Initialize backend
    backend = test_backend_initialization()
    if not backend:
        print("❌ Cannot proceed without successful backend initialization")
        return
    
    # Test entity extraction
    extraction_result = test_entity_extraction(backend)
    
    # Check database contents
    check_neo4j_database(backend)
    
    # Test query processing
    query_result = test_query_processing(backend)
    
    print("\n" + "=" * 50)
    print("📈 SUMMARY")
    print("=" * 50)
    
    # Summarize results
    if extraction_result:
        entities = len(extraction_result.get('entities', []))
        relationships = len(extraction_result.get('relationships', []))
        print(f"🧬 Entity Extraction: {entities} entities, {relationships} relationships")
        
        if relationships == 0:
            print("⚠️  WARNING: No relationships extracted from test text!")
            print("💡 This suggests relationship extraction logic needs improvement")
        else:
            print("✅ Relationship extraction is working!")
    else:
        print("❌ Entity extraction test failed")
    
    if query_result:
        vector_count = query_result.get('vector_results', 0)
        graph_count = query_result.get('graph_results', 0)
        print(f"🔍 Query Processing: {vector_count} vector + {graph_count} graph results")
        
        if vector_count == 0 and graph_count == 0:
            print("⚠️  WARNING: Query returned no results!")
        else:
            print("✅ Query processing is working!")
    else:
        print("❌ Query processing test failed")
    
    print("\n✅ Debug session complete!")

if __name__ == "__main__":
    main()