#!/usr/bin/env python3
"""
Check what methods are missing from Neo4jGraphBuilder
"""

def check_missing_methods():
    try:
        from weaviate_rag_pipeline_transformers import Neo4jGraphBuilder
        from backend.config import Config
        
        # Create an instance to check methods
        config = Config()
        builder = Neo4jGraphBuilder(config)
        
        # List of methods that should exist based on the error and our analysis
        required_methods = [
            'is_graph_populated',
            'extract_entities_and_relationships', 
            'populate_graph',
            'validate_graph_population',
            'graph_search',
            '_search_by_entities',
            '_search_by_relationships', 
            '_search_by_content',
            '_get_most_connected_nodes',
            '_query_project_participants',
            '_query_entity_roles',
            '_query_development_relationships',
            '_query_general_relationships'
        ]
        
        print("🔍 Checking for required methods...")
        missing_methods = []
        existing_methods = []
        
        for method in required_methods:
            if hasattr(builder, method):
                existing_methods.append(method)
                print(f"✅ {method}")
            else:
                missing_methods.append(method)
                print(f"❌ {method}")
        
        print(f"\n📊 Summary:")
        print(f"✅ Existing methods: {len(existing_methods)}")
        print(f"❌ Missing methods: {len(missing_methods)}")
        
        if missing_methods:
            print(f"\n🔧 Missing methods that need to be added:")
            for method in missing_methods:
                print(f"  - {method}")
        
        return missing_methods
        
    except Exception as e:
        print(f"❌ Error checking methods: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    missing = check_missing_methods()
    if not missing:
        print("\n🎉 All required methods are present!")
    else:
        print(f"\n⚠️  Need to add {len(missing)} missing methods")
