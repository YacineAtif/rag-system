#!/usr/bin/env python3
"""
Debug script to check where the knowledge graph data is coming from.
"""

from backend.config import Config
from weaviate_rag_pipeline_transformers import create_neo4j_driver
from neo4j import GraphDatabase

def check_local_neo4j():
    """Check local Neo4j database."""
    print("🏠 Checking Local Neo4j (bolt://localhost:7687)...")
    try:
        local_driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        
        with local_driver.session() as session:
            entity_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            entity_count = entity_result.single()["count"]
            
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            
            # Check relationship types
            rel_types = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type")
            unique_rels = [record["rel_type"] for record in rel_types]
            
            print(f"   📊 Local: {entity_count} entities, {rel_count} relationships")
            print(f"   🔗 Unique relationship types: {len(unique_rels)}")
            print(f"   📝 Relationship types: {unique_rels[:5]}..." if len(unique_rels) > 5 else f"   📝 Relationship types: {unique_rels}")
        
        local_driver.close()
        return entity_count, rel_count, len(unique_rels)
        
    except Exception as e:
        print(f"   ❌ Local connection failed: {e}")
        return 0, 0, 0

def check_aura_neo4j():
    """Check Aura Neo4j database."""
    print("☁️  Checking Aura Neo4j...")
    try:
        config = Config()
        config.environment = "aura"
        
        aura_driver = GraphDatabase.driver(
            config.neo4j.aura.uri,
            auth=(config.neo4j.aura.user, config.neo4j.aura.password)
        )
        
        with aura_driver.session() as session:
            entity_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            entity_count = entity_result.single()["count"]
            
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            
            # Check relationship types
            rel_types = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type")
            unique_rels = [record["rel_type"] for record in rel_types]
            
            print(f"   📊 Aura: {entity_count} entities, {rel_count} relationships")
            print(f"   🔗 Unique relationship types: {len(unique_rels)}")
            print(f"   📝 Relationship types: {unique_rels[:5]}..." if len(unique_rels) > 5 else f"   📝 Relationship types: {unique_rels}")
        
        aura_driver.close()
        return entity_count, rel_count, len(unique_rels)
        
    except Exception as e:
        print(f"   ❌ Aura connection failed: {e}")
        return 0, 0, 0

def check_web_app_backend():
    """Check what the RAGBackend is actually using."""
    print("🌐 Checking RAGBackend configuration...")
    try:
        from rag_backend import RAGBackend
        
        backend = RAGBackend()
        stats = backend.get_stats()
        
        print(f"   📊 RAGBackend reports: {stats.get('entities', 'unknown')} entities, {stats.get('relationships', 'unknown')} relationships")
        print(f"   🔧 Environment: {stats.get('environment', 'unknown')}")
        print(f"   🔗 URI: {stats.get('neo4j_uri', 'unknown')}")
        
        return stats
        
    except Exception as e:
        print(f"   ❌ RAGBackend check failed: {e}")
        return {}

def check_current_config():
    """Check current configuration."""
    print("⚙️  Checking current configuration...")
    config = Config()
    print(f"   🌍 Environment: {config.environment}")
    
    if config.environment == 'local':
        print(f"   🔗 Would connect to: {config.neo4j.local.uri}")
    elif config.environment == 'aura':
        print(f"   🔗 Would connect to: {config.neo4j.aura.uri}")
    
    return config

def main():
    """Main debug function."""
    print("🔍 Knowledge Graph Debug Analysis")
    print("=" * 50)
    
    # Check configuration
    config = check_current_config()
    print()
    
    # Check both databases
    local_entities, local_rels, local_types = check_local_neo4j()
    print()
    
    aura_entities, aura_rels, aura_types = check_aura_neo4j()
    print()
    
    # Check web app backend
    backend_stats = check_web_app_backend()
    print()
    
    # Analysis
    print("📋 ANALYSIS:")
    print("-" * 30)
    
    backend_entities = backend_stats.get('entities', 0)
    backend_rels = backend_stats.get('relationships', 0)
    
    if backend_entities == local_entities and backend_rels == local_rels:
        print("🎯 Web app is using LOCAL Neo4j database")
    elif backend_entities == aura_entities and backend_rels == aura_rels:
        print("🎯 Web app is using AURA Neo4j database")
    else:
        print("❓ Web app data doesn't match either database - possible caching or different source")
    
    print()
    print("💡 RECOMMENDATIONS:")
    if local_entities > aura_entities:
        print("   📈 Local database has more data - consider re-migrating from local to Aura")
    elif aura_entities > local_entities:
        print("   📈 Aura database has more data - current setup is good")
    else:
        print("   ✅ Databases are in sync")

if __name__ == "__main__":
    main()