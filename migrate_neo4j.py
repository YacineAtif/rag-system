#!/usr/bin/env python3
"""
Migration script to copy knowledge graph from local Neo4j to Railway Neo4j.
Adapted from your existing Aura migration script.
"""

from neo4j import GraphDatabase
import time

def get_neo4j_drivers():
    """Get both local and Railway drivers."""
    
    # Local connection (same as your original)
    local_driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )
    
    # Railway connection - using the correct proxied port
    railway_neo4j_url = "bolt://turntable.proxy.rlwy.net:43560"
    railway_driver = GraphDatabase.driver(
        railway_neo4j_url,
        auth=("neo4j", "mg89wxdi38d1xytau4yd40d7telcxtqo")  # Railway Neo4j credentials
    )
    
    return local_driver, railway_driver

def export_from_local(local_driver):
    """Export all entities and relationships from local Neo4j."""
    print("📤 Exporting data from local Neo4j...")
    
    entities = []
    relationships = []
    
    with local_driver.session() as session:
        # Export entities
        entity_result = session.run("""
        MATCH (n:Entity)
        RETURN n.name as name, 
               n.type as type, 
               n.source as source,
               properties(n) as properties
        """)
        
        for record in entity_result:
            entities.append({
                'name': record['name'],
                'type': record['type'],
                'source': record['source'],
                'properties': dict(record['properties'])
            })
        
        # Export relationships
        rel_result = session.run("""
        MATCH (a:Entity)-[r]->(b:Entity)
        RETURN a.name as source_name,
               b.name as target_name,
               type(r) as rel_type,
               r.source as source,
               properties(r) as properties
        """)
        
        for record in rel_result:
            relationships.append({
                'source_name': record['source_name'],
                'target_name': record['target_name'],
                'rel_type': record['rel_type'],
                'source': record['source'],
                'properties': dict(record['properties'])
            })
    
    print(f"✅ Exported {len(entities)} entities and {len(relationships)} relationships")
    return entities, relationships

def import_to_railway(railway_driver, entities, relationships):
    """Import entities and relationships to Railway Neo4j."""
    print("📥 Importing data to Railway Neo4j...")
    
    with railway_driver.session() as session:
        # Clear existing data in Railway Neo4j
        print("🗑️  Clearing existing data in Railway Neo4j...")
        session.run("MATCH (n) DETACH DELETE n")
        
        # Import entities
        print(f"📝 Creating {len(entities)} entities...")
        for entity in entities:
            session.run("""
            MERGE (e:Entity {name: $name})
            SET e.type = $type,
                e.source = $source,
                e += $properties
            """, 
            name=entity['name'],
            type=entity['type'],
            source=entity['source'],
            properties=entity['properties']
            )
        
        # Import relationships
        print(f"🔗 Creating {len(relationships)} relationships...")
        for rel in relationships:
            query = f"""
            MATCH (a:Entity {{name: $source_name}})
            MATCH (b:Entity {{name: $target_name}})
            MERGE (a)-[r:{rel['rel_type']}]->(b)
            SET r.source = $source,
                r += $properties
            """
            session.run(query,
                source_name=rel['source_name'],
                target_name=rel['target_name'],
                source=rel['source'],
                properties=rel['properties']
            )
    
    print("✅ Import completed!")

def validate_migration(local_driver, railway_driver):
    """Validate that migration was successful."""
    print("🔍 Validating migration...")
    
    # Count entities and relationships in both databases
    with local_driver.session() as session:
        local_entities = session.run("MATCH (n:Entity) RETURN count(n) as count").single()['count']
        local_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
    
    with railway_driver.session() as session:
        railway_entities = session.run("MATCH (n:Entity) RETURN count(n) as count").single()['count']
        railway_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
    
    print(f"📊 Local:   {local_entities} entities, {local_rels} relationships")
    print(f"🚀 Railway: {railway_entities} entities, {railway_rels} relationships")
    
    if local_entities == railway_entities and local_rels == railway_rels:
        print("✅ Migration validation successful!")
        return True
    else:
        print("❌ Migration validation failed - counts don't match")
        return False

def main():
    """Main migration function."""
    print("🚀 Starting Neo4j Local → Railway Migration")
    print("=" * 50)
    
    try:
        # Get drivers
        local_driver, railway_driver = get_neo4j_drivers()
        
        # Test connections
        print("🔌 Testing connections...")
        with local_driver.session() as session:
            session.run("RETURN 1")
        print("✅ Local Neo4j connected")
        
        with railway_driver.session() as session:
            session.run("RETURN 1")
        print("✅ Railway Neo4j connected")
        
        # Export from local
        entities, relationships = export_from_local(local_driver)
        
        if not entities and not relationships:
            print("❌ No data found in local Neo4j. Make sure your local knowledge graph is populated.")
            return
        
        # Import to Railway
        import_to_railway(railway_driver, entities, relationships)
        
        # Validate
        success = validate_migration(local_driver, railway_driver)
        
        # Close connections
        local_driver.close()
        railway_driver.close()
        
        if success:
            print("\n🎉 Migration completed successfully!")
            print("💡 You can now update your config to use Railway Neo4j")
        else:
            print("\n❌ Migration completed with errors")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()