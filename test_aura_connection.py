# test_aura_connection.py
from backend.config import Config
from weaviate_rag_pipeline_transformers import create_neo4j_driver

config = Config()
print(f"Testing connection to: {config.environment}")

try:
    driver = create_neo4j_driver(config)
    with driver.session() as session:
        result = session.run("RETURN 'Hello Aura!' as greeting")
        record = result.single()
        print(f"✅ Success: {record['greeting']}")
        
        # Check if database is empty
        count_result = session.run("MATCH (n) RETURN count(n) as count")
        count = count_result.single()["count"]
        print(f"📊 Current nodes in Aura: {count}")
    
    driver.close()
    print("🎉 Aura connection test passed!")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")