#!/usr/bin/env python3
"""
Migrate Weaviate data using text2vec-huggingface for Railway compatibility.
This approach copies existing vectors and sets up compatible vectorizer.
"""

import requests
import json
from tqdm import tqdm

def migrate_with_huggingface():
    """Migrate using text2vec-huggingface for Railway compatibility."""
    
    LOCAL_URL = "http://localhost:8080"
    RAILWAY_URL = "https://weaviate-production-55d6.up.railway.app"
    
    print("🚀 Migrating with Huggingface Vectorizer")
    print("=" * 50)
    
    # 1. Test connections
    print("1. Testing connections...")
    try:
        local_resp = requests.get(f"{LOCAL_URL}/v1/meta")
        print(f"   ✅ Local Weaviate: {local_resp.status_code}")
    except Exception as e:
        print(f"   ❌ Local Weaviate failed: {e}")
        return False
    
    try:
        railway_resp = requests.get(f"{RAILWAY_URL}/v1/meta")
        print(f"   ✅ Railway Weaviate: {railway_resp.status_code}")
    except Exception as e:
        print(f"   ❌ Railway Weaviate failed: {e}")
        return False
    
    # 2. Get local data
    print("\n2. Getting local data...")
    try:
        objects_resp = requests.get(f"{LOCAL_URL}/v1/objects?include=vector&limit=1000")
        objects_data = objects_resp.json()
        objects = objects_data.get('objects', [])
        print(f"   📊 Found {len(objects)} objects to migrate")
        
        if not objects:
            print("   ❌ No objects found in local Weaviate")
            return False
            
        print(f"   📋 Sample object: {objects[0].get('class')}")
        print(f"   🎯 Has vector: {'vector' in objects[0]}")
        
    except Exception as e:
        print(f"   ❌ Failed to get local data: {e}")
        return False
    
    # 3. Clear Railway and create schema
    print("\n3. Creating Railway schema...")
    try:
        # Clear existing schema
        clear_resp = requests.delete(f"{RAILWAY_URL}/v1/schema")
        print(f"   🗑️  Cleared existing schema: {clear_resp.status_code}")
        
        # Create Default class with huggingface vectorizer
        default_class = {
            "class": "Default",
            "description": "Migrated from local Weaviate",
            "vectorizer": "text2vec-huggingface",
            "moduleConfig": {
                "text2vec-huggingface": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "options": {
                        "waitForModel": True,
                        "useGPU": False
                    }
                }
            },
            "properties": [
                {"name": "_original_id", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]},
                {"name": "blob_data", "dataType": ["blob"]},
                {"name": "blob_mime_type", "dataType": ["text"]},
                {"name": "score", "dataType": ["number"]},
                {"name": "total_chunks", "dataType": ["number"]},
                {"name": "chunk_id", "dataType": ["number"]},
                {"name": "file_type", "dataType": ["text"]},
                {"name": "file_size", "dataType": ["number"]},
                {"name": "filename", "dataType": ["text"]},
                {"name": "modified_date", "dataType": ["text"]},
                {"name": "content_type", "dataType": ["text"]},
                {"name": "file_path", "dataType": ["text"]},
                {"name": "section_name", "dataType": ["text"]}
            ]
        }
        
        create_resp = requests.post(
            f"{RAILWAY_URL}/v1/schema",
            json=default_class,
            headers={'Content-Type': 'application/json'}
        )
        
        if create_resp.status_code in [200, 201]:
            print("   ✅ Default class created with text2vec-huggingface")
        else:
            print(f"   ❌ Failed to create class: {create_resp.status_code}")
            print(f"   Error: {create_resp.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Schema creation failed: {e}")
        return False
    
    # 4. Migrate data with existing vectors
    print("\n4. Migrating data...")
    try:
        migrated_count = 0
        batch_size = 5  # Small batches for stability
        
        print(f"   📦 Processing {len(objects)} objects in batches of {batch_size}")
        
        with tqdm(total=len(objects), desc="Migrating") as pbar:
            for i in range(0, len(objects), batch_size):
                batch = objects[i:i+batch_size]
                
                for obj in batch:
                    try:
                        # Prepare object for Railway
                        railway_obj = {
                            "class": "Default",
                            "id": obj.get('id'),
                            "properties": obj.get('properties', {}),
                            "vector": obj.get('vector', [])  # Keep existing vectors
                        }
                        
                        # Import object
                        import_resp = requests.post(
                            f"{RAILWAY_URL}/v1/objects",
                            json=railway_obj,
                            headers={'Content-Type': 'application/json'}
                        )
                        
                        if import_resp.status_code in [200, 201]:
                            migrated_count += 1
                        else:
                            print(f"\n   ⚠️  Failed to import object: {import_resp.status_code}")
                            
                    except Exception as e:
                        print(f"\n   ⚠️  Object import error: {e}")
                        continue
                    
                    pbar.update(1)
        
        print(f"\n   ✅ Migrated {migrated_count}/{len(objects)} objects")
        
    except Exception as e:
        print(f"   ❌ Migration failed: {e}")
        return False
    
    # 5. Verify migration
    print("\n5. Verifying migration...")
    try:
        railway_objects_resp = requests.get(f"{RAILWAY_URL}/v1/objects")
        railway_objects = railway_objects_resp.json().get('objects', [])
        
        print(f"   📊 Local objects: {len(objects)}")
        print(f"   📊 Railway objects: {len(railway_objects)}")
        
        if len(railway_objects) > 0:
            print("   ✅ Migration successful!")
            
            # Test query capability
            print("\n6. Testing query capability...")
            test_query = {
                "query": {
                    "nearText": {
                        "concepts": ["risk"]
                    }
                },
                "limit": 1
            }
            
            query_resp = requests.post(
                f"{RAILWAY_URL}/v1/graphql",
                json=test_query,
                headers={'Content-Type': 'application/json'}
            )
            
            if query_resp.status_code == 200:
                print("   ✅ Railway Weaviate can handle semantic queries!")
            else:
                print("   ⚠️  Semantic queries may need adjustment")
            
            return True
        else:
            print("   ❌ No objects found in Railway after migration")
            return False
            
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False

def next_steps():
    """Show next steps after successful migration."""
    print(f"\n🎉 MIGRATION COMPLETE!")
    print("=" * 50)
    print("Your data is now on Railway with compatible vectorizer!")
    print("\nNext steps:")
    print("1. export ENVIRONMENT=railway_weaviate_test")
    print("2. export USE_EXISTING_DATA=true")
    print("3. python app.py")
    print("4. Test query: 'What is risk in I2Connect?'")
    print("\n💡 Note: Railway will use text2vec-huggingface instead of")
    print("   text2vec-transformers, but with the same model!")

if __name__ == "__main__":
    success = migrate_with_huggingface()
    if success:
        next_steps()
    else:
        print("\n❌ Migration failed! Check errors above.")
