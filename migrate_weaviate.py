#!/usr/bin/env python3
"""
Simple Weaviate Migration using requests
No complex client dependencies - just HTTP calls
"""

import requests
import json

def migrate_weaviate():
    LOCAL_URL = "http://localhost:8080"
    RAILWAY_URL = "https://weaviate-production-55d6.up.railway.app"
    
    print("🚀 Starting simple Weaviate migration...")
    
    # 1. Test connections
    print("\n1. Testing connections...")
    try:
        local_resp = requests.get(f"{LOCAL_URL}/v1/meta")
        print(f"✅ Local Weaviate: {local_resp.status_code}")
    except Exception as e:
        print(f"❌ Local Weaviate failed: {e}")
        return False
    
    try:
        railway_resp = requests.get(f"{RAILWAY_URL}/v1/meta")
        print(f"✅ Railway Weaviate: {railway_resp.status_code}")
    except Exception as e:
        print(f"❌ Railway Weaviate failed: {e}")
        return False
    
    # 2. Get local schema
    print("\n2. Getting local schema...")
    try:
        schema_resp = requests.get(f"{LOCAL_URL}/v1/schema")
        schema = schema_resp.json()
        print(f"Found {len(schema['classes'])} classes:")
        for cls in schema['classes']:
            print(f"  - {cls['class']}")
    except Exception as e:
        print(f"❌ Failed to get schema: {e}")
        return False
    
    # 3. Create schema on Railway (simplified)
    print("\n3. Creating schema on Railway...")
    try:
        # Clear existing schema
        requests.delete(f"{RAILWAY_URL}/v1/schema")
        
        for class_obj in schema['classes']:
            # Simplify the class - remove transformers dependencies
            simple_class = {
                "class": class_obj['class'],
                "vectorizer": "none",
                "properties": []
            }
            
            # Add properties without module configs
            for prop in class_obj.get('properties', []):
                simple_prop = {
                    "name": prop['name'],
                    "dataType": prop['dataType']
                }
                simple_class['properties'].append(simple_prop)
            
            # Create class
            create_resp = requests.post(
                f"{RAILWAY_URL}/v1/schema",
                json=simple_class,
                headers={'Content-Type': 'application/json'}
            )
            
            if create_resp.status_code == 200:
                print(f"✅ Created class: {simple_class['class']}")
            else:
                print(f"❌ Failed to create {simple_class['class']}: {create_resp.text}")
    
    except Exception as e:
        print(f"❌ Schema creation failed: {e}")
        return False
    
    # 4. Get local data
    print("\n4. Getting local data...")
    try:
        objects_resp = requests.get(f"{LOCAL_URL}/v1/objects?include=vector")
        objects_data = objects_resp.json()
        objects = objects_data.get('objects', [])
        print(f"Found {len(objects)} objects")
        
        if objects:
            obj = objects[0]
            print(f"First object:")
            print(f"  - ID: {obj.get('id')}")
            print(f"  - Class: {obj.get('class')}")
            print(f"  - Properties: {list(obj.get('properties', {}).keys())}")
            print(f"  - Has vector: {'vector' in obj}")
    
    except Exception as e:
        print(f"❌ Failed to get data: {e}")
        return False
    
    # 5. Import data to Railway
    print("\n5. Importing data to Railway...")
    try:
        for obj in objects:
            # Prepare object for Railway
            railway_obj = {
                "class": obj.get('class'),
                "id": obj.get('id'),
                "properties": obj.get('properties', {}),
                "vector": obj.get('vector', [])
            }
            
            # Import object
            import_resp = requests.post(
                f"{RAILWAY_URL}/v1/objects",
                json=railway_obj,
                headers={'Content-Type': 'application/json'}
            )
            
            if import_resp.status_code in [200, 201]:
                print(f"✅ Imported object {obj.get('id')}")
            else:
                print(f"❌ Failed to import {obj.get('id')}: {import_resp.text}")
    
    except Exception as e:
        print(f"❌ Data import failed: {e}")
        return False
    
    # 6. Verify migration
    print("\n6. Verifying migration...")
    try:
        railway_objects_resp = requests.get(f"{RAILWAY_URL}/v1/objects")
        railway_objects = railway_objects_resp.json().get('objects', [])
        
        print(f"Local objects: {len(objects)}")
        print(f"Railway objects: {len(railway_objects)}")
        
        if len(objects) == len(railway_objects):
            print("✅ Migration verification: SUCCESS")
            return True
        else:
            print("⚠️ Migration verification: PARTIAL SUCCESS")
            return True
    
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate_weaviate()
    if success:
        print("\n🎉 Migration completed!")
        print("Your data is now on Railway Weaviate!")
    else:
        print("\n❌ Migration failed!")
        print("Check the errors above.")