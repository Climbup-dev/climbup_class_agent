import os
from app.core.supabase_client import supabase_new

def test_supabase_storage():
    bucket_name = "class_materials"
    
    print("Checking buckets...")
    try:
        buckets = supabase_new.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        print("Existing buckets:", bucket_names)
        
        if bucket_name not in bucket_names:
            print(f"Creating bucket '{bucket_name}'...")
            supabase_new.storage.create_bucket(bucket_name, options={"public": True})
            print("Bucket created.")
        else:
            print("Bucket already exists. Updating to public...")
            supabase_new.storage.update_bucket(bucket_name, options={"public": True})
            
        print("Success!")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_supabase_storage()
