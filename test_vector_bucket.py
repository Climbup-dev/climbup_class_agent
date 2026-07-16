import os
import shutil
import zipfile
from app.core.supabase_client import supabase_new

def check_vector_bucket():
    bucket_name = "vector_stores"
    try:
        buckets = supabase_new.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        if bucket_name not in bucket_names:
            supabase_new.storage.create_bucket(bucket_name, options={"public": False})
            print(f"Bucket {bucket_name} created.")
        else:
            print(f"Bucket {bucket_name} exists.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    check_vector_bucket()
