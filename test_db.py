from app.core.supabase_client import supabase_new
import json

res = supabase_new.table('classrooms').select('*').execute()
print(f"Total Classrooms: {len(res.data)}")
for c in res.data:
    print(json.dumps(c, indent=2))
