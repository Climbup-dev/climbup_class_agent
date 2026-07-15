from supabase import create_client, Client
from app.core.config import settings

# Initialize New Supabase Client (For Data & AI Memory)
supabase_new: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Initialize Old Supabase Client (For Auth/PYQ Verification)
supabase_old: Client = None
if settings.OLD_SUPABASE_URL and settings.OLD_SUPABASE_KEY:
    supabase_old = create_client(settings.OLD_SUPABASE_URL, settings.OLD_SUPABASE_KEY)
