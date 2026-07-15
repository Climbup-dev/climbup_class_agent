from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ClimbUP MVP"
    DATABASE_URL: str = "postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-SUPABASE-REF].supabase.co:5432/postgres"
    REDIS_URL: str = "redis://localhost:6379"
    SECRET_KEY: str = "supersecretkey_change_in_prod"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Supabase Settings
    SUPABASE_URL: str = "https://[YOUR-SUPABASE-REF].supabase.co"
    SUPABASE_KEY: str = ""
    OLD_SUPABASE_URL: str = ""
    OLD_SUPABASE_KEY: str = ""

    # LLM Settings
    OPENROUTER_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
