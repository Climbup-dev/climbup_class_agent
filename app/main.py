from app.core.ssl_fix import apply_ssl_fix
apply_ssl_fix()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, websockets, personal_websockets

app = FastAPI(
    title="ClimbUP AI Classroom API",
    description="Backend API for the ClimbUP virtual AI classroom MVP",
    version="0.1.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(websockets.router)
app.include_router(personal_websockets.router)

@app.get("/")
def root():
    return {"message": "Welcome to ClimbUP AI Classroom API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
