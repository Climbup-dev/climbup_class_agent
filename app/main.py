from app.core.ssl_fix import apply_ssl_fix
apply_ssl_fix()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, websockets, personal_websockets, classrooms, debug, tour

app = FastAPI(
    title="ClimbUP AI Classroom API",
    description="Backend API for the ClimbUP virtual AI classroom MVP",
    version="0.1.0"
)

from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Global Error: {str(exc)}\nTraceback: {traceback.format_exc()}"
    print(error_msg)
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg}
    )

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(websockets.router)
app.include_router(personal_websockets.router)
app.include_router(classrooms.router)
app.include_router(debug.router)
app.include_router(tour.router)

@app.get("/")
def root():
    return {"message": "Welcome to ClimbUP AI Classroom API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
