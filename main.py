"""
FastAPI main application entry point for RAG Comparison Platform
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from config import settings
from utils.database import create_tables
from utils.logging_config import setup_development_logging, setup_production_logging
from middleware.error_handlers import setup_error_handlers
from middleware.request_logging import RequestLoggingMiddleware, SecurityLoggingMiddleware
from routes import auth, upload, classic_rag, memvid_rag, performance, lightrag, comparison
from services.vector_store import initialize_vector_store

# Setup comprehensive logging
if settings.DEBUG:
    setup_development_logging()
else:
    setup_production_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting RAG Comparison Platform API")
    
    # Initialize database tables
    try:
        create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Initialize vector store
    try:
        await initialize_vector_store()
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise
    
    yield
    logger.info("Shutting down RAG Comparison Platform API")


# Create FastAPI application
app = FastAPI(
    title="RAG Comparison Platform API",
    description="API for comparing Classic RAG and MemVid-inspired RAG approaches",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware, log_body=False, log_headers=False)
app.add_middleware(SecurityLoggingMiddleware)

# Setup global error handlers
setup_error_handlers(app)

# Include routers
app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(classic_rag.router)
app.include_router(memvid_rag.router)
app.include_router(lightrag.router)
app.include_router(comparison.router)
app.include_router(performance.router)


@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "message": "RAG Comparison Platform API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )