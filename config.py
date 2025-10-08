"""
Configuration settings for the RAG Comparison Platform
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:4200", "http://127.0.0.1:4200"]
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:admin@localhost:5432/rag_platform"
    
    # PostgreSQL specific settings
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "admin"
    POSTGRES_DB: str = "rag_platform"
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = ["application/pdf", "text/plain"]
    UPLOAD_DIR: str = "uploads"
    
    # RAG Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    
    # LLM Configuration
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-3.5-turbo"
    
    # Google Gemini Configuration
    GEMINI_API_KEY: str = "your-secret-key-change-in-production"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    
    # Vector Database
    VECTOR_DB_PATH: str = "./vector_db"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
