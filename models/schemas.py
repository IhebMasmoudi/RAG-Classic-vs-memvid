"""
Pydantic models for request/response schemas
"""
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# User schemas
class UserCreate(BaseModel):
    """Schema for user registration"""
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user response (without password)"""
    id: int
    email: str
    created_at: datetime
    is_active: bool
    
    model_config = {"from_attributes": True}


class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for token data"""
    email: Optional[str] = None


# Document schemas
class DocumentUpload(BaseModel):
    """Schema for document upload metadata"""
    filename: str
    content_type: str
    size: int


class DocumentResponse(BaseModel):
    """Schema for document response"""
    id: str
    filename: str
    content_type: str
    upload_date: datetime
    chunk_count: int
    embedding_model: str
    
    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    """Schema for upload response"""
    document_id: str
    filename: str
    chunks_created: int
    embeddings_generated: int
    message: str


# Query schemas
class QueryRequest(BaseModel):
    """Schema for RAG query request"""
    query: str = Field(..., min_length=1, description="Query text cannot be empty")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional list of document IDs to search within")
    mode: Optional[str] = Field(default="hybrid", description="Query mode for LightRAG: local, global, hybrid, naive")


class MemVidQueryRequest(BaseModel):
    """Schema for MemVid RAG query request"""
    query: str = Field(..., min_length=1, description="Query text cannot be empty")
    context_window: int = Field(default=3, ge=1, le=10, description="Context window size")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional list of document IDs to search within")


class SourceChunk(BaseModel):
    """Schema for source chunk information"""
    chunk_id: str
    content: str
    similarity_score: float
    document_id: str
    chunk_index: int


class Source(BaseModel):
    """Schema for LightRAG source information"""
    document_id: str
    chunk_index: int
    content: str
    similarity_score: float
    metadata: Optional[Dict[str, Any]] = None


class RAGResponse(BaseModel):
    """Schema for RAG pipeline response"""
    answer: str
    sources: List[SourceChunk]
    response_time: float
    chunks_used: int
    query: str
    timestamp: datetime


class MemVidRAGResponse(BaseModel):
    """Schema for MemVid RAG pipeline response"""
    answer: str
    sources: List[SourceChunk]
    response_time: float
    chunks_used: int
    query: str
    timestamp: datetime
    memvid_metadata: Dict[str, Any]


class ComparisonResponse(BaseModel):
    """Schema for comparison between RAG approaches"""
    query: str
    classic_rag: RAGResponse
    memvid_rag: MemVidRAGResponse
    performance_comparison: Dict[str, Any]


# Error schemas
class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str
    detail: str
    timestamp: datetime
    
    model_config = {"from_attributes": True}


class ValidationErrorResponse(BaseModel):
    """Schema for validation error responses"""
    error: str
    detail: List[Dict[str, Any]]
    timestamp: datetime
    
    model_config = {"from_attributes": True}