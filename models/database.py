"""
SQLAlchemy database models for the RAG Comparison Platform
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid


Base = declarative_base()


class User(Base):
    """User model for authentication and user management"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    query_history = relationship("QueryHistory", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"


class Document(Base):
    """Document model for uploaded files"""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    chunk_count = Column(Integer, default=0)
    embedding_model = Column(String(255), nullable=False)
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    
    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id='{self.id}', filename='{self.filename}')>"


class DocumentChunk(Base):
    """Document chunk model for vector storage metadata"""
    __tablename__ = "document_chunks"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA-256 hash of content
    embedding_vector_id = Column(String(255))  # Reference to vector in FAISS/Chroma
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id='{self.id}', document_id='{self.document_id}', index={self.chunk_index})>"


class QueryHistory(Base):
    """Query history model for tracking user queries and responses"""
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query_text = Column(Text, nullable=False)
    query_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Classic RAG results
    classic_answer = Column(Text)
    classic_response_time = Column(Float)
    classic_chunks_used = Column(Integer)
    classic_sources = Column(Text)  # JSON string of source chunks
    
    # MemVid RAG results
    memvid_answer = Column(Text)
    memvid_response_time = Column(Float)
    memvid_chunks_used = Column(Integer)
    memvid_sources = Column(Text)  # JSON string of source chunks
    memvid_metadata = Column(Text)  # JSON string of MemVid-specific metadata
    
    # Relationships
    user = relationship("User", back_populates="query_history")
    
    def __repr__(self):
        return f"<QueryHistory(id={self.id}, user_id={self.user_id}, query='{self.query_text[:50]}...')>"


class SystemLog(Base):
    """System log model for tracking application events"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    component = Column(String(100), nullable=False)  # auth, upload, classic_rag, memvid_rag, etc.
    message = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    additional_data = Column(Text)  # JSON string for additional context
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level='{self.level}', component='{self.component}')>"