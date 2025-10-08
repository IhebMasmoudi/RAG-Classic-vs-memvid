"""
Document processing service for RAG Comparison Platform
Handles file upload, text extraction, chunking, and embedding generation
"""
import os
import hashlib
import logging
from typing import List, Tuple, Optional, BinaryIO, Dict, Any
from datetime import datetime
import aiofiles
import pypdf
from io import BytesIO
from pathlib import Path

from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
from sentence_transformers import SentenceTransformer

from models.database import Document, DocumentChunk, User
from models.schemas import UploadResponse
from utils.database import get_db_context
from services.embedding_service import generate_embeddings
from services.chunking_service import create_advanced_chunks, get_chunking_statistics
from config import settings

# Import LightRAG service with fallback
try:
    from services.lightrag_service import process_document_with_lightrag, LIGHTRAG_AVAILABLE
except ImportError:
    LIGHTRAG_AVAILABLE = False
    process_document_with_lightrag = None

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing operations"""
    
    def __init__(self):
        self.embedding_model = None
        self.supported_types = {
            'application/pdf': '.pdf',
            'text/plain': '.txt',
            'text/csv': '.csv'
        }
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    async def initialize_embedding_model(self):
        """Initialize the sentence transformer model"""
        if self.embedding_model is None:
            try:
                model_name = getattr(settings, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Initialized embedding model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise HTTPException(status_code=500, detail="Failed to initialize embedding model")
    
    def validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file"""
        # Check file size
        if hasattr(file, 'size') and file.size > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {self.max_file_size // (1024*1024)}MB"
            )
        
        # Check content type
        if file.content_type not in self.supported_types:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type. Supported types: {list(self.supported_types.keys())}"
            )
        
        # Check filename
        if not file.filename or len(file.filename) > 255:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename"
            )
    
    async def extract_text(self, file: UploadFile) -> str:
        """Extract text from uploaded file"""
        try:
            content = await file.read()
            
            if file.content_type == 'application/pdf':
                return await self._extract_pdf_text(content)
            elif file.content_type in ['text/plain', 'text/csv']:
                return await self._extract_text_content(content)
            else:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported content type: {file.content_type}"
                )
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract text from file")
    
    async def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = BytesIO(content)
            pdf_reader = pypdf.PdfReader(pdf_file)
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
            
            if not text_content:
                raise HTTPException(status_code=400, detail="No text content found in PDF")
            
            return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract text from PDF")
    
    async def _extract_text_content(self, content: bytes) -> str:
        """Extract text from plain text files"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    if text.strip():
                        return text
                except UnicodeDecodeError:
                    continue
            
            raise HTTPException(status_code=400, detail="Unable to decode text file")
        except Exception as e:
            logger.error(f"Text content extraction failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract text content")
    
    def create_chunks(self, text: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Create advanced chunks using LlamaIndex recursive text splitter"""
        if not text.strip():
            return []
        
        try:
            # Use advanced chunking service
            chunks = create_advanced_chunks(
                text=text,
                document_metadata=document_metadata,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Log chunking statistics
            stats = get_chunking_statistics(chunks)
            logger.info(f"Chunking statistics: {stats}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Advanced chunking failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to create text chunks")
    
    async def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks using the embedding service"""
        try:
            return await generate_embeddings(chunks)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def save_document_to_db(
        self,
        user_id: int,
        filename: str,
        content_type: str,
        file_size: int,
        chunks: List[Dict[str, Any]],
        db: Session
    ) -> Document:
        """Save document and chunks to database"""
        try:
            # Create document record
            document = Document(
                user_id=user_id,
                filename=filename,
                original_filename=filename,
                content_type=content_type,
                file_size=file_size,
                chunk_count=len(chunks),
                embedding_model=getattr(settings, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
                processing_status="processing"
            )
            
            db.add(document)
            db.flush()  # Get the document ID
            
            # Create chunk records
            chunk_objects = []
            for chunk_data in chunks:
                chunk_content = chunk_data['content']
                chunk_index = chunk_data['chunk_index']
                chunk_metadata = chunk_data.get('metadata', {})
                
                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    content_hash=self._generate_content_hash(chunk_content)
                )
                
                # Store additional metadata as JSON in a new field if needed
                # For now, we'll just use the existing fields
                chunk_objects.append(chunk)
            
            db.add_all(chunk_objects)
            db.commit()
            
            logger.info(f"Saved document {document.id} with {len(chunks)} chunks")
            return document
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save document to database: {e}")
            raise HTTPException(status_code=500, detail="Failed to save document")
    
    async def update_document_status(
        self,
        document_id: str,
        status: str,
        db: Session
    ) -> None:
        """Update document processing status"""
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.processing_status = status
                db.commit()
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")


# Global document processor instance
document_processor = DocumentProcessor()


async def process_document_upload(
    file: UploadFile,
    user_id: int
) -> UploadResponse:
    """
    Main function to process document upload
    """
    logger.info(f"Processing document upload: {file.filename} for user {user_id}")
    
    # Validate file
    document_processor.validate_file(file)
    
    # Get file size
    file_size = 0
    if hasattr(file, 'size'):
        file_size = file.size
    else:
        # Read content to get size
        content = await file.read()
        file_size = len(content)
        # Reset file pointer
        await file.seek(0)
    
    try:
        # Extract text
        text_content = await document_processor.extract_text(file)
        
        # Create chunks with metadata
        document_metadata = {
            'filename': file.filename,
            'content_type': file.content_type,
            'file_size': file_size,
            'user_id': user_id
        }
        
        chunks = document_processor.create_chunks(text_content, document_metadata)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content could be extracted from file")
        
        # Save to database
        with get_db_context() as db:
            document = await document_processor.save_document_to_db(
                user_id=user_id,
                filename=file.filename,
                content_type=file.content_type,
                file_size=file_size,
                chunks=chunks,
                db=db
            )
            
            # Extract chunk contents for embedding generation
            chunk_contents = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings (this will be handled by vector store service)
            embeddings = await document_processor.generate_embeddings(chunk_contents)
            
            # Store embeddings in vector store
            from services.vector_store import add_document_to_vector_store
            await add_document_to_vector_store(document.id, embeddings)
            
            # Also process with LightRAG if available
            if LIGHTRAG_AVAILABLE and process_document_with_lightrag:
                try:
                    await process_document_with_lightrag(
                        user_id=user_id,
                        document_id=document.id,
                        content=text_content,
                        metadata={
                            'filename': file.filename,
                            'content_type': file.content_type,
                            'file_size': file_size,
                            'upload_date': document.upload_date.isoformat()
                        }
                    )
                    logger.info(f"Document {document.id} also processed with LightRAG")
                except Exception as e:
                    logger.warning(f"LightRAG processing failed for document {document.id}: {e}")
            
            # Update status to completed
            await document_processor.update_document_status(
                document.id,
                "completed",
                db
            )
            
            return UploadResponse(
                document_id=document.id,
                filename=file.filename,
                chunks_created=len(chunks),
                embeddings_generated=len(embeddings),
                message="Document processed successfully"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail="Document processing failed")


async def get_user_documents(user_id: int) -> List[Document]:
    """Get all documents for a user"""
    with get_db_context() as db:
        documents = db.query(Document).filter(Document.user_id == user_id).all()
        # Create detached copies to avoid session issues
        detached_documents = []
        for doc in documents:
            # Create a new document instance with the same data
            detached_doc = Document(
                id=doc.id,
                user_id=doc.user_id,
                filename=doc.filename,
                original_filename=doc.original_filename,
                content_type=doc.content_type,
                file_size=doc.file_size,
                upload_date=doc.upload_date,
                chunk_count=doc.chunk_count,
                embedding_model=doc.embedding_model,
                processing_status=doc.processing_status
            )
            detached_documents.append(detached_doc)
        return detached_documents


async def delete_document(document_id: str, user_id: int) -> bool:
    """Delete a document and its chunks"""
    with get_db_context() as db:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
        
        if not document:
            return False
        
        db.delete(document)
        db.commit()
        return True