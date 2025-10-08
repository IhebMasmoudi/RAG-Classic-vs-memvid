"""
Document upload routes for RAG Comparison Platform
"""
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from models.database import User, Document
from models.schemas import UploadResponse, DocumentResponse, ErrorResponse
from services.document_service import process_document_upload, get_user_documents, delete_document
from services.vector_store import add_document_to_vector_store, remove_document_from_vector_store
from middleware.auth_middleware import get_current_user
from utils.database import get_db_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Document Upload"])
security = HTTPBearer()


@router.post(
    "/",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process document",
    description="Upload a PDF or text file for processing and embedding generation"
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or text file to upload"),
    current_user: User = Depends(get_current_user)
) -> UploadResponse:
    """
    Upload and process a document for RAG queries.
    
    - **file**: PDF or text file (max 50MB)
    - Returns document ID, processing statistics, and success message
    """
    try:
        logger.info(f"Document upload request from user {current_user.id}: {file.filename}")
        
        # Process the document
        upload_response = await process_document_upload(file, current_user.id)
        
        # Get the document and its chunks from database
        with get_db_context() as db:
            document = db.query(Document).filter(Document.id == upload_response.document_id).first()
            if not document:
                raise HTTPException(status_code=500, detail="Document not found after processing")
            
            chunks = document.chunks
            
            # Generate embeddings and add to vector store
            from services.document_service import document_processor
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await document_processor.generate_embeddings(chunk_texts)
            
            # Add to vector store
            vector_ids = await add_document_to_vector_store(
                document.id,
                embeddings,
                chunks
            )
            
            # Update chunk records with vector IDs
            for chunk, vector_id in zip(chunks, vector_ids):
                chunk.embedding_vector_id = vector_id
            db.commit()
        
        logger.info(f"Successfully processed document {upload_response.document_id}")
        return upload_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload and processing failed"
        )


@router.get(
    "/documents",
    response_model=List[DocumentResponse],
    summary="Get user documents",
    description="Retrieve all documents uploaded by the current user"
)
async def get_documents(
    current_user: User = Depends(get_current_user)
) -> List[DocumentResponse]:
    """
    Get all documents for the current user.
    
    Returns list of document metadata including processing status.
    """
    try:
        documents = await get_user_documents(current_user.id)
        return [
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                content_type=doc.content_type,
                upload_date=doc.upload_date,
                chunk_count=doc.chunk_count,
                embedding_model=doc.embedding_model
            )
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"Failed to get user documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Delete a document and all its associated data"
)
async def delete_user_document(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete a document and all its associated data.
    
    - **document_id**: ID of the document to delete
    - Removes document, chunks, and embeddings from vector store
    """
    try:
        # Remove from vector store first
        await remove_document_from_vector_store(document_id)
        
        # Delete from database
        success = await delete_document(document_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        logger.info(f"Successfully deleted document {document_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.get(
    "/stats",
    summary="Get upload statistics",
    description="Get statistics about uploaded documents and vector store"
)
async def get_upload_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get upload and processing statistics for the current user.
    """
    try:
        from services.vector_store import get_vector_store_stats
        
        # Get user document count
        documents = await get_user_documents(current_user.id)
        
        # Get vector store stats
        vector_stats = await get_vector_store_stats()
        
        user_stats = {
            'user_documents': len(documents),
            'total_chunks': sum(doc.chunk_count for doc in documents),
            'processing_status': {
                status: len([doc for doc in documents if doc.processing_status == status])
                for status in ['pending', 'processing', 'completed', 'failed']
            }
        }
        
        return {
            'user_stats': user_stats,
            'vector_store_stats': vector_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get upload stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


@router.post(
    "/test",
    summary="Test file upload",
    description="Test endpoint for file upload validation without processing"
)
async def test_upload(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Test file upload without processing.
    Useful for validating file format and size before actual upload.
    """
    try:
        from services.document_service import document_processor
        
        # Validate file
        document_processor.validate_file(file)
        
        # Get file info
        file_size = 0
        if hasattr(file, 'size'):
            file_size = file.size
        else:
            content = await file.read()
            file_size = len(content)
            await file.seek(0)
        
        return {
            'filename': file.filename,
            'content_type': file.content_type,
            'size': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2),
            'validation': 'passed',
            'message': 'File is valid for upload'
        }
        
    except HTTPException as e:
        return {
            'filename': file.filename,
            'content_type': file.content_type,
            'validation': 'failed',
            'error': e.detail
        }
    except Exception as e:
        logger.error(f"Test upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Test upload failed"
        )