"""
Classic RAG API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from models.schemas import QueryRequest, RAGResponse
from models.database import User
from middleware.auth_middleware import get_current_active_user
from services.classic_rag import process_classic_rag_query
from utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/classic_rag",
    tags=["Classic RAG"],
    responses={404: {"description": "Not found"}},
)


@router.post("/query", response_model=RAGResponse)
async def classic_rag_query(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Process a query using the Classic RAG pipeline
    
    This endpoint:
    1. Validates the user's authentication
    2. Generates embeddings for the query
    3. Retrieves similar document chunks from the vector store
    4. Assembles context for the LLM
    5. Generates an answer using the LLM
    6. Returns the response with sources and performance metrics
    
    Args:
        query_request: Query request with query text and parameters
        current_user: Authenticated user from JWT token
        db: Database session
        
    Returns:
        RAGResponse: Complete response with answer, sources, and metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"Classic RAG query request from user {current_user.id}")
        
        # Validate query
        if not query_request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Process query through Classic RAG pipeline
        response = await process_classic_rag_query(query_request, current_user)
        
        logger.info(f"Classic RAG query completed for user {current_user.id} in {response.response_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classic RAG query failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during query processing"
        )


@router.get("/health")
async def classic_rag_health():
    """
    Health check endpoint for Classic RAG service
    
    Returns:
        dict: Health status and service information
    """
    try:
        from services.llm_service import get_llm_info
        from services.vector_store import get_vector_store_stats
        
        # Get service information
        llm_info = get_llm_info()
        vector_stats = await get_vector_store_stats()
        
        return {
            "status": "healthy",
            "service": "Classic RAG Pipeline",
            "llm_service": llm_info,
            "vector_store": vector_stats,
            "features": [
                "Query embedding generation",
                "Vector similarity search",
                "Context assembly",
                "LLM response generation",
                "Response time measurement",
                "Query history logging"
            ]
        }
        
    except Exception as e:
        logger.error(f"Classic RAG health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classic RAG service is not healthy"
        )