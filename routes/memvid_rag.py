"""
MemVid-inspired RAG API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from models.schemas import MemVidQueryRequest, MemVidRAGResponse
from models.database import User
from middleware.auth_middleware import get_current_active_user
from services.memvid_rag import process_memvid_rag_query, get_memvid_memory_stats, clear_memvid_memory
from utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/memvid_rag",
    tags=["MemVid RAG"],
    responses={404: {"description": "Not found"}},
)


@router.post("/query", response_model=MemVidRAGResponse)
async def memvid_rag_query(
    query_request: MemVidQueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Process a query using the MemVid-inspired RAG pipeline
    
    This endpoint implements an enhanced RAG approach inspired by MemVid:
    1. Query enhancement with conversation memory
    2. Hierarchical retrieval with context windows
    3. Enhanced context assembly with document organization
    4. Memory-aware response generation
    5. Conversation history tracking
    
    Args:
        query_request: MemVid query request with enhanced parameters
        current_user: Authenticated user from JWT token
        db: Database session
        
    Returns:
        MemVidRAGResponse: Complete response with MemVid-specific metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"MemVid RAG query request from user {current_user.id}")
        
        # Validate query
        if not query_request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Validate parameters
        if query_request.context_window < 0 or query_request.context_window > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Context window must be between 0 and 10"
            )
        
        # Process query through MemVid RAG pipeline
        response = await process_memvid_rag_query(query_request, current_user)
        
        logger.info(f"MemVid RAG query completed for user {current_user.id} in {response.response_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MemVid RAG query failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during MemVid query processing"
        )


@router.get("/memory/stats")
async def get_memory_stats(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get MemVid memory cache statistics
    
    Returns information about the current state of the MemVid memory cache,
    including cache size, utilization, and entry timestamps.
    
    Args:
        current_user: Authenticated user from JWT token
        
    Returns:
        dict: Memory cache statistics
    """
    try:
        stats = get_memvid_memory_stats()
        
        return {
            "status": "success",
            "memory_stats": stats,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Failed to get MemVid memory stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve memory statistics"
        )


@router.post("/memory/clear")
async def clear_memory(
    current_user: User = Depends(get_current_active_user)
):
    """
    Clear the MemVid memory cache
    
    This endpoint clears the conversation memory cache used by the MemVid RAG pipeline.
    This can be useful for starting fresh conversations or managing memory usage.
    
    Args:
        current_user: Authenticated user from JWT token
        
    Returns:
        dict: Success message
    """
    try:
        clear_memvid_memory()
        
        logger.info(f"MemVid memory cache cleared by user {current_user.id}")
        
        return {
            "status": "success",
            "message": "MemVid memory cache cleared successfully",
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Failed to clear MemVid memory cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear memory cache"
        )


@router.get("/health")
async def memvid_rag_health():
    """
    Health check endpoint for MemVid RAG service
    
    Returns:
        dict: Health status and service information
    """
    try:
        from services.llm_service import get_llm_info
        from services.vector_store import get_vector_store_stats
        from services.embedding_service import get_embedding_info
        
        # Get service information
        llm_info = get_llm_info()
        vector_stats = await get_vector_store_stats()
        embedding_info = get_embedding_info()
        memory_stats = get_memvid_memory_stats()
        
        return {
            "status": "healthy",
            "service": "MemVid-Inspired RAG Pipeline",
            "llm_service": llm_info,
            "vector_store": vector_stats,
            "embedding_service": embedding_info,
            "memory_cache": memory_stats,
            "features": [
                "Query enhancement with conversation memory",
                "Hierarchical retrieval with context windows",
                "Enhanced context assembly",
                "Memory-aware response generation",
                "Conversation history tracking",
                "Performance optimization caching"
            ],
            "memvid_enhancements": [
                "Context window expansion",
                "Hierarchical document organization",
                "Query pattern memory",
                "Enhanced prompt engineering",
                "Conversation continuity"
            ]
        }
        
    except Exception as e:
        logger.error(f"MemVid RAG health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MemVid RAG service is not healthy"
        )


@router.get("/compare/{query_id}")
async def get_comparison_data(
    query_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get comparison data between Classic RAG and MemVid RAG for a specific query
    
    This endpoint retrieves the results from both RAG pipelines for comparison purposes.
    
    Args:
        query_id: Query ID to retrieve comparison data for
        current_user: Authenticated user from JWT token
        db: Database session
        
    Returns:
        dict: Comparison data between both RAG approaches
    """
    try:
        from models.database import QueryHistory
        
        # Get query history entry
        query_history = db.query(QueryHistory).filter(
            QueryHistory.id == query_id,
            QueryHistory.user_id == current_user.id
        ).first()
        
        if not query_history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found"
            )
        
        # Build comparison data
        comparison_data = {
            "query": query_history.query_text,
            "timestamp": query_history.query_timestamp,
            "classic_rag": {
                "answer": query_history.classic_answer,
                "response_time": query_history.classic_response_time,
                "chunks_used": query_history.classic_chunks_used,
                "sources": query_history.classic_sources
            } if query_history.classic_answer else None,
            "memvid_rag": {
                "answer": query_history.memvid_answer,
                "response_time": query_history.memvid_response_time,
                "chunks_used": query_history.memvid_chunks_used,
                "sources": query_history.memvid_sources,
                "metadata": query_history.memvid_metadata
            } if query_history.memvid_answer else None
        }
        
        # Calculate performance comparison if both results exist
        if comparison_data["classic_rag"] and comparison_data["memvid_rag"]:
            classic_time = comparison_data["classic_rag"]["response_time"] or 0
            memvid_time = comparison_data["memvid_rag"]["response_time"] or 0
            
            comparison_data["performance_comparison"] = {
                "response_time_difference": memvid_time - classic_time,
                "memvid_faster": memvid_time < classic_time,
                "speed_improvement_percent": ((classic_time - memvid_time) / classic_time * 100) if classic_time > 0 else 0,
                "chunks_difference": (comparison_data["memvid_rag"]["chunks_used"] or 0) - (comparison_data["classic_rag"]["chunks_used"] or 0)
            }
        
        return comparison_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get comparison data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve comparison data"
        )