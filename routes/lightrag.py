"""
LightRAG routes for graph-based RAG functionality
"""
import logging
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from models.schemas import QueryRequest, RAGResponse
from models.database import User
from services.lightrag_service import (
    query_with_lightrag, 
    get_lightrag_stats, 
    delete_lightrag_document,
    LIGHTRAG_AVAILABLE
)
from middleware.auth_middleware import get_current_active_user
# Request logging is handled by middleware
from utils.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lightrag", tags=["LightRAG"])

@router.get("/health")
async def health_check():
    """Health check for LightRAG service"""
    return {
        "service": "LightRAG Graph-based RAG Pipeline",
        "status": "healthy" if LIGHTRAG_AVAILABLE else "unavailable",
        "lightrag_available": LIGHTRAG_AVAILABLE,
        "timestamp": datetime.now(),
        "features": [
            "Graph-based retrieval",
            "Entity-relationship extraction", 
            "Multiple query modes",
            "Knowledge graph construction"
        ]
    }

@router.post("/query", response_model=RAGResponse)
async def query_lightrag(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Execute LightRAG query with graph-based retrieval
    Supports multiple query modes:
    - local: Context-dependent information
    - global: Global knowledge utilization  
    - hybrid: Combined local and global retrieval
    - naive: Basic search without advanced techniques
    """
    if not LIGHTRAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="LightRAG service not available")
    
    # Request logging is handled by middleware
    logger.info(f"LightRAG query request from user {current_user.id}")
    
    try:
        # Add mode to query request if not present
        if not hasattr(query_request, 'mode'):
            query_request.mode = 'hybrid'  # Default to hybrid mode
        
        result = await query_with_lightrag(current_user.id, query_request, current_user)
        logger.info(f"LightRAG query completed for user {current_user.id} in {result.response_time:.2f}s")
        return result
    except Exception as e:
        logger.error(f"LightRAG query failed for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/stats")
async def get_knowledge_graph_stats(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get knowledge graph statistics for the current user"""
    if not LIGHTRAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="LightRAG service not available")
    
    try:
        stats = await get_lightrag_stats(current_user.id)
        return {
            "user_id": current_user.id,
            "knowledge_graph": stats,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Failed to get LightRAG stats for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.delete("/documents/{document_id}")
async def delete_document_from_graph(
    document_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete document from knowledge graph"""
    if not LIGHTRAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="LightRAG service not available")
    
    try:
        success = await delete_lightrag_document(current_user.id, document_id)
        if success:
            return {
                "message": f"Document {document_id} deleted from knowledge graph",
                "document_id": document_id,
                "timestamp": datetime.now()
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found or deletion not supported")
    except Exception as e:
        logger.error(f"Failed to delete document {document_id} from LightRAG: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/modes")
async def get_query_modes():
    """Get available LightRAG query modes and their descriptions"""
    return {
        "modes": {
            "local": {
                "name": "Local Mode",
                "description": "Focuses on context-dependent information and local relationships",
                "use_case": "When you need specific, contextual answers about particular entities or events"
            },
            "global": {
                "name": "Global Mode", 
                "description": "Utilizes global knowledge and broad relationships across the entire graph",
                "use_case": "When you need comprehensive, high-level insights or summaries"
            },
            "hybrid": {
                "name": "Hybrid Mode",
                "description": "Combines local and global retrieval methods for balanced results",
                "use_case": "Default mode that provides both specific and comprehensive information"
            },
            "naive": {
                "name": "Naive Mode",
                "description": "Performs basic search without advanced graph techniques",
                "use_case": "Simple keyword-based search similar to traditional RAG"
            }
        },
        "default_mode": "hybrid",
        "recommendation": "Use 'hybrid' for most queries, 'local' for specific details, 'global' for overviews"
    }

@router.post("/query/local", response_model=RAGResponse)
async def query_local_mode(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Execute LightRAG query in local mode (context-dependent)"""
    query_request.mode = 'local'
    return await query_lightrag(query_request, current_user)

@router.post("/query/global", response_model=RAGResponse)
async def query_global_mode(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Execute LightRAG query in global mode (global knowledge)"""
    query_request.mode = 'global'
    return await query_lightrag(query_request, current_user)

@router.post("/query/hybrid", response_model=RAGResponse)
async def query_hybrid_mode(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Execute LightRAG query in hybrid mode (combined approach)"""
    query_request.mode = 'hybrid'
    return await query_lightrag(query_request, current_user)

@router.post("/query/naive", response_model=RAGResponse)
async def query_naive_mode(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Execute LightRAG query in naive mode (basic search)"""
    query_request.mode = 'naive'
    return await query_lightrag(query_request, current_user)