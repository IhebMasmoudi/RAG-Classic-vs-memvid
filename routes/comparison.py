"""
RAG Comparison routes for comparing different RAG implementations
"""
import logging
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.schemas import QueryRequest
from models.database import User
from services.rag_comparison_service import compare_rag_methods, get_rag_capabilities
from middleware.auth_middleware import get_current_active_user
# Request logging is handled by middleware
from utils.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/comparison", tags=["RAG Comparison"])

@router.get("/health")
async def health_check():
    """Health check for RAG comparison service"""
    return {
        "service": "RAG Comparison Service",
        "status": "healthy",
        "timestamp": datetime.now(),
        "available_methods": ["classic_rag", "memvid_rag", "lightrag"]
    }

@router.post("/compare")
async def compare_all_rag_methods(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Compare all available RAG methods with the same query
    This endpoint runs the query through:
    - Classic RAG (vector similarity search)
    - MemVid RAG (memory-enhanced hierarchical retrieval)  
    - LightRAG (graph-based retrieval with entity relationships)
    Returns detailed comparison including performance metrics and analysis.
    """
    # Request logging is handled by middleware
    logger.info(f"RAG comparison request from user {current_user.id}: {query_request.query[:100]}...")
    
    try:
        result = await compare_rag_methods(current_user, query_request)
        logger.info(f"RAG comparison completed for user {current_user.id} in {result['total_comparison_time']:.2f}s")
        return result
    except Exception as e:
        logger.error(f"RAG comparison failed for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.get("/capabilities")
async def get_method_capabilities() -> Dict[str, Any]:
    """
    Get detailed information about each RAG method's capabilities
    Returns information about:
    - Method descriptions and strengths
    - Recommended use cases
    - Available features and modes
    - Performance characteristics
    """
    try:
        capabilities = await get_rag_capabilities()
        return {
            "timestamp": datetime.now(),
            "methods": capabilities,
            "comparison_features": {
                "performance_metrics": ["response_time", "chunks_used", "sources_count"],
                "quality_metrics": ["response_length", "source_diversity", "context_relevance"],
                "analysis_features": ["fastest_method", "most_comprehensive", "best_for_use_case"]
            }
        }
    except Exception as e:
        logger.error(f"Failed to get RAG capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")

@router.get("/recommendations")
async def get_method_recommendations():
    """Get recommendations for when to use each RAG method"""
    return {
        "recommendations": {
            "classic_rag": {
                "best_for": [
                    "Quick factual lookups",
                    "Simple question-answering",
                    "When speed is priority",
                    "Straightforward document search"
                ],
                "avoid_when": [
                    "Complex reasoning required",
                    "Multi-step queries",
                    "Relationship discovery needed"
                ]
            },
            "memvid_rag": {
                "best_for": [
                    "Complex reasoning tasks",
                    "Multi-step problem solving", 
                    "Context-dependent queries",
                    "When memory/history matters"
                ],
                "avoid_when": [
                    "Simple fact lookup",
                    "Speed is critical",
                    "No context needed"
                ]
            },
            "lightrag": {
                "best_for": [
                    "Knowledge discovery",
                    "Relationship exploration",
                    "Comprehensive analysis",
                    "Entity-centric queries",
                    "Complex domain knowledge"
                ],
                "modes": {
                    "local": "Specific entity/event details",
                    "global": "High-level overviews and summaries", 
                    "hybrid": "Balanced approach (recommended default)",
                    "naive": "Simple keyword search"
                },
                "avoid_when": [
                    "Simple document retrieval",
                    "When graph construction overhead not justified"
                ]
            }
        },
        "decision_tree": {
            "simple_facts": "classic_rag",
            "complex_reasoning": "memvid_rag", 
            "knowledge_discovery": "lightrag",
            "relationship_queries": "lightrag",
            "speed_priority": "classic_rag",
            "comprehensive_analysis": "lightrag (global mode)"
        }
    }