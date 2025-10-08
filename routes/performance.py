"""
Performance monitoring API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from sqlalchemy.orm import Session
from typing import Optional
import logging

from models.database import User
from middleware.auth_middleware import get_current_active_user
from services.performance_service import performance_service
from utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/performance",
    tags=["Performance Monitoring"],
    responses={404: {"description": "Not found"}},
)


@router.get("/stats")
async def get_performance_stats(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    pipeline_type: Optional[str] = Query(default=None, regex="^(classic|memvid)$", description="Filter by pipeline type"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get performance statistics for the current user
    
    This endpoint provides comprehensive performance analytics including:
    - Response time statistics for both pipelines
    - Comparison metrics between Classic and MemVid RAG
    - Performance trends over time
    - Query volume and efficiency metrics
    
    Args:
        days: Number of days to analyze (1-365)
        pipeline_type: Optional filter for specific pipeline
        current_user: Authenticated user from JWT token
        db: Database session
        
    Returns:
        dict: Performance statistics and metrics
    """
    try:
        logger.info(f"Getting performance stats for user {current_user.id} ({days} days)")
        
        stats = performance_service.get_user_performance_stats(
            user_id=current_user.id,
            pipeline_type=pipeline_type,
            days=days
        )
        
        return {
            "status": "success",
            "user_id": current_user.id,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance stats for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance statistics"
        )


@router.get("/session")
async def get_session_performance(
    session_id: Optional[str] = Query(default=None, description="Session ID (optional)"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get performance data for a specific session
    
    This endpoint provides session-specific performance metrics including:
    - Query history for the session
    - Performance comparison between pipelines
    - Session trends and patterns
    
    Args:
        session_id: Optional session ID (defaults to current session)
        current_user: Authenticated user from JWT token
        
    Returns:
        dict: Session performance data
    """
    try:
        logger.info(f"Getting session performance for user {current_user.id}")
        
        session_data = performance_service.get_session_performance_data(
            user_id=current_user.id,
            session_id=session_id
        )
        
        # Convert dataclass to dict for JSON serialization
        return {
            "status": "success",
            "session_data": {
                "session_id": session_data.session_id,
                "user_id": session_data.user_id,
                "session_start": session_data.session_start.isoformat(),
                "session_end": session_data.session_end.isoformat() if session_data.session_end else None,
                "total_queries": session_data.total_queries,
                "avg_response_time_classic": session_data.avg_response_time_classic,
                "avg_response_time_memvid": session_data.avg_response_time_memvid,
                "performance_trend": session_data.performance_trend,
                "queries": [
                    {
                        "query": q.query,
                        "timestamp": q.timestamp.isoformat(),
                        "performance_difference": q.performance_difference,
                        "chunks_difference": q.chunks_difference,
                        "answer_length_difference": q.answer_length_difference,
                        "classic_metrics": {
                            "response_time": q.classic_metrics.response_time if q.classic_metrics else None,
                            "chunks_used": q.classic_metrics.chunks_used if q.classic_metrics else None,
                            "query_length": q.classic_metrics.query_length if q.classic_metrics else None,
                            "answer_length": q.classic_metrics.answer_length if q.classic_metrics else None
                        },
                        "memvid_metrics": {
                            "response_time": q.memvid_metrics.response_time if q.memvid_metrics else None,
                            "chunks_used": q.memvid_metrics.chunks_used if q.memvid_metrics else None,
                            "query_length": q.memvid_metrics.query_length if q.memvid_metrics else None,
                            "answer_length": q.memvid_metrics.answer_length if q.memvid_metrics else None
                        }
                    }
                    for q in session_data.queries
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get session performance for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session performance data"
        )


@router.get("/summary")
async def get_performance_summary(
    days: int = Query(default=7, ge=1, le=30, description="Number of days to analyze"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a performance summary with insights and recommendations
    
    This endpoint provides a high-level performance summary including:
    - Key performance insights
    - Recommendations for optimization
    - Trend analysis
    - Usage patterns
    
    Args:
        days: Number of days to analyze (1-30)
        current_user: Authenticated user from JWT token
        
    Returns:
        dict: Performance summary with insights
    """
    try:
        logger.info(f"Getting performance summary for user {current_user.id} ({days} days)")
        
        summary = performance_service.get_performance_summary(
            user_id=current_user.id,
            days=days
        )
        
        return {
            "status": "success",
            "user_id": current_user.id,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance summary for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance summary"
        )


@router.get("/export")
async def export_performance_data(
    format_type: str = Query(default="csv", regex="^(csv|json)$", description="Export format"),
    days: int = Query(default=30, ge=1, le=365, description="Number of days to export"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Export performance data in CSV or JSON format
    
    This endpoint allows users to export their performance data for external analysis.
    The exported data includes all query metrics, response times, and comparison data.
    
    Args:
        format_type: Export format ('csv' or 'json')
        days: Number of days to export (1-365)
        current_user: Authenticated user from JWT token
        
    Returns:
        Response: File download with performance data
    """
    try:
        logger.info(f"Exporting performance data for user {current_user.id} ({format_type}, {days} days)")
        
        exported_data = performance_service.export_performance_data(
            user_id=current_user.id,
            format_type=format_type,
            days=days
        )
        
        # Set appropriate headers for file download
        filename = f"performance_data_{current_user.id}_{days}days.{format_type}"
        media_type = "text/csv" if format_type == "csv" else "application/json"
        
        return Response(
            content=exported_data,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to export performance data for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export performance data"
        )


@router.get("/health")
async def performance_health():
    """
    Health check endpoint for performance monitoring service
    
    Returns:
        dict: Health status and service information
    """
    try:
        return {
            "status": "healthy",
            "service": "Performance Monitoring Service",
            "features": [
                "Response time tracking",
                "Pipeline comparison metrics",
                "Session performance analysis",
                "Performance trend analysis",
                "Data export functionality",
                "Automated insights and recommendations"
            ],
            "supported_formats": ["csv", "json"],
            "max_export_days": 365,
            "max_analysis_days": 365
        }
        
    except Exception as e:
        logger.error(f"Performance service health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Performance monitoring service is not healthy"
        )