"""
Performance monitoring service for RAG pipelines
"""
import time
import logging
import json
import csv
import io
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from statistics import mean, median, stdev

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from models.database import QueryHistory, User
from utils.database import get_db_context

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single query"""
    response_time: float
    chunks_used: int
    query_length: int
    answer_length: int
    timestamp: datetime
    pipeline_type: str


@dataclass
class ComparisonMetrics:
    """Comparison metrics between two RAG pipelines"""
    query: str
    classic_metrics: Optional[PerformanceMetrics]
    memvid_metrics: Optional[PerformanceMetrics]
    performance_difference: float  # Positive if MemVid is faster
    chunks_difference: int  # Difference in chunks used
    answer_length_difference: int  # Difference in answer lengths
    timestamp: datetime


@dataclass
class PerformanceStats:
    """Aggregated performance statistics"""
    total_queries: int
    avg_response_time: float
    median_response_time: float
    min_response_time: float
    max_response_time: float
    std_response_time: float
    avg_chunks_used: float
    total_processing_time: float


@dataclass
class PerformanceSummary:
    """Performance summary with insights and recommendations"""
    period: str
    total_queries: int
    insights: List[str]
    recommendations: List[str]
    classic_stats: Optional[PerformanceStats]
    memvid_stats: Optional[PerformanceStats]
    comparison_stats: Optional[Dict[str, Any]]


@dataclass
class SessionPerformanceData:
    """Performance data for a user session"""
    session_id: str
    user_id: int
    queries: List[ComparisonMetrics]
    session_start: datetime
    session_end: Optional[datetime]
    total_queries: int
    avg_response_time_classic: float
    avg_response_time_memvid: float
    performance_trend: str  # "improving", "declining", "stable"


class PerformanceMonitoringService:
    """Service for monitoring and analyzing RAG pipeline performance"""
    
    def __init__(self):
        self.session_cache = {}  # Cache for session-based metrics
        self.max_cache_size = 1000
    
    def get_user_performance_stats(
        self, 
        user_id: int, 
        pipeline_type: str = None,
        days: int = 30
    ) -> Dict[str, Any]: 
        """
        Get performance statistics for a user
        
        Args:
            user_id: User ID
            pipeline_type: Optional pipeline type filter ('classic' or 'memvid')
            days: Number of days to look back
            
        Returns:
            Dict containing performance statistics
        """
        try:
            with get_db_context() as db:
                # Calculate date range
                start_date = datetime.now() - timedelta(days=days)
                
                # Base query
                query = db.query(QueryHistory).filter(
                    QueryHistory.user_id == user_id,
                    QueryHistory.query_timestamp >= start_date
                )
                
                query_history = query.all()
                
                if not query_history:
                    return {
                        "total_queries": 0,
                        "classic_stats": None,
                        "memvid_stats": None,
                        "comparison_stats": None
                    }
                
                # Separate classic and memvid results
                classic_results = []
                memvid_results = []
                comparison_results = []
                
                for entry in query_history:
                    if entry.classic_response_time is not None:
                        classic_results.append({
                            "response_time": entry.classic_response_time,
                            "chunks_used": entry.classic_chunks_used or 0,
                            "query_length": len(entry.query_text),
                            "answer_length": len(entry.classic_answer or ""),
                            "timestamp": entry.query_timestamp
                        })
                    
                    if entry.memvid_response_time is not None:
                        memvid_results.append({
                            "response_time": entry.memvid_response_time,
                            "chunks_used": entry.memvid_chunks_used or 0,
                            "query_length": len(entry.query_text),
                            "answer_length": len(entry.memvid_answer or ""),
                            "timestamp": entry.query_timestamp
                        })
                    
                    # If both pipelines have results, create comparison
                    if (entry.classic_response_time is not None and 
                        entry.memvid_response_time is not None):
                        comparison_results.append({
                            "query": entry.query_text,
                            "classic_time": entry.classic_response_time,
                            "memvid_time": entry.memvid_response_time,
                            "time_difference": entry.memvid_response_time - entry.classic_response_time,
                            "classic_chunks": entry.classic_chunks_used or 0,
                            "memvid_chunks": entry.memvid_chunks_used or 0,
                            "chunks_difference": (entry.memvid_chunks_used or 0) - (entry.classic_chunks_used or 0),
                            "timestamp": entry.query_timestamp
                        })
                
                # Calculate statistics
                classic_stats = self._calculate_pipeline_stats(classic_results) if classic_results else None
                memvid_stats = self._calculate_pipeline_stats(memvid_results) if memvid_results else None
                comparison_stats = self._calculate_comparison_stats(comparison_results) if comparison_results else None
                
                return {
                    "total_queries": len(query_history),
                    "date_range": {
                        "start": start_date.isoformat(),
                        "end": datetime.now().isoformat(),
                        "days": days
                    },
                    "classic_stats": classic_stats,
                    "memvid_stats": memvid_stats,
                    "comparison_stats": comparison_stats
                }
                
        except Exception as e:
            logger.error(f"Error getting user performance stats: {e}")
            raise
    
    def _calculate_pipeline_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics for a single pipeline"""
        if not results:
            return None
        
        response_times = [r["response_time"] for r in results]
        chunks_used = [r["chunks_used"] for r in results]
        
        return {
            "total_queries": len(results),
            "avg_response_time": mean(response_times),
            "median_response_time": median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "std_response_time": stdev(response_times) if len(response_times) > 1 else 0,
            "avg_chunks_used": mean(chunks_used),
            "total_processing_time": sum(response_times)
        }
    
    def _calculate_comparison_stats(self, comparisons: List[Dict]) -> Dict[str, Any]:
        """Calculate comparison statistics between pipelines"""
        if not comparisons:
            return None
        
        time_differences = [c["time_difference"] for c in comparisons]
        chunks_differences = [c["chunks_difference"] for c in comparisons]
        
        classic_faster_count = sum(1 for diff in time_differences if diff > 0)
        memvid_faster_count = sum(1 for diff in time_differences if diff < 0)
        
        return {
            "total_comparisons": len(comparisons),
            "classic_faster_count": classic_faster_count,
            "memvid_faster_count": memvid_faster_count,
            "avg_time_difference": mean(time_differences),
            "avg_chunks_difference": mean(chunks_differences),
            "classic_win_rate": classic_faster_count / len(comparisons) * 100,
            "memvid_win_rate": memvid_faster_count / len(comparisons) * 100,
            "performance_trend": self._calculate_performance_trend(time_differences)
        }
    
    def _calculate_performance_trend(self, time_differences: List[float]) -> str:
        """Calculate performance trend over time"""
        if len(time_differences) < 3:
            return "insufficient_data"
        
        # Split into first and second half
        mid_point = len(time_differences) // 2
        first_half = time_differences[:mid_point]
        second_half = time_differences[mid_point:]
        
        first_avg = mean(first_half)
        second_avg = mean(second_half)
        
        # If MemVid is getting relatively faster (more negative differences)
        if second_avg < first_avg - 0.1:
            return "memvid_improving"
        elif second_avg > first_avg + 0.1:
            return "classic_improving"
        else:
            return "stable"
    
    def get_session_performance_data(
        self, 
        user_id: int, 
        session_id: str = None
    ) -> SessionPerformanceData:
        """
        Get performance data for a specific session
        
        Args:
            user_id: User ID
            session_id: Optional session ID (if None, uses current session)
            
        Returns:
            SessionPerformanceData object
        """
        try:
            # For now, we'll use a simple session based on recent queries
            # In a real implementation, you'd track actual sessions
            session_id = session_id or f"session_{user_id}_{datetime.now().strftime('%Y%m%d')}"
            
            with get_db_context() as db:
                # Get recent queries (last 2 hours as a session)
                session_start = datetime.now() - timedelta(hours=2)
                
                query_history = db.query(QueryHistory).filter(
                    QueryHistory.user_id == user_id,
                    QueryHistory.query_timestamp >= session_start
                ).order_by(QueryHistory.query_timestamp.desc()).all()
                
                queries = []
                classic_times = []
                memvid_times = []
                
                for entry in query_history:
                    if (entry.classic_response_time is not None and 
                        entry.memvid_response_time is not None):
                        
                        classic_metrics = PerformanceMetrics(
                            response_time=entry.classic_response_time,
                            chunks_used=entry.classic_chunks_used or 0,
                            query_length=len(entry.query_text),
                            answer_length=len(entry.classic_answer or ""),
                            timestamp=entry.query_timestamp,
                            pipeline_type="classic"
                        )
                        
                        memvid_metrics = PerformanceMetrics(
                            response_time=entry.memvid_response_time,
                            chunks_used=entry.memvid_chunks_used or 0,
                            query_length=len(entry.query_text),
                            answer_length=len(entry.memvid_answer or ""),
                            timestamp=entry.query_timestamp,
                            pipeline_type="memvid"
                        )
                        
                        comparison = ComparisonMetrics(
                            query=entry.query_text,
                            classic_metrics=classic_metrics,
                            memvid_metrics=memvid_metrics,
                            performance_difference=entry.memvid_response_time - entry.classic_response_time,
                            chunks_difference=(entry.memvid_chunks_used or 0) - (entry.classic_chunks_used or 0),
                            answer_length_difference=len(entry.memvid_answer or "") - len(entry.classic_answer or ""),
                            timestamp=entry.query_timestamp
                        )
                        
                        queries.append(comparison)
                        classic_times.append(entry.classic_response_time)
                        memvid_times.append(entry.memvid_response_time)
                
                # Calculate trend
                time_differences = [q.performance_difference for q in queries]
                trend = self._calculate_performance_trend(time_differences) if time_differences else "no_data"
                
                return SessionPerformanceData(
                    session_id=session_id,
                    user_id=user_id,
                    queries=queries,
                    session_start=session_start,
                    session_end=datetime.now() if queries else None,
                    total_queries=len(queries),
                    avg_response_time_classic=mean(classic_times) if classic_times else 0,
                    avg_response_time_memvid=mean(memvid_times) if memvid_times else 0,
                    performance_trend=trend
                )
                
        except Exception as e:
            logger.error(f"Error getting session performance data: {e}")
            raise
    
    def export_performance_data(
        self, 
        user_id: int, 
        format_type: str = "csv",
        days: int = 30
    ) -> str:
        """
        Export performance data in specified format
        
        Args:
            user_id: User ID
            format_type: Export format ('csv' or 'json')
            days: Number of days to export
            
        Returns:
            Exported data as string
        """
        try:
            with get_db_context() as db:
                start_date = datetime.now() - timedelta(days=days)
                
                query_history = db.query(QueryHistory).filter(
                    QueryHistory.user_id == user_id,
                    QueryHistory.query_timestamp >= start_date
                ).order_by(QueryHistory.query_timestamp.desc()).all()
                
                export_data = []
                
                for entry in query_history:
                    row = {
                        "timestamp": entry.query_timestamp.isoformat(),
                        "query": entry.query_text,
                        "query_length": len(entry.query_text),
                        "classic_response_time": entry.classic_response_time,
                        "classic_chunks_used": entry.classic_chunks_used,
                        "classic_answer_length": len(entry.classic_answer or ""),
                        "memvid_response_time": entry.memvid_response_time,
                        "memvid_chunks_used": entry.memvid_chunks_used,
                        "memvid_answer_length": len(entry.memvid_answer or ""),
                        "time_difference": (
                            entry.memvid_response_time - entry.classic_response_time
                            if entry.classic_response_time and entry.memvid_response_time
                            else None
                        ),
                        "chunks_difference": (
                            (entry.memvid_chunks_used or 0) - (entry.classic_chunks_used or 0)
                            if entry.classic_chunks_used is not None and entry.memvid_chunks_used is not None
                            else None
                        )
                    }
                    export_data.append(row)
                
                if format_type.lower() == "csv":
                    return self._export_to_csv(export_data)
                elif format_type.lower() == "json":
                    return json.dumps(export_data, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
                    
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            raise
    
    def _export_to_csv(self, data: List[Dict]) -> str:
        """Export data to CSV format"""
        if not data:
            return ""
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()
    
    def get_performance_summary(self, user_id: int, days: int = 7) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary
        
        Args:
            user_id: User ID
            days: Number of days to analyze
            
        Returns:
            Performance summary with key metrics and insights
        """
        try:
            stats = self.get_user_performance_stats(user_id, days=days)
            
            summary = {
                "period": f"Last {days} days",
                "total_queries": stats["total_queries"],
                "insights": [],
                "recommendations": []
            }
            
            if stats["comparison_stats"]:
                comp_stats = stats["comparison_stats"]
                
                # Performance insights
                if comp_stats["memvid_win_rate"] > 60:
                    summary["insights"].append("MemVid RAG consistently outperforms Classic RAG")
                    summary["recommendations"].append("Consider using MemVid RAG as your primary pipeline")
                elif comp_stats["classic_win_rate"] > 60:
                    summary["insights"].append("Classic RAG consistently outperforms MemVid RAG")
                    summary["recommendations"].append("Classic RAG may be better suited for your query patterns")
                else:
                    summary["insights"].append("Both pipelines show similar performance")
                    summary["recommendations"].append("Consider query-specific pipeline selection")
                
                # Trend insights
                trend = comp_stats["performance_trend"]
                if trend == "memvid_improving":
                    summary["insights"].append("MemVid RAG performance is improving over time")
                elif trend == "classic_improving":
                    summary["insights"].append("Classic RAG performance is improving over time")
                
                # Efficiency insights
                avg_time_diff = comp_stats["avg_time_difference"]
                if abs(avg_time_diff) > 1.0:
                    faster_pipeline = "Classic" if avg_time_diff > 0 else "MemVid"
                    summary["insights"].append(f"{faster_pipeline} RAG is significantly faster on average")
            
            # Usage patterns
            if stats["total_queries"] > 50:
                summary["insights"].append("High query volume - consider performance optimization")
            elif stats["total_queries"] < 5:
                summary["insights"].append("Low query volume - more data needed for reliable insights")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            raise


# Global instance
performance_service = PerformanceMonitoringService()