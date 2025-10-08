"""
RAG Comparison Service for comparing different RAG implementations
"""
import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from models.schemas import QueryRequest, RAGResponse
from models.database import User
from services.classic_rag import ClassicRAGPipeline
from services.memvid_rag import MemVidRAGPipeline
from services.lightrag_service import query_with_lightrag, LIGHTRAG_AVAILABLE

logger = logging.getLogger(__name__)

class RAGComparisonService:
    """Service for comparing different RAG implementations"""
    
    def __init__(self):
        self.classic_rag = ClassicRAGPipeline()
        self.memvid_rag = MemVidRAGPipeline()

    async def compare_all_methods(self, user: User, query_request: QueryRequest) -> Dict[str, Any]:
        """Compare all available RAG methods"""
        start_time = datetime.now()
        results = {}
        
        try:
            # Run all methods concurrently
            tasks = []
            
            # Classic RAG
            tasks.append(self._run_classic_rag(user, query_request))
            
            # MemVid RAG  
            tasks.append(self._run_memvid_rag(user, query_request))
            
            # LightRAG (if available)
            if LIGHTRAG_AVAILABLE:
                tasks.append(self._run_lightrag(user, query_request))
            
            # Execute all tasks
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            method_names = ["classic_rag", "memvid_rag"]
            if LIGHTRAG_AVAILABLE:
                method_names.append("lightrag")
            
            for i, result in enumerate(task_results):
                method_name = method_names[i]
                if isinstance(result, Exception):
                    results[method_name] = {
                        "error": str(result),
                        "status": "failed"
                    }
                else:
                    results[method_name] = {
                        "response": result,
                        "status": "success"
                    }
            
            # Calculate total comparison time
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Generate comparison analysis
            analysis = self._analyze_results(results)
            
            return {
                "query": query_request.query,
                "timestamp": datetime.now(),
                "total_comparison_time": total_time,
                "methods_compared": len(results),
                "results": results,
                "analysis": analysis,
                "metadata": {
                    "user_id": user.id,
                    "top_k": query_request.top_k,
                    "document_ids": query_request.document_ids
                }
            }
        except Exception as e:
            logger.error(f"RAG comparison failed: {e}")
            raise

    async def _run_classic_rag(self, user: User, query_request: QueryRequest) -> RAGResponse:
        """Run Classic RAG"""
        try:
            return await self.classic_rag.process_query(query_request, user)
        except Exception as e:
            logger.error(f"Classic RAG failed: {e}")
            raise

    async def _run_memvid_rag(self, user: User, query_request: QueryRequest) -> RAGResponse:
        """Run MemVid RAG"""
        try:
            # Create MemVid-specific request
            from models.schemas import MemVidQueryRequest
            memvid_request = MemVidQueryRequest(
                query=query_request.query,
                context_window=3,  # Default context window
                top_k=query_request.top_k,
                document_ids=query_request.document_ids
            )
            return await self.memvid_rag.process_query(memvid_request, user)
        except Exception as e:
            logger.error(f"MemVid RAG failed: {e}")
            raise

    async def _run_lightrag(self, user: User, query_request: QueryRequest) -> RAGResponse:
        """Run LightRAG"""
        try:
            return await query_with_lightrag(user.id, query_request, user)
        except Exception as e:
            logger.error(f"LightRAG failed: {e}")
            raise

    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare results from different methods"""
        analysis = {
            "performance_comparison": {},
            "response_length_comparison": {},
            "sources_comparison": {},
            "recommendations": []
        }
        
        successful_results = {k: v for k, v in results.items() if v.get("status") == "success"}
        
        if not successful_results:
            return analysis
        
        # Performance comparison
        for method, result in successful_results.items():
            response = result["response"]
            analysis["performance_comparison"][method] = {
                "response_time": response.response_time,
                "chunks_used": response.chunks_used
            }
        
        # Response length comparison
        for method, result in successful_results.items():
            response = result["response"]
            analysis["response_length_comparison"][method] = len(response.answer)
        
        # Sources comparison
        for method, result in successful_results.items():
            response = result["response"]
            analysis["sources_comparison"][method] = len(response.sources)
        
        # Generate recommendations
        if len(successful_results) > 1:
            # Find fastest method
            fastest_method = min(
                successful_results.keys(),
                key=lambda x: successful_results[x]["response"]["response_time"]
            )
            analysis["recommendations"].append(f"Fastest response: {fastest_method}")
            
            # Find method with most sources
            most_sources_method = max(
                successful_results.keys(),
                key=lambda x: len(successful_results[x]["response"]["sources"])
            )
            analysis["recommendations"].append(f"Most comprehensive sources: {most_sources_method}")
            
            # Find longest response
            longest_response_method = max(
                successful_results.keys(),
                key=lambda x: len(successful_results[x]["response"]["answer"])
            )
            analysis["recommendations"].append(f"Most detailed response: {longest_response_method}")
        
        return analysis

    async def get_method_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of each RAG method"""
        return {
            "classic_rag": {
                "name": "Classic RAG",
                "description": "Traditional vector similarity search with embeddings",
                "strengths": ["Fast retrieval", "Simple implementation", "Good for factual queries"],
                "use_cases": ["Quick fact lookup", "Simple Q&A", "Document search"]
            },
            "memvid_rag": {
                "name": "MemVid RAG", 
                "description": "Memory-enhanced RAG with hierarchical retrieval",
                "strengths": ["Context awareness", "Memory mechanisms", "Hierarchical search"],
                "use_cases": ["Complex reasoning", "Multi-step queries", "Context-dependent answers"]
            },
            "lightrag": {
                "name": "LightRAG",
                "description": "Graph-based RAG with entity-relationship extraction",
                "available": LIGHTRAG_AVAILABLE,
                "strengths": ["Knowledge graphs", "Entity relationships", "Multiple query modes"],
                "use_cases": ["Complex knowledge queries", "Relationship discovery", "Comprehensive analysis"],
                "modes": ["local", "global", "hybrid", "naive"]
            }
        }

# Global service instance
rag_comparison_service = RAGComparisonService()

async def compare_rag_methods(user: User, query_request: QueryRequest) -> Dict[str, Any]:
    """Compare all available RAG methods"""
    return await rag_comparison_service.compare_all_methods(user, query_request)

async def get_rag_capabilities() -> Dict[str, Any]:
    """Get capabilities of all RAG methods"""
    return await rag_comparison_service.get_method_capabilities()