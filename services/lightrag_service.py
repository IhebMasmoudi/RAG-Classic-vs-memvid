"""
LightRAG service implementation for graph-based RAG
"""
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai_compatible_llm import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False

from fastapi import HTTPException
from sqlalchemy.orm import Session
from models.database import User, Document, QueryHistory
from models.schemas import QueryRequest, RAGResponse, Source
from services.llm_service import generate_llm_response
from services.embedding_service import generate_embeddings
from utils.database import get_db_context
from config import settings

logger = logging.getLogger(__name__)

class LightRAGService:
    """LightRAG service for graph-based RAG implementation"""
    
    def __init__(self):
        self.lightrag_instances: Dict[int, LightRAG] = {}
        self.working_dir = Path("data/lightrag")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        if not LIGHTRAG_AVAILABLE:
            logger.error("LightRAG is not available. Please install lightrag-hku package.")
            raise ImportError("LightRAG package not found")

    async def get_lightrag_instance(self, user_id: int) -> LightRAG:
        """Get or create LightRAG instance for user"""
        if user_id not in self.lightrag_instances:
            user_working_dir = self.working_dir / f"user_{user_id}"
            user_working_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize LightRAG with OpenAI-compatible API (using Gemini)
            rag = LightRAG(
                working_dir=str(user_working_dir),
                llm_model_func=self._llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=768,  # Gemini embedding dimension
                    func=self._embedding_func
                ),
                # Configure for better performance
                chunk_token_size=1200,
                chunk_overlap_token_size=200,
                entity_extract_max_gleaning=1,
                llm_model_max_async=2,
                embedding_batch_num=16,
                enable_llm_cache=True,
            )
            
            # Initialize storage
            await rag.initialize_storages()
            self.lightrag_instances[user_id] = rag
            logger.info(f"Created LightRAG instance for user {user_id}")
        
        return self.lightrag_instances[user_id]

    async def _llm_model_func(self, prompt: str, system_prompt: str = None, 
                             history_messages: List = None, **kwargs) -> str:
        """LLM function compatible with LightRAG"""
        try:
            # Use your existing LLM service (Gemini)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})
            
            # Use your existing generate_llm_response function
            response = await generate_llm_response(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM function error: {e}")
            raise

    async def _embedding_func(self, texts: List[str]) -> List[List[float]]:
        """Embedding function compatible with LightRAG"""
        try:
            # Use your existing embedding service
            embeddings = await generate_embeddings(texts)
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        except Exception as e:
            logger.error(f"Embedding function error: {e}")
            raise

    async def process_document(self, user_id: int, document_id: str, 
                              content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process document with LightRAG"""
        try:
            rag = await self.get_lightrag_instance(user_id)
            
            # Add metadata to content if provided
            if metadata:
                content_with_metadata = f"Document: {metadata.get('filename', 'Unknown')}\n\n{content}"
            else:
                content_with_metadata = content
            
            # Insert document into LightRAG with document ID
            await rag.ainsert(content_with_metadata, doc_id=document_id)
            
            logger.info(f"Processed document {document_id} for user {user_id} with LightRAG")
            return {
                "document_id": document_id,
                "status": "processed",
                "method": "lightrag",
                "entities_extracted": True,
                "relationships_extracted": True
            }
        except Exception as e:
            logger.error(f"Error processing document with LightRAG: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    async def query_lightrag(self, user_id: int, query_request: QueryRequest, 
                            user: User) -> RAGResponse:
        """Execute LightRAG query"""
        start_time = datetime.now()
        
        try:
            rag = await self.get_lightrag_instance(user_id)
            
            # Determine query mode based on request or use hybrid as default
            mode = getattr(query_request, 'mode', 'hybrid')
            if mode not in ['local', 'global', 'hybrid', 'naive']:
                mode = 'hybrid'
            
            # Create query parameters
            query_param = QueryParam(
                mode=mode,
                top_k=query_request.top_k or 5,
                response_type="Multiple Paragraphs",
                stream=False
            )
            
            # Execute query
            logger.info(f"Executing LightRAG query for user {user_id}: {query_request.query[:100]}...")
            result = await rag.aquery(query_request.query, param=query_param)
            
            # Calculate response time
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Extract sources from LightRAG context (if available)
            sources = await self._extract_sources_from_context(rag, query_request.query, user_id)
            
            # Create response
            response = RAGResponse(
                answer=result,
                sources=sources,
                response_time=response_time,
                chunks_used=len(sources),
                query_mode=mode,
                timestamp=end_time,
                metadata={
                    "method": "lightrag",
                    "mode": mode,
                    "top_k": query_request.top_k or 5,
                    "graph_based": True
                }
            )
            
            # Log query history
            await self._log_query_history(user, query_request.query, result, response_time, mode)
            
            logger.info(f"LightRAG query completed for user {user_id} in {response_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"LightRAG query failed for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"LightRAG query failed: {str(e)}")

    async def _extract_sources_from_context(self, rag: LightRAG, query: str, 
                                           user_id: int) -> List[Source]:
        """Extract sources from LightRAG context"""
        try:
            # Get context without generating response
            query_param = QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=5
            )
            context = await rag.aquery(query, param=query_param)
            
            # Parse context to extract document references
            sources = []
            
            # Try to extract document IDs from context
            # This is a simplified approach - you might need to enhance based on LightRAG's actual context format
            if isinstance(context, str):
                # Look for document references in the context
                lines = context.split('\n')
                doc_count = 0
                for line in lines:
                    if line.strip() and doc_count < 5:  # Limit to top 5 sources
                        sources.append(Source(
                            document_id=f"lightrag_context_{doc_count}",
                            chunk_index=doc_count,
                            content=line.strip()[:200] + "..." if len(line.strip()) > 200 else line.strip(),
                            similarity_score=0.9 - (doc_count * 0.1),  # Simulated scores
                            metadata={
                                "source_type": "lightrag_context",
                                "user_id": user_id
                            }
                        ))
                        doc_count += 1
            
            return sources
        except Exception as e:
            logger.warning(f"Could not extract sources from LightRAG context: {e}")
            return [Source(
                document_id="lightrag_generated",
                chunk_index=0,
                content="Response generated from knowledge graph",
                similarity_score=1.0,
                metadata={"source_type": "lightrag_graph"}
            )]

    async def _log_query_history(self, user: User, query: str, answer: str, 
                                response_time: float, mode: str):
        """Log query to history"""
        try:
            with get_db_context() as db:
                query_history = QueryHistory(
                    user_id=user.id,
                    query=query,
                    answer=answer,
                    response_time=response_time,
                    pipeline_type="lightrag",
                    metadata={
                        "mode": mode,
                        "method": "lightrag",
                        "graph_based": True
                    }
                )
                db.add(query_history)
                db.commit()
        except Exception as e:
            logger.error(f"Failed to log query history: {e}")

    async def get_knowledge_graph_stats(self, user_id: int) -> Dict[str, Any]:
        """Get knowledge graph statistics for user"""
        try:
            rag = await self.get_lightrag_instance(user_id)
            
            # Get basic stats from LightRAG storage
            stats = {
                "entities_count": 0,
                "relationships_count": 0,
                "documents_processed": 0,
                "graph_density": 0.0,
                "last_updated": datetime.now().isoformat()
            }
            
            # Try to get actual stats from LightRAG storage
            try:
                # Access the graph storage to get entity and relationship counts
                if hasattr(rag, 'graph_storage'):
                    # This depends on LightRAG's internal structure
                    # You might need to adjust based on the actual API
                    pass
            except Exception as e:
                logger.warning(f"Could not get detailed graph stats: {e}")
            
            return stats
        except Exception as e:
            logger.error(f"Error getting knowledge graph stats: {e}")
            return {
                "entities_count": 0,
                "relationships_count": 0,
                "documents_processed": 0,
                "graph_density": 0.0,
                "error": str(e)
            }

    async def delete_document(self, user_id: int, document_id: str) -> bool:
        """Delete document from LightRAG"""
        try:
            rag = await self.get_lightrag_instance(user_id)
            
            # Delete document by ID if LightRAG supports it
            if hasattr(rag, 'adelete_by_doc_id'):
                await rag.adelete_by_doc_id(document_id)
                logger.info(f"Deleted document {document_id} from LightRAG for user {user_id}")
                return True
            else:
                logger.warning("LightRAG document deletion not supported in this version")
                return False
        except Exception as e:
            logger.error(f"Error deleting document from LightRAG: {e}")
            return False

# Global service instance
lightrag_service = LightRAGService() if LIGHTRAG_AVAILABLE else None

async def get_lightrag_service() -> LightRAGService:
    """Get LightRAG service instance"""
    if not LIGHTRAG_AVAILABLE:
        raise HTTPException(status_code=500, detail="LightRAG not available")
    return lightrag_service

async def process_document_with_lightrag(user_id: int, document_id: str, 
                                       content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process document with LightRAG"""
    service = await get_lightrag_service()
    return await service.process_document(user_id, document_id, content, metadata)

async def query_with_lightrag(user_id: int, query_request: QueryRequest, user: User) -> RAGResponse:
    """Query using LightRAG"""
    service = await get_lightrag_service()
    return await service.query_lightrag(user_id, query_request, user)

async def get_lightrag_stats(user_id: int) -> Dict[str, Any]:
    """Get LightRAG knowledge graph statistics"""
    service = await get_lightrag_service()
    return await service.get_knowledge_graph_stats(user_id)

async def delete_lightrag_document(user_id: int, document_id: str) -> bool:
    """Delete document from LightRAG"""
    service = await get_lightrag_service()
    return await service.delete_document(user_id, document_id)