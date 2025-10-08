"""
MemVid-inspired RAG pipeline service with enhanced chunking and hierarchical retrieval
"""
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from fastapi import HTTPException
from sqlalchemy.orm import Session

from models.schemas import MemVidQueryRequest, MemVidRAGResponse, SourceChunk
from models.database import User, QueryHistory
from services.vector_store import search_similar_chunks
from services.embedding_service import generate_query_embedding
from services.llm_service import LLMService
from utils.database import get_db_context
from config import settings

logger = logging.getLogger(__name__)


class MemVidRAGPipeline:
    """MemVid-inspired RAG pipeline with enhanced retrieval and memory features"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.max_context_length = 6000  # Larger context than Classic RAG
        self.chunk_separator = "\n\n---\n\n"
        self.memory_cache = {}  # Simple memory cache for query patterns
        self.max_cache_size = 100
    
    async def process_query(
        self,
        query_request: MemVidQueryRequest,
        user: User
    ) -> MemVidRAGResponse:
        """
        Process a query through the MemVid-inspired RAG pipeline
        
        Args:
            query_request: MemVid query request with enhanced parameters
            user: Authenticated user
            
        Returns:
            MemVidRAGResponse: Complete response with MemVid-specific metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing MemVid RAG query for user {user.id}: {query_request.query[:100]}...")
            
            # Step 1: Enhanced query processing with memory
            processed_query, query_context = await self._enhance_query_with_memory(
                query_request.query, user.id
            )
            
            # Step 2: Generate query embedding
            query_embedding = await self._generate_query_embedding(processed_query)
            
            # Step 3: Hierarchical retrieval with context window
            similar_chunks = await self._hierarchical_retrieval(
                query_embedding,
                query_request.top_k,
                query_request.context_window,
                user.id,
                query_request.document_ids
            )
            
            if not similar_chunks:
                logger.warning(f"No relevant chunks found for MemVid query: {query_request.query}")
                return MemVidRAGResponse(
                    answer="I couldn't find any relevant information in your documents to answer this question.",
                    sources=[],
                    response_time=time.time() - start_time,
                    chunks_used=0,
                    query=query_request.query,
                    timestamp=datetime.utcnow(),
                    memvid_metadata={
                        "enhanced_query": processed_query,
                        "query_context": query_context,
                        "retrieval_strategy": "hierarchical",
                        "context_window_used": query_request.context_window,
                        "memory_cache_hits": 0
                    }
                )
            
            # Step 4: Enhanced context assembly with hierarchical organization
            context, source_chunks, assembly_metadata = await self._assemble_hierarchical_context(
                similar_chunks, query_request.context_window
            )
            
            # Step 5: Generate answer with enhanced prompt
            answer = await self._generate_enhanced_answer(
                query_request.query, processed_query, context, query_context
            )
            
            # Step 6: Update memory cache
            await self._update_memory_cache(query_request.query, answer, source_chunks, user.id)
            
            # Step 7: Calculate response time
            response_time = time.time() - start_time
            
            # Step 8: Create MemVid response with enhanced metadata
            memvid_metadata = {
                "enhanced_query": processed_query,
                "query_context": query_context,
                "retrieval_strategy": "hierarchical",
                "context_window_used": query_request.context_window,
                "assembly_metadata": assembly_metadata,
                "memory_cache_size": len(self.memory_cache),
                "processing_stages": [
                    "query_enhancement",
                    "hierarchical_retrieval", 
                    "context_assembly",
                    "enhanced_generation",
                    "memory_update"
                ]
            }
            
            response = MemVidRAGResponse(
                answer=answer,
                sources=source_chunks,
                response_time=response_time,
                chunks_used=len(source_chunks),
                query=query_request.query,
                timestamp=datetime.utcnow(),
                memvid_metadata=memvid_metadata
            )
            
            # Step 9: Log query to history
            await self._log_query_history(user.id, query_request.query, response)
            
            logger.info(f"MemVid RAG query completed in {response_time:.2f}s with {len(source_chunks)} chunks")
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"MemVid RAG pipeline failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"MemVid RAG pipeline failed: {str(e)}"
            )
    
    async def _enhance_query_with_memory(
        self, 
        query: str, 
        user_id: int
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance query using conversation memory and context
        
        Args:
            query: Original query
            user_id: User ID for memory lookup
            
        Returns:
            Tuple of (enhanced_query, query_context)
        """
        try:
            # Get recent query history for context
            query_context = await self._get_query_context(user_id)
            
            # Check memory cache for similar queries
            cache_hits = self._check_memory_cache(query)
            
            # Enhance query based on context and memory
            enhanced_query = query
            if query_context.get("recent_topics"):
                # Add context from recent queries
                recent_topics = ", ".join(query_context["recent_topics"][:3])
                enhanced_query = f"Context: Recent topics discussed include {recent_topics}. Current query: {query}"
            
            context_metadata = {
                "recent_queries": query_context.get("recent_queries", []),
                "recent_topics": query_context.get("recent_topics", []),
                "cache_hits": len(cache_hits),
                "enhancement_applied": enhanced_query != query
            }
            
            return enhanced_query, context_metadata
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return query, {"error": str(e)}
    
    async def _get_query_context(self, user_id: int) -> Dict[str, Any]:
        """Get recent query context for the user"""
        try:
            with get_db_context() as db:
                recent_queries = db.query(QueryHistory).filter(
                    QueryHistory.user_id == user_id
                ).order_by(QueryHistory.query_timestamp.desc()).limit(5).all()
                
                context = {
                    "recent_queries": [q.query_text for q in recent_queries],
                    "recent_topics": self._extract_topics([q.query_text for q in recent_queries])
                }
                
                return context
                
        except Exception as e:
            logger.error(f"Failed to get query context: {e}")
            return {}
    
    def _extract_topics(self, queries: List[str]) -> List[str]:
        """Extract key topics from recent queries"""
        # Simple keyword extraction - could be enhanced with NLP
        topics = set()
        common_words = {"what", "how", "when", "where", "why", "is", "are", "the", "a", "an", "and", "or", "but"}
        
        for query in queries:
            words = query.lower().split()
            for word in words:
                if len(word) > 3 and word not in common_words:
                    topics.add(word)
        
        return list(topics)[:10]  # Return top 10 topics
    
    def _check_memory_cache(self, query: str) -> List[Dict[str, Any]]:
        """Check memory cache for similar queries"""
        cache_hits = []
        query_lower = query.lower()
        
        for cached_query, cached_data in self.memory_cache.items():
            # Simple similarity check - could be enhanced with embeddings
            if any(word in cached_query.lower() for word in query_lower.split() if len(word) > 3):
                cache_hits.append({
                    "cached_query": cached_query,
                    "timestamp": cached_data.get("timestamp"),
                    "answer_preview": cached_data.get("answer", "")[:100]
                })
        
        return cache_hits
    
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the enhanced query"""
        try:
            return await generate_query_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
    
    async def _hierarchical_retrieval(
        self,
        query_embedding: List[float],
        top_k: int,
        context_window: int,
        user_id: int,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hierarchical retrieval with context window expansion
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of primary results
            context_window: Context window size for expansion
            user_id: User ID for filtering
            
        Returns:
            List of retrieved chunks with hierarchical context
        """
        try:
            # Step 1: Primary retrieval
            primary_chunks = await search_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k,
                user_id=user_id,
                document_ids=document_ids
            )
            
            if not primary_chunks:
                return []
            
            # Step 2: Context window expansion
            expanded_chunks = []
            processed_chunk_ids = set()
            
            for chunk_data in primary_chunks:
                chunk_id = chunk_data['chunk_id']
                chunk_index = chunk_data['chunk_index']
                document_id = chunk_data['document_id']
                
                # Add primary chunk
                if chunk_id not in processed_chunk_ids:
                    chunk_data['retrieval_type'] = 'primary'
                    chunk_data['context_position'] = 0
                    expanded_chunks.append(chunk_data)
                    processed_chunk_ids.add(chunk_id)
                
                # Add context window chunks
                for offset in range(-context_window, context_window + 1):
                    if offset == 0:  # Skip the primary chunk
                        continue
                    
                    context_index = chunk_index + offset
                    if context_index < 0:  # Skip negative indices
                        continue
                    
                    # Get context chunk (this would need to be implemented in vector_store)
                    context_chunk = await self._get_chunk_by_index(
                        document_id, context_index, user_id
                    )
                    
                    if context_chunk and context_chunk['chunk_id'] not in processed_chunk_ids:
                        context_chunk['retrieval_type'] = 'context'
                        context_chunk['context_position'] = offset
                        context_chunk['similarity_score'] = chunk_data['similarity_score'] * 0.8  # Reduce score for context
                        expanded_chunks.append(context_chunk)
                        processed_chunk_ids.add(context_chunk['chunk_id'])
            
            # Sort by document order and similarity
            expanded_chunks.sort(key=lambda x: (
                x['document_id'], 
                x['chunk_index'], 
                -x['similarity_score']
            ))
            
            logger.debug(f"Hierarchical retrieval: {len(primary_chunks)} primary + {len(expanded_chunks) - len(primary_chunks)} context chunks")
            return expanded_chunks
            
        except Exception as e:
            logger.error(f"Hierarchical retrieval failed: {e}")
            raise HTTPException(status_code=500, detail="Hierarchical retrieval failed")
    
    async def _get_chunk_by_index(
        self, 
        document_id: str, 
        chunk_index: int, 
        user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get chunk by document ID and index
        This is a simplified implementation - would need proper database queries
        """
        try:
            # This would need to be implemented with proper database queries
            # For now, return None to indicate chunk not found
            return None
        except Exception as e:
            logger.error(f"Failed to get chunk by index: {e}")
            return None
    
    async def _assemble_hierarchical_context(
        self,
        chunks: List[Dict[str, Any]],
        context_window: int
    ) -> Tuple[str, List[SourceChunk], Dict[str, Any]]:
        """
        Assemble context with hierarchical organization
        
        Args:
            chunks: Retrieved chunks with metadata
            context_window: Context window size
            
        Returns:
            Tuple of (context_string, source_chunks, assembly_metadata)
        """
        try:
            # Group chunks by document and organize hierarchically
            document_groups = {}
            for chunk_data in chunks:
                doc_id = chunk_data['document_id']
                if doc_id not in document_groups:
                    document_groups[doc_id] = {
                        'primary': [],
                        'context': []
                    }
                
                retrieval_type = chunk_data.get('retrieval_type', 'primary')
                document_groups[doc_id][retrieval_type].append(chunk_data)
            
            # Assemble context with hierarchical structure
            context_parts = []
            source_chunks = []
            current_length = 0
            
            for doc_id, groups in document_groups.items():
                # Add primary chunks first
                for chunk_data in groups['primary']:
                    if current_length >= self.max_context_length:
                        break
                    
                    chunk_content = chunk_data['content']
                    if current_length + len(chunk_content) > self.max_context_length:
                        remaining_space = self.max_context_length - current_length
                        if remaining_space > 100:
                            chunk_content = chunk_content[:remaining_space] + "..."
                        else:
                            break
                    
                    context_parts.append(f"[PRIMARY] {chunk_content}")
                    current_length += len(chunk_content) + 20  # Account for prefix
                    
                    source_chunk = SourceChunk(
                        chunk_id=chunk_data['chunk_id'],
                        content=chunk_content,
                        similarity_score=chunk_data['similarity_score'],
                        document_id=chunk_data['document_id'],
                        chunk_index=chunk_data['chunk_index']
                    )
                    source_chunks.append(source_chunk)
                
                # Add context chunks
                for chunk_data in groups['context']:
                    if current_length >= self.max_context_length:
                        break
                    
                    chunk_content = chunk_data['content']
                    if current_length + len(chunk_content) > self.max_context_length:
                        break
                    
                    context_parts.append(f"[CONTEXT] {chunk_content}")
                    current_length += len(chunk_content) + 20
                    
                    source_chunk = SourceChunk(
                        chunk_id=chunk_data['chunk_id'],
                        content=chunk_content,
                        similarity_score=chunk_data['similarity_score'],
                        document_id=chunk_data['document_id'],
                        chunk_index=chunk_data['chunk_index']
                    )
                    source_chunks.append(source_chunk)
            
            context = self.chunk_separator.join(context_parts)
            
            assembly_metadata = {
                "total_documents": len(document_groups),
                "primary_chunks": sum(len(g['primary']) for g in document_groups.values()),
                "context_chunks": sum(len(g['context']) for g in document_groups.values()),
                "total_context_length": len(context),
                "context_utilization": len(context) / self.max_context_length
            }
            
            logger.debug(f"Assembled hierarchical context: {len(source_chunks)} chunks, {len(context)} characters")
            return context, source_chunks, assembly_metadata
            
        except Exception as e:
            logger.error(f"Failed to assemble hierarchical context: {e}")
            raise HTTPException(status_code=500, detail="Failed to assemble context")
    
    async def _generate_enhanced_answer(
        self, 
        original_query: str, 
        enhanced_query: str, 
        context: str, 
        query_context: Dict[str, Any]
    ) -> str:
        """Generate answer using enhanced MemVid prompt"""
        try:
            # Create enhanced prompt for MemVid RAG
            prompt = self._create_memvid_prompt(original_query, enhanced_query, context, query_context)
            
            # Generate answer using LLM service
            answer = await self.llm_service.generate_response(prompt)
            
            logger.debug(f"Generated MemVid answer with {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced answer: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate answer")
    
    def _create_memvid_prompt(
        self, 
        original_query: str, 
        enhanced_query: str, 
        context: str, 
        query_context: Dict[str, Any]
    ) -> str:
        """Create enhanced MemVid RAG prompt"""
        
        # Build context information
        context_info = ""
        if query_context.get("recent_topics"):
            context_info = f"\nRecent conversation topics: {', '.join(query_context['recent_topics'][:3])}"
        
        prompt = f"""You are an advanced AI assistant with enhanced memory and contextual understanding. 
You have access to a hierarchically organized knowledge base with both primary and contextual information.

{context_info}

Knowledge Base Context (PRIMARY chunks are most relevant, CONTEXT chunks provide additional background):
{context}

Original Question: {original_query}

Instructions:
1. Use both PRIMARY and CONTEXT information to provide a comprehensive answer
2. Consider the conversation history and recent topics when relevant
3. Clearly distinguish between information from the knowledge base and general knowledge
4. If the context provides partial information, acknowledge what's available and what might be missing
5. Provide a well-structured, informative response that builds on the hierarchical context

Answer:"""
        
        return prompt
    
    async def _update_memory_cache(
        self, 
        query: str, 
        answer: str, 
        source_chunks: List[SourceChunk], 
        user_id: int
    ):
        """Update memory cache with query results"""
        try:
            # Manage cache size
            if len(self.memory_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.memory_cache.keys(), 
                               key=lambda k: self.memory_cache[k].get("timestamp", 0))
                del self.memory_cache[oldest_key]
            
            # Add new entry
            self.memory_cache[query] = {
                "answer": answer,
                "source_count": len(source_chunks),
                "user_id": user_id,
                "timestamp": time.time()
            }
            
            logger.debug(f"Updated memory cache: {len(self.memory_cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to update memory cache: {e}")
            # Non-critical error, continue execution
    
    async def _log_query_history(
        self,
        user_id: int,
        query: str,
        response: MemVidRAGResponse
    ) -> None:
        """Log MemVid query and response to history"""
        try:
            with get_db_context() as db:
                # Check if there's an existing query history entry for this query
                existing_history = db.query(QueryHistory).filter(
                    QueryHistory.user_id == user_id,
                    QueryHistory.query_text == query
                ).order_by(QueryHistory.query_timestamp.desc()).first()
                
                if existing_history:
                    # Update existing entry with MemVid RAG results
                    existing_history.memvid_answer = response.answer
                    existing_history.memvid_response_time = response.response_time
                    existing_history.memvid_chunks_used = response.chunks_used
                    existing_history.memvid_sources = self._serialize_sources(response.sources)
                    existing_history.memvid_metadata = json.dumps(response.memvid_metadata)
                else:
                    # Create new query history entry
                    query_history = QueryHistory(
                        user_id=user_id,
                        query_text=query,
                        memvid_answer=response.answer,
                        memvid_response_time=response.response_time,
                        memvid_chunks_used=response.chunks_used,
                        memvid_sources=self._serialize_sources(response.sources),
                        memvid_metadata=json.dumps(response.memvid_metadata)
                    )
                    db.add(query_history)
                
                db.commit()
                logger.debug(f"Logged MemVid RAG query history for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to log query history: {e}")
            # Don't raise exception as this is not critical for the response
    
    def _serialize_sources(self, sources: List[SourceChunk]) -> str:
        """Serialize source chunks to JSON string"""
        try:
            sources_data = [
                {
                    "chunk_id": source.chunk_id,
                    "content": source.content[:200] + "..." if len(source.content) > 200 else source.content,
                    "similarity_score": source.similarity_score,
                    "document_id": source.document_id,
                    "chunk_index": source.chunk_index
                }
                for source in sources
            ]
            return json.dumps(sources_data)
        except Exception as e:
            logger.error(f"Failed to serialize sources: {e}")
            return "[]"
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        return {
            "cache_size": len(self.memory_cache),
            "max_cache_size": self.max_cache_size,
            "cache_utilization": len(self.memory_cache) / self.max_cache_size,
            "oldest_entry": min(
                (entry.get("timestamp", 0) for entry in self.memory_cache.values()),
                default=0
            ),
            "newest_entry": max(
                (entry.get("timestamp", 0) for entry in self.memory_cache.values()),
                default=0
            )
        }
    
    def clear_memory_cache(self):
        """Clear the memory cache"""
        self.memory_cache.clear()
        logger.info("Cleared MemVid memory cache")


# Global MemVid RAG pipeline instance
memvid_rag_pipeline = MemVidRAGPipeline()


async def process_memvid_rag_query(
    query_request: MemVidQueryRequest,
    user: User
) -> MemVidRAGResponse:
    """
    Main function to process MemVid RAG query
    
    Args:
        query_request: MemVid query request with enhanced parameters
        user: Authenticated user
        
    Returns:
        MemVidRAGResponse: Complete MemVid RAG response
    """
    return await memvid_rag_pipeline.process_query(query_request, user)


def get_memvid_memory_stats() -> Dict[str, Any]:
    """Get MemVid memory statistics"""
    return memvid_rag_pipeline.get_memory_stats()


def clear_memvid_memory() -> None:
    """Clear MemVid memory cache"""
    memvid_rag_pipeline.clear_memory_cache()