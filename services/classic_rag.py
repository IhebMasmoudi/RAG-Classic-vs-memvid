"""
Classic RAG pipeline service for traditional retrieval-augmented generation
"""
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session

from models.schemas import QueryRequest, RAGResponse, SourceChunk
from models.database import User, QueryHistory
from services.vector_store import search_similar_chunks
from services.embedding_service import generate_query_embedding
from services.llm_service import LLMService
from utils.database import get_db_context
from config import settings

logger = logging.getLogger(__name__)


class ClassicRAGPipeline:
    """Classic RAG pipeline implementation"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.max_context_length = 4000  # Maximum context length for LLM
        self.chunk_separator = "\n\n---\n\n"
    
    async def process_query(
        self,
        query_request: QueryRequest,
        user: User
    ) -> RAGResponse:
        """
        Process a query through the Classic RAG pipeline
        
        Args:
            query_request: Query request with query text and parameters
            user: Authenticated user
            
        Returns:
            RAGResponse: Complete response with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing Classic RAG query for user {user.id}: {query_request.query[:100]}...")
            
            # Step 1: Generate query embedding
            query_embedding = await self._generate_query_embedding(query_request.query)
            
            # Step 2: Retrieve similar chunks
            similar_chunks = await self._retrieve_similar_chunks(
                query_embedding,
                query_request.top_k,
                user.id,
                query_request.document_ids
            )
            
            if not similar_chunks:
                logger.warning(f"No relevant chunks found for query: {query_request.query}")
                return RAGResponse(
                    answer="I couldn't find any relevant information in your documents to answer this question.",
                    sources=[],
                    response_time=time.time() - start_time,
                    chunks_used=0,
                    query=query_request.query,
                    timestamp=datetime.utcnow()
                )
            
            # Step 3: Assemble context for LLM
            context, source_chunks = await self._assemble_context(similar_chunks)
            
            # Step 4: Generate answer using LLM
            answer = await self._generate_answer(query_request.query, context)
            
            # Step 5: Calculate response time
            response_time = time.time() - start_time
            
            # Step 6: Create response
            response = RAGResponse(
                answer=answer,
                sources=source_chunks,
                response_time=response_time,
                chunks_used=len(source_chunks),
                query=query_request.query,
                timestamp=datetime.utcnow()
            )
            
            # Step 7: Log query to history
            await self._log_query_history(user.id, query_request.query, response)
            
            logger.info(f"Classic RAG query completed in {response_time:.2f}s with {len(source_chunks)} chunks")
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Classic RAG pipeline failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Classic RAG pipeline failed: {str(e)}"
            )
    
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query"""
        try:
            return await generate_query_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
    
    async def _retrieve_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int,
        user_id: int,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve similar chunks from vector store"""
        try:
            similar_chunks = await search_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k,
                user_id=user_id,
                document_ids=document_ids
            )
            
            logger.debug(f"Retrieved {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar chunks: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve similar chunks")
    
    async def _assemble_context(
        self,
        similar_chunks: List[Dict[str, Any]]
    ) -> tuple[str, List[SourceChunk]]:
        """Assemble context from retrieved chunks"""
        try:
            context_parts = []
            source_chunks = []
            current_length = 0
            
            for chunk_data in similar_chunks:
                chunk_content = chunk_data['content']
                
                # Check if adding this chunk would exceed context limit
                if current_length + len(chunk_content) > self.max_context_length:
                    # Try to fit partial content
                    remaining_space = self.max_context_length - current_length
                    if remaining_space > 100:  # Only add if we have meaningful space
                        chunk_content = chunk_content[:remaining_space] + "..."
                    else:
                        break
                
                context_parts.append(chunk_content)
                current_length += len(chunk_content) + len(self.chunk_separator)
                
                # Create source chunk
                source_chunk = SourceChunk(
                    chunk_id=chunk_data['chunk_id'],
                    content=chunk_content,
                    similarity_score=chunk_data['similarity_score'],
                    document_id=chunk_data['document_id'],
                    chunk_index=chunk_data['chunk_index']
                )
                source_chunks.append(source_chunk)
            
            context = self.chunk_separator.join(context_parts)
            
            logger.debug(f"Assembled context with {len(source_chunks)} chunks, {len(context)} characters")
            return context, source_chunks
            
        except Exception as e:
            logger.error(f"Failed to assemble context: {e}")
            raise HTTPException(status_code=500, detail="Failed to assemble context")
    
    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        try:
            # Create prompt for LLM
            prompt = self._create_rag_prompt(query, context)
            
            # Generate answer using LLM service
            answer = await self.llm_service.generate_response(prompt)
            
            logger.debug(f"Generated answer with {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate answer")
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create RAG prompt for LLM"""
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. If the context doesn't contain 
enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    async def _log_query_history(
        self,
        user_id: int,
        query: str,
        response: RAGResponse
    ) -> None:
        """Log query and response to history"""
        try:
            with get_db_context() as db:
                # Check if there's an existing query history entry for this query
                existing_history = db.query(QueryHistory).filter(
                    QueryHistory.user_id == user_id,
                    QueryHistory.query_text == query
                ).order_by(QueryHistory.query_timestamp.desc()).first()
                
                if existing_history:
                    # Update existing entry with Classic RAG results
                    existing_history.classic_answer = response.answer
                    existing_history.classic_response_time = response.response_time
                    existing_history.classic_chunks_used = response.chunks_used
                    existing_history.classic_sources = self._serialize_sources(response.sources)
                else:
                    # Create new query history entry
                    query_history = QueryHistory(
                        user_id=user_id,
                        query_text=query,
                        classic_answer=response.answer,
                        classic_response_time=response.response_time,
                        classic_chunks_used=response.chunks_used,
                        classic_sources=self._serialize_sources(response.sources)
                    )
                    db.add(query_history)
                
                db.commit()
                logger.debug(f"Logged Classic RAG query history for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to log query history: {e}")
            # Don't raise exception as this is not critical for the response
    
    def _serialize_sources(self, sources: List[SourceChunk]) -> str:
        """Serialize source chunks to JSON string"""
        import json
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


# Global Classic RAG pipeline instance
classic_rag_pipeline = ClassicRAGPipeline()


async def process_classic_rag_query(
    query_request: QueryRequest,
    user: User
) -> RAGResponse:
    """
    Main function to process Classic RAG query
    
    Args:
        query_request: Query request with parameters
        user: Authenticated user
        
    Returns:
        RAGResponse: Complete RAG response
    """
    return await classic_rag_pipeline.process_query(query_request, user)