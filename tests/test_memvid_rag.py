"""
Unit tests for MemVid-inspired RAG pipeline
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from datetime import datetime

from services.memvid_rag import MemVidRAGPipeline, process_memvid_rag_query
from models.schemas import MemVidQueryRequest, MemVidRAGResponse, SourceChunk
from models.database import User


class TestMemVidRAGPipeline:
    """Test MemVid RAG pipeline functionality"""
    
    @pytest.fixture
    def mock_user(self):
        """Create a mock user for testing"""
        user = Mock(spec=User)
        user.id = 1
        user.email = "test@example.com"
        return user
    
    @pytest.fixture
    def memvid_query_request(self):
        """Create a test MemVid query request"""
        return MemVidQueryRequest(
            query="What is machine learning?",
            top_k=5,
            context_window=3
        )
    
    @pytest.fixture
    def mock_similar_chunks(self):
        """Create mock similar chunks data with MemVid metadata"""
        return [
            {
                'chunk_id': 'chunk_1',
                'document_id': 'doc_1',
                'chunk_index': 0,
                'content': 'Machine learning is a subset of artificial intelligence.',
                'similarity_score': 0.95,
                'retrieval_type': 'primary',
                'context_position': 0
            },
            {
                'chunk_id': 'chunk_2',
                'document_id': 'doc_1',
                'chunk_index': 1,
                'content': 'It involves training algorithms on data to make predictions.',
                'similarity_score': 0.87,
                'retrieval_type': 'context',
                'context_position': 1
            }
        ]
    
    @pytest.fixture
    def memvid_rag_pipeline(self):
        """Create MemVid RAG pipeline instance"""
        return MemVidRAGPipeline()
    
    @pytest.mark.asyncio
    async def test_process_query_success(
        self,
        memvid_rag_pipeline,
        memvid_query_request,
        mock_user,
        mock_similar_chunks
    ):
        """Test successful MemVid query processing"""
        with patch.object(memvid_rag_pipeline, '_enhance_query_with_memory') as mock_enhance, \
             patch.object(memvid_rag_pipeline, '_generate_query_embedding') as mock_embed, \
             patch.object(memvid_rag_pipeline, '_hierarchical_retrieval') as mock_retrieve, \
             patch.object(memvid_rag_pipeline, '_assemble_hierarchical_context') as mock_assemble, \
             patch.object(memvid_rag_pipeline, '_generate_enhanced_answer') as mock_answer, \
             patch.object(memvid_rag_pipeline, '_update_memory_cache') as mock_cache, \
             patch.object(memvid_rag_pipeline, '_log_query_history') as mock_log:
            
            # Setup mocks
            mock_enhance.return_value = ("Enhanced: What is machine learning?", {"recent_topics": ["AI"]})
            mock_embed.return_value = [0.1, 0.2, 0.3]
            mock_retrieve.return_value = mock_similar_chunks
            mock_assemble.return_value = (
                "Machine learning context",
                [SourceChunk(
                    chunk_id='chunk_1',
                    content='Machine learning is a subset of artificial intelligence.',
                    similarity_score=0.95,
                    document_id='doc_1',
                    chunk_index=0
                )],
                {"total_documents": 1, "primary_chunks": 1, "context_chunks": 1}
            )
            mock_answer.return_value = "Machine learning is a field of AI that uses algorithms to learn from data."
            mock_cache.return_value = None
            mock_log.return_value = None
            
            # Execute
            response = await memvid_rag_pipeline.process_query(memvid_query_request, mock_user)
            
            # Verify
            assert isinstance(response, MemVidRAGResponse)
            assert response.answer == "Machine learning is a field of AI that uses algorithms to learn from data."
            assert len(response.sources) == 1
            assert response.chunks_used == 1
            assert response.query == memvid_query_request.query
            assert response.response_time > 0
            assert "memvid_metadata" in response.__dict__
            assert response.memvid_metadata["enhanced_query"] == "Enhanced: What is machine learning?"
            
            # Verify method calls
            mock_enhance.assert_called_once_with(memvid_query_request.query, mock_user.id)
            mock_embed.assert_called_once_with("Enhanced: What is machine learning?")
            mock_retrieve.assert_called_once()
            mock_assemble.assert_called_once()
            mock_answer.assert_called_once()
            mock_cache.assert_called_once()
            mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enhance_query_with_memory(self, memvid_rag_pipeline):
        """Test query enhancement with memory"""
        with patch.object(memvid_rag_pipeline, '_get_query_context') as mock_context, \
             patch.object(memvid_rag_pipeline, '_check_memory_cache') as mock_cache:
            
            mock_context.return_value = {
                "recent_queries": ["What is AI?", "How does ML work?"],
                "recent_topics": ["artificial intelligence", "machine learning"]
            }
            mock_cache.return_value = []
            
            enhanced_query, context = await memvid_rag_pipeline._enhance_query_with_memory(
                "What is deep learning?", 1
            )
            
            assert "artificial intelligence" in enhanced_query or "machine learning" in enhanced_query
            assert context["recent_topics"] == ["artificial intelligence", "machine learning"]
            assert context["enhancement_applied"] is True
    
    @pytest.mark.asyncio
    async def test_get_query_context(self, memvid_rag_pipeline):
        """Test getting query context from history"""
        with patch('services.memvid_rag.get_db_context') as mock_db_context:
            mock_db = Mock()
            mock_db_context.return_value.__enter__.return_value = mock_db
            
            # Mock query history
            mock_query1 = Mock()
            mock_query1.query_text = "What is machine learning?"
            mock_query2 = Mock()
            mock_query2.query_text = "How does AI work?"
            
            mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
                mock_query1, mock_query2
            ]
            
            context = await memvid_rag_pipeline._get_query_context(1)
            
            assert "recent_queries" in context
            assert "recent_topics" in context
            assert len(context["recent_queries"]) == 2
    
    def test_extract_topics(self, memvid_rag_pipeline):
        """Test topic extraction from queries"""
        queries = [
            "What is machine learning?",
            "How does artificial intelligence work?",
            "Explain neural networks"
        ]
        
        topics = memvid_rag_pipeline._extract_topics(queries)
        
        assert "machine" in topics
        assert "learning" in topics
        assert "artificial" in topics
        assert "intelligence" in topics
        assert "neural" in topics
        assert "networks" in topics
    
    def test_check_memory_cache(self, memvid_rag_pipeline):
        """Test memory cache checking"""
        # Setup cache
        memvid_rag_pipeline.memory_cache = {
            "What is machine learning?": {
                "answer": "ML is a subset of AI",
                "timestamp": 1234567890
            },
            "How does AI work?": {
                "answer": "AI uses algorithms",
                "timestamp": 1234567891
            }
        }
        
        # Test cache hit
        hits = memvid_rag_pipeline._check_memory_cache("machine learning algorithms")
        assert len(hits) > 0
        assert any("machine learning" in hit["cached_query"].lower() for hit in hits)
        
        # Test cache miss
        hits = memvid_rag_pipeline._check_memory_cache("quantum computing")
        assert len(hits) == 0
    
    @pytest.mark.asyncio
    async def test_hierarchical_retrieval(self, memvid_rag_pipeline):
        """Test hierarchical retrieval with context window"""
        with patch('services.memvid_rag.search_similar_chunks') as mock_search, \
             patch.object(memvid_rag_pipeline, '_get_chunk_by_index') as mock_get_chunk:
            
            # Mock primary search results
            mock_search.return_value = [
                {
                    'chunk_id': 'chunk_1',
                    'document_id': 'doc_1',
                    'chunk_index': 5,
                    'content': 'Primary content',
                    'similarity_score': 0.95
                }
            ]
            
            # Mock context chunks
            mock_get_chunk.return_value = {
                'chunk_id': 'chunk_2',
                'document_id': 'doc_1',
                'chunk_index': 6,
                'content': 'Context content',
                'similarity_score': 0.80
            }
            
            result = await memvid_rag_pipeline._hierarchical_retrieval(
                [0.1, 0.2, 0.3], 5, 2, 1
            )
            
            assert len(result) >= 1  # At least primary chunk
            assert any(chunk.get('retrieval_type') == 'primary' for chunk in result)
    
    @pytest.mark.asyncio
    async def test_assemble_hierarchical_context(self, memvid_rag_pipeline, mock_similar_chunks):
        """Test hierarchical context assembly"""
        context, source_chunks, metadata = await memvid_rag_pipeline._assemble_hierarchical_context(
            mock_similar_chunks, 3
        )
        
        assert isinstance(context, str)
        assert len(source_chunks) == 2
        assert all(isinstance(chunk, SourceChunk) for chunk in source_chunks)
        assert "[PRIMARY]" in context
        assert "[CONTEXT]" in context
        assert metadata["total_documents"] == 1
        assert metadata["primary_chunks"] >= 1
        assert metadata["context_chunks"] >= 1
    
    def test_create_memvid_prompt(self, memvid_rag_pipeline):
        """Test MemVid prompt creation"""
        original_query = "What is AI?"
        enhanced_query = "Context: Recent topics include ML. Current query: What is AI?"
        context = "AI is artificial intelligence."
        query_context = {"recent_topics": ["machine learning"]}
        
        prompt = memvid_rag_pipeline._create_memvid_prompt(
            original_query, enhanced_query, context, query_context
        )
        
        assert original_query in prompt
        assert context in prompt
        assert "PRIMARY" in prompt or "CONTEXT" in prompt
        assert "machine learning" in prompt
        assert "enhanced memory" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_update_memory_cache(self, memvid_rag_pipeline):
        """Test memory cache updating"""
        source_chunks = [
            SourceChunk(
                chunk_id='chunk_1',
                content='Test content',
                similarity_score=0.95,
                document_id='doc_1',
                chunk_index=0
            )
        ]
        
        await memvid_rag_pipeline._update_memory_cache(
            "Test query", "Test answer", source_chunks, 1
        )
        
        assert "Test query" in memvid_rag_pipeline.memory_cache
        cache_entry = memvid_rag_pipeline.memory_cache["Test query"]
        assert cache_entry["answer"] == "Test answer"
        assert cache_entry["source_count"] == 1
        assert cache_entry["user_id"] == 1
    
    @pytest.mark.asyncio
    async def test_update_memory_cache_size_limit(self, memvid_rag_pipeline):
        """Test memory cache size limiting"""
        # Fill cache to max size
        memvid_rag_pipeline.max_cache_size = 2
        
        await memvid_rag_pipeline._update_memory_cache("Query 1", "Answer 1", [], 1)
        await memvid_rag_pipeline._update_memory_cache("Query 2", "Answer 2", [], 1)
        
        assert len(memvid_rag_pipeline.memory_cache) == 2
        
        # Add one more - should remove oldest
        await memvid_rag_pipeline._update_memory_cache("Query 3", "Answer 3", [], 1)
        
        assert len(memvid_rag_pipeline.memory_cache) == 2
        assert "Query 3" in memvid_rag_pipeline.memory_cache
    
    def test_get_memory_stats(self, memvid_rag_pipeline):
        """Test memory statistics retrieval"""
        # Add some cache entries
        memvid_rag_pipeline.memory_cache = {
            "Query 1": {"timestamp": 1000},
            "Query 2": {"timestamp": 2000}
        }
        
        stats = memvid_rag_pipeline.get_memory_stats()
        
        assert stats["cache_size"] == 2
        assert stats["max_cache_size"] == memvid_rag_pipeline.max_cache_size
        assert stats["cache_utilization"] == 2 / memvid_rag_pipeline.max_cache_size
        assert stats["oldest_entry"] == 1000
        assert stats["newest_entry"] == 2000
    
    def test_clear_memory_cache(self, memvid_rag_pipeline):
        """Test memory cache clearing"""
        # Add some cache entries
        memvid_rag_pipeline.memory_cache = {
            "Query 1": {"answer": "Answer 1"},
            "Query 2": {"answer": "Answer 2"}
        }
        
        assert len(memvid_rag_pipeline.memory_cache) == 2
        
        memvid_rag_pipeline.clear_memory_cache()
        
        assert len(memvid_rag_pipeline.memory_cache) == 0


class TestMemVidRAGIntegration:
    """Integration tests for MemVid RAG pipeline"""
    
    @pytest.mark.asyncio
    async def test_process_memvid_rag_query_function(self):
        """Test the main process_memvid_rag_query function"""
        query_request = MemVidQueryRequest(query="Test query", top_k=3, context_window=2)
        mock_user = Mock(spec=User)
        mock_user.id = 1
        
        with patch('services.memvid_rag.memvid_rag_pipeline') as mock_pipeline:
            mock_response = MemVidRAGResponse(
                answer="Test answer",
                sources=[],
                response_time=1.0,
                chunks_used=0,
                query="Test query",
                timestamp=datetime.utcnow(),
                memvid_metadata={"enhanced_query": "Test query"}
            )
            mock_pipeline.process_query.return_value = mock_response
            
            result = await process_memvid_rag_query(query_request, mock_user)
            
            assert result == mock_response
            mock_pipeline.process_query.assert_called_once_with(query_request, mock_user)


class TestMemVidRAGErrorHandling:
    """Test error handling in MemVid RAG pipeline"""
    
    @pytest.fixture
    def memvid_rag_pipeline(self):
        return MemVidRAGPipeline()
    
    @pytest.mark.asyncio
    async def test_query_enhancement_error(self, memvid_rag_pipeline):
        """Test handling of query enhancement errors"""
        with patch.object(memvid_rag_pipeline, '_enhance_query_with_memory') as mock_enhance:
            mock_enhance.side_effect = Exception("Enhancement error")
            
            query_request = MemVidQueryRequest(query="test", top_k=5, context_window=2)
            mock_user = Mock(spec=User)
            mock_user.id = 1
            
            with pytest.raises(HTTPException) as exc_info:
                await memvid_rag_pipeline.process_query(query_request, mock_user)
            
            assert exc_info.value.status_code == 500
    
    @pytest.mark.asyncio
    async def test_hierarchical_retrieval_error(self, memvid_rag_pipeline):
        """Test handling of hierarchical retrieval errors"""
        with patch.object(memvid_rag_pipeline, '_enhance_query_with_memory') as mock_enhance, \
             patch.object(memvid_rag_pipeline, '_generate_query_embedding') as mock_embed, \
             patch.object(memvid_rag_pipeline, '_hierarchical_retrieval') as mock_retrieve:
            
            mock_enhance.return_value = ("enhanced", {})
            mock_embed.return_value = [0.1, 0.2, 0.3]
            mock_retrieve.side_effect = HTTPException(status_code=500, detail="Retrieval error")
            
            query_request = MemVidQueryRequest(query="test", top_k=5, context_window=2)
            mock_user = Mock(spec=User)
            mock_user.id = 1
            
            with pytest.raises(HTTPException) as exc_info:
                await memvid_rag_pipeline.process_query(query_request, mock_user)
            
            assert exc_info.value.status_code == 500
    
    @pytest.mark.asyncio
    async def test_context_assembly_error(self, memvid_rag_pipeline):
        """Test handling of context assembly errors"""
        with patch.object(memvid_rag_pipeline, '_enhance_query_with_memory') as mock_enhance, \
             patch.object(memvid_rag_pipeline, '_generate_query_embedding') as mock_embed, \
             patch.object(memvid_rag_pipeline, '_hierarchical_retrieval') as mock_retrieve, \
             patch.object(memvid_rag_pipeline, '_assemble_hierarchical_context') as mock_assemble:
            
            mock_enhance.return_value = ("enhanced", {})
            mock_embed.return_value = [0.1, 0.2, 0.3]
            mock_retrieve.return_value = [{'chunk_id': 'test', 'content': 'test', 'similarity_score': 0.9}]
            mock_assemble.side_effect = HTTPException(status_code=500, detail="Assembly error")
            
            query_request = MemVidQueryRequest(query="test", top_k=5, context_window=2)
            mock_user = Mock(spec=User)
            mock_user.id = 1
            
            with pytest.raises(HTTPException) as exc_info:
                await memvid_rag_pipeline.process_query(query_request, mock_user)
            
            assert exc_info.value.status_code == 500
    
    @pytest.mark.asyncio
    async def test_no_chunks_found_response(self, memvid_rag_pipeline):
        """Test response when no chunks are found"""
        with patch.object(memvid_rag_pipeline, '_enhance_query_with_memory') as mock_enhance, \
             patch.object(memvid_rag_pipeline, '_generate_query_embedding') as mock_embed, \
             patch.object(memvid_rag_pipeline, '_hierarchical_retrieval') as mock_retrieve:
            
            mock_enhance.return_value = ("enhanced query", {"recent_topics": []})
            mock_embed.return_value = [0.1, 0.2, 0.3]
            mock_retrieve.return_value = []
            
            query_request = MemVidQueryRequest(query="test", top_k=5, context_window=2)
            mock_user = Mock(spec=User)
            mock_user.id = 1
            
            response = await memvid_rag_pipeline.process_query(query_request, mock_user)
            
            assert isinstance(response, MemVidRAGResponse)
            assert "couldn't find any relevant information" in response.answer
            assert len(response.sources) == 0
            assert response.chunks_used == 0
            assert "memvid_metadata" in response.__dict__


class TestMemVidRAGMemoryFeatures:
    """Test MemVid-specific memory features"""
    
    @pytest.fixture
    def memvid_rag_pipeline(self):
        return MemVidRAGPipeline()
    
    @pytest.mark.asyncio
    async def test_memory_cache_functionality(self, memvid_rag_pipeline):
        """Test complete memory cache functionality"""
        # Test cache update
        source_chunks = [
            SourceChunk(
                chunk_id='chunk_1',
                content='Test content',
                similarity_score=0.95,
                document_id='doc_1',
                chunk_index=0
            )
        ]
        
        await memvid_rag_pipeline._update_memory_cache(
            "What is AI?", "AI is artificial intelligence", source_chunks, 1
        )
        
        # Test cache check
        hits = memvid_rag_pipeline._check_memory_cache("artificial intelligence")
        assert len(hits) > 0
        
        # Test cache stats
        stats = memvid_rag_pipeline.get_memory_stats()
        assert stats["cache_size"] == 1
        
        # Test cache clear
        memvid_rag_pipeline.clear_memory_cache()
        assert len(memvid_rag_pipeline.memory_cache) == 0
    
    def test_topic_extraction_edge_cases(self, memvid_rag_pipeline):
        """Test topic extraction with edge cases"""
        # Empty queries
        topics = memvid_rag_pipeline._extract_topics([])
        assert topics == []
        
        # Short words only
        topics = memvid_rag_pipeline._extract_topics(["is a an the"])
        assert topics == []
        
        # Mixed case and punctuation
        topics = memvid_rag_pipeline._extract_topics([
            "What is Machine-Learning?",
            "How does ARTIFICIAL intelligence work!"
        ])
        assert "machine-learning" in topics
        assert "artificial" in topics
        assert "intelligence" in topics