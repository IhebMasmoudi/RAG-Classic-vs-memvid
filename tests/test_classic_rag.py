"""
Unit tests for Classic RAG pipeline
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from datetime import datetime

from services.classic_rag import ClassicRAGPipeline, process_classic_rag_query
from models.schemas import QueryRequest, RAGResponse, SourceChunk
from models.database import User


class TestClassicRAGPipeline:
    """Test Classic RAG pipeline functionality"""
    
    @pytest.fixture
    def mock_user(self):
        """Create a mock user for testing"""
        user = Mock(spec=User)
        user.id = 1
        user.email = "test@example.com"
        return user
    
    @pytest.fixture
    def query_request(self):
        """Create a test query request"""
        return QueryRequest(
            query="What is machine learning?",
            top_k=5
        )
    
    @pytest.fixture
    def mock_similar_chunks(self):
        """Create mock similar chunks data"""
        return [
            {
                'chunk_id': 'chunk_1',
                'document_id': 'doc_1',
                'chunk_index': 0,
                'content': 'Machine learning is a subset of artificial intelligence.',
                'similarity_score': 0.95
            },
            {
                'chunk_id': 'chunk_2',
                'document_id': 'doc_1',
                'chunk_index': 1,
                'content': 'It involves training algorithms on data to make predictions.',
                'similarity_score': 0.87
            }
        ]
    
    @pytest.fixture
    def classic_rag_pipeline(self):
        """Create Classic RAG pipeline instance"""
        return ClassicRAGPipeline()
    
    @pytest.mark.asyncio
    async def test_process_query_success(
        self,
        classic_rag_pipeline,
        query_request,
        mock_user,
        mock_similar_chunks
    ):
        """Test successful query processing"""
        with patch.object(classic_rag_pipeline, '_generate_query_embedding') as mock_embed, \
             patch.object(classic_rag_pipeline, '_retrieve_similar_chunks') as mock_retrieve, \
             patch.object(classic_rag_pipeline, '_assemble_context') as mock_assemble, \
             patch.object(classic_rag_pipeline, '_generate_answer') as mock_answer, \
             patch.object(classic_rag_pipeline, '_log_query_history') as mock_log:
            
            # Setup mocks
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
                )]
            )
            mock_answer.return_value = "Machine learning is a field of AI that uses algorithms to learn from data."
            mock_log.return_value = None
            
            # Execute
            response = await classic_rag_pipeline.process_query(query_request, mock_user)
            
            # Verify
            assert isinstance(response, RAGResponse)
            assert response.answer == "Machine learning is a field of AI that uses algorithms to learn from data."
            assert len(response.sources) == 1
            assert response.chunks_used == 1
            assert response.query == query_request.query
            assert response.response_time > 0
            
            # Verify method calls
            mock_embed.assert_called_once_with(query_request.query)
            mock_retrieve.assert_called_once_with([0.1, 0.2, 0.3], query_request.top_k, mock_user.id)
            mock_assemble.assert_called_once_with(mock_similar_chunks)
            mock_answer.assert_called_once_with(query_request.query, "Machine learning context")
            mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_query_no_chunks_found(
        self,
        classic_rag_pipeline,
        query_request,
        mock_user
    ):
        """Test query processing when no relevant chunks are found"""
        with patch.object(classic_rag_pipeline, '_generate_query_embedding') as mock_embed, \
             patch.object(classic_rag_pipeline, '_retrieve_similar_chunks') as mock_retrieve:
            
            # Setup mocks
            mock_embed.return_value = [0.1, 0.2, 0.3]
            mock_retrieve.return_value = []
            
            # Execute
            response = await classic_rag_pipeline.process_query(query_request, mock_user)
            
            # Verify
            assert isinstance(response, RAGResponse)
            assert "couldn't find any relevant information" in response.answer
            assert len(response.sources) == 0
            assert response.chunks_used == 0
            assert response.response_time > 0
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding(self, classic_rag_pipeline):
        """Test query embedding generation"""
        with patch('services.classic_rag.generate_query_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            result = await classic_rag_pipeline._generate_query_embedding("test query")
            
            assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_embed.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding_failure(self, classic_rag_pipeline):
        """Test query embedding generation failure"""
        with patch('services.classic_rag.generate_query_embedding') as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")
            
            with pytest.raises(HTTPException) as exc_info:
                await classic_rag_pipeline._generate_query_embedding("test query")
            
            assert exc_info.value.status_code == 500
            assert "Failed to generate query embedding" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_retrieve_similar_chunks(self, classic_rag_pipeline, mock_similar_chunks):
        """Test similar chunks retrieval"""
        with patch('services.classic_rag.search_similar_chunks') as mock_search:
            mock_search.return_value = mock_similar_chunks
            
            result = await classic_rag_pipeline._retrieve_similar_chunks(
                [0.1, 0.2, 0.3], 5, 1
            )
            
            assert result == mock_similar_chunks
            mock_search.assert_called_once_with(
                query_embedding=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=1
            )
    
    @pytest.mark.asyncio
    async def test_assemble_context(self, classic_rag_pipeline, mock_similar_chunks):
        """Test context assembly from chunks"""
        context, source_chunks = await classic_rag_pipeline._assemble_context(mock_similar_chunks)
        
        assert isinstance(context, str)
        assert len(source_chunks) == 2
        assert all(isinstance(chunk, SourceChunk) for chunk in source_chunks)
        assert "Machine learning is a subset of artificial intelligence." in context
        assert "It involves training algorithms on data to make predictions." in context
    
    @pytest.mark.asyncio
    async def test_assemble_context_with_length_limit(self, classic_rag_pipeline):
        """Test context assembly with length limit"""
        # Create chunks that exceed the context limit
        large_chunks = [
            {
                'chunk_id': f'chunk_{i}',
                'document_id': 'doc_1',
                'chunk_index': i,
                'content': 'A' * 2000,  # Large content
                'similarity_score': 0.9
            }
            for i in range(5)
        ]
        
        context, source_chunks = await classic_rag_pipeline._assemble_context(large_chunks)
        
        # Should respect the max context length
        assert len(context) <= classic_rag_pipeline.max_context_length
        assert len(source_chunks) < len(large_chunks)  # Some chunks should be excluded
    
    @pytest.mark.asyncio
    async def test_generate_answer(self, classic_rag_pipeline):
        """Test answer generation using LLM"""
        with patch.object(classic_rag_pipeline.llm_service, 'generate_response') as mock_llm:
            mock_llm.return_value = "This is the generated answer."
            
            result = await classic_rag_pipeline._generate_answer(
                "What is AI?", 
                "AI is artificial intelligence."
            )
            
            assert result == "This is the generated answer."
            mock_llm.assert_called_once()
            
            # Verify the prompt contains both query and context
            call_args = mock_llm.call_args[0][0]
            assert "What is AI?" in call_args
            assert "AI is artificial intelligence." in call_args
    
    def test_create_rag_prompt(self, classic_rag_pipeline):
        """Test RAG prompt creation"""
        query = "What is machine learning?"
        context = "Machine learning is a subset of AI."
        
        prompt = classic_rag_pipeline._create_rag_prompt(query, context)
        
        assert query in prompt
        assert context in prompt
        assert "helpful assistant" in prompt.lower()
        assert "context" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_log_query_history_new_entry(self, classic_rag_pipeline):
        """Test logging query history for new entry"""
        response = RAGResponse(
            answer="Test answer",
            sources=[],
            response_time=1.5,
            chunks_used=2,
            query="Test query",
            timestamp=datetime.utcnow()
        )
        
        with patch('services.classic_rag.get_db_context') as mock_db_context:
            mock_db = Mock()
            mock_db_context.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
            
            await classic_rag_pipeline._log_query_history(1, "Test query", response)
            
            # Verify new QueryHistory object was added
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
    
    def test_serialize_sources(self, classic_rag_pipeline):
        """Test source chunks serialization"""
        sources = [
            SourceChunk(
                chunk_id='chunk_1',
                content='Short content',
                similarity_score=0.95,
                document_id='doc_1',
                chunk_index=0
            ),
            SourceChunk(
                chunk_id='chunk_2',
                content='A' * 300,  # Long content that should be truncated
                similarity_score=0.87,
                document_id='doc_1',
                chunk_index=1
            )
        ]
        
        result = classic_rag_pipeline._serialize_sources(sources)
        
        assert isinstance(result, str)
        import json
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]['chunk_id'] == 'chunk_1'
        assert parsed[0]['content'] == 'Short content'
        assert len(parsed[1]['content']) <= 203  # 200 + "..."


class TestClassicRAGIntegration:
    """Integration tests for Classic RAG pipeline"""
    
    @pytest.mark.asyncio
    async def test_process_classic_rag_query_function(self):
        """Test the main process_classic_rag_query function"""
        query_request = QueryRequest(query="Test query", top_k=3)
        mock_user = Mock(spec=User)
        mock_user.id = 1
        
        with patch('services.classic_rag.classic_rag_pipeline') as mock_pipeline:
            mock_response = RAGResponse(
                answer="Test answer",
                sources=[],
                response_time=1.0,
                chunks_used=0,
                query="Test query",
                timestamp=datetime.utcnow()
            )
            mock_pipeline.process_query.return_value = mock_response
            
            result = await process_classic_rag_query(query_request, mock_user)
            
            assert result == mock_response
            mock_pipeline.process_query.assert_called_once_with(query_request, mock_user)


class TestClassicRAGErrorHandling:
    """Test error handling in Classic RAG pipeline"""
    
    @pytest.fixture
    def classic_rag_pipeline(self):
        return ClassicRAGPipeline()
    
    @pytest.mark.asyncio
    async def test_embedding_generation_error(self, classic_rag_pipeline):
        """Test handling of embedding generation errors"""
        with patch.object(classic_rag_pipeline, '_generate_query_embedding') as mock_embed:
            mock_embed.side_effect = HTTPException(status_code=500, detail="Embedding error")
            
            query_request = QueryRequest(query="test", top_k=5)
            mock_user = Mock(spec=User)
            mock_user.id = 1
            
            with pytest.raises(HTTPException) as exc_info:
                await classic_rag_pipeline.process_query(query_request, mock_user)
            
            assert exc_info.value.status_code == 500
    
    @pytest.mark.asyncio
    async def test_retrieval_error(self, classic_rag_pipeline):
        """Test handling of retrieval errors"""
        with patch.object(classic_rag_pipeline, '_generate_query_embedding') as mock_embed, \
             patch.object(classic_rag_pipeline, '_retrieve_similar_chunks') as mock_retrieve:
            
            mock_embed.return_value = [0.1, 0.2, 0.3]
            mock_retrieve.side_effect = HTTPException(status_code=500, detail="Retrieval error")
            
            query_request = QueryRequest(query="test", top_k=5)
            mock_user = Mock(spec=User)
            mock_user.id = 1
            
            with pytest.raises(HTTPException) as exc_info:
                await classic_rag_pipeline.process_query(query_request, mock_user)
            
            assert exc_info.value.status_code == 500
    
    @pytest.mark.asyncio
    async def test_llm_generation_error(self, classic_rag_pipeline):
        """Test handling of LLM generation errors"""
        with patch.object(classic_rag_pipeline, '_generate_query_embedding') as mock_embed, \
             patch.object(classic_rag_pipeline, '_retrieve_similar_chunks') as mock_retrieve, \
             patch.object(classic_rag_pipeline, '_assemble_context') as mock_assemble, \
             patch.object(classic_rag_pipeline, '_generate_answer') as mock_answer:
            
            mock_embed.return_value = [0.1, 0.2, 0.3]
            mock_retrieve.return_value = [{'chunk_id': 'test', 'content': 'test', 'similarity_score': 0.9, 'document_id': 'doc', 'chunk_index': 0}]
            mock_assemble.return_value = ("context", [])
            mock_answer.side_effect = HTTPException(status_code=500, detail="LLM error")
            
            query_request = QueryRequest(query="test", top_k=5)
            mock_user = Mock(spec=User)
            mock_user.id = 1
            
            with pytest.raises(HTTPException) as exc_info:
                await classic_rag_pipeline.process_query(query_request, mock_user)
            
            assert exc_info.value.status_code == 500