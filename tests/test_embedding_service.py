"""
Unit tests for embedding service
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException

from services.embedding_service import (
    EmbeddingService, 
    generate_embeddings, 
    generate_query_embedding,
    get_embedding_dimension,
    get_embedding_info,
    test_embedding_service
)


class TestEmbeddingService:
    """Test embedding service functionality"""
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance"""
        return EmbeddingService()
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_with_gemini(self, embedding_service):
        """Test embedding generation using Gemini"""
        texts = ["Hello world", "Machine learning is great"]
        
        with patch('google.generativeai.embed_content') as mock_embed:
            mock_embed.side_effect = [
                {'embedding': [0.1, 0.2, 0.3]},
                {'embedding': [0.4, 0.5, 0.6]}
            ]
            embedding_service.gemini_embedding_model = "models/text-embedding-004"
            
            result = await embedding_service.generate_embeddings(texts)
            
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            assert mock_embed.call_count == 2
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_with_sentence_transformer(self, embedding_service):
        """Test embedding generation using Sentence Transformers"""
        texts = ["Hello world", "Machine learning is great"]
        embedding_service.gemini_embedding_model = None
        
        with patch.object(embedding_service, 'sentence_transformer_model') as mock_model:
            mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            
            result = await embedding_service.generate_embeddings(texts)
            
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            mock_model.encode.assert_called_once_with(
                texts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self, embedding_service):
        """Test embedding generation with empty list"""
        result = await embedding_service.generate_embeddings([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_no_model_available(self, embedding_service):
        """Test embedding generation when no model is available"""
        embedding_service.gemini_embedding_model = None
        embedding_service.sentence_transformer_model = None
        
        with pytest.raises(HTTPException) as exc_info:
            await embedding_service.generate_embeddings(["test"])
        
        assert exc_info.value.status_code == 500
        assert "No embedding model available" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_generate_gemini_embeddings_success(self, embedding_service):
        """Test successful Gemini embedding generation"""
        texts = ["Hello", "World"]
        
        with patch('google.generativeai.embed_content') as mock_embed:
            mock_embed.side_effect = [
                {'embedding': [0.1, 0.2]},
                {'embedding': [0.3, 0.4]}
            ]
            
            result = await embedding_service._generate_gemini_embeddings(texts)
            
            assert result == [[0.1, 0.2], [0.3, 0.4]]
            assert mock_embed.call_count == 2
            
            # Verify correct parameters
            calls = mock_embed.call_args_list
            assert calls[0][1]['content'] == 'Hello'
            assert calls[0][1]['task_type'] == 'retrieval_document'
            assert calls[1][1]['content'] == 'World'
    
    @pytest.mark.asyncio
    async def test_generate_gemini_embeddings_batch_processing(self, embedding_service):
        """Test Gemini embedding generation with batch processing"""
        texts = [f"Text {i}" for i in range(15)]  # More than batch size of 10
        
        with patch('google.generativeai.embed_content') as mock_embed:
            mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3]}
            
            result = await embedding_service._generate_gemini_embeddings(texts)
            
            assert len(result) == 15
            assert all(emb == [0.1, 0.2, 0.3] for emb in result)
            assert mock_embed.call_count == 15
    
    @pytest.mark.asyncio
    async def test_generate_gemini_embeddings_error_fallback(self, embedding_service):
        """Test Gemini embedding error with Sentence Transformer fallback"""
        texts = ["Hello", "World"]
        
        with patch('google.generativeai.embed_content') as mock_embed, \
             patch.object(embedding_service, '_generate_sentence_transformer_embeddings') as mock_st:
            
            mock_embed.side_effect = Exception("Gemini error")
            mock_st.return_value = [[0.5, 0.6], [0.7, 0.8]]
            embedding_service.sentence_transformer_model = Mock()
            
            result = await embedding_service._generate_gemini_embeddings(texts)
            
            assert result == [[0.5, 0.6], [0.7, 0.8]]
            mock_st.assert_called_once_with(texts)
    
    @pytest.mark.asyncio
    async def test_generate_sentence_transformer_embeddings_success(self, embedding_service):
        """Test successful Sentence Transformer embedding generation"""
        texts = ["Hello", "World"]
        
        with patch.object(embedding_service, 'sentence_transformer_model') as mock_model:
            mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
            
            result = await embedding_service._generate_sentence_transformer_embeddings(texts)
            
            assert result == [[0.1, 0.2], [0.3, 0.4]]
            mock_model.encode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_sentence_transformer_embeddings_numpy_array(self, embedding_service):
        """Test Sentence Transformer embedding with numpy array return"""
        texts = ["Hello", "World"]
        
        with patch.object(embedding_service, 'sentence_transformer_model') as mock_model:
            # Mock numpy array with tolist method
            mock_array = Mock()
            mock_array.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
            mock_model.encode.return_value = mock_array
            
            result = await embedding_service._generate_sentence_transformer_embeddings(texts)
            
            assert result == [[0.1, 0.2], [0.3, 0.4]]
            mock_array.tolist.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_sentence_transformer_embeddings_batch_processing(self, embedding_service):
        """Test Sentence Transformer embedding with batch processing"""
        texts = [f"Text {i}" for i in range(50)]  # More than batch size of 32
        
        with patch.object(embedding_service, 'sentence_transformer_model') as mock_model:
            # Mock two batch calls
            mock_model.encode.side_effect = [
                [[0.1, 0.2]] * 32,  # First batch
                [[0.3, 0.4]] * 18   # Second batch
            ]
            
            result = await embedding_service._generate_sentence_transformer_embeddings(texts)
            
            assert len(result) == 50
            assert mock_model.encode.call_count == 2
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding_with_gemini(self, embedding_service):
        """Test query embedding generation using Gemini"""
        query = "What is machine learning?"
        
        with patch('google.generativeai.embed_content') as mock_embed:
            mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3]}
            embedding_service.gemini_embedding_model = "models/text-embedding-004"
            
            result = await embedding_service.generate_query_embedding(query)
            
            assert result == [0.1, 0.2, 0.3]
            mock_embed.assert_called_once_with(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding_with_sentence_transformer(self, embedding_service):
        """Test query embedding generation using Sentence Transformers"""
        query = "What is machine learning?"
        embedding_service.gemini_embedding_model = None
        
        with patch.object(embedding_service, 'sentence_transformer_model') as mock_model:
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            
            result = await embedding_service.generate_query_embedding(query)
            
            assert result == [0.1, 0.2, 0.3]
            mock_model.encode.assert_called_once_with(
                [query],
                convert_to_tensor=False,
                show_progress_bar=False
            )
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding_numpy_array(self, embedding_service):
        """Test query embedding with numpy array return"""
        query = "What is machine learning?"
        embedding_service.gemini_embedding_model = None
        
        with patch.object(embedding_service, 'sentence_transformer_model') as mock_model:
            # Mock numpy array with tolist method
            mock_array = Mock()
            mock_array.tolist.return_value = [[0.1, 0.2, 0.3]]
            mock_model.encode.return_value = mock_array
            
            result = await embedding_service.generate_query_embedding(query)
            
            assert result == [0.1, 0.2, 0.3]
            mock_array.tolist.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding_no_model(self, embedding_service):
        """Test query embedding when no model is available"""
        embedding_service.gemini_embedding_model = None
        embedding_service.sentence_transformer_model = None
        
        with pytest.raises(HTTPException) as exc_info:
            await embedding_service.generate_query_embedding("test")
        
        assert exc_info.value.status_code == 500
        assert "No embedding model available" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding_error(self, embedding_service):
        """Test query embedding generation error"""
        with patch('google.generativeai.embed_content') as mock_embed:
            mock_embed.side_effect = Exception("Embedding error")
            embedding_service.gemini_embedding_model = "models/text-embedding-004"
            
            with pytest.raises(HTTPException) as exc_info:
                await embedding_service.generate_query_embedding("test")
            
            assert exc_info.value.status_code == 500
            assert "Failed to generate query embedding" in str(exc_info.value.detail)
    
    def test_get_embedding_dimension(self, embedding_service):
        """Test getting embedding dimension"""
        embedding_service.embedding_dimension = 768
        
        result = embedding_service.get_embedding_dimension()
        
        assert result == 768
    
    def test_get_model_info_with_gemini(self, embedding_service):
        """Test getting model info when Gemini is available"""
        embedding_service.gemini_embedding_model = "models/text-embedding-004"
        embedding_service.sentence_transformer_model = Mock()
        embedding_service.embedding_dimension = 768
        
        info = embedding_service.get_model_info()
        
        assert info['gemini_available'] is True
        assert info['sentence_transformer_available'] is True
        assert info['primary_service'] == 'gemini'
        assert info['embedding_dimension'] == 768
        assert info['gemini_model'] == "models/text-embedding-004"
    
    def test_get_model_info_sentence_transformer_only(self, embedding_service):
        """Test getting model info when only Sentence Transformers is available"""
        embedding_service.gemini_embedding_model = None
        embedding_service.sentence_transformer_model = Mock()
        embedding_service.embedding_dimension = 384
        
        info = embedding_service.get_model_info()
        
        assert info['gemini_available'] is False
        assert info['sentence_transformer_available'] is True
        assert info['primary_service'] == 'sentence_transformer'
        assert info['embedding_dimension'] == 384
    
    @pytest.mark.asyncio
    async def test_test_embedding_success(self, embedding_service):
        """Test embedding service testing"""
        with patch.object(embedding_service, 'generate_query_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            embedding_service.gemini_embedding_model = "models/text-embedding-004"
            
            result = await embedding_service.test_embedding()
            
            assert result['status'] == 'success'
            assert result['service'] == 'gemini'
            assert result['embedding_dimension'] == 6
            assert result['sample_embedding'] == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    @pytest.mark.asyncio
    async def test_test_embedding_error(self, embedding_service):
        """Test embedding service testing with error"""
        with patch.object(embedding_service, 'generate_query_embedding') as mock_embed:
            mock_embed.side_effect = Exception("Test error")
            
            result = await embedding_service.test_embedding()
            
            assert result['status'] == 'error'
            assert 'Test error' in result['error']


class TestEmbeddingServiceInitialization:
    """Test embedding service initialization"""
    
    def test_initialization_with_gemini_key(self):
        """Test initialization when Gemini API key is available"""
        with patch('services.embedding_service.settings') as mock_settings, \
             patch('google.generativeai.configure') as mock_configure, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_settings.GEMINI_API_KEY = "test_gemini_key"
            mock_settings.GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
            mock_st.return_value = Mock()
            
            service = EmbeddingService()
            
            mock_configure.assert_called_once_with(api_key="test_gemini_key")
            assert service.gemini_embedding_model == "models/text-embedding-004"
            assert service.embedding_dimension == 768
    
    def test_initialization_sentence_transformer_only(self):
        """Test initialization when only Sentence Transformers is available"""
        with patch('services.embedding_service.settings') as mock_settings, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_settings.GEMINI_API_KEY = ""
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            service = EmbeddingService()
            
            assert service.gemini_embedding_model is None
            assert service.sentence_transformer_model is not None
            assert service.embedding_dimension == 384
    
    def test_initialization_sentence_transformer_error(self):
        """Test initialization when Sentence Transformers fails"""
        with patch('services.embedding_service.settings') as mock_settings, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_settings.GEMINI_API_KEY = ""
            mock_st.side_effect = Exception("ST initialization failed")
            
            service = EmbeddingService()
            
            assert service.gemini_embedding_model is None
            assert service.sentence_transformer_model is None
    
    def test_initialization_error_handling(self):
        """Test initialization error handling"""
        with patch('services.embedding_service.settings') as mock_settings, \
             patch('google.generativeai.configure') as mock_configure:
            
            mock_settings.GEMINI_API_KEY = "test_key"
            mock_configure.side_effect = Exception("Initialization failed")
            
            # Should not raise exception, just log error
            service = EmbeddingService()
            
            assert service.gemini_embedding_model is None


class TestEmbeddingServiceFunctions:
    """Test module-level functions"""
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_function(self):
        """Test generate_embeddings function"""
        with patch('services.embedding_service.embedding_service') as mock_service:
            mock_service.generate_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
            
            result = await generate_embeddings(["text1", "text2"])
            
            assert result == [[0.1, 0.2], [0.3, 0.4]]
            mock_service.generate_embeddings.assert_called_once_with(["text1", "text2"])
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding_function(self):
        """Test generate_query_embedding function"""
        with patch('services.embedding_service.embedding_service') as mock_service:
            mock_service.generate_query_embedding.return_value = [0.1, 0.2, 0.3]
            
            result = await generate_query_embedding("test query")
            
            assert result == [0.1, 0.2, 0.3]
            mock_service.generate_query_embedding.assert_called_once_with("test query")
    
    def test_get_embedding_dimension_function(self):
        """Test get_embedding_dimension function"""
        with patch('services.embedding_service.embedding_service') as mock_service:
            mock_service.get_embedding_dimension.return_value = 768
            
            result = get_embedding_dimension()
            
            assert result == 768
            mock_service.get_embedding_dimension.assert_called_once()
    
    def test_get_embedding_info_function(self):
        """Test get_embedding_info function"""
        with patch('services.embedding_service.embedding_service') as mock_service:
            mock_service.get_model_info.return_value = {"model": "test"}
            
            result = get_embedding_info()
            
            assert result == {"model": "test"}
            mock_service.get_model_info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_embedding_service_function(self):
        """Test test_embedding_service function"""
        with patch('services.embedding_service.embedding_service') as mock_service:
            mock_service.test_embedding.return_value = {"status": "success"}
            
            result = await test_embedding_service()
            
            assert result == {"status": "success"}
            mock_service.test_embedding.assert_called_once()