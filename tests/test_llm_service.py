"""
Unit tests for LLM service
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException

from services.llm_service import LLMService, generate_llm_response, test_llm_service, get_llm_info


class TestLLMService:
    """Test LLM service functionality"""
    
    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance"""
        return LLMService()
    
    @pytest.mark.asyncio
    async def test_generate_response_with_gemini(self, llm_service):
        """Test response generation using Gemini"""
        with patch.object(llm_service, 'gemini_model') as mock_gemini:
            mock_response = Mock()
            mock_response.text = "This is a Gemini response."
            mock_gemini.generate_content.return_value = mock_response
            
            result = await llm_service.generate_response("Test prompt")
            
            assert result == "This is a Gemini response."
            mock_gemini.generate_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_with_openai_fallback(self, llm_service):
        """Test response generation falling back to OpenAI"""
        llm_service.gemini_model = None
        
        with patch.object(llm_service, 'openai_client') as mock_openai, \
             patch('services.llm_service.settings') as mock_settings:
            
            mock_settings.OPENAI_API_KEY = "test_key"
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "This is an OpenAI response."
            mock_openai.chat.completions.create.return_value = mock_response
            
            result = await llm_service.generate_response("Test prompt")
            
            assert result == "This is an OpenAI response."
            mock_openai.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_fallback_mode(self, llm_service):
        """Test response generation in fallback mode"""
        llm_service.gemini_model = None
        llm_service.openai_client = None
        
        result = await llm_service.generate_response("What is AI?")
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "configure an OpenAI API key" in result
    
    @pytest.mark.asyncio
    async def test_generate_gemini_response_success(self, llm_service):
        """Test successful Gemini response generation"""
        with patch('google.generativeai.types.GenerationConfig') as mock_config, \
             patch.object(llm_service, 'gemini_model') as mock_model:
            
            mock_response = Mock()
            mock_response.text = "Gemini generated response"
            mock_model.generate_content.return_value = mock_response
            
            result = await llm_service._generate_gemini_response("Test prompt", 100, 0.7)
            
            assert result == "Gemini generated response"
            mock_model.generate_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_gemini_response_empty_response(self, llm_service):
        """Test Gemini response when response is empty"""
        with patch('google.generativeai.types.GenerationConfig') as mock_config, \
             patch.object(llm_service, 'gemini_model') as mock_model, \
             patch.object(llm_service, '_generate_fallback_response') as mock_fallback:
            
            mock_response = Mock()
            mock_response.text = None
            mock_model.generate_content.return_value = mock_response
            mock_fallback.return_value = "Fallback response"
            
            result = await llm_service._generate_gemini_response("Test prompt", 100, 0.7)
            
            assert result == "Fallback response"
            mock_fallback.assert_called_once_with("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_gemini_response_error_fallback(self, llm_service):
        """Test Gemini response error with OpenAI fallback"""
        llm_service.openai_client = Mock()
        
        with patch('google.generativeai.types.GenerationConfig') as mock_config, \
             patch.object(llm_service, 'gemini_model') as mock_model, \
             patch.object(llm_service, '_generate_openai_response') as mock_openai, \
             patch('services.llm_service.settings') as mock_settings:
            
            mock_model.generate_content.side_effect = Exception("Gemini error")
            mock_settings.OPENAI_API_KEY = "test_key"
            mock_openai.return_value = "OpenAI fallback response"
            
            result = await llm_service._generate_gemini_response("Test prompt", 100, 0.7)
            
            assert result == "OpenAI fallback response"
            mock_openai.assert_called_once_with("Test prompt", 100, 0.7)
    
    @pytest.mark.asyncio
    async def test_generate_openai_response_success(self, llm_service):
        """Test successful OpenAI response generation"""
        with patch.object(llm_service, 'openai_client') as mock_client:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "  OpenAI response  "
            mock_client.chat.completions.create.return_value = mock_response
            
            result = await llm_service._generate_openai_response("Test prompt", 100, 0.7)
            
            assert result == "OpenAI response"
            mock_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_openai_response_rate_limit_error(self, llm_service):
        """Test OpenAI rate limit error handling"""
        import openai
        
        with patch.object(llm_service, 'openai_client') as mock_client:
            mock_client.chat.completions.create.side_effect = openai.RateLimitError(
                "Rate limit exceeded", response=Mock(), body={}
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await llm_service._generate_openai_response("Test prompt", 100, 0.7)
            
            assert exc_info.value.status_code == 429
            assert "Rate limit exceeded" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_generate_openai_response_auth_error(self, llm_service):
        """Test OpenAI authentication error handling"""
        import openai
        
        with patch.object(llm_service, 'openai_client') as mock_client:
            mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
                "Invalid API key", response=Mock(), body={}
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await llm_service._generate_openai_response("Test prompt", 100, 0.7)
            
            assert exc_info.value.status_code == 401
            assert "authentication failed" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_generate_openai_response_bad_request_error(self, llm_service):
        """Test OpenAI bad request error handling"""
        import openai
        
        with patch.object(llm_service, 'openai_client') as mock_client:
            mock_client.chat.completions.create.side_effect = openai.BadRequestError(
                "Invalid request", response=Mock(), body={}
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await llm_service._generate_openai_response("Test prompt", 100, 0.7)
            
            assert exc_info.value.status_code == 400
            assert "Invalid request" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_generate_openai_response_general_error_fallback(self, llm_service):
        """Test OpenAI general error with fallback"""
        with patch.object(llm_service, 'openai_client') as mock_client, \
             patch.object(llm_service, '_generate_fallback_response') as mock_fallback:
            
            mock_client.chat.completions.create.side_effect = Exception("General error")
            mock_fallback.return_value = "Fallback response"
            
            result = await llm_service._generate_openai_response("Test prompt", 100, 0.7)
            
            assert result == "Fallback response"
            mock_fallback.assert_called_once_with("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_fallback_response_what_question(self, llm_service):
        """Test fallback response for 'what' questions"""
        result = await llm_service._generate_fallback_response("What is machine learning?")
        
        assert isinstance(result, str)
        assert "what" in result.lower() or "addresses your question" in result
        assert "configure an OpenAI API key" in result
    
    @pytest.mark.asyncio
    async def test_generate_fallback_response_when_question(self, llm_service):
        """Test fallback response for 'when' questions"""
        result = await llm_service._generate_fallback_response("When was AI invented?")
        
        assert isinstance(result, str)
        assert ("timing" in result or "location" in result) or "configure an OpenAI API key" in result
    
    @pytest.mark.asyncio
    async def test_generate_fallback_response_general(self, llm_service):
        """Test fallback response for general questions"""
        result = await llm_service._generate_fallback_response("Tell me about this topic.")
        
        assert isinstance(result, str)
        assert "relevant information" in result
        assert "configure an OpenAI API key" in result
    
    def test_get_model_info_with_gemini(self, llm_service):
        """Test getting model info when Gemini is available"""
        llm_service.gemini_model = Mock()
        llm_service.openai_client = None
        
        info = llm_service.get_model_info()
        
        assert info['gemini_available'] is True
        assert info['openai_available'] is False
        assert info['primary_service'] == 'gemini'
        assert info['fallback_mode'] is False
    
    def test_get_model_info_with_openai(self, llm_service):
        """Test getting model info when only OpenAI is available"""
        llm_service.gemini_model = None
        llm_service.openai_client = Mock()
        
        with patch('services.llm_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test_key"
            
            info = llm_service.get_model_info()
            
            assert info['gemini_available'] is False
            assert info['openai_available'] is True
            assert info['primary_service'] == 'openai'
            assert info['fallback_mode'] is False
    
    def test_get_model_info_fallback_mode(self, llm_service):
        """Test getting model info in fallback mode"""
        llm_service.gemini_model = None
        llm_service.openai_client = None
        
        info = llm_service.get_model_info()
        
        assert info['gemini_available'] is False
        assert info['openai_available'] is False
        assert info['primary_service'] == 'fallback'
        assert info['fallback_mode'] is True
    
    @pytest.mark.asyncio
    async def test_test_connection_with_gemini(self, llm_service):
        """Test connection testing with Gemini"""
        with patch.object(llm_service, '_generate_gemini_response') as mock_gemini:
            llm_service.gemini_model = Mock()
            mock_gemini.return_value = "Hello"
            
            result = await llm_service.test_connection()
            
            assert result['status'] == 'success'
            assert result['service'] == 'gemini'
            assert result['test_response'] == 'Hello'
            mock_gemini.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_connection_with_openai(self, llm_service):
        """Test connection testing with OpenAI"""
        llm_service.gemini_model = None
        llm_service.openai_client = Mock()
        
        with patch.object(llm_service, '_generate_openai_response') as mock_openai, \
             patch('services.llm_service.settings') as mock_settings:
            
            mock_settings.OPENAI_API_KEY = "test_key"
            mock_openai.return_value = "Hello"
            
            result = await llm_service.test_connection()
            
            assert result['status'] == 'success'
            assert result['service'] == 'openai'
            assert result['test_response'] == 'Hello'
            mock_openai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_connection_fallback(self, llm_service):
        """Test connection testing in fallback mode"""
        llm_service.gemini_model = None
        llm_service.openai_client = None
        
        with patch.object(llm_service, '_generate_fallback_response') as mock_fallback:
            mock_fallback.return_value = "Fallback response"
            
            result = await llm_service.test_connection()
            
            assert result['status'] == 'success'
            assert result['service'] == 'fallback'
            assert result['test_response'] == 'Fallback response'
    
    @pytest.mark.asyncio
    async def test_test_connection_error(self, llm_service):
        """Test connection testing with error"""
        with patch.object(llm_service, '_generate_gemini_response') as mock_gemini:
            llm_service.gemini_model = Mock()
            mock_gemini.side_effect = Exception("Connection failed")
            
            result = await llm_service.test_connection()
            
            assert result['status'] == 'error'
            assert 'Connection failed' in result['error']


class TestLLMServiceFunctions:
    """Test module-level functions"""
    
    @pytest.mark.asyncio
    async def test_generate_llm_response(self):
        """Test generate_llm_response function"""
        with patch('services.llm_service.llm_service') as mock_service:
            mock_service.generate_response.return_value = "Test response"
            
            result = await generate_llm_response("Test prompt", 100, 0.5)
            
            assert result == "Test response"
            mock_service.generate_response.assert_called_once_with("Test prompt", 100, 0.5)
    
    @pytest.mark.asyncio
    async def test_test_llm_service(self):
        """Test test_llm_service function"""
        with patch('services.llm_service.llm_service') as mock_service:
            mock_service.test_connection.return_value = {"status": "success"}
            
            result = await test_llm_service()
            
            assert result == {"status": "success"}
            mock_service.test_connection.assert_called_once()
    
    def test_get_llm_info(self):
        """Test get_llm_info function"""
        with patch('services.llm_service.llm_service') as mock_service:
            mock_service.get_model_info.return_value = {"model": "test"}
            
            result = get_llm_info()
            
            assert result == {"model": "test"}
            mock_service.get_model_info.assert_called_once()


class TestLLMServiceInitialization:
    """Test LLM service initialization"""
    
    def test_initialization_with_gemini_key(self):
        """Test initialization when Gemini API key is available"""
        with patch('services.llm_service.settings') as mock_settings, \
             patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            mock_settings.GEMINI_API_KEY = "test_gemini_key"
            mock_settings.GEMINI_MODEL = "gemini-1.5-flash"
            mock_settings.OPENAI_API_KEY = ""
            
            service = LLMService()
            
            mock_configure.assert_called_once_with(api_key="test_gemini_key")
            mock_model.assert_called_once_with("gemini-1.5-flash")
            assert service.gemini_model is not None
    
    def test_initialization_with_openai_key(self):
        """Test initialization when OpenAI API key is available"""
        with patch('services.llm_service.settings') as mock_settings, \
             patch('openai.OpenAI') as mock_openai:
            
            mock_settings.GEMINI_API_KEY = ""
            mock_settings.OPENAI_API_KEY = "test_openai_key"
            
            service = LLMService()
            
            mock_openai.assert_called_once_with(api_key="test_openai_key")
            assert service.openai_client is not None
    
    def test_initialization_no_keys(self):
        """Test initialization when no API keys are available"""
        with patch('services.llm_service.settings') as mock_settings:
            mock_settings.GEMINI_API_KEY = ""
            mock_settings.OPENAI_API_KEY = ""
            
            service = LLMService()
            
            assert service.gemini_model is None
            assert service.openai_client is None
    
    def test_initialization_error_handling(self):
        """Test initialization error handling"""
        with patch('services.llm_service.settings') as mock_settings, \
             patch('google.generativeai.configure') as mock_configure:
            
            mock_settings.GEMINI_API_KEY = "test_key"
            mock_configure.side_effect = Exception("Initialization failed")
            
            # Should not raise exception, just log error
            service = LLMService()
            
            assert service.gemini_model is None