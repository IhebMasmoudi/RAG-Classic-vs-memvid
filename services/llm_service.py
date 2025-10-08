"""
LLM service for generating responses using Google Gemini or OpenAI models
"""
import logging
from typing import Optional, Dict, Any
import google.generativeai as genai
import openai
from fastapi import HTTPException

from config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM interactions"""
    
    def __init__(self):
        self.openai_client = None
        self.gemini_model = None
        self.model_name = settings.GEMINI_MODEL
        self.max_tokens = 1000
        self.temperature = 0.7
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Gemini and OpenAI clients if API keys are available"""
        try:
            # Initialize Gemini client (preferred)
            if settings.GEMINI_API_KEY:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)
                logger.info(f"Initialized Gemini client with model: {settings.GEMINI_MODEL}")
            
            # Initialize OpenAI client as fallback
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info(f"Initialized OpenAI client with model: {settings.LLM_MODEL}")
            
            if not self.gemini_model and not self.openai_client:
                logger.warning("No API keys found. LLM service will use fallback responses.")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM clients: {e}")
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate response using LLM
        
        Args:
            prompt: Input prompt for the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: Generated response
        """
        try:
            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            # Try Gemini first (preferred)
            if self.gemini_model:
                return await self._generate_gemini_response(prompt, max_tokens, temperature)
            # Fall back to OpenAI
            elif self.openai_client and settings.OPENAI_API_KEY:
                return await self._generate_openai_response(prompt, max_tokens, temperature)
            else:
                return await self._generate_fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate LLM response")
    
    async def _generate_gemini_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate response using Google Gemini API"""
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Generate response
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                answer = response.text.strip()
                logger.debug(f"Generated Gemini response with {len(answer)} characters")
                return answer
            else:
                logger.warning("Gemini returned empty response")
                return await self._generate_fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            # Fall back to OpenAI or fallback response
            if self.openai_client and settings.OPENAI_API_KEY:
                return await self._generate_openai_response(prompt, max_tokens, temperature)
            else:
                return await self._generate_fallback_response(prompt)
    
    async def _generate_openai_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate response using OpenAI API"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=30
            )
            
            answer = response.choices[0].message.content.strip()
            logger.debug(f"Generated OpenAI response with {len(answer)} characters")
            return answer
            
        except openai.RateLimitError:
            logger.error("OpenAI rate limit exceeded")
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        except openai.AuthenticationError:
            logger.error("OpenAI authentication failed")
            raise HTTPException(status_code=401, detail="LLM service authentication failed")
        except openai.BadRequestError as e:
            logger.error(f"Invalid OpenAI request: {e}")
            raise HTTPException(status_code=400, detail="Invalid request to LLM service")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fall back to fallback response
            return await self._generate_fallback_response(prompt)
    
    async def _generate_fallback_response(self, prompt: str) -> str:
        """Generate fallback response when OpenAI is not available"""
        logger.info("Using fallback response generation")
        
        # Simple keyword-based response generation for demo purposes
        query_lower = prompt.lower()
        
        if "what" in query_lower or "how" in query_lower or "why" in query_lower:
            return ("Based on the provided context, I can see relevant information that addresses your question. "
                   "However, I'm currently using a simplified response system. For more detailed answers, "
                   "please configure an OpenAI API key in the system settings.")
        elif "when" in query_lower or "where" in query_lower:
            return ("I can see information in the context that relates to your question about timing or location. "
                   "For more precise answers, please configure an OpenAI API key in the system settings.")
        else:
            return ("I found relevant information in your documents that relates to your question. "
                   "For more detailed and accurate responses, please configure an OpenAI API key in the system settings.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM configuration"""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "gemini_available": bool(self.gemini_model),
            "openai_available": bool(self.openai_client and settings.OPENAI_API_KEY),
            "primary_service": "gemini" if self.gemini_model else ("openai" if self.openai_client else "fallback"),
            "fallback_mode": not bool(self.gemini_model or (self.openai_client and settings.OPENAI_API_KEY))
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test LLM service connection"""
        try:
            if self.gemini_model:
                # Test with a simple prompt
                test_response = await self._generate_gemini_response(
                    "Say 'Hello' if you can hear me.",
                    max_tokens=10,
                    temperature=0.1
                )
                return {
                    "status": "success",
                    "service": "gemini",
                    "model": settings.GEMINI_MODEL,
                    "test_response": test_response
                }
            elif self.openai_client and settings.OPENAI_API_KEY:
                # Test with a simple prompt
                test_response = await self._generate_openai_response(
                    "Say 'Hello' if you can hear me.",
                    max_tokens=10,
                    temperature=0.1
                )
                return {
                    "status": "success",
                    "service": "openai",
                    "model": settings.LLM_MODEL,
                    "test_response": test_response
                }
            else:
                test_response = await self._generate_fallback_response("Test prompt")
                return {
                    "status": "success",
                    "service": "fallback",
                    "model": "fallback",
                    "test_response": test_response
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Global LLM service instance
llm_service = LLMService()


async def generate_llm_response(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> str:
    """
    Main function to generate LLM response
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        str: Generated response
    """
    return await llm_service.generate_response(prompt, max_tokens, temperature)


async def test_llm_service() -> Dict[str, Any]:
    """Test LLM service connection"""
    return await llm_service.test_connection()


def get_llm_info() -> Dict[str, Any]:
    """Get LLM service information"""
    return llm_service.get_model_info()