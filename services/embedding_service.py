"""
Embedding service for generating embeddings using Google Gemini or Sentence Transformers
"""
import logging
from typing import List, Optional
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings"""
    
    def __init__(self):
        self.gemini_embedding_model = None
        self.sentence_transformer_model = None
        self.embedding_dimension = 768  # Default dimension
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models"""
        try:
            # Initialize Gemini embeddings (preferred)
            if settings.GEMINI_API_KEY:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_embedding_model = settings.GEMINI_EMBEDDING_MODEL
                self.embedding_dimension = 768  # Gemini text-embedding-004 dimension
                logger.info(f"Initialized Gemini embeddings with model: {self.gemini_embedding_model}")
            
            # Initialize Sentence Transformers as fallback
            try:
                model_name = getattr(settings, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
                self.sentence_transformer_model = SentenceTransformer(model_name)
                if not self.gemini_embedding_model:
                    self.embedding_dimension = self.sentence_transformer_model.get_sentence_embedding_dimension()
                logger.info(f"Initialized Sentence Transformers with model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Sentence Transformers: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            # Try Gemini embeddings first
            if self.gemini_embedding_model:
                return await self._generate_gemini_embeddings(texts)
            # Fall back to Sentence Transformers
            elif self.sentence_transformer_model:
                return await self._generate_sentence_transformer_embeddings(texts)
            else:
                raise HTTPException(status_code=500, detail="No embedding model available")
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
    
    async def _generate_gemini_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google Gemini"""
        try:
            embeddings = []
            
            # Process texts in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                for text in batch:
                    # Generate embedding for single text
                    result = genai.embed_content(
                        model=self.gemini_embedding_model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
            
            logger.debug(f"Generated {len(embeddings)} Gemini embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Gemini embedding generation failed: {e}")
            # Fall back to Sentence Transformers if available
            if self.sentence_transformer_model:
                return await self._generate_sentence_transformer_embeddings(texts)
            else:
                raise
    
    async def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Sentence Transformers"""
        try:
            # Generate embeddings in batches to avoid memory issues
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.sentence_transformer_model.encode(
                    batch,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
                
                # Handle both numpy arrays and lists
                if hasattr(batch_embeddings, 'tolist'):
                    embeddings.extend(batch_embeddings.tolist())
                else:
                    embeddings.extend(batch_embeddings)
            
            logger.debug(f"Generated {len(embeddings)} Sentence Transformer embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Sentence Transformer embedding generation failed: {e}")
            raise
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            if self.gemini_embedding_model:
                result = genai.embed_content(
                    model=self.gemini_embedding_model,
                    content=query,
                    task_type="retrieval_query"
                )
                return result['embedding']
            elif self.sentence_transformer_model:
                embedding = self.sentence_transformer_model.encode(
                    [query],
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()[0]
                else:
                    return embedding[0]
            else:
                raise HTTPException(status_code=500, detail="No embedding model available")
                
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dimension
    
    def get_model_info(self) -> dict:
        """Get information about the embedding models"""
        return {
            "gemini_available": bool(self.gemini_embedding_model),
            "sentence_transformer_available": bool(self.sentence_transformer_model),
            "primary_service": "gemini" if self.gemini_embedding_model else "sentence_transformer",
            "embedding_dimension": self.embedding_dimension,
            "gemini_model": self.gemini_embedding_model,
            "sentence_transformer_model": getattr(settings, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        }
    
    async def test_embedding(self) -> dict:
        """Test embedding generation"""
        try:
            test_text = "This is a test sentence for embedding generation."
            embedding = await self.generate_query_embedding(test_text)
            
            return {
                "status": "success",
                "service": "gemini" if self.gemini_embedding_model else "sentence_transformer",
                "embedding_dimension": len(embedding),
                "sample_embedding": embedding[:5]  # First 5 dimensions
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Global embedding service instance
embedding_service = EmbeddingService()


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Main function to generate embeddings
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    return await embedding_service.generate_embeddings(texts)


async def generate_query_embedding(query: str) -> List[float]:
    """
    Main function to generate query embedding
    
    Args:
        query: Query text to embed
        
    Returns:
        Embedding vector
    """
    return await embedding_service.generate_query_embedding(query)


def get_embedding_dimension() -> int:
    """Get embedding dimension"""
    return embedding_service.get_embedding_dimension()


def get_embedding_info() -> dict:
    """Get embedding service information"""
    return embedding_service.get_model_info()


async def test_embedding_service() -> dict:
    """Test embedding service"""
    return await embedding_service.test_embedding()