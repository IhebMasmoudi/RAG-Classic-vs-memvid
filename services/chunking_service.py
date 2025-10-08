"""
Advanced chunking service using LlamaIndex recursive text splitter
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from llama_index.core.text_splitter import SentenceSplitter
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.core.node_parser import SimpleNodeParser
    LLAMA_INDEX_AVAILABLE = True
except ImportError as e:
    LLAMA_INDEX_AVAILABLE = False
    # Don't initialize logger here to avoid circular imports

from fastapi import HTTPException

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking parameters"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n\n"
    secondary_separators: List[str] = None
    
    def __post_init__(self):
        if self.secondary_separators is None:
            self.secondary_separators = ["\n", ". ", "! ", "? ", " "]


class AdvancedChunkingService:
    """Advanced chunking service using LlamaIndex recursive text splitter"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.text_splitter = None
        self.node_parser = None
        
        if LLAMA_INDEX_AVAILABLE:
            self._initialize_llama_index_splitter()
        else:
            logger.warning("LlamaIndex not available, using fallback chunking")
    
    def _initialize_llama_index_splitter(self):
        """Initialize LlamaIndex sentence splitter"""
        try:
            # Create sentence splitter (LlamaIndex's recursive splitter)
            self.text_splitter = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator=self.config.separator
            )
            
            # Node parser is optional for direct text splitting
            self.node_parser = None
            
            logger.info("LlamaIndex sentence splitter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex splitter: {e}")
            self.text_splitter = None
            self.node_parser = None
    
    def create_chunks(self, text: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Create chunks using LlamaIndex recursive text splitter
        
        Args:
            text: Text to chunk
            document_metadata: Optional metadata to include with chunks
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text.strip():
            return []
        
        if LLAMA_INDEX_AVAILABLE and self.text_splitter:
            return self._create_llama_index_chunks(text, document_metadata)
        else:
            return self._create_fallback_chunks(text, document_metadata)
    
    def _create_llama_index_chunks(self, text: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Create chunks using LlamaIndex sentence splitter"""
        try:
            # Use the text splitter directly to split text
            text_chunks = self.text_splitter.split_text(text)
            
            # Convert to our chunk format
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunk_data = {
                    'content': chunk_text,
                    'chunk_index': i,
                    'metadata': {
                        **(document_metadata or {}),
                        'chunk_type': 'llama_index',
                        'chunk_method': 'sentence_splitter',
                        'chunk_size': len(chunk_text),
                    }
                }
                chunks.append(chunk_data)
            
            logger.info(f"Created {len(chunks)} chunks using LlamaIndex sentence splitter")
            return chunks
            
        except Exception as e:
            logger.error(f"LlamaIndex chunking failed: {e}")
            # Fall back to simple chunking
            return self._create_fallback_chunks(text, document_metadata)
    
    def _create_fallback_chunks(self, text: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fallback chunking method when LlamaIndex is not available"""
        try:
            chunks = []
            start = 0
            text_length = len(text)
            chunk_index = 0
            
            while start < text_length:
                end = start + self.config.chunk_size
                
                # Try to break at natural boundaries
                if end < text_length:
                    # Look for paragraph break first
                    para_break = text.rfind('\n\n', start, end)
                    if para_break > start:
                        end = para_break + 2
                    else:
                        # Look for sentence boundary
                        sentence_end = max(
                            text.rfind('. ', start, end),
                            text.rfind('! ', start, end),
                            text.rfind('? ', start, end)
                        )
                        
                        if sentence_end > start:
                            end = sentence_end + 2
                        else:
                            # Look for word boundary
                            word_end = text.rfind(' ', start, end)
                            if word_end > start:
                                end = word_end
                
                chunk_content = text[start:end].strip()
                if chunk_content:
                    chunk_data = {
                        'content': chunk_content,
                        'chunk_index': chunk_index,
                        'metadata': {
                            **(document_metadata or {}),
                            'chunk_type': 'fallback',
                            'chunk_method': 'recursive_fallback',
                            'start_char_idx': start,
                            'end_char_idx': end,
                        }
                    }
                    chunks.append(chunk_data)
                    chunk_index += 1
                
                # Move start position with overlap
                start = max(start + 1, end - self.config.chunk_overlap)
                
                # Prevent infinite loop
                if start >= text_length:
                    break
            
            logger.info(f"Created {len(chunks)} chunks using fallback recursive method")
            return chunks
            
        except Exception as e:
            logger.error(f"Fallback chunking failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to create text chunks")
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the created chunks"""
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_characters': 0
            }
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'chunking_method': chunks[0]['metadata'].get('chunk_method', 'unknown') if chunks else 'unknown'
        }
    
    def update_config(self, **kwargs):
        """Update chunking configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Reinitialize splitter with new config
        if LLAMA_INDEX_AVAILABLE:
            self._initialize_llama_index_splitter()
        
        logger.info(f"Updated chunking configuration: {kwargs}")


# Global chunking service instance
chunking_service = AdvancedChunkingService()


def create_advanced_chunks(
    text: str, 
    document_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Main function to create advanced chunks
    
    Args:
        text: Text to chunk
        document_metadata: Optional metadata to include
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries
    """
    # Update config if different from defaults
    if chunk_size != chunking_service.config.chunk_size or chunk_overlap != chunking_service.config.chunk_overlap:
        chunking_service.update_config(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return chunking_service.create_chunks(text, document_metadata)


def get_chunking_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics about chunks"""
    return chunking_service.get_chunk_statistics(chunks)


def is_llama_index_available() -> bool:
    """Check if LlamaIndex is available"""
    return LLAMA_INDEX_AVAILABLE