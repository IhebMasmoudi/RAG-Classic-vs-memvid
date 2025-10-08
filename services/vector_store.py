"""
Vector store service for managing FAISS embeddings
"""
import os
import pickle
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss
from pathlib import Path

from sqlalchemy.orm import Session
from fastapi import HTTPException

from models.database import Document, DocumentChunk
from utils.database import get_db_context
from config import settings
from services.embedding_service import get_embedding_dimension

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self):
        self.index = None
        self.dimension = get_embedding_dimension()  # Get dimension from embedding service
        self.document_chunks = {}  # Maps vector_id to chunk metadata
        self.vector_counter = 0
        self.index_path = Path("data/vector_store")
        self.index_path.mkdir(parents=True, exist_ok=True)
        
    def initialize_index(self, dimension: Optional[int] = None):
        """Initialize FAISS index"""
        try:
            if dimension is None:
                dimension = get_embedding_dimension()
            self.dimension = dimension
            # Use IndexFlatIP for cosine similarity (after normalization)
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"Initialized FAISS index with dimension {dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize vector store")
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        document_id: str,
        chunks: List[DocumentChunk]
    ) -> List[str]:
        """Add embeddings to the vector store"""
        if self.index is None:
            self.initialize_index(len(embeddings[0]) if embeddings else 384)
        
        try:
            # Convert to numpy array and normalize for cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            start_id = self.vector_counter
            self.index.add(embeddings_array)
            
            # Store chunk metadata
            vector_ids = []
            for i, chunk in enumerate(chunks):
                vector_id = f"{document_id}_{chunk.chunk_index}"
                vector_ids.append(vector_id)
                
                self.document_chunks[start_id + i] = {
                    'vector_id': vector_id,
                    'document_id': document_id,
                    'chunk_id': chunk.id,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content
                }
            
            self.vector_counter += len(embeddings)
            
            # Save index and metadata
            self._save_index()
            
            logger.info(f"Added {len(embeddings)} embeddings for document {document_id}")
            return vector_ids
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise HTTPException(status_code=500, detail="Failed to add embeddings to vector store")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        user_id: Optional[int] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query embedding
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                chunk_metadata = self.document_chunks.get(idx)
                if chunk_metadata:
                    # Filter by user if specified
                    if user_id:
                        with get_db_context() as db:
                            document = db.query(Document).filter(
                                Document.id == chunk_metadata['document_id']
                            ).first()
                            if not document or document.user_id != user_id:
                                continue
                    
                    # Filter by document IDs if specified
                    if document_ids and chunk_metadata['document_id'] not in document_ids:
                        continue
                    
                    results.append({
                        'chunk_id': chunk_metadata['chunk_id'],
                        'document_id': chunk_metadata['document_id'],
                        'chunk_index': chunk_metadata['chunk_index'],
                        'content': chunk_metadata['content'],
                        'similarity_score': float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise HTTPException(status_code=500, detail="Vector search failed")
    
    def remove_document_embeddings(self, document_id: str):
        """Remove all embeddings for a document"""
        try:
            # Find indices to remove
            indices_to_remove = []
            for idx, metadata in self.document_chunks.items():
                if metadata['document_id'] == document_id:
                    indices_to_remove.append(idx)
            
            # Remove from metadata
            for idx in indices_to_remove:
                del self.document_chunks[idx]
            
            # Note: FAISS doesn't support efficient removal of specific vectors
            # In a production system, you might want to rebuild the index periodically
            # or use a different vector database that supports deletion
            
            logger.info(f"Removed embeddings for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove document embeddings: {e}")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.index is not None:
                # Save FAISS index
                index_file = self.index_path / "faiss_index.bin"
                faiss.write_index(self.index, str(index_file))
                
                # Save metadata
                metadata_file = self.index_path / "metadata.pkl"
                with open(metadata_file, 'wb') as f:
                    pickle.dump({
                        'document_chunks': self.document_chunks,
                        'vector_counter': self.vector_counter,
                        'dimension': self.dimension
                    }, f)
                
                logger.debug("Saved vector store to disk")
                
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            index_file = self.index_path / "faiss_index.bin"
            metadata_file = self.index_path / "metadata.pkl"
            
            if index_file.exists() and metadata_file.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))
                
                # Load metadata
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self.document_chunks = metadata['document_chunks']
                    self.vector_counter = metadata['vector_counter']
                    self.dimension = metadata['dimension']
                
                logger.info(f"Loaded vector store with {self.index.ntotal} vectors")
            else:
                logger.info("No existing vector store found, will create new one")
                
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            # Initialize empty index on failure
            self.initialize_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'total_documents': len(set(
                metadata['document_id'] 
                for metadata in self.document_chunks.values()
            ))
        }


# Global vector store instance
vector_store = VectorStore()


async def initialize_vector_store():
    """Initialize the vector store on startup"""
    try:
        vector_store.load_index()
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise


async def add_document_to_vector_store(
    document_id: str,
    embeddings: List[List[float]],
    chunks: List[DocumentChunk]
) -> List[str]:
    """Add document embeddings to vector store"""
    return vector_store.add_embeddings(embeddings, document_id, chunks)


async def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = 5,
    user_id: Optional[int] = None,
    document_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Search for similar chunks"""
    return vector_store.search(query_embedding, top_k, user_id, document_ids)


async def remove_document_from_vector_store(document_id: str):
    """Remove document from vector store"""
    vector_store.remove_document_embeddings(document_id)


async def get_vector_store_stats() -> Dict[str, Any]:
    """Get vector store statistics"""
    return vector_store.get_stats()