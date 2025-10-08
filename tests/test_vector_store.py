"""
Unit tests for vector store service
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path

from services.vector_store import VectorStore, vector_store
from models.database import DocumentChunk, Document


class TestVectorStore:
    """Test cases for VectorStore class"""
    
    @pytest.fixture
    def temp_vector_store(self):
        """Create a temporary vector store for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore()
            store.index_path = Path(temp_dir) / "test_vector_store"
            store.index_path.mkdir(parents=True, exist_ok=True)
            yield store
    
    def test_initialize_index(self, temp_vector_store):
        """Test FAISS index initialization"""
        store = temp_vector_store
        store.initialize_index(dimension=384)
        
        assert store.index is not None
        assert store.dimension == 384
        assert store.index.ntotal == 0
    
    def test_add_embeddings_success(self, temp_vector_store):
        """Test successful embedding addition"""
        store = temp_vector_store
        store.initialize_index(dimension=3)
        
        # Create test data
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        document_id = "test-doc-1"
        
        # Create mock chunks
        chunks = []
        for i in range(len(embeddings)):
            chunk = Mock(spec=DocumentChunk)
            chunk.id = f"chunk-{i}"
            chunk.chunk_index = i
            chunk.content = f"Test content {i}"
            chunks.append(chunk)
        
        # Add embeddings
        vector_ids = store.add_embeddings(embeddings, document_id, chunks)
        
        assert len(vector_ids) == 2
        assert store.index.ntotal == 2
        assert store.vector_counter == 2
        assert len(store.document_chunks) == 2
    
    def test_add_embeddings_auto_initialize(self, temp_vector_store):
        """Test that index is auto-initialized when adding embeddings"""
        store = temp_vector_store
        
        embeddings = [[0.1, 0.2, 0.3]]
        document_id = "test-doc-1"
        chunk = Mock(spec=DocumentChunk)
        chunk.id = "chunk-1"
        chunk.chunk_index = 0
        chunk.content = "Test content"
        
        vector_ids = store.add_embeddings(embeddings, document_id, [chunk])
        
        assert store.index is not None
        assert store.dimension == 3
        assert len(vector_ids) == 1
    
    def test_search_embeddings(self, temp_vector_store):
        """Test embedding search functionality"""
        store = temp_vector_store
        store.initialize_index(dimension=3)
        
        # Add test embeddings
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        document_id = "test-doc-1"
        
        chunks = []
        for i in range(len(embeddings)):
            chunk = Mock(spec=DocumentChunk)
            chunk.id = f"chunk-{i}"
            chunk.chunk_index = i
            chunk.content = f"Test content {i}"
            chunks.append(chunk)
        
        store.add_embeddings(embeddings, document_id, chunks)
        
        # Search for similar embedding
        query_embedding = [0.9, 0.1, 0.0]  # Should be closest to first embedding
        results = store.search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        assert results[0]['chunk_id'] == 'chunk-0'  # Should be most similar
        assert results[0]['similarity_score'] > results[1]['similarity_score']
    
    def test_search_empty_index(self, temp_vector_store):
        """Test search on empty index"""
        store = temp_vector_store
        
        query_embedding = [0.1, 0.2, 0.3]
        results = store.search(query_embedding, top_k=5)
        
        assert results == []
    
    def test_search_with_user_filter(self, temp_vector_store):
        """Test search with user ID filtering"""
        store = temp_vector_store
        store.initialize_index(dimension=3)
        
        # Add embeddings
        embeddings = [[1.0, 0.0, 0.0]]
        document_id = "test-doc-1"
        chunk = Mock(spec=DocumentChunk)
        chunk.id = "chunk-1"
        chunk.chunk_index = 0
        chunk.content = "Test content"
        
        store.add_embeddings(embeddings, document_id, [chunk])
        
        # Mock database query
        with patch('services.vector_store.get_db_context') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db
            
            # Mock document with correct user
            mock_document = Mock(spec=Document)
            mock_document.user_id = 1
            mock_db.query.return_value.filter.return_value.first.return_value = mock_document
            
            query_embedding = [1.0, 0.0, 0.0]
            results = store.search(query_embedding, top_k=5, user_id=1)
            
            assert len(results) == 1
            assert results[0]['chunk_id'] == 'chunk-1'
    
    def test_search_user_filter_no_access(self, temp_vector_store):
        """Test search with user filter when user has no access"""
        store = temp_vector_store
        store.initialize_index(dimension=3)
        
        # Add embeddings
        embeddings = [[1.0, 0.0, 0.0]]
        document_id = "test-doc-1"
        chunk = Mock(spec=DocumentChunk)
        chunk.id = "chunk-1"
        chunk.chunk_index = 0
        chunk.content = "Test content"
        
        store.add_embeddings(embeddings, document_id, [chunk])
        
        # Mock database query
        with patch('services.vector_store.get_db_context') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db
            
            # Mock document with different user
            mock_document = Mock(spec=Document)
            mock_document.user_id = 2
            mock_db.query.return_value.filter.return_value.first.return_value = mock_document
            
            query_embedding = [1.0, 0.0, 0.0]
            results = store.search(query_embedding, top_k=5, user_id=1)
            
            assert len(results) == 0
    
    def test_remove_document_embeddings(self, temp_vector_store):
        """Test removing document embeddings"""
        store = temp_vector_store
        store.initialize_index(dimension=3)
        
        # Add embeddings for two documents
        embeddings1 = [[1.0, 0.0, 0.0]]
        embeddings2 = [[0.0, 1.0, 0.0]]
        
        chunk1 = Mock(spec=DocumentChunk)
        chunk1.id = "chunk-1"
        chunk1.chunk_index = 0
        chunk1.content = "Content 1"
        
        chunk2 = Mock(spec=DocumentChunk)
        chunk2.id = "chunk-2"
        chunk2.chunk_index = 0
        chunk2.content = "Content 2"
        
        store.add_embeddings(embeddings1, "doc-1", [chunk1])
        store.add_embeddings(embeddings2, "doc-2", [chunk2])
        
        assert len(store.document_chunks) == 2
        
        # Remove first document
        store.remove_document_embeddings("doc-1")
        
        # Check that only doc-2 chunks remain
        remaining_docs = set(
            metadata['document_id'] 
            for metadata in store.document_chunks.values()
        )
        assert remaining_docs == {"doc-2"}
    
    def test_save_and_load_index(self, temp_vector_store):
        """Test saving and loading index from disk"""
        store = temp_vector_store
        store.initialize_index(dimension=3)
        
        # Add some data
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        document_id = "test-doc-1"
        
        chunks = []
        for i in range(len(embeddings)):
            chunk = Mock(spec=DocumentChunk)
            chunk.id = f"chunk-{i}"
            chunk.chunk_index = i
            chunk.content = f"Test content {i}"
            chunks.append(chunk)
        
        store.add_embeddings(embeddings, document_id, chunks)
        
        # Save index
        store._save_index()
        
        # Create new store and load
        new_store = VectorStore()
        new_store.index_path = store.index_path
        new_store.load_index()
        
        assert new_store.index.ntotal == 2
        assert new_store.vector_counter == 2
        assert len(new_store.document_chunks) == 2
        assert new_store.dimension == 3
    
    def test_load_index_no_existing_files(self, temp_vector_store):
        """Test loading index when no files exist"""
        store = temp_vector_store
        
        # Should not raise exception but index will be None until first use
        store.load_index()
        
        # Index should be None since no files exist and no initialization happened
        assert store.index is None
    
    def test_get_stats(self, temp_vector_store):
        """Test getting vector store statistics"""
        store = temp_vector_store
        store.initialize_index(dimension=3)
        
        # Add embeddings from two documents
        embeddings1 = [[1.0, 0.0, 0.0]]
        embeddings2 = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        
        chunk1 = Mock(spec=DocumentChunk)
        chunk1.id = "chunk-1"
        chunk1.chunk_index = 0
        chunk1.content = "Content 1"
        
        chunk2 = Mock(spec=DocumentChunk)
        chunk2.id = "chunk-2"
        chunk2.chunk_index = 0
        chunk2.content = "Content 2"
        
        chunk3 = Mock(spec=DocumentChunk)
        chunk3.id = "chunk-3"
        chunk3.chunk_index = 1
        chunk3.content = "Content 3"
        
        store.add_embeddings(embeddings1, "doc-1", [chunk1])
        store.add_embeddings(embeddings2, "doc-2", [chunk2, chunk3])
        
        stats = store.get_stats()
        
        assert stats['total_vectors'] == 3
        assert stats['dimension'] == 3
        assert stats['total_documents'] == 2


class TestVectorStoreIntegration:
    """Integration tests for vector store functions"""
    
    @pytest.mark.asyncio
    async def test_initialize_vector_store(self):
        """Test vector store initialization"""
        with patch('services.vector_store.vector_store') as mock_store:
            mock_store.load_index = Mock()
            
            from services.vector_store import initialize_vector_store
            await initialize_vector_store()
            
            mock_store.load_index.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_document_to_vector_store(self):
        """Test adding document to vector store"""
        with patch('services.vector_store.vector_store') as mock_store:
            mock_store.add_embeddings.return_value = ["vec-1", "vec-2"]
            
            from services.vector_store import add_document_to_vector_store
            
            embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            chunks = [Mock(), Mock()]
            
            result = await add_document_to_vector_store("doc-1", embeddings, chunks)
            
            assert result == ["vec-1", "vec-2"]
            mock_store.add_embeddings.assert_called_once_with(embeddings, "doc-1", chunks)
    
    @pytest.mark.asyncio
    async def test_search_similar_chunks(self):
        """Test searching for similar chunks"""
        with patch('services.vector_store.vector_store') as mock_store:
            mock_store.search.return_value = [
                {
                    'chunk_id': 'chunk-1',
                    'content': 'Test content',
                    'similarity_score': 0.9
                }
            ]
            
            from services.vector_store import search_similar_chunks
            
            query_embedding = [0.1, 0.2, 0.3]
            result = await search_similar_chunks(query_embedding, top_k=5, user_id=1)
            
            assert len(result) == 1
            assert result[0]['chunk_id'] == 'chunk-1'
            mock_store.search.assert_called_once_with(query_embedding, 5, 1)
    
    @pytest.mark.asyncio
    async def test_remove_document_from_vector_store(self):
        """Test removing document from vector store"""
        with patch('services.vector_store.vector_store') as mock_store:
            from services.vector_store import remove_document_from_vector_store
            
            await remove_document_from_vector_store("doc-1")
            
            mock_store.remove_document_embeddings.assert_called_once_with("doc-1")
    
    @pytest.mark.asyncio
    async def test_get_vector_store_stats(self):
        """Test getting vector store statistics"""
        with patch('services.vector_store.vector_store') as mock_store:
            mock_store.get_stats.return_value = {
                'total_vectors': 100,
                'dimension': 384,
                'total_documents': 10
            }
            
            from services.vector_store import get_vector_store_stats
            
            result = await get_vector_store_stats()
            
            assert result['total_vectors'] == 100
            assert result['dimension'] == 384
            assert result['total_documents'] == 10
            mock_store.get_stats.assert_called_once()


class TestVectorStoreErrorHandling:
    """Test error handling in vector store operations"""
    
    def test_initialize_index_error(self, temp_vector_store):
        """Test error handling during index initialization"""
        store = temp_vector_store
        
        with patch('services.vector_store.faiss.IndexFlatIP') as mock_index:
            mock_index.side_effect = Exception("FAISS initialization failed")
            
            with pytest.raises(Exception):
                store.initialize_index()
    
    def test_add_embeddings_error(self, temp_vector_store):
        """Test error handling during embedding addition"""
        store = temp_vector_store
        store.initialize_index(dimension=3)
        
        # Invalid embeddings (wrong dimension)
        embeddings = [[0.1, 0.2]]  # 2D instead of 3D
        chunk = Mock(spec=DocumentChunk)
        chunk.id = "chunk-1"
        chunk.chunk_index = 0
        chunk.content = "Test content"
        
        with pytest.raises(Exception):
            store.add_embeddings(embeddings, "doc-1", [chunk])
    
    def test_search_error(self, temp_vector_store):
        """Test error handling during search"""
        store = temp_vector_store
        store.initialize_index(dimension=3)
        
        # Add valid embedding
        embeddings = [[1.0, 0.0, 0.0]]
        chunk = Mock(spec=DocumentChunk)
        chunk.id = "chunk-1"
        chunk.chunk_index = 0
        chunk.content = "Test content"
        store.add_embeddings(embeddings, "doc-1", [chunk])
        
        # Search with wrong dimension
        query_embedding = [1.0, 0.0]  # 2D instead of 3D
        
        with pytest.raises(Exception):
            store.search(query_embedding, top_k=5)