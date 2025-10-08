"""
Unit tests for MemVid RAG routes
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, Mock
from datetime import datetime

from main import app
from models.database import Base
from models.schemas import MemVidRAGResponse, SourceChunk
from utils.database import get_db
from services.user_service import UserService
from models.schemas import UserCreate


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_memvid_rag_routes.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def client():
    """Create test client"""
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def existing_user():
    """Create an existing user for testing"""
    db = TestingSessionLocal()
    try:
        user_data = UserCreate(
            email="memvid_test@example.com",
            password="test_password_123"
        )
        user = UserService.create_user(db, user_data)
        return user
    finally:
        db.close()


@pytest.fixture
def auth_headers(client, existing_user):
    """Get authentication headers for testing"""
    login_data = {
        "email": existing_user.email,
        "password": "test_password_123"
    }
    response = client.post("/auth/login", json=login_data)
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestMemVidRAGQueryEndpoint:
    """Test MemVid RAG query endpoint"""
    
    def test_memvid_rag_query_success(self, client, auth_headers):
        """Test successful MemVid RAG query"""
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5,
            "context_window": 3
        }
        
        mock_response = MemVidRAGResponse(
            answer="Machine learning is a subset of artificial intelligence that enables computers to learn.",
            sources=[
                SourceChunk(
                    chunk_id="chunk_1",
                    content="Machine learning involves training algorithms on data.",
                    similarity_score=0.95,
                    document_id="doc_1",
                    chunk_index=0
                )
            ],
            response_time=2.1,
            chunks_used=1,
            query="What is machine learning?",
            timestamp=datetime.utcnow(),
            memvid_metadata={
                "enhanced_query": "Context: Recent topics include AI. Current query: What is machine learning?",
                "query_context": {"recent_topics": ["AI"]},
                "retrieval_strategy": "hierarchical",
                "context_window_used": 3,
                "assembly_metadata": {"total_documents": 1, "primary_chunks": 1, "context_chunks": 0},
                "memory_cache_size": 0,
                "processing_stages": ["query_enhancement", "hierarchical_retrieval", "context_assembly", "enhanced_generation", "memory_update"]
            }
        )
        
        with patch('routes.memvid_rag.process_memvid_rag_query') as mock_process:
            mock_process.return_value = mock_response
            
            response = client.post(
                "/memvid_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Machine learning is a subset of artificial intelligence that enables computers to learn."
            assert len(data["sources"]) == 1
            assert data["chunks_used"] == 1
            assert data["response_time"] == 2.1
            assert data["query"] == "What is machine learning?"
            assert "memvid_metadata" in data
            assert data["memvid_metadata"]["retrieval_strategy"] == "hierarchical"
            assert data["memvid_metadata"]["context_window_used"] == 3
            
            mock_process.assert_called_once()
    
    def test_memvid_rag_query_empty_query(self, client, auth_headers):
        """Test MemVid RAG query with empty query"""
        query_data = {
            "query": "",
            "top_k": 5,
            "context_window": 3
        }
        
        response = client.post(
            "/memvid_rag/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Query cannot be empty" in data["detail"]
    
    def test_memvid_rag_query_invalid_context_window(self, client, auth_headers):
        """Test MemVid RAG query with invalid context window"""
        # Test negative context window
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5,
            "context_window": -1
        }
        
        response = client.post(
            "/memvid_rag/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Context window must be between 0 and 10" in data["detail"]
        
        # Test context window too large
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5,
            "context_window": 15
        }
        
        response = client.post(
            "/memvid_rag/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Context window must be between 0 and 10" in data["detail"]
    
    def test_memvid_rag_query_no_authentication(self, client):
        """Test MemVid RAG query without authentication"""
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5,
            "context_window": 3
        }
        
        response = client.post("/memvid_rag/query", json=query_data)
        
        assert response.status_code == 403
    
    def test_memvid_rag_query_invalid_token(self, client):
        """Test MemVid RAG query with invalid token"""
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5,
            "context_window": 3
        }
        headers = {"Authorization": "Bearer invalid_token"}
        
        response = client.post(
            "/memvid_rag/query",
            json=query_data,
            headers=headers
        )
        
        assert response.status_code == 401
    
    def test_memvid_rag_query_default_context_window(self, client, auth_headers):
        """Test MemVid RAG query with default context window"""
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5
            # context_window should default to 3
        }
        
        mock_response = MemVidRAGResponse(
            answer="Test answer",
            sources=[],
            response_time=1.0,
            chunks_used=0,
            query="What is machine learning?",
            timestamp=datetime.utcnow(),
            memvid_metadata={"context_window_used": 3}
        )
        
        with patch('routes.memvid_rag.process_memvid_rag_query') as mock_process:
            mock_process.return_value = mock_response
            
            response = client.post(
                "/memvid_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            
            # Verify that the default context_window was used
            call_args = mock_process.call_args[0][0]  # MemVidQueryRequest object
            assert call_args.context_window == 3
    
    def test_memvid_rag_query_processing_error(self, client, auth_headers):
        """Test MemVid RAG query when processing fails"""
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5,
            "context_window": 3
        }
        
        with patch('routes.memvid_rag.process_memvid_rag_query') as mock_process:
            mock_process.side_effect = Exception("Processing failed")
            
            response = client.post(
                "/memvid_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]
    
    def test_memvid_rag_query_http_exception(self, client, auth_headers):
        """Test MemVid RAG query when HTTPException is raised"""
        from fastapi import HTTPException
        
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5,
            "context_window": 3
        }
        
        with patch('routes.memvid_rag.process_memvid_rag_query') as mock_process:
            mock_process.side_effect = HTTPException(
                status_code=503,
                detail="MemVid service temporarily unavailable"
            )
            
            response = client.post(
                "/memvid_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 503
            data = response.json()
            assert "MemVid service temporarily unavailable" in data["detail"]


class TestMemVidRAGMemoryEndpoints:
    """Test MemVid RAG memory management endpoints"""
    
    def test_get_memory_stats_success(self, client, auth_headers):
        """Test successful memory stats retrieval"""
        with patch('routes.memvid_rag.get_memvid_memory_stats') as mock_stats:
            mock_stats.return_value = {
                "cache_size": 5,
                "max_cache_size": 100,
                "cache_utilization": 0.05,
                "oldest_entry": 1234567890,
                "newest_entry": 1234567900
            }
            
            response = client.get("/memvid_rag/memory/stats", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "memory_stats" in data
            assert data["memory_stats"]["cache_size"] == 5
            assert data["memory_stats"]["cache_utilization"] == 0.05
    
    def test_get_memory_stats_no_auth(self, client):
        """Test memory stats without authentication"""
        response = client.get("/memvid_rag/memory/stats")
        assert response.status_code == 403
    
    def test_get_memory_stats_error(self, client, auth_headers):
        """Test memory stats with service error"""
        with patch('routes.memvid_rag.get_memvid_memory_stats') as mock_stats:
            mock_stats.side_effect = Exception("Stats error")
            
            response = client.get("/memvid_rag/memory/stats", headers=auth_headers)
            
            assert response.status_code == 500
            data = response.json()
            assert "Failed to retrieve memory statistics" in data["detail"]
    
    def test_clear_memory_success(self, client, auth_headers):
        """Test successful memory cache clearing"""
        with patch('routes.memvid_rag.clear_memvid_memory') as mock_clear:
            mock_clear.return_value = None
            
            response = client.post("/memvid_rag/memory/clear", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "cleared successfully" in data["message"]
            mock_clear.assert_called_once()
    
    def test_clear_memory_no_auth(self, client):
        """Test memory clearing without authentication"""
        response = client.post("/memvid_rag/memory/clear")
        assert response.status_code == 403
    
    def test_clear_memory_error(self, client, auth_headers):
        """Test memory clearing with service error"""
        with patch('routes.memvid_rag.clear_memvid_memory') as mock_clear:
            mock_clear.side_effect = Exception("Clear error")
            
            response = client.post("/memvid_rag/memory/clear", headers=auth_headers)
            
            assert response.status_code == 500
            data = response.json()
            assert "Failed to clear memory cache" in data["detail"]


class TestMemVidRAGHealthEndpoint:
    """Test MemVid RAG health endpoint"""
    
    def test_memvid_rag_health_success(self, client):
        """Test successful health check"""
        with patch('routes.memvid_rag.get_llm_info') as mock_llm_info, \
             patch('routes.memvid_rag.get_vector_store_stats') as mock_vector_stats, \
             patch('routes.memvid_rag.get_embedding_info') as mock_embedding_info, \
             patch('routes.memvid_rag.get_memvid_memory_stats') as mock_memory_stats:
            
            mock_llm_info.return_value = {
                "model_name": "gemini-1.5-flash",
                "gemini_available": True,
                "primary_service": "gemini"
            }
            mock_vector_stats.return_value = {
                "total_vectors": 1000,
                "dimension": 768,
                "total_documents": 10
            }
            mock_embedding_info.return_value = {
                "gemini_available": True,
                "primary_service": "gemini",
                "embedding_dimension": 768
            }
            mock_memory_stats.return_value = {
                "cache_size": 5,
                "max_cache_size": 100
            }
            
            response = client.get("/memvid_rag/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "MemVid-Inspired RAG Pipeline"
            assert "llm_service" in data
            assert "vector_store" in data
            assert "embedding_service" in data
            assert "memory_cache" in data
            assert "features" in data
            assert "memvid_enhancements" in data
            assert len(data["features"]) > 0
            assert len(data["memvid_enhancements"]) > 0
    
    def test_memvid_rag_health_service_error(self, client):
        """Test health check when service has errors"""
        with patch('routes.memvid_rag.get_llm_info') as mock_llm_info:
            mock_llm_info.side_effect = Exception("Service error")
            
            response = client.get("/memvid_rag/health")
            
            assert response.status_code == 503
            data = response.json()
            assert "MemVid RAG service is not healthy" in data["detail"]


class TestMemVidRAGComparisonEndpoint:
    """Test MemVid RAG comparison endpoint"""
    
    def test_get_comparison_data_success(self, client, auth_headers, existing_user):
        """Test successful comparison data retrieval"""
        with patch('routes.memvid_rag.QueryHistory') as mock_query_history:
            # Mock database query
            mock_history = Mock()
            mock_history.query_text = "What is AI?"
            mock_history.query_timestamp = datetime.utcnow()
            mock_history.classic_answer = "AI is artificial intelligence"
            mock_history.classic_response_time = 1.5
            mock_history.classic_chunks_used = 3
            mock_history.classic_sources = '[]'
            mock_history.memvid_answer = "AI is artificial intelligence with enhanced context"
            mock_history.memvid_response_time = 1.2
            mock_history.memvid_chunks_used = 5
            mock_history.memvid_sources = '[]'
            mock_history.memvid_metadata = '{"enhanced_query": "Enhanced AI query"}'
            
            # Mock database session
            with patch('routes.memvid_rag.get_db') as mock_get_db:
                mock_db = Mock()
                mock_db.query.return_value.filter.return_value.first.return_value = mock_history
                mock_get_db.return_value = mock_db
                
                response = client.get("/memvid_rag/compare/123", headers=auth_headers)
                
                assert response.status_code == 200
                data = response.json()
                assert data["query"] == "What is AI?"
                assert "classic_rag" in data
                assert "memvid_rag" in data
                assert "performance_comparison" in data
                assert data["performance_comparison"]["memvid_faster"] is True
    
    def test_get_comparison_data_not_found(self, client, auth_headers):
        """Test comparison data when query not found"""
        with patch('routes.memvid_rag.get_db') as mock_get_db:
            mock_db = Mock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_get_db.return_value = mock_db
            
            response = client.get("/memvid_rag/compare/999", headers=auth_headers)
            
            assert response.status_code == 404
            data = response.json()
            assert "Query not found" in data["detail"]
    
    def test_get_comparison_data_no_auth(self, client):
        """Test comparison data without authentication"""
        response = client.get("/memvid_rag/compare/123")
        assert response.status_code == 403
    
    def test_get_comparison_data_partial_results(self, client, auth_headers):
        """Test comparison data with only one RAG result"""
        with patch('routes.memvid_rag.get_db') as mock_get_db:
            mock_db = Mock()
            mock_history = Mock()
            mock_history.query_text = "What is AI?"
            mock_history.query_timestamp = datetime.utcnow()
            mock_history.classic_answer = None  # No classic result
            mock_history.memvid_answer = "AI is artificial intelligence"
            mock_history.memvid_response_time = 1.2
            mock_history.memvid_chunks_used = 5
            mock_history.memvid_sources = '[]'
            mock_history.memvid_metadata = '{}'
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_history
            mock_get_db.return_value = mock_db
            
            response = client.get("/memvid_rag/compare/123", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["classic_rag"] is None
            assert data["memvid_rag"] is not None
            assert "performance_comparison" not in data  # No comparison without both results


class TestMemVidRAGIntegration:
    """Integration tests for MemVid RAG routes"""
    
    def test_complete_memvid_workflow(self, client, auth_headers):
        """Test complete MemVid workflow from query to memory management"""
        # First check health
        health_response = client.get("/memvid_rag/health")
        assert health_response.status_code in [200, 503]  # May fail if services not available
        
        # Make a query
        query_data = {
            "query": "Explain artificial intelligence and machine learning",
            "top_k": 5,
            "context_window": 2
        }
        
        mock_response = MemVidRAGResponse(
            answer="AI and ML are related fields in computer science.",
            sources=[],
            response_time=1.8,
            chunks_used=3,
            query="Explain artificial intelligence and machine learning",
            timestamp=datetime.utcnow(),
            memvid_metadata={
                "enhanced_query": "Context: Recent topics include technology. Current query: Explain artificial intelligence and machine learning",
                "retrieval_strategy": "hierarchical",
                "context_window_used": 2,
                "memory_cache_size": 1
            }
        )
        
        with patch('routes.memvid_rag.process_memvid_rag_query') as mock_process:
            mock_process.return_value = mock_response
            
            response = client.post(
                "/memvid_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "memvid_metadata" in data
            assert data["memvid_metadata"]["retrieval_strategy"] == "hierarchical"
        
        # Check memory stats
        with patch('routes.memvid_rag.get_memvid_memory_stats') as mock_stats:
            mock_stats.return_value = {"cache_size": 1, "max_cache_size": 100}
            
            stats_response = client.get("/memvid_rag/memory/stats", headers=auth_headers)
            assert stats_response.status_code == 200
            stats_data = stats_response.json()
            assert stats_data["memory_stats"]["cache_size"] == 1
        
        # Clear memory
        with patch('routes.memvid_rag.clear_memvid_memory') as mock_clear:
            clear_response = client.post("/memvid_rag/memory/clear", headers=auth_headers)
            assert clear_response.status_code == 200
            mock_clear.assert_called_once()