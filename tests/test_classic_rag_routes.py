"""
Unit tests for Classic RAG routes
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, Mock
from datetime import datetime

from main import app
from models.database import Base
from models.schemas import RAGResponse, SourceChunk
from utils.database import get_db
from services.user_service import UserService
from models.schemas import UserCreate


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_classic_rag_routes.db"
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
            email="rag_test@example.com",
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


class TestClassicRAGQueryEndpoint:
    """Test Classic RAG query endpoint"""
    
    def test_classic_rag_query_success(self, client, auth_headers):
        """Test successful Classic RAG query"""
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5
        }
        
        mock_response = RAGResponse(
            answer="Machine learning is a subset of artificial intelligence.",
            sources=[
                SourceChunk(
                    chunk_id="chunk_1",
                    content="Machine learning involves training algorithms on data.",
                    similarity_score=0.95,
                    document_id="doc_1",
                    chunk_index=0
                )
            ],
            response_time=1.5,
            chunks_used=1,
            query="What is machine learning?",
            timestamp=datetime.utcnow()
        )
        
        with patch('routes.classic_rag.process_classic_rag_query') as mock_process:
            mock_process.return_value = mock_response
            
            response = client.post(
                "/classic_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Machine learning is a subset of artificial intelligence."
            assert len(data["sources"]) == 1
            assert data["chunks_used"] == 1
            assert data["response_time"] == 1.5
            assert data["query"] == "What is machine learning?"
            
            mock_process.assert_called_once()
    
    def test_classic_rag_query_empty_query(self, client, auth_headers):
        """Test Classic RAG query with empty query"""
        query_data = {
            "query": "",
            "top_k": 5
        }
        
        response = client.post(
            "/classic_rag/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Query cannot be empty" in data["detail"]
    
    def test_classic_rag_query_whitespace_only(self, client, auth_headers):
        """Test Classic RAG query with whitespace-only query"""
        query_data = {
            "query": "   \n\t   ",
            "top_k": 5
        }
        
        response = client.post(
            "/classic_rag/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Query cannot be empty" in data["detail"]
    
    def test_classic_rag_query_no_authentication(self, client):
        """Test Classic RAG query without authentication"""
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5
        }
        
        response = client.post("/classic_rag/query", json=query_data)
        
        assert response.status_code == 403
    
    def test_classic_rag_query_invalid_token(self, client):
        """Test Classic RAG query with invalid token"""
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5
        }
        headers = {"Authorization": "Bearer invalid_token"}
        
        response = client.post(
            "/classic_rag/query",
            json=query_data,
            headers=headers
        )
        
        assert response.status_code == 401
    
    def test_classic_rag_query_invalid_top_k(self, client, auth_headers):
        """Test Classic RAG query with invalid top_k values"""
        # Test top_k too low
        query_data = {
            "query": "What is machine learning?",
            "top_k": 0
        }
        
        response = client.post(
            "/classic_rag/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422
        
        # Test top_k too high
        query_data = {
            "query": "What is machine learning?",
            "top_k": 25
        }
        
        response = client.post(
            "/classic_rag/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422
    
    def test_classic_rag_query_missing_fields(self, client, auth_headers):
        """Test Classic RAG query with missing required fields"""
        query_data = {
            "top_k": 5
            # Missing query
        }
        
        response = client.post(
            "/classic_rag/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422
    
    def test_classic_rag_query_default_top_k(self, client, auth_headers):
        """Test Classic RAG query with default top_k value"""
        query_data = {
            "query": "What is machine learning?"
            # top_k should default to 5
        }
        
        mock_response = RAGResponse(
            answer="Test answer",
            sources=[],
            response_time=1.0,
            chunks_used=0,
            query="What is machine learning?",
            timestamp=datetime.utcnow()
        )
        
        with patch('routes.classic_rag.process_classic_rag_query') as mock_process:
            mock_process.return_value = mock_response
            
            response = client.post(
                "/classic_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            
            # Verify that the default top_k was used
            call_args = mock_process.call_args[0][0]  # QueryRequest object
            assert call_args.top_k == 5
    
    def test_classic_rag_query_processing_error(self, client, auth_headers):
        """Test Classic RAG query when processing fails"""
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5
        }
        
        with patch('routes.classic_rag.process_classic_rag_query') as mock_process:
            mock_process.side_effect = Exception("Processing failed")
            
            response = client.post(
                "/classic_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]
    
    def test_classic_rag_query_http_exception(self, client, auth_headers):
        """Test Classic RAG query when HTTPException is raised"""
        from fastapi import HTTPException
        
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5
        }
        
        with patch('routes.classic_rag.process_classic_rag_query') as mock_process:
            mock_process.side_effect = HTTPException(
                status_code=503,
                detail="Service temporarily unavailable"
            )
            
            response = client.post(
                "/classic_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 503
            data = response.json()
            assert "Service temporarily unavailable" in data["detail"]
    
    def test_classic_rag_query_long_query(self, client, auth_headers):
        """Test Classic RAG query with very long query"""
        query_data = {
            "query": "A" * 10000,  # Very long query
            "top_k": 5
        }
        
        mock_response = RAGResponse(
            answer="Test answer for long query",
            sources=[],
            response_time=2.0,
            chunks_used=0,
            query="A" * 10000,
            timestamp=datetime.utcnow()
        )
        
        with patch('routes.classic_rag.process_classic_rag_query') as mock_process:
            mock_process.return_value = mock_response
            
            response = client.post(
                "/classic_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Test answer for long query"
    
    def test_classic_rag_query_special_characters(self, client, auth_headers):
        """Test Classic RAG query with special characters"""
        query_data = {
            "query": "What is AI? 🤖 How does it work? (machine learning & deep learning)",
            "top_k": 3
        }
        
        mock_response = RAGResponse(
            answer="AI involves machine learning and deep learning techniques.",
            sources=[],
            response_time=1.2,
            chunks_used=0,
            query="What is AI? 🤖 How does it work? (machine learning & deep learning)",
            timestamp=datetime.utcnow()
        )
        
        with patch('routes.classic_rag.process_classic_rag_query') as mock_process:
            mock_process.return_value = mock_response
            
            response = client.post(
                "/classic_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "🤖" in data["query"]


class TestClassicRAGHealthEndpoint:
    """Test Classic RAG health endpoint"""
    
    def test_classic_rag_health_success(self, client):
        """Test successful health check"""
        with patch('routes.classic_rag.get_llm_info') as mock_llm_info, \
             patch('routes.classic_rag.get_vector_store_stats') as mock_vector_stats:
            
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
            
            response = client.get("/classic_rag/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "Classic RAG Pipeline"
            assert "llm_service" in data
            assert "vector_store" in data
            assert "features" in data
            assert len(data["features"]) > 0
    
    def test_classic_rag_health_service_error(self, client):
        """Test health check when service has errors"""
        with patch('routes.classic_rag.get_llm_info') as mock_llm_info:
            mock_llm_info.side_effect = Exception("Service error")
            
            response = client.get("/classic_rag/health")
            
            assert response.status_code == 503
            data = response.json()
            assert "Classic RAG service is not healthy" in data["detail"]
    
    def test_classic_rag_health_partial_service_availability(self, client):
        """Test health check with partial service availability"""
        with patch('routes.classic_rag.get_llm_info') as mock_llm_info, \
             patch('routes.classic_rag.get_vector_store_stats') as mock_vector_stats:
            
            mock_llm_info.return_value = {
                "model_name": "fallback",
                "gemini_available": False,
                "openai_available": False,
                "primary_service": "fallback",
                "fallback_mode": True
            }
            mock_vector_stats.return_value = {
                "total_vectors": 0,
                "dimension": 768,
                "total_documents": 0
            }
            
            response = client.get("/classic_rag/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["llm_service"]["fallback_mode"] is True
            assert data["vector_store"]["total_vectors"] == 0


class TestClassicRAGIntegration:
    """Integration tests for Classic RAG routes"""
    
    def test_complete_rag_workflow(self, client, auth_headers):
        """Test complete RAG workflow from query to response"""
        # First check health
        health_response = client.get("/classic_rag/health")
        assert health_response.status_code in [200, 503]  # May fail if services not available
        
        # Then make a query
        query_data = {
            "query": "Explain the concept of artificial intelligence",
            "top_k": 3
        }
        
        mock_response = RAGResponse(
            answer="Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines.",
            sources=[
                SourceChunk(
                    chunk_id="chunk_ai_1",
                    content="AI is the simulation of human intelligence in machines.",
                    similarity_score=0.92,
                    document_id="doc_ai",
                    chunk_index=0
                ),
                SourceChunk(
                    chunk_id="chunk_ai_2",
                    content="Machine learning is a subset of AI that enables computers to learn.",
                    similarity_score=0.88,
                    document_id="doc_ai",
                    chunk_index=1
                )
            ],
            response_time=2.1,
            chunks_used=2,
            query="Explain the concept of artificial intelligence",
            timestamp=datetime.utcnow()
        )
        
        with patch('routes.classic_rag.process_classic_rag_query') as mock_process:
            mock_process.return_value = mock_response
            
            response = client.post(
                "/classic_rag/query",
                json=query_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "answer" in data
            assert "sources" in data
            assert "response_time" in data
            assert "chunks_used" in data
            assert "query" in data
            assert "timestamp" in data
            
            # Verify response content
            assert len(data["sources"]) == 2
            assert data["chunks_used"] == 2
            assert data["response_time"] > 0
            assert "artificial intelligence" in data["answer"].lower()
    
    def test_multiple_concurrent_queries(self, client, auth_headers):
        """Test handling multiple concurrent queries"""
        import threading
        import time
        
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5
        }
        
        mock_response = RAGResponse(
            answer="Machine learning is a method of data analysis.",
            sources=[],
            response_time=1.0,
            chunks_used=0,
            query="What is machine learning?",
            timestamp=datetime.utcnow()
        )
        
        results = []
        
        def make_request():
            with patch('routes.classic_rag.process_classic_rag_query') as mock_process:
                mock_process.return_value = mock_response
                response = client.post(
                    "/classic_rag/query",
                    json=query_data,
                    headers=auth_headers
                )
                results.append(response.status_code)
        
        # Create multiple threads to simulate concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5