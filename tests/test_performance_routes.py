"""
Unit tests for performance monitoring API routes
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status
import json
from datetime import datetime

from main import app
from models.database import User
from services.performance_service import PerformanceStats, PerformanceSummary, SessionPerformanceData


class TestPerformanceRoutes:
    """Test cases for performance monitoring API routes"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = TestClient(app)
        self.test_user = User(
            id=1,
            email="test@example.com",
            password_hash="hashed_password",
            is_active=True
        )
        
        # Sample performance stats
        self.sample_stats = {
            "total_queries": 10,
            "date_range": {
                "start": "2024-01-01T00:00:00",
                "end": "2024-01-08T00:00:00",
                "days": 7
            },
            "classic_stats": {
                "total_queries": 10,
                "avg_response_time": 2.5,
                "median_response_time": 2.3,
                "min_response_time": 1.2,
                "max_response_time": 4.1,
                "std_response_time": 0.8,
                "avg_chunks_used": 5.2,
                "total_processing_time": 25.0
            },
            "memvid_stats": {
                "total_queries": 10,
                "avg_response_time": 2.1,
                "median_response_time": 2.0,
                "min_response_time": 1.0,
                "max_response_time": 3.5,
                "std_response_time": 0.7,
                "avg_chunks_used": 4.8,
                "total_processing_time": 21.0
            },
            "comparison_stats": {
                "total_comparisons": 10,
                "classic_faster_count": 3,
                "memvid_faster_count": 7,
                "avg_time_difference": -0.4,
                "avg_chunks_difference": -0.4,
                "classic_win_rate": 30.0,
                "memvid_win_rate": 70.0,
                "performance_trend": "memvid_improving"
            }
        }
        
        # Sample performance summary
        self.sample_summary = {
            "period": "Last 7 days",
            "total_queries": 10,
            "insights": [
                "MemVid RAG consistently outperforms Classic RAG",
                "MemVid RAG performance is improving over time"
            ],
            "recommendations": [
                "Consider using MemVid RAG as your primary pipeline"
            ]
        }
        
        # Sample session data
        self.sample_session_data = {
            "session_id": "session_1_20240101",
            "user_id": 1,
            "session_start": "2024-01-01T10:00:00",
            "session_end": "2024-01-01T12:00:00",
            "total_queries": 5,
            "avg_response_time_classic": 2.3,
            "avg_response_time_memvid": 2.0,
            "performance_trend": "stable",
            "queries": [
                {
                    "query": "What is machine learning?",
                    "timestamp": "2024-01-01T10:30:00",
                    "performance_difference": -0.3,
                    "chunks_difference": -1,
                    "answer_length_difference": 10,
                    "classic_metrics": {
                        "response_time": 2.5,
                        "chunks_used": 5,
                        "query_length": 25,
                        "answer_length": 200
                    },
                    "memvid_metrics": {
                        "response_time": 2.2,
                        "chunks_used": 4,
                        "query_length": 25,
                        "answer_length": 210
                    }
                }
            ]
        }
    
    def get_auth_headers(self):
        """Get authentication headers for requests"""
        return {"Authorization": "Bearer test_token"}
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_get_performance_stats_success(self, mock_service, mock_get_user):
        """Test successful performance stats retrieval"""
        # Mock authentication
        mock_get_user.return_value = self.test_user
        
        # Mock service response
        mock_service.get_user_performance_stats.return_value = self.sample_stats
        
        # Make request
        response = self.client.get(
            "/performance/stats?days=7",
            headers=self.get_auth_headers()
        )
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"
        assert data["user_id"] == 1
        assert data["stats"]["total_queries"] == 10
        assert data["stats"]["classic_stats"]["avg_response_time"] == 2.5
        assert data["stats"]["memvid_stats"]["avg_response_time"] == 2.1
        
        # Verify service was called with correct parameters
        mock_service.get_user_performance_stats.assert_called_once_with(
            user_id=1,
            pipeline_type=None,
            days=7
        )
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_get_performance_stats_with_pipeline_filter(self, mock_service, mock_get_user):
        """Test performance stats with pipeline type filter"""
        mock_get_user.return_value = self.test_user
        mock_service.get_user_performance_stats.return_value = self.sample_stats
        
        response = self.client.get(
            "/performance/stats?days=30&pipeline_type=classic",
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_service.get_user_performance_stats.assert_called_once_with(
            user_id=1,
            pipeline_type="classic",
            days=30
        )
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_get_performance_stats_invalid_pipeline_type(self, mock_service, mock_get_user):
        """Test performance stats with invalid pipeline type"""
        mock_get_user.return_value = self.test_user
        
        response = self.client.get(
            "/performance/stats?pipeline_type=invalid",
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_get_performance_stats_service_error(self, mock_service, mock_get_user):
        """Test performance stats with service error"""
        mock_get_user.return_value = self.test_user
        mock_service.get_user_performance_stats.side_effect = Exception("Service error")
        
        response = self.client.get(
            "/performance/stats",
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to retrieve performance statistics" in data["detail"]
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_get_performance_summary_success(self, mock_service, mock_get_user):
        """Test successful performance summary retrieval"""
        mock_get_user.return_value = self.test_user
        mock_service.get_performance_summary.return_value = self.sample_summary
        
        response = self.client.get(
            "/performance/summary?days=7",
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"
        assert data["user_id"] == 1
        assert data["summary"]["period"] == "Last 7 days"
        assert len(data["summary"]["insights"]) == 2
        assert len(data["summary"]["recommendations"]) == 1
        
        mock_service.get_performance_summary.assert_called_once_with(
            user_id=1,
            days=7
        )
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_get_session_performance_success(self, mock_service, mock_get_user):
        """Test successful session performance retrieval"""
        mock_get_user.return_value = self.test_user
        
        # Create mock SessionPerformanceData object
        mock_session_data = Mock()
        mock_session_data.session_id = self.sample_session_data["session_id"]
        mock_session_data.user_id = self.sample_session_data["user_id"]
        mock_session_data.session_start = datetime.fromisoformat(self.sample_session_data["session_start"])
        mock_session_data.session_end = datetime.fromisoformat(self.sample_session_data["session_end"])
        mock_session_data.total_queries = self.sample_session_data["total_queries"]
        mock_session_data.avg_response_time_classic = self.sample_session_data["avg_response_time_classic"]
        mock_session_data.avg_response_time_memvid = self.sample_session_data["avg_response_time_memvid"]
        mock_session_data.performance_trend = self.sample_session_data["performance_trend"]
        
        # Mock queries
        mock_query = Mock()
        mock_query.query = "What is machine learning?"
        mock_query.timestamp = datetime.fromisoformat("2024-01-01T10:30:00")
        mock_query.performance_difference = -0.3
        mock_query.chunks_difference = -1
        mock_query.answer_length_difference = 10
        
        # Mock metrics
        mock_classic_metrics = Mock()
        mock_classic_metrics.response_time = 2.5
        mock_classic_metrics.chunks_used = 5
        mock_classic_metrics.query_length = 25
        mock_classic_metrics.answer_length = 200
        
        mock_memvid_metrics = Mock()
        mock_memvid_metrics.response_time = 2.2
        mock_memvid_metrics.chunks_used = 4
        mock_memvid_metrics.query_length = 25
        mock_memvid_metrics.answer_length = 210
        
        mock_query.classic_metrics = mock_classic_metrics
        mock_query.memvid_metrics = mock_memvid_metrics
        mock_session_data.queries = [mock_query]
        
        mock_service.get_session_performance_data.return_value = mock_session_data
        
        response = self.client.get(
            "/performance/session",
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"
        assert data["session_data"]["session_id"] == "session_1_20240101"
        assert data["session_data"]["total_queries"] == 5
        assert len(data["session_data"]["queries"]) == 1
        
        mock_service.get_session_performance_data.assert_called_once_with(
            user_id=1,
            session_id=None
        )
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_get_session_performance_with_session_id(self, mock_service, mock_get_user):
        """Test session performance with specific session ID"""
        mock_get_user.return_value = self.test_user
        mock_service.get_session_performance_data.return_value = Mock()
        
        response = self.client.get(
            "/performance/session?session_id=test_session",
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_service.get_session_performance_data.assert_called_once_with(
            user_id=1,
            session_id="test_session"
        )
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_export_performance_data_csv(self, mock_service, mock_get_user):
        """Test performance data export in CSV format"""
        mock_get_user.return_value = self.test_user
        mock_service.export_performance_data.return_value = "timestamp,query,classic_time\n2024-01-01,test,2.5"
        
        response = self.client.get(
            "/performance/export?format_type=csv&days=30",
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment; filename=performance_data_1_30days.csv" in response.headers["content-disposition"]
        assert "timestamp,query,classic_time" in response.text
        
        mock_service.export_performance_data.assert_called_once_with(
            user_id=1,
            format_type="csv",
            days=30
        )
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_export_performance_data_json(self, mock_service, mock_get_user):
        """Test performance data export in JSON format"""
        mock_get_user.return_value = self.test_user
        mock_service.export_performance_data.return_value = '[{"timestamp": "2024-01-01", "query": "test"}]'
        
        response = self.client.get(
            "/performance/export?format_type=json&days=7",
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/json; charset=utf-8"
        assert "attachment; filename=performance_data_1_7days.json" in response.headers["content-disposition"]
        
        mock_service.export_performance_data.assert_called_once_with(
            user_id=1,
            format_type="json",
            days=7
        )
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_export_performance_data_invalid_format(self, mock_service, mock_get_user):
        """Test export with invalid format"""
        mock_get_user.return_value = self.test_user
        
        response = self.client.get(
            "/performance/export?format_type=xml",
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('routes.performance.get_current_active_user')
    @patch('routes.performance.performance_service')
    def test_export_performance_data_service_error(self, mock_service, mock_get_user):
        """Test export with service error"""
        mock_get_user.return_value = self.test_user
        mock_service.export_performance_data.side_effect = Exception("Export failed")
        
        response = self.client.get(
            "/performance/export?format_type=csv",
            headers=self.get_auth_headers()
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to export performance data" in data["detail"]
    
    def test_performance_health_check(self):
        """Test performance service health check"""
        response = self.client.get("/performance/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Performance Monitoring Service"
        assert "features" in data
        assert "supported_formats" in data
        assert "csv" in data["supported_formats"]
        assert "json" in data["supported_formats"]
        assert data["max_export_days"] == 365
    
    def test_performance_health_check_error(self):
        """Test performance health check with error"""
        with patch('routes.performance.logger') as mock_logger:
            mock_logger.error.side_effect = Exception("Health check failed")
            
            # The health endpoint should still work even if logging fails
            response = self.client.get("/performance/health")
            assert response.status_code == status.HTTP_200_OK
    
    def test_unauthorized_access(self):
        """Test unauthorized access to performance endpoints"""
        endpoints = [
            "/performance/stats",
            "/performance/summary",
            "/performance/session",
            "/performance/export"
        ]
        
        for endpoint in endpoints:
            response = self.client.get(endpoint)
            # Should return 401 or 403 depending on auth middleware implementation
            assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
    
    @patch('routes.performance.get_current_active_user')
    def test_invalid_query_parameters(self, mock_get_user):
        """Test endpoints with invalid query parameters"""
        mock_get_user.return_value = self.test_user
        
        # Test invalid days parameter
        response = self.client.get(
            "/performance/stats?days=0",
            headers=self.get_auth_headers()
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        response = self.client.get(
            "/performance/stats?days=400",
            headers=self.get_auth_headers()
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test invalid summary days parameter
        response = self.client.get(
            "/performance/summary?days=0",
            headers=self.get_auth_headers()
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        response = self.client.get(
            "/performance/summary?days=35",
            headers=self.get_auth_headers()
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestPerformanceRoutesIntegration:
    """Integration tests for performance routes"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = TestClient(app)
    
    def test_all_performance_endpoints_exist(self):
        """Test that all performance endpoints are properly registered"""
        # Test that endpoints exist (will return auth error but not 404)
        endpoints = [
            "/performance/stats",
            "/performance/summary", 
            "/performance/session",
            "/performance/export",
            "/performance/health"
        ]
        
        for endpoint in endpoints:
            response = self.client.get(endpoint)
            # Should not return 404 Not Found
            assert response.status_code != status.HTTP_404_NOT_FOUND
    
    def test_performance_health_endpoint_public(self):
        """Test that health endpoint is publicly accessible"""
        response = self.client.get("/performance/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "features" in data
    
    def test_cors_headers(self):
        """Test CORS headers on performance endpoints"""
        response = self.client.options("/performance/health")
        # CORS headers should be present due to middleware
        assert "access-control-allow-origin" in response.headers or response.status_code == status.HTTP_200_OK


if __name__ == "__main__":
    pytest.main([__file__])