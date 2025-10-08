"""
Unit tests for performance monitoring service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from statistics import mean, median, stdev

from services.performance_service import (
    PerformanceMonitoringService,
    PerformanceMetrics,
    ComparisonMetrics,
    PerformanceStats,
    SessionPerformanceData,
    performance_service
)
from models.database import QueryHistory, User


class TestPerformanceMonitoringService:
    """Test cases for PerformanceMonitoringService"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = PerformanceMonitoringService()
        self.test_user_id = 1
        
        # Sample query history data
        self.sample_queries = [
            {
                'id': 1,
                'user_id': 1,
                'query_text': 'What is machine learning?',
                'query_timestamp': datetime.now() - timedelta(hours=1),
                'classic_response_time': 2.5,
                'classic_chunks_used': 5,
                'classic_answer': 'Machine learning is a subset of AI...',
                'memvid_response_time': 2.1,
                'memvid_chunks_used': 4,
                'memvid_answer': 'Machine learning involves algorithms...'
            },
            {
                'id': 2,
                'user_id': 1,
                'query_text': 'How does neural network work?',
                'query_timestamp': datetime.now() - timedelta(minutes=30),
                'classic_response_time': 3.2,
                'classic_chunks_used': 6,
                'classic_answer': 'Neural networks are computing systems...',
                'memvid_response_time': 2.8,
                'memvid_chunks_used': 5,
                'memvid_answer': 'Neural networks consist of layers...'
            },
            {
                'id': 3,
                'user_id': 1,
                'query_text': 'What is deep learning?',
                'query_timestamp': datetime.now() - timedelta(minutes=10),
                'classic_response_time': 1.8,
                'classic_chunks_used': 3,
                'classic_answer': 'Deep learning is a subset of ML...',
                'memvid_response_time': 2.2,
                'memvid_chunks_used': 4,
                'memvid_answer': 'Deep learning uses neural networks...'
            }
        ]
    
    @patch('services.performance_service.get_db_context')
    def test_get_user_performance_stats_success(self, mock_db_context):
        """Test successful retrieval of user performance stats"""
        # Mock database context and query results
        mock_db = Mock()
        mock_db_context.return_value.__enter__.return_value = mock_db
        
        # Create mock QueryHistory objects
        mock_queries = []
        for query_data in self.sample_queries:
            mock_query = Mock(spec=QueryHistory)
            for key, value in query_data.items():
                setattr(mock_query, key, value)
            mock_queries.append(mock_query)
        
        mock_db.query.return_value.filter.return_value.all.return_value = mock_queries
        
        # Test the method
        result = self.service.get_user_performance_stats(self.test_user_id, days=7)
        
        # Assertions
        assert result['total_queries'] == 3
        assert result['classic_stats'] is not None
        assert result['memvid_stats'] is not None
        assert result['comparison_stats'] is not None
        
        # Check classic stats
        classic_stats = result['classic_stats']
        expected_classic_times = [2.5, 3.2, 1.8]
        assert classic_stats['total_queries'] == 3
        assert classic_stats['avg_response_time'] == mean(expected_classic_times)
        assert classic_stats['median_response_time'] == median(expected_classic_times)
        assert classic_stats['min_response_time'] == min(expected_classic_times)
        assert classic_stats['max_response_time'] == max(expected_classic_times)
        
        # Check memvid stats
        memvid_stats = result['memvid_stats']
        expected_memvid_times = [2.1, 2.8, 2.2]
        assert memvid_stats['total_queries'] == 3
        assert memvid_stats['avg_response_time'] == mean(expected_memvid_times)
        
        # Check comparison stats
        comparison_stats = result['comparison_stats']
        assert comparison_stats['total_comparisons'] == 3
        assert comparison_stats['memvid_faster_count'] == 2  # MemVid faster in 2 out of 3
        assert comparison_stats['classic_faster_count'] == 1  # Classic faster in 1 out of 3
    
    @patch('services.performance_service.get_db_context')
    def test_get_user_performance_stats_no_data(self, mock_db_context):
        """Test performance stats with no query history"""
        mock_db = Mock()
        mock_db_context.return_value.__enter__.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        result = self.service.get_user_performance_stats(self.test_user_id)
        
        assert result['total_queries'] == 0
        assert result['classic_stats'] is None
        assert result['memvid_stats'] is None
        assert result['comparison_stats'] is None
    
    def test_calculate_pipeline_stats(self):
        """Test pipeline statistics calculation"""
        results = [
            {"response_time": 2.5, "chunks_used": 5},
            {"response_time": 3.2, "chunks_used": 6},
            {"response_time": 1.8, "chunks_used": 3}
        ]
        
        stats = self.service._calculate_pipeline_stats(results)
        
        response_times = [2.5, 3.2, 1.8]
        chunks_used = [5, 6, 3]
        
        assert stats['total_queries'] == 3
        assert stats['avg_response_time'] == mean(response_times)
        assert stats['median_response_time'] == median(response_times)
        assert stats['min_response_time'] == min(response_times)
        assert stats['max_response_time'] == max(response_times)
        assert stats['std_response_time'] == stdev(response_times)
        assert stats['avg_chunks_used'] == mean(chunks_used)
        assert stats['total_processing_time'] == sum(response_times)
    
    def test_calculate_pipeline_stats_empty(self):
        """Test pipeline statistics with empty results"""
        stats = self.service._calculate_pipeline_stats([])
        assert stats is None
    
    def test_calculate_comparison_stats(self):
        """Test comparison statistics calculation"""
        comparisons = [
            {"time_difference": -0.4, "chunks_difference": -1},  # MemVid faster
            {"time_difference": 0.4, "chunks_difference": 1},    # Classic faster
            {"time_difference": -0.4, "chunks_difference": -1}   # MemVid faster
        ]
        
        stats = self.service._calculate_comparison_stats(comparisons)
        
        assert stats['total_comparisons'] == 3
        assert stats['classic_faster_count'] == 1
        assert stats['memvid_faster_count'] == 2
        assert stats['classic_win_rate'] == pytest.approx(33.33, rel=1e-2)
        assert stats['memvid_win_rate'] == pytest.approx(66.67, rel=1e-2)
        assert stats['avg_time_difference'] == pytest.approx(-0.133, rel=1e-2)
        assert stats['avg_chunks_difference'] == pytest.approx(-0.333, rel=1e-2)
    
    def test_calculate_performance_trend(self):
        """Test performance trend calculation"""
        # MemVid improving (more negative differences over time)
        improving_trend = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
        trend = self.service._calculate_performance_trend(improving_trend)
        assert trend == "memvid_improving"
        
        # Classic improving (more positive differences over time)
        declining_trend = [-0.6, -0.5, -0.4, -0.1, 0.2, 0.3]
        trend = self.service._calculate_performance_trend(declining_trend)
        assert trend == "classic_improving"
        
        # Stable performance
        stable_trend = [-0.2, -0.1, -0.2, -0.1, -0.2, -0.1]
        trend = self.service._calculate_performance_trend(stable_trend)
        assert trend == "stable"
        
        # Insufficient data
        insufficient_data = [-0.1, -0.2]
        trend = self.service._calculate_performance_trend(insufficient_data)
        assert trend == "insufficient_data"
    
    @patch('services.performance_service.get_db_context')
    def test_get_session_performance_data(self, mock_db_context):
        """Test session performance data retrieval"""
        mock_db = Mock()
        mock_db_context.return_value.__enter__.return_value = mock_db
        
        # Create mock QueryHistory objects for recent session
        mock_queries = []
        for query_data in self.sample_queries:
            mock_query = Mock(spec=QueryHistory)
            for key, value in query_data.items():
                setattr(mock_query, key, value)
            mock_queries.append(mock_query)
        
        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_queries
        
        result = self.service.get_session_performance_data(self.test_user_id)
        
        assert isinstance(result, SessionPerformanceData)
        assert result.user_id == self.test_user_id
        assert result.total_queries == 3
        assert len(result.queries) == 3
        assert result.avg_response_time_classic > 0
        assert result.avg_response_time_memvid > 0
    
    def test_export_to_csv(self):
        """Test CSV export functionality"""
        data = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "query": "Test query",
                "classic_response_time": 2.5,
                "memvid_response_time": 2.1
            },
            {
                "timestamp": "2024-01-01T11:00:00",
                "query": "Another query",
                "classic_response_time": 3.2,
                "memvid_response_time": 2.8
            }
        ]
        
        csv_output = self.service._export_to_csv(data)
        
        assert "timestamp,query,classic_response_time,memvid_response_time" in csv_output
        assert "2024-01-01T10:00:00,Test query,2.5,2.1" in csv_output
        assert "2024-01-01T11:00:00,Another query,3.2,2.8" in csv_output
    
    def test_export_to_csv_empty(self):
        """Test CSV export with empty data"""
        csv_output = self.service._export_to_csv([])
        assert csv_output == ""
    
    @patch('services.performance_service.get_db_context')
    def test_export_performance_data_csv(self, mock_db_context):
        """Test performance data export in CSV format"""
        mock_db = Mock()
        mock_db_context.return_value.__enter__.return_value = mock_db
        
        # Mock query results
        mock_queries = []
        for query_data in self.sample_queries:
            mock_query = Mock(spec=QueryHistory)
            for key, value in query_data.items():
                setattr(mock_query, key, value)
            mock_queries.append(mock_query)
        
        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_queries
        
        result = self.service.export_performance_data(self.test_user_id, format_type="csv", days=7)
        
        assert isinstance(result, str)
        assert "timestamp" in result
        assert "query" in result
        assert "classic_response_time" in result
        assert "memvid_response_time" in result
    
    @patch('services.performance_service.get_db_context')
    def test_export_performance_data_json(self, mock_db_context):
        """Test performance data export in JSON format"""
        mock_db = Mock()
        mock_db_context.return_value.__enter__.return_value = mock_db
        
        # Mock query results
        mock_queries = []
        for query_data in self.sample_queries:
            mock_query = Mock(spec=QueryHistory)
            for key, value in query_data.items():
                setattr(mock_query, key, value)
            mock_queries.append(mock_query)
        
        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_queries
        
        result = self.service.export_performance_data(self.test_user_id, format_type="json", days=7)
        
        assert isinstance(result, str)
        import json
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 3
        assert "timestamp" in data[0]
        assert "query" in data[0]
    
    def test_export_performance_data_invalid_format(self):
        """Test export with invalid format"""
        with pytest.raises(ValueError, match="Unsupported format"):
            self.service.export_performance_data(self.test_user_id, format_type="xml")
    
    @patch('services.performance_service.get_db_context')
    def test_get_performance_summary(self, mock_db_context):
        """Test performance summary generation"""
        mock_db = Mock()
        mock_db_context.return_value.__enter__.return_value = mock_db
        
        # Mock query results
        mock_queries = []
        for query_data in self.sample_queries:
            mock_query = Mock(spec=QueryHistory)
            for key, value in query_data.items():
                setattr(mock_query, key, value)
            mock_queries.append(mock_query)
        
        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_queries
        
        # Mock the get_user_performance_stats method
        with patch.object(self.service, 'get_user_performance_stats') as mock_get_stats:
            mock_get_stats.return_value = {
                'total_queries': 3,
                'comparison_stats': {
                    'memvid_win_rate': 70.0,
                    'classic_win_rate': 30.0,
                    'performance_trend': 'memvid_improving',
                    'avg_time_difference': -0.2
                }
            }
            
            result = self.service.get_performance_summary(self.test_user_id, days=7)
            
            assert result['period'] == "Last 7 days"
            assert result['total_queries'] == 3
            assert len(result['insights']) > 0
            assert len(result['recommendations']) > 0
            assert "MemVid RAG consistently outperforms Classic RAG" in result['insights']
    
    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass"""
        metrics = PerformanceMetrics(
            response_time=2.5,
            chunks_used=5,
            query_length=50,
            answer_length=200,
            timestamp=datetime.now(),
            pipeline_type="classic"
        )
        
        assert metrics.response_time == 2.5
        assert metrics.chunks_used == 5
        assert metrics.pipeline_type == "classic"
    
    def test_comparison_metrics_dataclass(self):
        """Test ComparisonMetrics dataclass"""
        classic_metrics = PerformanceMetrics(
            response_time=2.5,
            chunks_used=5,
            query_length=50,
            answer_length=200,
            timestamp=datetime.now(),
            pipeline_type="classic"
        )
        
        memvid_metrics = PerformanceMetrics(
            response_time=2.1,
            chunks_used=4,
            query_length=50,
            answer_length=180,
            timestamp=datetime.now(),
            pipeline_type="memvid"
        )
        
        comparison = ComparisonMetrics(
            query="Test query",
            classic_metrics=classic_metrics,
            memvid_metrics=memvid_metrics,
            performance_difference=-0.4,
            chunks_difference=-1,
            answer_length_difference=-20,
            timestamp=datetime.now()
        )
        
        assert comparison.performance_difference == -0.4
        assert comparison.chunks_difference == -1
        assert comparison.classic_metrics.response_time == 2.5
        assert comparison.memvid_metrics.response_time == 2.1
    
    @patch('services.performance_service.get_db_context')
    def test_error_handling(self, mock_db_context):
        """Test error handling in performance service"""
        # Mock database error
        mock_db_context.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception, match="Database connection failed"):
            self.service.get_user_performance_stats(self.test_user_id)
    
    def test_cache_management(self):
        """Test session cache management"""
        # Test cache size limit
        service = PerformanceMonitoringService()
        service.max_cache_size = 2
        
        # Add items to cache
        service.session_cache = {
            "session1": {"data": "test1"},
            "session2": {"data": "test2"}
        }
        
        # Cache should be at limit
        assert len(service.session_cache) == 2
        
        # This would trigger cache cleanup in a real implementation
        # For now, just verify the cache structure
        assert "session1" in service.session_cache
        assert "session2" in service.session_cache


class TestPerformanceServiceIntegration:
    """Integration tests for performance service"""
    
    def test_global_service_instance(self):
        """Test that global service instance is properly initialized"""
        assert performance_service is not None
        assert isinstance(performance_service, PerformanceMonitoringService)
        assert performance_service.max_cache_size == 1000
    
    def test_service_methods_exist(self):
        """Test that all required methods exist on the service"""
        required_methods = [
            'get_user_performance_stats',
            'get_session_performance_data',
            'export_performance_data',
            'get_performance_summary'
        ]
        
        for method_name in required_methods:
            assert hasattr(performance_service, method_name)
            assert callable(getattr(performance_service, method_name))


if __name__ == "__main__":
    pytest.main([__file__])