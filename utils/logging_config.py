"""
Comprehensive logging configuration for the RAG Comparison Platform
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON-like logging"""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'additional_data'):
            log_entry['additional_data'] = record.additional_data
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return str(log_entry)


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_directory: str = "logs",
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to files
        log_to_console: Whether to log to console
        log_directory: Directory for log files
        max_file_size: Maximum size of each log file
        backup_count: Number of backup files to keep
    """
    
    # Create logs directory if it doesn't exist
    if log_to_file:
        log_path = Path(log_directory)
        log_path.mkdir(exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    if log_to_file:
        # General application log file (rotating)
        app_log_file = log_path / "app.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        app_handler.setLevel(logging.INFO)
        
        app_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        app_handler.setFormatter(app_formatter)
        root_logger.addHandler(app_handler)
        
        # Error log file (errors and critical only)
        error_log_file = log_path / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        
        error_formatter = StructuredFormatter()
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)
        
        # Access log file (for API requests)
        access_log_file = log_path / "access.log"
        access_handler = logging.handlers.RotatingFileHandler(
            access_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        access_handler.setLevel(logging.INFO)
        
        access_formatter = logging.Formatter(
            fmt='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        access_handler.setFormatter(access_formatter)
        
        # Create access logger
        access_logger = logging.getLogger('access')
        access_logger.addHandler(access_handler)
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False
    
    # Configure specific loggers
    configure_specific_loggers()
    
    logging.info("Logging configuration completed successfully")


def configure_specific_loggers():
    """Configure specific loggers for different components"""
    
    # Database logger
    db_logger = logging.getLogger('sqlalchemy.engine')
    db_logger.setLevel(logging.WARNING)  # Reduce database query noise
    
    # HTTP client loggers
    http_loggers = [
        'httpx',
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool'
    ]
    for logger_name in http_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
    
    # FastAPI logger
    fastapi_logger = logging.getLogger('fastapi')
    fastapi_logger.setLevel(logging.INFO)
    
    # Uvicorn logger
    uvicorn_logger = logging.getLogger('uvicorn')
    uvicorn_logger.setLevel(logging.INFO)


class RequestLogger:
    """Utility class for logging API requests"""
    
    @staticmethod
    def log_request(
        method: str,
        url: str,
        user_id: Optional[int] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log API request"""
        access_logger = logging.getLogger('access')
        
        log_data = {
            'method': method,
            'url': str(url),
            'user_id': user_id,
            'request_id': request_id,
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        
        access_logger.info(f"API Request: {log_data}")
    
    @staticmethod
    def log_response(
        method: str,
        url: str,
        status_code: int,
        response_time: float,
        user_id: Optional[int] = None,
        request_id: Optional[str] = None
    ):
        """Log API response"""
        access_logger = logging.getLogger('access')
        
        log_data = {
            'method': method,
            'url': str(url),
            'status_code': status_code,
            'response_time_ms': round(response_time * 1000, 2),
            'user_id': user_id,
            'request_id': request_id
        }
        
        access_logger.info(f"API Response: {log_data}")


class PerformanceLogger:
    """Utility class for logging performance metrics"""
    
    @staticmethod
    def log_rag_performance(
        pipeline_type: str,
        query: str,
        response_time: float,
        chunks_used: int,
        user_id: Optional[int] = None,
        success: bool = True
    ):
        """Log RAG pipeline performance"""
        perf_logger = logging.getLogger('performance')
        
        log_data = {
            'pipeline_type': pipeline_type,
            'query_length': len(query),
            'response_time_ms': round(response_time * 1000, 2),
            'chunks_used': chunks_used,
            'user_id': user_id,
            'success': success
        }
        
        perf_logger.info(f"RAG Performance: {log_data}")
    
    @staticmethod
    def log_database_performance(
        operation: str,
        table: str,
        execution_time: float,
        record_count: Optional[int] = None
    ):
        """Log database operation performance"""
        perf_logger = logging.getLogger('performance')
        
        log_data = {
            'operation': operation,
            'table': table,
            'execution_time_ms': round(execution_time * 1000, 2),
            'record_count': record_count
        }
        
        perf_logger.info(f"DB Performance: {log_data}")


# Global logger instances
request_logger = RequestLogger()
performance_logger = PerformanceLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


def setup_development_logging():
    """Setup logging for development environment"""
    setup_logging(
        log_level="DEBUG",
        log_to_file=True,
        log_to_console=True,
        log_directory="logs"
    )


def setup_production_logging():
    """Setup logging for production environment"""
    setup_logging(
        log_level="INFO",
        log_to_file=True,
        log_to_console=False,
        log_directory="/var/log/rag-platform"
    )


def setup_testing_logging():
    """Setup logging for testing environment"""
    setup_logging(
        log_level="WARNING",
        log_to_file=False,
        log_to_console=True
    )