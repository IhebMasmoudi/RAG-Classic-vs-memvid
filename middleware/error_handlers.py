"""
Global error handlers for FastAPI application
"""
import logging
import traceback
from datetime import datetime
from typing import Union

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from pydantic import ValidationError

from models.schemas import ErrorResponse, ValidationErrorResponse

logger = logging.getLogger(__name__)


def setup_error_handlers(app: FastAPI):
    """Setup global error handlers for the FastAPI application"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.warning(
            f"HTTP {exc.status_code} error on {request.method} {request.url}: {exc.detail}"
        )
        
        error_response = ErrorResponse(
            error=f"HTTP {exc.status_code}",
            detail=exc.detail,
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions"""
        logger.warning(
            f"Starlette HTTP {exc.status_code} error on {request.method} {request.url}: {exc.detail}"
        )
        
        error_response = ErrorResponse(
            error=f"HTTP {exc.status_code}",
            detail=exc.detail,
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.warning(
            f"Validation error on {request.method} {request.url}: {exc.errors()}"
        )
        
        error_response = ValidationErrorResponse(
            error="Validation Error",
            detail=exc.errors(),
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors"""
        logger.warning(
            f"Pydantic validation error on {request.method} {request.url}: {exc.errors()}"
        )
        
        error_response = ValidationErrorResponse(
            error="Data Validation Error",
            detail=exc.errors(),
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
        """Handle SQLAlchemy database errors"""
        logger.error(
            f"Database error on {request.method} {request.url}: {str(exc)}",
            exc_info=True
        )
        
        # Don't expose internal database errors to users
        error_response = ErrorResponse(
            error="Database Error",
            detail="An internal database error occurred. Please try again later.",
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(IntegrityError)
    async def integrity_error_handler(request: Request, exc: IntegrityError):
        """Handle database integrity constraint errors"""
        logger.warning(
            f"Database integrity error on {request.method} {request.url}: {str(exc)}"
        )
        
        # Common integrity errors
        detail = "A database constraint was violated."
        if "UNIQUE constraint failed" in str(exc) or "duplicate key" in str(exc).lower():
            detail = "This record already exists. Please check your input."
        elif "FOREIGN KEY constraint failed" in str(exc):
            detail = "Referenced record does not exist."
        elif "NOT NULL constraint failed" in str(exc):
            detail = "Required field is missing."
        
        error_response = ErrorResponse(
            error="Data Integrity Error",
            detail=detail,
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        """Handle file not found errors"""
        logger.error(
            f"File not found error on {request.method} {request.url}: {str(exc)}"
        )
        
        error_response = ErrorResponse(
            error="File Not Found",
            detail="The requested file could not be found.",
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(PermissionError)
    async def permission_error_handler(request: Request, exc: PermissionError):
        """Handle permission errors"""
        logger.error(
            f"Permission error on {request.method} {request.url}: {str(exc)}"
        )
        
        error_response = ErrorResponse(
            error="Permission Denied",
            detail="You don't have permission to access this resource.",
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(TimeoutError)
    async def timeout_error_handler(request: Request, exc: TimeoutError):
        """Handle timeout errors"""
        logger.error(
            f"Timeout error on {request.method} {request.url}: {str(exc)}"
        )
        
        error_response = ErrorResponse(
            error="Request Timeout",
            detail="The request took too long to process. Please try again.",
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(ConnectionError)
    async def connection_error_handler(request: Request, exc: ConnectionError):
        """Handle connection errors"""
        logger.error(
            f"Connection error on {request.method} {request.url}: {str(exc)}"
        )
        
        error_response = ErrorResponse(
            error="Service Unavailable",
            detail="Unable to connect to external service. Please try again later.",
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other unhandled exceptions"""
        logger.error(
            f"Unhandled exception on {request.method} {request.url}: {str(exc)}",
            exc_info=True
        )
        
        # Log the full traceback for debugging
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        error_response = ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred. Please try again later.",
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(mode='json')
        )
    
    logger.info("Global error handlers configured successfully")


class ErrorLogger:
    """Utility class for structured error logging"""
    
    @staticmethod
    def log_error(
        error: Exception,
        context: str = "",
        user_id: Union[int, None] = None,
        request_id: Union[str, None] = None,
        additional_data: Union[dict, None] = None
    ):
        """
        Log error with structured information
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            user_id: ID of the user if available
            request_id: Request ID if available
            additional_data: Additional context data
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "user_id": user_id,
            "request_id": request_id,
            "additional_data": additional_data or {}
        }
        
        logger.error(
            f"Error in {context}: {error_data}",
            exc_info=True,
            extra=error_data
        )
    
    @staticmethod
    def log_warning(
        message: str,
        context: str = "",
        user_id: Union[int, None] = None,
        additional_data: Union[dict, None] = None
    ):
        """
        Log warning with structured information
        
        Args:
            message: Warning message
            context: Context where the warning occurred
            user_id: ID of the user if available
            additional_data: Additional context data
        """
        warning_data = {
            "message": message,
            "context": context,
            "user_id": user_id,
            "additional_data": additional_data or {}
        }
        
        logger.warning(f"Warning in {context}: {warning_data}", extra=warning_data)
    
    @staticmethod
    def log_info(
        message: str,
        context: str = "",
        user_id: Union[int, None] = None,
        additional_data: Union[dict, None] = None
    ):
        """
        Log info with structured information
        
        Args:
            message: Info message
            context: Context where the info occurred
            user_id: ID of the user if available
            additional_data: Additional context data
        """
        info_data = {
            "message": message,
            "context": context,
            "user_id": user_id,
            "additional_data": additional_data or {}
        }
        
        logger.info(f"Info in {context}: {info_data}", extra=info_data)


# Global error logger instance
error_logger = ErrorLogger()