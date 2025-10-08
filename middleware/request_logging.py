"""
Request logging middleware for FastAPI
"""
import time
import uuid
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logging_config import request_logger

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests and responses"""
    
    def __init__(self, app, log_body: bool = False, log_headers: bool = False):
        super().__init__(app)
        self.log_body = log_body
        self.log_headers = log_headers
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details"""
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "Unknown")
        
        # Get user ID if available (from JWT token)
        user_id = getattr(request.state, 'user_id', None)
        
        # Log request start
        start_time = time.time()
        
        request_logger.log_request(
            method=request.method,
            url=request.url,
            user_id=user_id,
            request_id=request_id,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Log request headers if enabled
        if self.log_headers:
            logger.debug(f"Request {request_id} headers: {dict(request.headers)}")
        
        # Log request body if enabled (be careful with sensitive data)
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Don't log sensitive endpoints
                    sensitive_paths = ["/auth/login", "/auth/register"]
                    if not any(path in str(request.url) for path in sensitive_paths):
                        logger.debug(f"Request {request_id} body: {body.decode('utf-8')[:1000]}")
            except Exception as e:
                logger.warning(f"Could not log request body for {request_id}: {e}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log response
            request_logger.log_response(
                method=request.method,
                url=request.url,
                status_code=response.status_code,
                response_time=response_time,
                user_id=user_id,
                request_id=request_id
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log error
            response_time = time.time() - start_time
            
            logger.error(
                f"Request {request_id} failed after {response_time:.3f}s: {str(e)}",
                exc_info=True,
                extra={
                    'request_id': request_id,
                    'method': request.method,
                    'url': str(request.url),
                    'user_id': user_id,
                    'response_time': response_time
                }
            )
            
            # Re-raise the exception
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers (when behind proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs, take the first one
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "Unknown"


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log security-related events"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log security events"""
        
        # Log authentication attempts
        if request.url.path in ["/auth/login", "/auth/register"]:
            client_ip = self._get_client_ip(request)
            logger.info(
                f"Authentication attempt from {client_ip} to {request.url.path}",
                extra={
                    'event_type': 'auth_attempt',
                    'ip_address': client_ip,
                    'endpoint': request.url.path,
                    'user_agent': request.headers.get("user-agent", "Unknown")
                }
            )
        
        # Log access to protected endpoints
        if request.headers.get("Authorization"):
            logger.debug(
                f"Authenticated request to {request.url.path}",
                extra={
                    'event_type': 'authenticated_access',
                    'endpoint': request.url.path,
                    'method': request.method
                }
            )
        
        # Check for suspicious patterns
        self._check_suspicious_patterns(request)
        
        response = await call_next(request)
        
        # Log failed authentication
        if request.url.path in ["/auth/login"] and response.status_code == 401:
            client_ip = self._get_client_ip(request)
            logger.warning(
                f"Failed login attempt from {client_ip}",
                extra={
                    'event_type': 'failed_login',
                    'ip_address': client_ip,
                    'user_agent': request.headers.get("user-agent", "Unknown")
                }
            )
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "Unknown"
    
    def _check_suspicious_patterns(self, request: Request):
        """Check for suspicious request patterns"""
        
        # Check for SQL injection patterns in query parameters
        query_string = str(request.url.query).lower()
        sql_patterns = ["'", "union", "select", "drop", "insert", "delete", "update"]
        
        if any(pattern in query_string for pattern in sql_patterns):
            client_ip = self._get_client_ip(request)
            logger.warning(
                f"Potential SQL injection attempt from {client_ip}: {request.url}",
                extra={
                    'event_type': 'potential_sql_injection',
                    'ip_address': client_ip,
                    'url': str(request.url),
                    'user_agent': request.headers.get("user-agent", "Unknown")
                }
            )
        
        # Check for path traversal attempts
        path = str(request.url.path)
        if "../" in path or "..%2F" in path or "..%5C" in path:
            client_ip = self._get_client_ip(request)
            logger.warning(
                f"Potential path traversal attempt from {client_ip}: {path}",
                extra={
                    'event_type': 'potential_path_traversal',
                    'ip_address': client_ip,
                    'path': path,
                    'user_agent': request.headers.get("user-agent", "Unknown")
                }
            )
        
        # Check for unusually large requests
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 50 * 1024 * 1024:  # 50MB
            client_ip = self._get_client_ip(request)
            logger.warning(
                f"Unusually large request from {client_ip}: {content_length} bytes",
                extra={
                    'event_type': 'large_request',
                    'ip_address': client_ip,
                    'content_length': content_length,
                    'endpoint': request.url.path
                }
            )