"""
Unit tests for authentication utilities
"""
import pytest
from datetime import datetime, timedelta
from jose import jwt
from fastapi import HTTPException

from utils.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    verify_token,
    extract_email_from_token
)
from config import settings


class TestPasswordHashing:
    """Test password hashing and verification"""
    
    def test_password_hashing(self):
        """Test password hashing"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")
    
    def test_password_verification_success(self):
        """Test successful password verification"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_password_verification_failure(self):
        """Test failed password verification"""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = get_password_hash(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_different_passwords_different_hashes(self):
        """Test that different passwords produce different hashes"""
        password1 = "password1"
        password2 = "password2"
        
        hash1 = get_password_hash(password1)
        hash2 = get_password_hash(password2)
        
        assert hash1 != hash2


class TestJWTTokens:
    """Test JWT token creation and verification"""
    
    def test_create_access_token(self):
        """Test JWT token creation"""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode token to verify structure
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert payload["sub"] == "test@example.com"
        assert "exp" in payload
    
    def test_create_access_token_with_expiry(self):
        """Test JWT token creation with custom expiry"""
        data = {"sub": "test@example.com"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta)
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        exp_time = datetime.utcfromtimestamp(payload["exp"])
        expected_time = datetime.utcnow() + expires_delta
        
        # Allow 2 minute tolerance for test execution time
        assert abs((exp_time - expected_time).total_seconds()) < 120
    
    def test_verify_token_success(self):
        """Test successful token verification"""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)
        
        payload = verify_token(token)
        assert payload["sub"] == "test@example.com"
    
    def test_verify_token_invalid(self):
        """Test verification of invalid token"""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(invalid_token)
        
        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in exc_info.value.detail
    
    def test_verify_token_expired(self):
        """Test verification of expired token"""
        data = {"sub": "test@example.com"}
        # Create token that expires immediately
        expires_delta = timedelta(seconds=-1)
        token = create_access_token(data, expires_delta)
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)
        
        assert exc_info.value.status_code == 401
    
    def test_extract_email_from_token_success(self):
        """Test successful email extraction from token"""
        email = "test@example.com"
        data = {"sub": email}
        token = create_access_token(data)
        
        extracted_email = extract_email_from_token(token)
        assert extracted_email == email
    
    def test_extract_email_from_token_no_subject(self):
        """Test email extraction from token without subject"""
        data = {"user_id": 123}  # No 'sub' field
        token = create_access_token(data)
        
        with pytest.raises(HTTPException) as exc_info:
            extract_email_from_token(token)
        
        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in exc_info.value.detail
    
    def test_extract_email_from_token_invalid(self):
        """Test email extraction from invalid token"""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(HTTPException) as exc_info:
            extract_email_from_token(invalid_token)
        
        assert exc_info.value.status_code == 401


class TestTokenSecurity:
    """Test token security features"""
    
    def test_token_with_wrong_secret(self):
        """Test token verification with wrong secret key"""
        data = {"sub": "test@example.com"}
        token = jwt.encode(data, "wrong_secret", algorithm=settings.ALGORITHM)
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)
        
        assert exc_info.value.status_code == 401
    
    def test_token_with_wrong_algorithm(self):
        """Test token verification with wrong algorithm"""
        data = {"sub": "test@example.com"}
        token = jwt.encode(data, settings.SECRET_KEY, algorithm="HS512")
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)
        
        assert exc_info.value.status_code == 401