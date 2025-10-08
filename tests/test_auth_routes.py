"""
Unit tests for authentication routes
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from main import app
from models.database import Base
from utils.database import get_db
from services.user_service import UserService
from models.schemas import UserCreate


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_auth_routes.db"
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
            email="existing@example.com",
            password="existing_password"
        )
        user = UserService.create_user(db, user_data)
        return user
    finally:
        db.close()


class TestUserRegistration:
    """Test user registration endpoint"""
    
    def test_register_user_success(self, client):
        """Test successful user registration"""
        user_data = {
            "email": "test@example.com",
            "password": "test_password_123"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"]
        assert "id" in data
        assert "created_at" in data
        assert data["is_active"] is True
        assert "password" not in data
    
    def test_register_user_duplicate_email(self, client, existing_user):
        """Test registration with duplicate email"""
        user_data = {
            "email": existing_user.email,
            "password": "test_password_123"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "Email already registered" in data["detail"]
    
    def test_register_user_invalid_email(self, client):
        """Test registration with invalid email"""
        user_data = {
            "email": "invalid_email",
            "password": "test_password_123"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == 422
    
    def test_register_user_short_password(self, client):
        """Test registration with short password"""
        user_data = {
            "email": "test@example.com",
            "password": "short"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == 422
    
    def test_register_user_missing_fields(self, client):
        """Test registration with missing fields"""
        user_data = {
            "email": "test@example.com"
            # Missing password
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == 422


class TestUserLogin:
    """Test user login endpoint"""
    
    def test_login_user_success(self, client, existing_user):
        """Test successful user login"""
        login_data = {
            "email": existing_user.email,
            "password": "existing_password"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 0
    
    def test_login_user_wrong_password(self, client, existing_user):
        """Test login with wrong password"""
        login_data = {
            "email": existing_user.email,
            "password": "wrong_password"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "Incorrect email or password" in data["detail"]
    
    def test_login_user_nonexistent_email(self, client):
        """Test login with non-existent email"""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "any_password"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "Incorrect email or password" in data["detail"]
    
    def test_login_user_invalid_email(self, client):
        """Test login with invalid email format"""
        login_data = {
            "email": "invalid_email",
            "password": "any_password"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 422
    
    def test_login_user_missing_fields(self, client):
        """Test login with missing fields"""
        login_data = {
            "email": "test@example.com"
            # Missing password
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 422


class TestProtectedRoutes:
    """Test protected routes with authentication"""
    
    def get_auth_headers(self, client, user_email, password):
        """Helper method to get authentication headers"""
        login_data = {
            "email": user_email,
            "password": password
        }
        response = client.post("/auth/login", json=login_data)
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_get_current_user_success(self, client, existing_user):
        """Test getting current user info with valid token"""
        headers = self.get_auth_headers(client, existing_user.email, "existing_password")
        
        response = client.get("/auth/me", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == existing_user.email
        assert data["id"] == existing_user.id
        assert data["is_active"] is True
    
    def test_get_current_user_no_token(self, client):
        """Test getting current user info without token"""
        response = client.get("/auth/me")
        
        assert response.status_code == 403
    
    def test_get_current_user_invalid_token(self, client):
        """Test getting current user info with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}
        
        response = client.get("/auth/me", headers=headers)
        
        assert response.status_code == 401
    
    def test_logout_user_success(self, client, existing_user):
        """Test user logout"""
        headers = self.get_auth_headers(client, existing_user.email, "existing_password")
        
        response = client.post("/auth/logout", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "Successfully logged out" in data["message"]
    
    def test_refresh_token_success(self, client, existing_user):
        """Test token refresh"""
        headers = self.get_auth_headers(client, existing_user.email, "existing_password")
        
        response = client.post("/auth/refresh", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 0
    
    def test_refresh_token_no_auth(self, client):
        """Test token refresh without authentication"""
        response = client.post("/auth/refresh")
        
        assert response.status_code == 403


class TestAuthenticationFlow:
    """Test complete authentication flow"""
    
    def test_complete_auth_flow(self, client):
        """Test complete registration -> login -> protected route flow"""
        # 1. Register user
        register_data = {
            "email": "flow@example.com",
            "password": "flow_password_123"
        }
        response = client.post("/auth/register", json=register_data)
        assert response.status_code == 201
        
        # 2. Login user
        login_data = {
            "email": "flow@example.com",
            "password": "flow_password_123"
        }
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        token = response.json()["access_token"]
        
        # 3. Access protected route
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "flow@example.com"
        
        # 4. Logout
        response = client.post("/auth/logout", headers=headers)
        assert response.status_code == 200