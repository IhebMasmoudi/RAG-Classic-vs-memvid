"""
Unit tests for user service
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

from models.database import Base, User
from models.schemas import UserCreate
from services.user_service import UserService
from utils.auth import get_password_hash


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create test database session"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return UserCreate(
        email="test@example.com",
        password="test_password_123"
    )


@pytest.fixture
def existing_user(db_session):
    """Create an existing user in the database"""
    user_data = UserCreate(
        email="existing@example.com",
        password="existing_password"
    )
    return UserService.create_user(db_session, user_data)


class TestUserCreation:
    """Test user creation functionality"""
    
    def test_create_user_success(self, db_session, sample_user_data):
        """Test successful user creation"""
        user = UserService.create_user(db_session, sample_user_data)
        
        assert user.id is not None
        assert user.email == sample_user_data.email
        assert user.password_hash != sample_user_data.password
        assert user.is_active is True
        assert user.created_at is not None
    
    def test_create_user_duplicate_email(self, db_session, sample_user_data):
        """Test user creation with duplicate email"""
        # Create first user
        UserService.create_user(db_session, sample_user_data)
        
        # Try to create second user with same email
        with pytest.raises(HTTPException) as exc_info:
            UserService.create_user(db_session, sample_user_data)
        
        assert exc_info.value.status_code == 400
        assert "Email already registered" in exc_info.value.detail
    
    def test_create_user_password_hashed(self, db_session, sample_user_data):
        """Test that user password is properly hashed"""
        user = UserService.create_user(db_session, sample_user_data)
        
        # Password should be hashed, not stored as plain text
        assert user.password_hash != sample_user_data.password
        assert user.password_hash.startswith("$2b$")


class TestUserAuthentication:
    """Test user authentication functionality"""
    
    def test_authenticate_user_success(self, db_session, existing_user):
        """Test successful user authentication"""
        authenticated_user = UserService.authenticate_user(
            db_session, 
            existing_user.email, 
            "existing_password"
        )
        
        assert authenticated_user is not None
        assert authenticated_user.id == existing_user.id
        assert authenticated_user.email == existing_user.email
    
    def test_authenticate_user_wrong_password(self, db_session, existing_user):
        """Test authentication with wrong password"""
        authenticated_user = UserService.authenticate_user(
            db_session, 
            existing_user.email, 
            "wrong_password"
        )
        
        assert authenticated_user is None
    
    def test_authenticate_user_nonexistent_email(self, db_session):
        """Test authentication with non-existent email"""
        authenticated_user = UserService.authenticate_user(
            db_session, 
            "nonexistent@example.com", 
            "any_password"
        )
        
        assert authenticated_user is None
    
    def test_authenticate_inactive_user(self, db_session, existing_user):
        """Test authentication of inactive user"""
        # Deactivate user
        existing_user.is_active = False
        db_session.commit()
        
        authenticated_user = UserService.authenticate_user(
            db_session, 
            existing_user.email, 
            "existing_password"
        )
        
        assert authenticated_user is None


class TestUserRetrieval:
    """Test user retrieval functionality"""
    
    def test_get_user_by_email_success(self, db_session, existing_user):
        """Test successful user retrieval by email"""
        user = UserService.get_user_by_email(db_session, existing_user.email)
        
        assert user is not None
        assert user.id == existing_user.id
        assert user.email == existing_user.email
    
    def test_get_user_by_email_not_found(self, db_session):
        """Test user retrieval with non-existent email"""
        user = UserService.get_user_by_email(db_session, "nonexistent@example.com")
        
        assert user is None
    
    def test_get_user_by_id_success(self, db_session, existing_user):
        """Test successful user retrieval by ID"""
        user = UserService.get_user_by_id(db_session, existing_user.id)
        
        assert user is not None
        assert user.id == existing_user.id
        assert user.email == existing_user.email
    
    def test_get_user_by_id_not_found(self, db_session):
        """Test user retrieval with non-existent ID"""
        user = UserService.get_user_by_id(db_session, 99999)
        
        assert user is None


class TestUserActivityUpdate:
    """Test user activity update functionality"""
    
    def test_update_user_activity_success(self, db_session, existing_user):
        """Test successful user activity update"""
        # Deactivate user
        result = UserService.update_user_activity(db_session, existing_user.id, False)
        
        assert result is True
        
        # Verify user is deactivated
        updated_user = UserService.get_user_by_id(db_session, existing_user.id)
        assert updated_user.is_active is False
        
        # Reactivate user
        result = UserService.update_user_activity(db_session, existing_user.id, True)
        
        assert result is True
        
        # Verify user is reactivated
        updated_user = UserService.get_user_by_id(db_session, existing_user.id)
        assert updated_user.is_active is True
    
    def test_update_user_activity_not_found(self, db_session):
        """Test user activity update with non-existent user"""
        result = UserService.update_user_activity(db_session, 99999, False)
        
        assert result is False