"""
Authentication routes for user registration and login
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from datetime import timedelta
import logging

from utils.database import get_db
from utils.auth import create_access_token
from services.user_service import UserService
from models.schemas import UserCreate, UserLogin, Token, UserResponse
from middleware.auth_middleware import get_current_active_user
from models.database import User
from config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Security scheme
security = HTTPBearer()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user
    
    Args:
        user_data: User registration data
        db: Database session
        
    Returns:
        UserResponse: Created user information
        
    Raises:
        HTTPException: If registration fails
    """
    try:
        logger.info(f"User registration attempt: {user_data.email}")
        
        # Create user
        user = UserService.create_user(db, user_data)
        
        logger.info(f"User registered successfully: {user.email}")
        return UserResponse.model_validate(user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login_user(
    user_data: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT token
    
    Args:
        user_data: User login credentials
        db: Database session
        
    Returns:
        Token: JWT access token
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        logger.info(f"Login attempt: {user_data.email}")
        
        # Authenticate user
        user = UserService.authenticate_user(db, user_data.email, user_data.password)
        if not user:
            logger.warning(f"Failed login attempt: {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email},
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in successfully: {user.email}")
        return Token(access_token=access_token, token_type="bearer")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user information
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        UserResponse: Current user information
    """
    logger.info(f"User info requested: {current_user.email}")
    return UserResponse.model_validate(current_user)


@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_active_user)
):
    """
    Logout user (client-side token removal)
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        dict: Logout confirmation message
    """
    logger.info(f"User logged out: {current_user.email}")
    return {"message": "Successfully logged out"}


@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_active_user)
):
    """
    Refresh JWT token for current user
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Token: New JWT access token
    """
    try:
        # Create new access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": current_user.email},
            expires_delta=access_token_expires
        )
        
        logger.info(f"Token refreshed for user: {current_user.email}")
        return Token(access_token=access_token, token_type="bearer")
        
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )