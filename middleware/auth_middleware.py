"""
JWT authentication middleware for protecting routes
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional
import logging

from utils.database import get_db
from utils.auth import extract_email_from_token
from services.user_service import UserService
from models.database import User

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get current authenticated user from JWT token
    
    Args:
        credentials: HTTP Bearer credentials
        db: Database session
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Extract email from token
        email = extract_email_from_token(credentials.credentials)
        
        # Get user from database
        user = UserService.get_user_by_email(db, email)
        if user is None:
            logger.warning(f"User not found in database: {email}")
            raise credentials_exception
        
        if not user.is_active:
            logger.warning(f"Inactive user attempted access: {email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_user: {e}")
        raise credentials_exception


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get current active user
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        User: Current active user
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Dependency to optionally get current user (for endpoints that work with or without auth)
    
    Args:
        credentials: Optional HTTP Bearer credentials
        db: Database session
        
    Returns:
        User or None: Current user if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    try:
        email = extract_email_from_token(credentials.credentials)
        user = UserService.get_user_by_email(db, email)
        
        if user and user.is_active:
            return user
        return None
        
    except Exception as e:
        logger.debug(f"Optional auth failed: {e}")
        return None