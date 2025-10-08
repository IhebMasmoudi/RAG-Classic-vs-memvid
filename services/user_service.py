"""
User service for database operations
"""
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from typing import Optional
import logging

from models.database import User
from models.schemas import UserCreate
from utils.auth import get_password_hash, verify_password

logger = logging.getLogger(__name__)


class UserService:
    """Service class for user-related database operations"""
    
    @staticmethod
    def create_user(db: Session, user_data: UserCreate) -> User:
        """
        Create a new user in the database
        
        Args:
            db: Database session
            user_data: User creation data
            
        Returns:
            User: Created user object
            
        Raises:
            HTTPException: If email already exists
        """
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(User.email == user_data.email).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Hash password and create user
            hashed_password = get_password_hash(user_data.password)
            db_user = User(
                email=user_data.email,
                password_hash=hashed_password
            )
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"User created successfully: {user_data.email}")
            return db_user
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Integrity error creating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    
    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
        """
        Authenticate a user with email and password
        
        Args:
            db: Database session
            email: User email
            password: Plain text password
            
        Returns:
            User: Authenticated user object or None if authentication fails
        """
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user:
                logger.warning(f"Authentication failed: User not found - {email}")
                return None
            
            if not user.is_active:
                logger.warning(f"Authentication failed: User inactive - {email}")
                return None
            
            if not verify_password(password, user.password_hash):
                logger.warning(f"Authentication failed: Invalid password - {email}")
                return None
            
            logger.info(f"User authenticated successfully: {email}")
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """
        Get user by email address
        
        Args:
            db: Database session
            email: User email
            
        Returns:
            User: User object or None if not found
        """
        try:
            return db.query(User).filter(User.email == email).first()
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """
        Get user by ID
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            User: User object or None if not found
        """
        try:
            return db.query(User).filter(User.id == user_id).first()
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    @staticmethod
    def update_user_activity(db: Session, user_id: int, is_active: bool) -> bool:
        """
        Update user active status
        
        Args:
            db: Database session
            user_id: User ID
            is_active: New active status
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            
            user.is_active = is_active
            db.commit()
            
            logger.info(f"User activity updated: {user.email} - Active: {is_active}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating user activity: {e}")
            return False