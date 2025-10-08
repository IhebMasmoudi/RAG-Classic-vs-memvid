"""
Database connection and session utilities
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from typing import Generator
import logging
from config import settings
from models.database import Base

logger = logging.getLogger(__name__)

# Create database engine
if "postgresql" in settings.DATABASE_URL:
    # PostgreSQL configuration
    engine = create_engine(
        settings.DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=300
    )
else:
    # SQLite configuration (fallback)
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False}
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database session
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        logger.error(f"Database context error: {e}")
        db.rollback()
        raise
    finally:
        db.close()