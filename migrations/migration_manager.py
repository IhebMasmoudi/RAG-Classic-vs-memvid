"""
Database migration manager for the RAG Comparison Platform
"""
import logging
import os
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy import text
from sqlalchemy.orm import Session

from utils.database import SessionLocal, engine
from models.database import Base

logger = logging.getLogger(__name__)


class Migration:
    """Base migration class"""
    
    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
        self.timestamp = datetime.now()
    
    def up(self, db: Session):
        """Apply migration"""
        raise NotImplementedError("Migration must implement up() method")
    
    def down(self, db: Session):
        """Rollback migration"""
        raise NotImplementedError("Migration must implement down() method")
    
    def __str__(self):
        return f"Migration {self.version}: {self.description}"


class MigrationManager:
    """Manages database migrations"""
    
    def __init__(self):
        self.migrations: List[Migration] = []
        self._init_migration_table()
    
    def _init_migration_table(self):
        """Initialize migration tracking table"""
        try:
            db = SessionLocal()
            # Create migrations table if it doesn't exist
            db.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(50) PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            db.commit()
            db.close()
            logger.info("Migration tracking table initialized")
        except Exception as e:
            logger.error(f"Error initializing migration table: {e}")
            raise
    
    def register_migration(self, migration: Migration):
        """Register a migration"""
        self.migrations.append(migration)
        logger.info(f"Registered migration: {migration}")
    
    def get_applied_migrations(self) -> Lis