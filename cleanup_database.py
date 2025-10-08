#!/usr/bin/env python3
"""
Database cleanup script - truncates all tables except users to start fresh
"""
import sys
import os
from pathlib import Path

# Add the BackEnd directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import text
from utils.database import get_db_context
from models.database import Base
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_database():
    """Clean up database tables while preserving users"""
    try:
        with get_db_context() as db:
            logger.info("Starting database cleanup...")
            
            # Truncate tables in order (respecting foreign key constraints)
            tables_to_truncate = [
                'query_history',
                'document_chunks', 
                'documents',
                'system_logs'
            ]
            
            for table in tables_to_truncate:
                try:
                    # Use TRUNCATE for PostgreSQL or DELETE for SQLite
                    db.execute(text(f"DELETE FROM {table}"))
                    logger.info(f"Cleared table: {table}")
                except Exception as e:
                    logger.warning(f"Could not clear table {table}: {e}")
            
            # Commit the changes
            db.commit()
            logger.info("Database cleanup completed successfully")
            
            # Show remaining user count
            result = db.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()
            logger.info(f"Users table preserved with {user_count} users")
            
    except Exception as e:
        logger.error(f"Database cleanup failed: {e}")
        raise

def cleanup_vector_store():
    """Clean up vector store files"""
    try:
        import shutil
        vector_store_path = Path("data/vector_store")
        
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
            logger.info("Vector store files cleaned up")
        
        # Recreate the directory
        vector_store_path.mkdir(parents=True, exist_ok=True)
        logger.info("Vector store directory recreated")
        
    except Exception as e:
        logger.error(f"Vector store cleanup failed: {e}")

def cleanup_uploads():
    """Clean up uploaded files"""
    try:
        uploads_path = Path("uploads")
        
        if uploads_path.exists():
            import shutil
            shutil.rmtree(uploads_path)
            logger.info("Upload files cleaned up")
        
        # Recreate the directory
        uploads_path.mkdir(parents=True, exist_ok=True)
        logger.info("Uploads directory recreated")
        
    except Exception as e:
        logger.error(f"Uploads cleanup failed: {e}")

def main():
    """Main cleanup function"""
    print("RAG Comparison Platform - Database Cleanup")
    print("=" * 45)
    print("This will remove all documents, chunks, and query history")
    print("Users will be preserved")
    print()
    
    # Ask for confirmation
    confirm = input("Are you sure you want to proceed? (yes/no): ").lower().strip()
    
    if confirm != 'yes':
        print("Cleanup cancelled")
        return
    
    try:
        # Clean up database
        cleanup_database()
        
        # Clean up vector store
        cleanup_vector_store()
        
        # Clean up uploads
        cleanup_uploads()
        
        print("\n✅ Cleanup completed successfully!")
        print("- All documents and chunks removed")
        print("- Query history cleared")
        print("- Vector store reset")
        print("- Upload files removed")
        print("- Users preserved")
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()