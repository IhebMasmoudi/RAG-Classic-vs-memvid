"""
PostgreSQL database initialization script
Creates the database and tables for the RAG Comparison Platform
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from config import settings

logger = logging.getLogger(__name__)


def create_database():
    """Create the PostgreSQL database if it doesn't exist"""
    try:
        # Connect to PostgreSQL server (not to a specific database)
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database='postgres'  # Connect to default postgres database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
            (settings.POSTGRES_DB,)
        )
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(f'CREATE DATABASE "{settings.POSTGRES_DB}"')
            logger.info(f"Created database: {settings.POSTGRES_DB}")
        else:
            logger.info(f"Database {settings.POSTGRES_DB} already exists")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        logger.error(f"Error creating database: {e}")
        raise


def test_connection():
    """Test connection to the RAG platform database"""
    try:
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DB
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"Connected to PostgreSQL: {version[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return False


def main():
    """Main initialization function"""
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing PostgreSQL database for RAG Comparison Platform...")
    print(f"Host: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
    print(f"Database: {settings.POSTGRES_DB}")
    print(f"User: {settings.POSTGRES_USER}")
    
    try:
        # Create database
        create_database()
        
        # Test connection
        if test_connection():
            print("✅ Database initialization successful!")
            
            # Create tables using SQLAlchemy
            from utils.database import create_tables
            create_tables()
            print("✅ Database tables created successfully!")
            
        else:
            print("❌ Database connection failed!")
            
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise


if __name__ == "__main__":
    main()