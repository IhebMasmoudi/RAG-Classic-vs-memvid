#!/usr/bin/env python3
"""
Verify database cleanup
"""
import sys
from pathlib import Path

# Add the BackEnd directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import text
from utils.database import get_db_context
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_cleanup():
    """Verify the cleanup was successful"""
    try:
        with get_db_context() as db:
            # Check table counts
            tables = {
                'users': 'SELECT COUNT(*) FROM users',
                'documents': 'SELECT COUNT(*) FROM documents', 
                'document_chunks': 'SELECT COUNT(*) FROM document_chunks',
                'query_history': 'SELECT COUNT(*) FROM query_history',
                'system_logs': 'SELECT COUNT(*) FROM system_logs'
            }
            
            print("Database Status After Cleanup:")
            print("=" * 35)
            
            for table_name, query in tables.items():
                try:
                    result = db.execute(text(query))
                    count = result.scalar()
                    status = "✅ Preserved" if table_name == 'users' and count > 0 else "🧹 Cleaned" if count == 0 else f"⚠️  {count} records"
                    print(f"{table_name:15}: {count:3d} records - {status}")
                except Exception as e:
                    print(f"{table_name:15}: Error - {e}")
            
            print("\nVector Store Status:")
            vector_path = Path("data/vector_store")
            if vector_path.exists():
                files = list(vector_path.glob("*"))
                if files:
                    print(f"⚠️  {len(files)} files remaining: {[f.name for f in files]}")
                else:
                    print("🧹 Empty directory - cleaned")
            else:
                print("🧹 Directory doesn't exist - cleaned")
            
            print("\nUploads Status:")
            uploads_path = Path("uploads")
            if uploads_path.exists():
                files = list(uploads_path.glob("*"))
                if files:
                    print(f"⚠️  {len(files)} files remaining: {[f.name for f in files]}")
                else:
                    print("🧹 Empty directory - cleaned")
            else:
                print("🧹 Directory doesn't exist - cleaned")
                
    except Exception as e:
        logger.error(f"Verification failed: {e}")

if __name__ == "__main__":
    verify_cleanup()