#!/usr/bin/env python3
"""
Debug script to test upload functionality and identify issues
"""
import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the BackEnd directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from main import app
from config import settings
from utils.database import create_tables
from services.vector_store import initialize_vector_store

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_basic_endpoints():
    """Test basic endpoints"""
    client = TestClient(app)
    
    print("=== Testing Basic Endpoints ===")
    
    # Test root endpoint
    response = client.get("/")
    print(f"Root endpoint: {response.status_code} - {response.json()}")
    
    # Test health endpoint
    response = client.get("/health")
    print(f"Health endpoint: {response.status_code} - {response.json()}")
    
    return client

def test_auth_endpoints(client):
    """Test authentication endpoints"""
    print("\n=== Testing Authentication ===")
    
    # Test user registration
    user_data = {
        "email": "test@example.com",
        "password": "testpassword123"
    }
    
    response = client.post("/auth/register", json=user_data)
    print(f"Register: {response.status_code}")
    if response.status_code != 201:
        print(f"Register error: {response.text}")
        return None
    
    # Test user login
    response = client.post("/auth/login", json=user_data)
    print(f"Login: {response.status_code}")
    if response.status_code != 200:
        print(f"Login error: {response.text}")
        return None
    
    token_data = response.json()
    token = token_data.get("access_token")
    print(f"Got token: {token[:20]}...")
    
    return token

def test_upload_endpoint(client, token):
    """Test upload endpoint"""
    print("\n=== Testing Upload Endpoint ===")
    
    if not token:
        print("No token available, skipping upload test")
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create a test file
    test_content = "This is a test document for upload testing. It contains some sample text that will be processed by the RAG system."
    
    files = {
        "file": ("test.txt", test_content, "text/plain")
    }
    
    try:
        response = client.post("/upload/", files=files, headers=headers)
        print(f"Upload: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 201:
            print("Upload successful!")
            return response.json()
        else:
            print(f"Upload failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"Upload exception: {e}")
        return None

def test_services():
    """Test individual services"""
    print("\n=== Testing Services ===")
    
    # Test embedding service
    try:
        from services.embedding_service import test_embedding_service
        result = asyncio.run(test_embedding_service())
        print(f"Embedding service test: {result}")
    except Exception as e:
        print(f"Embedding service error: {e}")
    
    # Test vector store
    try:
        from services.vector_store import get_vector_store_stats
        stats = asyncio.run(get_vector_store_stats())
        print(f"Vector store stats: {stats}")
    except Exception as e:
        print(f"Vector store error: {e}")

def check_dependencies():
    """Check if all dependencies are available"""
    print("\n=== Checking Dependencies ===")
    
    try:
        import fastapi
        print(f"FastAPI: {fastapi.__version__}")
    except ImportError as e:
        print(f"FastAPI not available: {e}")
    
    try:
        import sentence_transformers
        print(f"Sentence Transformers: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"Sentence Transformers not available: {e}")
    
    try:
        import faiss
        print(f"FAISS: Available")
    except ImportError as e:
        print(f"FAISS not available: {e}")
    
    try:
        import google.generativeai as genai
        print(f"Google Generative AI: Available")
    except ImportError as e:
        print(f"Google Generative AI not available: {e}")
    
    try:
        import pypdf
        print(f"PyPDF: {pypdf.__version__}")
    except ImportError as e:
        print(f"PyPDF not available: {e}")

def check_database():
    """Check database connection"""
    print("\n=== Checking Database ===")
    
    try:
        create_tables()
        print("Database tables created successfully")
    except Exception as e:
        print(f"Database error: {e}")

def main():
    """Main test function"""
    print("RAG Comparison Platform - Upload Debug Test")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Check database
    check_database()
    
    # Test services
    test_services()
    
    # Test endpoints
    client = test_basic_endpoints()
    token = test_auth_endpoints(client)
    test_upload_endpoint(client, token)

if __name__ == "__main__":
    main()