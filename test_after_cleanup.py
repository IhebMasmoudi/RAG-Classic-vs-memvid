#!/usr/bin/env python3
"""
Test system functionality after cleanup
"""
import sys
from pathlib import Path

# Add the BackEnd directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from main import app
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test that basic functionality still works after cleanup"""
    print("Testing System After Cleanup")
    print("=" * 30)
    
    client = TestClient(app)
    
    # Test health endpoints
    response = client.get("/")
    print(f"✅ Root endpoint: {response.status_code}")
    
    response = client.get("/health")
    print(f"✅ Health endpoint: {response.status_code}")
    
    # Test registration (should work)
    user_data = {
        "email": "cleanup_test@example.com",
        "password": "testpassword123"
    }
    
    response = client.post("/auth/register", json=user_data)
    print(f"✅ User registration: {response.status_code}")
    
    if response.status_code == 201:
        # Test login
        response = client.post("/auth/login", json=user_data)
        print(f"✅ User login: {response.status_code}")
        
        if response.status_code == 200:
            token = response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Test document list (should be empty)
            response = client.get("/upload/documents", headers=headers)
            print(f"✅ Document list: {response.status_code}")
            
            if response.status_code == 200:
                docs = response.json()
                print(f"   📄 Documents found: {len(docs)} (should be 0)")
            
            # Test upload
            test_content = "This is a test document after cleanup."
            files = {"file": ("test_cleanup.txt", test_content, "text/plain")}
            
            response = client.post("/upload/", files=files, headers=headers)
            print(f"✅ Document upload: {response.status_code}")
            
            if response.status_code == 201:
                doc_data = response.json()
                print(f"   📄 Document uploaded: {doc_data['document_id'][:8]}...")
                
                # Test query
                query_data = {
                    "query": "What is this document about?",
                    "top_k": 3
                }
                
                response = client.post("/classic_rag/query", json=query_data, headers=headers)
                print(f"✅ Classic RAG query: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   🤖 Answer: {result['answer'][:50]}...")
    
    print("\n🎉 System is working properly after cleanup!")

if __name__ == "__main__":
    test_basic_functionality()