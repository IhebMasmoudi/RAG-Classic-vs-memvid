#!/usr/bin/env python3
"""
Test script for RAG pipelines
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
from utils.database import create_tables

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_test_data(client):
    """Setup test user and document"""
    print("=== Setting up test data ===")
    
    # Register user
    user_data = {
        "email": "ragtest@example.com",
        "password": "testpassword123"
    }
    
    response = client.post("/auth/register", json=user_data)
    if response.status_code != 201:
        print(f"Registration failed: {response.text}")
        return None
    
    # Login
    response = client.post("/auth/login", json=user_data)
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return None
    
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Upload a test document
    test_content = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. 
    Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.
    Deep learning is a subset of machine learning that uses neural networks with multiple layers.
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language.
    Computer vision is another important area of AI that deals with how computers can gain understanding from digital images or videos.
    """
    
    files = {
        "file": ("ai_document.txt", test_content, "text/plain")
    }
    
    response = client.post("/upload/", files=files, headers=headers)
    if response.status_code != 201:
        print(f"Upload failed: {response.text}")
        return None
    
    document_data = response.json()
    print(f"Uploaded document: {document_data['document_id']}")
    
    return headers, document_data

def test_classic_rag(client, headers):
    """Test Classic RAG pipeline"""
    print("\n=== Testing Classic RAG ===")
    
    query_data = {
        "query": "What is machine learning?",
        "top_k": 3
    }
    
    try:
        response = client.post("/classic_rag/query", json=query_data, headers=headers)
        print(f"Classic RAG status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Response time: {result['response_time']:.2f}s")
            print(f"Chunks used: {result['chunks_used']}")
            print(f"Sources: {len(result['sources'])}")
            return result
        else:
            print(f"Classic RAG failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"Classic RAG exception: {e}")
        return None

def test_memvid_rag(client, headers):
    """Test MemVid RAG pipeline"""
    print("\n=== Testing MemVid RAG ===")
    
    query_data = {
        "query": "What is machine learning?",
        "top_k": 3,
        "context_window": 2
    }
    
    try:
        response = client.post("/memvid_rag/query", json=query_data, headers=headers)
        print(f"MemVid RAG status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Response time: {result['response_time']:.2f}s")
            print(f"Chunks used: {result['chunks_used']}")
            print(f"Sources: {len(result['sources'])}")
            print(f"MemVid metadata: {list(result['memvid_metadata'].keys())}")
            return result
        else:
            print(f"MemVid RAG failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"MemVid RAG exception: {e}")
        return None

def test_health_endpoints(client):
    """Test health endpoints"""
    print("\n=== Testing Health Endpoints ===")
    
    # Classic RAG health
    response = client.get("/classic_rag/health")
    print(f"Classic RAG health: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"Classic RAG service: {health_data['service']}")
    
    # MemVid RAG health
    response = client.get("/memvid_rag/health")
    print(f"MemVid RAG health: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"MemVid RAG service: {health_data['service']}")

def test_document_selection_feature(client, headers):
    """Test document selection feature (if implemented)"""
    print("\n=== Testing Document Selection ===")
    
    # Get user documents
    response = client.get("/upload/documents", headers=headers)
    print(f"Get documents status: {response.status_code}")
    
    if response.status_code == 200:
        documents = response.json()
        print(f"Found {len(documents)} documents")
        for doc in documents:
            print(f"  - {doc['filename']} ({doc['id'][:8]}...)")
        
        # Test query with document selection (if the feature exists)
        if documents:
            query_data = {
                "query": "What is deep learning?",
                "top_k": 3,
                "document_ids": [documents[0]["id"]]  # This might not be implemented yet
            }
            
            # This will likely fail since document selection isn't implemented
            response = client.post("/classic_rag/query", json=query_data, headers=headers)
            print(f"Document selection test: {response.status_code}")
    else:
        print(f"Failed to get documents: {response.text}")

def main():
    """Main test function"""
    print("RAG Comparison Platform - RAG Pipeline Test")
    print("=" * 50)
    
    # Initialize database
    create_tables()
    
    # Create test client
    client = TestClient(app)
    
    # Setup test data
    test_data = setup_test_data(client)
    if not test_data:
        print("Failed to setup test data")
        return
    
    headers, document_data = test_data
    
    # Test health endpoints
    test_health_endpoints(client)
    
    # Test RAG pipelines
    classic_result = test_classic_rag(client, headers)
    memvid_result = test_memvid_rag(client, headers)
    
    # Test document selection feature
    test_document_selection_feature(client, headers)
    
    # Compare results
    if classic_result and memvid_result:
        print("\n=== Comparison ===")
        print(f"Classic RAG time: {classic_result['response_time']:.2f}s")
        print(f"MemVid RAG time: {memvid_result['response_time']:.2f}s")
        
        time_diff = memvid_result['response_time'] - classic_result['response_time']
        if time_diff > 0:
            print(f"MemVid is {time_diff:.2f}s slower")
        else:
            print(f"MemVid is {abs(time_diff):.2f}s faster")

if __name__ == "__main__":
    main()