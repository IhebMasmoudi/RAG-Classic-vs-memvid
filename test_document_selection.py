#!/usr/bin/env python3
"""
Test script for document selection functionality
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

def setup_test_user_and_documents(client):
    """Setup test user and multiple documents"""
    print("=== Setting up test user and documents ===")
    
    # Register user with timestamp to make it unique
    import time
    timestamp = int(time.time())
    user_data = {
        "email": f"docselect{timestamp}@example.com",
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
    
    # Upload multiple test documents
    documents = []
    
    # Document 1: AI/ML content
    ai_content = """
    Artificial Intelligence (AI) is the simulation of human intelligence in machines.
    Machine Learning (ML) is a subset of AI that enables computers to learn without explicit programming.
    Deep Learning uses neural networks with multiple layers to process data.
    Natural Language Processing (NLP) helps computers understand human language.
    """
    
    files = {"file": ("ai_basics.txt", ai_content, "text/plain")}
    response = client.post("/upload/", files=files, headers=headers)
    if response.status_code == 201:
        documents.append(response.json())
        print(f"Uploaded AI document: {documents[-1]['document_id'][:8]}...")
    
    # Document 2: Programming content
    prog_content = """
    Python is a high-level programming language known for its simplicity.
    JavaScript is widely used for web development and runs in browsers.
    SQL is used for managing and querying relational databases.
    Git is a version control system for tracking changes in code.
    """
    
    files = {"file": ("programming.txt", prog_content, "text/plain")}
    response = client.post("/upload/", files=files, headers=headers)
    if response.status_code == 201:
        documents.append(response.json())
        print(f"Uploaded Programming document: {documents[-1]['document_id'][:8]}...")
    
    # Document 3: Data Science content
    ds_content = """
    Data Science combines statistics, programming, and domain expertise.
    Data visualization helps communicate insights from data analysis.
    Statistical analysis is crucial for understanding data patterns.
    Big Data refers to extremely large datasets that require special tools.
    """
    
    files = {"file": ("data_science.txt", ds_content, "text/plain")}
    response = client.post("/upload/", files=files, headers=headers)
    if response.status_code == 201:
        documents.append(response.json())
        print(f"Uploaded Data Science document: {documents[-1]['document_id'][:8]}...")
    
    return headers, documents

def test_document_list(client, headers):
    """Test getting document list"""
    print("\n=== Testing Document List ===")
    
    response = client.get("/upload/documents", headers=headers)
    print(f"Get documents status: {response.status_code}")
    
    if response.status_code == 200:
        documents = response.json()
        print(f"Found {len(documents)} documents:")
        for doc in documents:
            print(f"  - {doc['filename']} (ID: {doc['id'][:8]}..., Chunks: {doc['chunk_count']})")
        return documents
    else:
        print(f"Failed to get documents: {response.text}")
        return []

def test_query_without_selection(client, headers):
    """Test query without document selection (should search all documents)"""
    print("\n=== Testing Query Without Document Selection ===")
    
    query_data = {
        "query": "What is machine learning?",
        "top_k": 5
    }
    
    response = client.post("/classic_rag/query", json=query_data, headers=headers)
    print(f"Classic RAG (all docs) status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer'][:100]}...")
        print(f"Sources from documents: {[src['document_id'][:8] for src in result['sources']]}")
        return result
    else:
        print(f"Query failed: {response.text}")
        return None

def test_query_with_document_selection(client, headers, documents):
    """Test query with specific document selection"""
    print("\n=== Testing Query With Document Selection ===")
    
    if not documents:
        print("No documents available for selection test")
        return
    
    # Test with AI document only
    ai_doc_id = documents[0]['id']
    query_data = {
        "query": "What is machine learning?",
        "top_k": 5,
        "document_ids": [ai_doc_id]
    }
    
    response = client.post("/classic_rag/query", json=query_data, headers=headers)
    print(f"Classic RAG (AI doc only) status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer'][:100]}...")
        print(f"Sources from documents: {[src['document_id'][:8] for src in result['sources']]}")
        
        # Verify all sources are from the selected document
        all_from_selected = all(src['document_id'] == ai_doc_id for src in result['sources'])
        print(f"All sources from selected document: {all_from_selected}")
    else:
        print(f"Query with selection failed: {response.text}")
    
    # Test with programming document only
    if len(documents) > 1:
        prog_doc_id = documents[1]['id']
        query_data = {
            "query": "What is Python?",
            "top_k": 5,
            "document_ids": [prog_doc_id]
        }
        
        response = client.post("/classic_rag/query", json=query_data, headers=headers)
        print(f"\nClassic RAG (Programming doc only) status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answer'][:100]}...")
            print(f"Sources from documents: {[src['document_id'][:8] for src in result['sources']]}")

def test_memvid_with_document_selection(client, headers, documents):
    """Test MemVid RAG with document selection"""
    print("\n=== Testing MemVid RAG With Document Selection ===")
    
    if not documents:
        print("No documents available for MemVid selection test")
        return
    
    # Test with Data Science document only
    if len(documents) > 2:
        ds_doc_id = documents[2]['id']
        query_data = {
            "query": "What is data visualization?",
            "top_k": 3,
            "context_window": 2,
            "document_ids": [ds_doc_id]
        }
        
        response = client.post("/memvid_rag/query", json=query_data, headers=headers)
        print(f"MemVid RAG (Data Science doc only) status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answer'][:100]}...")
            print(f"Sources from documents: {[src['document_id'][:8] for src in result['sources']]}")
            
            # Verify all sources are from the selected document
            all_from_selected = all(src['document_id'] == ds_doc_id for src in result['sources'])
            print(f"All sources from selected document: {all_from_selected}")
        else:
            print(f"MemVid query with selection failed: {response.text}")

def test_multiple_document_selection(client, headers, documents):
    """Test query with multiple document selection"""
    print("\n=== Testing Multiple Document Selection ===")
    
    if len(documents) < 2:
        print("Need at least 2 documents for multiple selection test")
        return
    
    # Select first two documents
    selected_docs = [documents[0]['id'], documents[1]['id']]
    query_data = {
        "query": "Tell me about programming and AI",
        "top_k": 5,
        "document_ids": selected_docs
    }
    
    response = client.post("/classic_rag/query", json=query_data, headers=headers)
    print(f"Classic RAG (multiple docs) status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer'][:100]}...")
        source_docs = [src['document_id'][:8] for src in result['sources']]
        print(f"Sources from documents: {source_docs}")
        
        # Verify all sources are from selected documents
        all_from_selected = all(src['document_id'] in selected_docs for src in result['sources'])
        print(f"All sources from selected documents: {all_from_selected}")

def main():
    """Main test function"""
    print("RAG Comparison Platform - Document Selection Test")
    print("=" * 55)
    
    # Initialize database
    create_tables()
    
    # Create test client
    client = TestClient(app)
    
    # Setup test data
    test_data = setup_test_user_and_documents(client)
    if not test_data:
        print("Failed to setup test data")
        return
    
    headers, uploaded_documents = test_data
    
    # Test document list retrieval
    documents = test_document_list(client, headers)
    
    # Test queries without document selection
    test_query_without_selection(client, headers)
    
    # Test queries with document selection
    test_query_with_document_selection(client, headers, documents)
    
    # Test MemVid with document selection
    test_memvid_with_document_selection(client, headers, documents)
    
    # Test multiple document selection
    test_multiple_document_selection(client, headers, documents)
    
    print("\n=== Document Selection Test Complete ===")

if __name__ == "__main__":
    main()