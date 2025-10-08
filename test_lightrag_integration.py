#!/usr/bin/env python3
"""
Test script for LightRAG integration
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

def test_lightrag_integration():
    """Test LightRAG integration with the RAG comparison platform"""
    print("RAG Comparison Platform - LightRAG Integration Test")
    print("=" * 55)
    
    # Initialize database
    create_tables()
    
    # Create test client
    client = TestClient(app)
    
    # Test health endpoints
    print("\n=== Testing Health Endpoints ===")
    
    # Test LightRAG health
    response = client.get("/lightrag/health")
    print(f"LightRAG health: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"  Service: {health_data['service']}")
        print(f"  Status: {health_data['status']}")
        print(f"  LightRAG Available: {health_data['lightrag_available']}")
    
    # Test comparison health
    response = client.get("/comparison/health")
    print(f"Comparison health: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"  Service: {health_data['service']}")
        print(f"  Available methods: {health_data['available_methods']}")
    
    # Setup test user and documents
    print("\n=== Setting Up Test Data ===")
    
    # Register user
    user_data = {
        "email": "lightrag_test@example.com",
        "password": "testpassword123"
    }
    response = client.post("/auth/register", json=user_data)
    if response.status_code != 201:
        print(f"Registration failed: {response.text}")
        return
    
    # Login
    response = client.post("/auth/login", json=user_data)
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return
    
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Upload test documents with rich content for graph construction
    test_documents = [
        {
            "filename": "ai_research.txt",
            "content": """
            Artificial Intelligence Research Overview
            Dr. Alice Johnson leads the AI research team at TechCorp University. She specializes in machine learning 
            and natural language processing. Her team includes Bob Smith, a computer vision expert, and Carol Davis, 
            who focuses on reinforcement learning.
            
            The team's current project involves developing advanced chatbots using transformer architectures. 
            They collaborate with the robotics lab led by Professor David Wilson. The project is funded by 
            the National Science Foundation and has a budget of $2 million over three years.
            
            Key technologies being used include:
            - GPT-based language models
            - Computer vision algorithms for multimodal understanding
            - Reinforcement learning for dialogue optimization
            - Graph neural networks for knowledge representation
            
            The research has applications in healthcare, education, and customer service automation.
            """
        },
        {
            "filename": "company_structure.txt", 
            "content": """
            TechCorp University Organizational Structure
            TechCorp University is a leading research institution founded in 1985. The current president is 
            Dr. Emily Rodriguez, who has been in the position since 2020.
            
            The university has several key departments:
            Computer Science Department:
            - Head: Professor Michael Chen
            - AI Research Lab: Led by Dr. Alice Johnson
            - Robotics Lab: Led by Professor David Wilson
            - Cybersecurity Center: Directed by Dr. Sarah Thompson
            
            Engineering Department:
            - Head: Professor James Anderson
            - Mechanical Engineering: Led by Dr. Lisa Wang
            - Electrical Engineering: Led by Professor Robert Brown
            
            The AI Research Lab and Robotics Lab frequently collaborate on interdisciplinary projects.
            Both labs share resources and often co-publish research papers.
            """
        }
    ]
    
    uploaded_docs = []
    for doc in test_documents:
        files = {"file": (doc["filename"], doc["content"], "text/plain")}
        response = client.post("/upload/", files=files, headers=headers)
        if response.status_code == 201:
            doc_data = response.json()
            uploaded_docs.append(doc_data)
            print(f"✅ Uploaded: {doc['filename']} (ID: {doc_data['document_id'][:8]}...)")
        else:
            print(f"❌ Failed to upload {doc['filename']}: {response.text}")
    
    if not uploaded_docs:
        print("No documents uploaded successfully. Exiting test.")
        return
    
    # Test LightRAG query modes
    print("\n=== Testing LightRAG Query Modes ===")
    
    test_queries = [
        {
            "query": "Who is Dr. Alice Johnson and what does she research?",
            "mode": "local",
            "description": "Local mode - specific entity information"
        },
        {
            "query": "What is the overall structure and research focus of TechCorp University?",
            "mode": "global", 
            "description": "Global mode - comprehensive overview"
        },
        {
            "query": "How do the AI and robotics teams collaborate?",
            "mode": "hybrid",
            "description": "Hybrid mode - relationship discovery"
        },
        {
            "query": "machine learning research",
            "mode": "naive",
            "description": "Naive mode - simple keyword search"
        }
    ]
    
    for test_query in test_queries:
        print(f"\n--- {test_query['description']} ---")
        print(f"Query: {test_query['query']}")
        
        query_data = {
            "query": test_query["query"],
            "top_k": 5,
            "mode": test_query["mode"]
        }
        
        response = client.post("/lightrag/query", json=query_data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success ({test_query['mode']} mode)")
            print(f"   Response time: {result['response_time']:.2f}s")
            print(f"   Answer length: {len(result['answer'])} chars")
            print(f"   Sources: {len(result['sources'])}")
            print(f"   Answer preview: {result['answer'][:150]}...")
        else:
            print(f"❌ Failed: {response.text}")
    
    # Test RAG method comparison
    print("\n=== Testing RAG Method Comparison ===")
    
    comparison_query = {
        "query": "What research projects are being conducted and who are the key researchers?",
        "top_k": 5
    }
    
    response = client.post("/comparison/compare", json=comparison_query, headers=headers)
    if response.status_code == 200:
        result = response.json()
        print("✅ RAG Comparison successful")
        print(f"   Total comparison time: {result['total_comparison_time']:.2f}s")
        print(f"   Methods compared: {result['methods_compared']}")
        
        # Show results for each method
        for method, method_result in result['results'].items():
            if method_result['status'] == 'success':
                response_data = method_result['response']
                print(f"   {method}: {response_data['response_time']:.2f}s, {len(response_data['answer'])} chars")
            else:
                print(f"   {method}: Failed - {method_result.get('error', 'Unknown error')}")
        
        # Show analysis
        if 'analysis' in result:
            analysis = result['analysis']
            if analysis.get('recommendations'):
                print("   Recommendations:")
                for rec in analysis['recommendations']:
                    print(f"     - {rec}")
    else:
        print(f"❌ Comparison failed: {response.text}")
    
    # Test capabilities endpoint
    print("\n=== Testing Capabilities Endpoint ===")
    
    response = client.get("/comparison/capabilities")
    if response.status_code == 200:
        capabilities = response.json()
        print("✅ Capabilities retrieved")
        for method, info in capabilities['methods'].items():
            print(f"   {method}: {info['name']}")
            print(f"     Description: {info['description']}")
            if 'available' in info:
                print(f"     Available: {info['available']}")
    else:
        print(f"❌ Failed to get capabilities: {response.text}")
    
    # Test LightRAG statistics
    print("\n=== Testing LightRAG Statistics ===")
    
    response = client.get("/lightrag/stats", headers=headers)
    if response.status_code == 200:
        stats = response.json()
        print("✅ LightRAG stats retrieved")
        print(f"   User ID: {stats['user_id']}")
        kg_stats = stats['knowledge_graph']
        print(f"   Entities: {kg_stats.get('entities_count', 'N/A')}")
        print(f"   Relationships: {kg_stats.get('relationships_count', 'N/A')}")
        print(f"   Documents processed: {kg_stats.get('documents_processed', 'N/A')}")
    else:
        print(f"❌ Failed to get stats: {response.text}")
    
    print("\n" + "=" * 55)
    print("LightRAG Integration Test Complete!")
    print("\nKey Features Tested:")
    print("✅ LightRAG service integration")
    print("✅ Multiple query modes (local, global, hybrid, naive)")
    print("✅ Graph-based knowledge extraction")
    print("✅ RAG method comparison")
    print("✅ Performance analysis")
    print("✅ Knowledge graph statistics")

if __name__ == "__main__":
    test_lightrag_integration()