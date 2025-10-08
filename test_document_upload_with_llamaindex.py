#!/usr/bin/env python3
"""
Test document upload with LlamaIndex chunking
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

def test_upload_with_llamaindex():
    """Test document upload with LlamaIndex chunking"""
    print("Testing Document Upload with LlamaIndex Chunking")
    print("=" * 50)
    
    client = TestClient(app)
    
    # Register and login user
    user_data = {
        "email": "llamatest@example.com",
        "password": "testpassword123"
    }
    
    response = client.post("/auth/register", json=user_data)
    if response.status_code != 201:
        print(f"Registration failed: {response.text}")
        return
    
    response = client.post("/auth/login", json=user_data)
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return
    
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create test document with more content
    test_content = """
    Artificial Intelligence and Machine Learning Overview
    
    Artificial Intelligence (AI) is a broad field of computer science focused on creating intelligent machines. 
    These systems can perform tasks that typically require human intelligence, such as visual perception, 
    speech recognition, decision-making, and language translation.
    
    Machine Learning Fundamentals
    
    Machine learning is a subset of AI that enables computers to learn and improve from experience without 
    being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences 
    from patterns in data. The three main types of machine learning are supervised learning, unsupervised 
    learning, and reinforcement learning.
    
    Deep Learning and Neural Networks
    
    Deep learning is a subset of machine learning that uses neural networks with multiple layers to 
    progressively extract higher-level features from raw input. These deep neural networks can automatically 
    discover representations needed for feature detection or classification from raw data.
    
    Natural Language Processing Applications
    
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers 
    and human language. It involves programming computers to process and analyze large amounts of natural 
    language data. Applications include chatbots, language translation, sentiment analysis, and text summarization.
    
    Computer Vision and Image Recognition
    
    Computer vision is another important area of AI that deals with how computers can gain understanding 
    from digital images or videos. It seeks to automate tasks that the human visual system can do, such as 
    recognizing objects, faces, or activities in images and videos.
    """
    
    files = {"file": ("ai_comprehensive.txt", test_content, "text/plain")}
    
    print("Uploading document...")
    response = client.post("/upload/", files=files, headers=headers)
    
    if response.status_code == 201:
        upload_result = response.json()
        print(f"✅ Upload successful!")
        print(f"   Document ID: {upload_result['document_id']}")
        print(f"   Filename: {upload_result['filename']}")
        print(f"   Chunks created: {upload_result['chunks_created']}")
        print(f"   Embeddings generated: {upload_result['embeddings_generated']}")
        
        # Test query with the uploaded document
        print("\nTesting query with uploaded document...")
        query_data = {
            "query": "What is machine learning and how does it work?",
            "top_k": 3
        }
        
        response = client.post("/classic_rag/query", json=query_data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful!")
            print(f"   Answer: {result['answer'][:200]}...")
            print(f"   Response time: {result['response_time']:.2f}s")
            print(f"   Chunks used: {result['chunks_used']}")
            print(f"   Sources: {len(result['sources'])}")
        else:
            print(f"❌ Query failed: {response.text}")
    else:
        print(f"❌ Upload failed: {response.text}")

if __name__ == "__main__":
    test_upload_with_llamaindex()