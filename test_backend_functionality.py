#!/usr/bin/env python3
"""
Simple test script to verify backend functionality
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.embedding_service import embedding_service, test_embedding_service
from services.llm_service import llm_service, test_llm_service
from services.vector_store import vector_store, initialize_vector_store
from utils.database import create_tables
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_embedding_service_functionality():
    """Test embedding service"""
    print("\n=== Testing Embedding Service ===")
    try:
        # Test embedding generation
        test_texts = ["Hello world", "This is a test"]
        embeddings = await embedding_service.generate_embeddings(test_texts)
        print(f"✓ Generated embeddings for {len(test_texts)} texts")
        print(f"  Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
        
        # Test query embedding
        query_embedding = await embedding_service.generate_query_embedding("test query")
        print(f"✓ Generated query embedding with dimension: {len(query_embedding)}")
        
        # Test service info
        info = embedding_service.get_model_info()
        print(f"✓ Embedding service info: {info}")
        
        return True
    except Exception as e:
        print(f"✗ Embedding service failed: {e}")
        return False

async def test_llm_service_functionality():
    """Test LLM service"""
    print("\n=== Testing LLM Service ===")
    try:
        # Test response generation
        prompt = "What is machine learning? Please provide a brief explanation."
        response = await llm_service.generate_response(prompt)
        print(f"✓ Generated LLM response: {response[:100]}...")
        
        # Test service info
        info = llm_service.get_model_info()
        print(f"✓ LLM service info: {info}")
        
        return True
    except Exception as e:
        print(f"✗ LLM service failed: {e}")
        return False

async def test_vector_store_functionality():
    """Test vector store"""
    print("\n=== Testing Vector Store ===")
    try:
        # Initialize vector store
        await initialize_vector_store()
        print("✓ Vector store initialized")
        
        # Test adding embeddings
        from models.database import DocumentChunk
        
        test_embeddings = [[0.1, 0.2, 0.3, 0.4] * 192, [0.4, 0.5, 0.6, 0.7] * 192]  # 768 dimensions
        test_chunks = [
            DocumentChunk(
                document_id="test_doc_1",
                content="This is test content 1",
                chunk_index=0,
                content_hash="hash1"
            ),
            DocumentChunk(
                document_id="test_doc_1",
                content="This is test content 2",
                chunk_index=1,
                content_hash="hash2"
            )
        ]
        
        vector_store.add_embeddings(test_embeddings, "test_doc_1", test_chunks)
        print("✓ Added test embeddings to vector store")
        
        # Test search
        query_embedding = [0.15, 0.25, 0.35, 0.45] * 192  # 768 dimensions
        results = vector_store.search(query_embedding, top_k=2, user_id=1)
        print(f"✓ Search returned {len(results)} results")
        
        # Test stats
        stats = vector_store.get_stats()
        print(f"✓ Vector store stats: {stats}")
        
        return True
    except Exception as e:
        print(f"✗ Vector store failed: {e}")
        return False

def test_database_functionality():
    """Test database"""
    print("\n=== Testing Database ===")
    try:
        create_tables()
        print("✓ Database tables created/verified")
        return True
    except Exception as e:
        print(f"✗ Database failed: {e}")
        return False

async def test_rag_pipeline_integration():
    """Test RAG pipeline integration"""
    print("\n=== Testing RAG Pipeline Integration ===")
    try:
        from services.classic_rag import classic_rag_pipeline
        from models.schemas import QueryRequest
        from models.database import User
        from unittest.mock import Mock
        
        # Create mock user
        mock_user = Mock(spec=User)
        mock_user.id = 1
        
        # Create query request
        query_request = QueryRequest(query="What is artificial intelligence?", top_k=3)
        
        # Test classic RAG pipeline
        print("Testing Classic RAG pipeline...")
        # This will likely fail due to no documents, but we can see if the pipeline structure works
        try:
            response = await classic_rag_pipeline.process_query(query_request, mock_user)
            print(f"✓ Classic RAG pipeline executed: {response.answer[:100]}...")
        except Exception as e:
            if "couldn't find any relevant information" in str(e) or "No relevant chunks found" in str(e):
                print("✓ Classic RAG pipeline structure works (no documents uploaded yet)")
            else:
                print(f"✗ Classic RAG pipeline failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ RAG pipeline integration failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Backend Functionality Tests")
    print("=" * 50)
    
    results = []
    
    # Test database
    results.append(test_database_functionality())
    
    # Test embedding service
    results.append(await test_embedding_service_functionality())
    
    # Test LLM service  
    results.append(await test_llm_service_functionality())
    
    # Test vector store
    results.append(await test_vector_store_functionality())
    
    # Test RAG pipeline integration
    results.append(await test_rag_pipeline_integration())
    
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Backend is functional.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)