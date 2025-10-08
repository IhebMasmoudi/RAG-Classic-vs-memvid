#!/usr/bin/env python3
"""
Test script for recursive chunking using LlamaIndex
"""
import sys
import os
import logging
from pathlib import Path

# Add the BackEnd directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from services.chunking_service import (
    create_advanced_chunks, 
    get_chunking_statistics, 
    is_llama_index_available,
    AdvancedChunkingService,
    ChunkingConfig
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llama_index_availability():
    """Test if LlamaIndex is available"""
    print("=== Testing LlamaIndex Availability ===")
    available = is_llama_index_available()
    print(f"LlamaIndex available: {available}")
    return available

def test_basic_chunking():
    """Test basic chunking functionality"""
    print("\n=== Testing Basic Chunking ===")
    
    # Sample text for testing
    test_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. 
    These machines are designed to think and act like humans, performing tasks that typically require human intelligence.
    
    Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.
    It uses algorithms and statistical models to analyze and draw inferences from patterns in data.
    
    Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to progressively extract higher-level features from raw input.
    For example, in image processing, lower layers may identify edges, while higher layers may identify concepts relevant to a human such as digits or letters or faces.
    
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language.
    It involves programming computers to process and analyze large amounts of natural language data.
    
    Computer vision is another important area of AI that deals with how computers can gain understanding from digital images or videos.
    It seeks to automate tasks that the human visual system can do, such as recognizing objects, faces, or activities in images.
    """
    
    # Test with different chunk sizes
    chunk_sizes = [500, 1000, 1500]
    
    for chunk_size in chunk_sizes:
        print(f"\n--- Testing with chunk size: {chunk_size} ---")
        
        chunks = create_advanced_chunks(
            text=test_text,
            document_metadata={'test_doc': True, 'source': 'test'},
            chunk_size=chunk_size,
            chunk_overlap=200
        )
        
        print(f"Number of chunks created: {len(chunks)}")
        
        # Display first few chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i + 1}:")
            print(f"  Content length: {len(chunk['content'])}")
            print(f"  Content preview: {chunk['content'][:100]}...")
            print(f"  Metadata: {chunk['metadata']}")
        
        # Get statistics
        stats = get_chunking_statistics(chunks)
        print(f"\nChunking Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

def test_different_configurations():
    """Test different chunking configurations"""
    print("\n=== Testing Different Configurations ===")
    
    test_text = """
    The field of artificial intelligence has evolved significantly over the past decades. 
    From rule-based systems to machine learning algorithms, AI has transformed how we approach complex problems.
    
    Today's AI systems can process vast amounts of data, recognize patterns, and make predictions with remarkable accuracy.
    Applications range from recommendation systems and autonomous vehicles to medical diagnosis and financial trading.
    
    However, challenges remain in areas such as explainability, bias, and ethical considerations.
    As AI becomes more prevalent in society, addressing these challenges becomes increasingly important.
    """
    
    # Test different separator configurations
    configs = [
        ChunkingConfig(chunk_size=300, chunk_overlap=50, separator="\n\n"),
        ChunkingConfig(chunk_size=300, chunk_overlap=50, separator="\n", secondary_separators=[". ", "! ", "? "]),
        ChunkingConfig(chunk_size=500, chunk_overlap=100, separator="\n\n", secondary_separators=["\n", ". ", " "])
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i + 1} ---")
        print(f"Chunk size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
        print(f"Primary separator: '{config.separator}'")
        print(f"Secondary separators: {config.secondary_separators}")
        
        chunking_service = AdvancedChunkingService(config)
        chunks = chunking_service.create_chunks(test_text, {'config_test': i + 1})
        
        print(f"Chunks created: {len(chunks)}")
        
        # Show chunk boundaries
        for j, chunk in enumerate(chunks):
            print(f"  Chunk {j + 1}: {len(chunk['content'])} chars - '{chunk['content'][:50]}...'")

def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    # Empty text
    print("Testing empty text:")
    chunks = create_advanced_chunks("", {})
    print(f"Empty text chunks: {len(chunks)}")
    
    # Very short text
    print("\nTesting very short text:")
    chunks = create_advanced_chunks("Short text.", {})
    print(f"Short text chunks: {len(chunks)}")
    if chunks:
        print(f"Content: '{chunks[0]['content']}'")
    
    # Text with no separators
    print("\nTesting text with no separators:")
    no_sep_text = "Thisisaverylongtextwithnoseparatorsatallitjustgoesononandonwithoutanybreaks"
    chunks = create_advanced_chunks(no_sep_text, {}, chunk_size=20, chunk_overlap=5)
    print(f"No separator chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}: '{chunk['content']}'")

def test_metadata_preservation():
    """Test metadata preservation"""
    print("\n=== Testing Metadata Preservation ===")
    
    test_text = "This is a test document for metadata preservation. It should maintain the metadata across all chunks."
    
    metadata = {
        'document_id': 'test-123',
        'filename': 'test.txt',
        'author': 'Test Author',
        'created_date': '2024-01-01'
    }
    
    chunks = create_advanced_chunks(test_text, metadata, chunk_size=50, chunk_overlap=10)
    
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Content: '{chunk['content']}'")
        print(f"  Original metadata preserved: {all(key in chunk['metadata'] for key in metadata.keys())}")
        print(f"  Chunk-specific metadata: {[key for key in chunk['metadata'].keys() if key not in metadata.keys()]}")

def main():
    """Main test function"""
    print("LlamaIndex Recursive Chunking Test")
    print("=" * 40)
    
    # Test availability
    if not test_llama_index_availability():
        print("LlamaIndex not available, testing fallback chunking...")
    
    # Run tests
    test_basic_chunking()
    test_different_configurations()
    test_edge_cases()
    test_metadata_preservation()
    
    print("\n" + "=" * 40)
    print("Recursive Chunking Test Complete!")

if __name__ == "__main__":
    main()