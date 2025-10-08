#!/usr/bin/env python3
"""
Simple test for LlamaIndex chunking
"""
import sys
from pathlib import Path

# Add the BackEnd directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from services.chunking_service import create_advanced_chunks

def test_llamaindex_chunking():
    """Test LlamaIndex chunking"""
    text = """
    This is a test document for LlamaIndex chunking. It should work properly now with sentence splitting.
    
    This is another paragraph to test the chunking behavior. The sentence splitter should handle this well.
    
    Here's a third paragraph with more content to see how the chunking works with different sizes and overlaps.
    """
    
    print("Testing LlamaIndex Chunking")
    print("=" * 30)
    
    # Test with different chunk sizes
    for chunk_size in [100, 200]:
        print(f"\nTesting with chunk size: {chunk_size}")
        chunks = create_advanced_chunks(
            text=text,
            document_metadata={'test': True},
            chunk_size=chunk_size,
            chunk_overlap=20
        )
        
        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            method = chunk['metadata']['chunk_method']
            content_preview = chunk['content'][:60].replace('\n', ' ').strip()
            print(f"  Chunk {i+1} ({method}): {content_preview}...")

if __name__ == "__main__":
    test_llamaindex_chunking()