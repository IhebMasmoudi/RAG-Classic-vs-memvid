"""
Unit tests for document service
"""
import pytest
import tempfile
import os
from io import BytesIO
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException, UploadFile

from services.document_service import DocumentProcessor, process_document_upload
from models.database import Document, DocumentChunk, User


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class"""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance for testing"""
        return DocumentProcessor()
    
    @pytest.fixture
    def mock_upload_file(self):
        """Create a mock UploadFile for testing"""
        def create_mock_file(filename, content_type, content=b"test content"):
            mock_file = Mock(spec=UploadFile)
            mock_file.filename = filename
            mock_file.content_type = content_type
            mock_file.size = len(content)
            mock_file.read = AsyncMock(return_value=content)
            mock_file.seek = AsyncMock()
            return mock_file
        return create_mock_file
    
    def test_validate_file_success(self, processor, mock_upload_file):
        """Test successful file validation"""
        file = mock_upload_file("test.pdf", "application/pdf")
        # Should not raise any exception
        processor.validate_file(file)
    
    def test_validate_file_too_large(self, processor, mock_upload_file):
        """Test file size validation"""
        file = mock_upload_file("large.pdf", "application/pdf")
        file.size = processor.max_file_size + 1
        
        with pytest.raises(HTTPException) as exc_info:
            processor.validate_file(file)
        assert exc_info.value.status_code == 413
    
    def test_validate_file_unsupported_type(self, processor, mock_upload_file):
        """Test unsupported file type validation"""
        file = mock_upload_file("test.doc", "application/msword")
        
        with pytest.raises(HTTPException) as exc_info:
            processor.validate_file(file)
        assert exc_info.value.status_code == 415
    
    def test_validate_file_invalid_filename(self, processor, mock_upload_file):
        """Test invalid filename validation"""
        file = mock_upload_file("", "application/pdf")
        
        with pytest.raises(HTTPException) as exc_info:
            processor.validate_file(file)
        assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_extract_text_plain_text(self, processor, mock_upload_file):
        """Test text extraction from plain text file"""
        content = b"This is a test document with some content."
        file = mock_upload_file("test.txt", "text/plain", content)
        
        result = await processor.extract_text(file)
        assert result == "This is a test document with some content."
    
    @pytest.mark.asyncio
    async def test_extract_text_unsupported_type(self, processor, mock_upload_file):
        """Test text extraction from unsupported file type"""
        file = mock_upload_file("test.doc", "application/msword")
        
        with pytest.raises(HTTPException) as exc_info:
            await processor.extract_text(file)
        # The error is caught and re-raised as 500 in the extract_text method
        assert exc_info.value.status_code == 500
    
    def test_create_chunks_basic(self, processor):
        """Test basic text chunking"""
        text = "This is a test. " * 100  # Create text longer than chunk_size
        chunks = processor.create_chunks(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= processor.chunk_size + 100 for chunk in chunks)  # Allow some flexibility
        assert all(chunk.strip() for chunk in chunks)  # No empty chunks
    
    def test_create_chunks_short_text(self, processor):
        """Test chunking of short text"""
        text = "This is a short text."
        chunks = processor.create_chunks(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_create_chunks_empty_text(self, processor):
        """Test chunking of empty text"""
        text = ""
        chunks = processor.create_chunks(text)
        
        assert len(chunks) == 0
    
    def test_create_chunks_sentence_boundary(self, processor):
        """Test that chunking respects sentence boundaries"""
        # Create text with clear sentence boundaries
        sentences = ["This is sentence one. ", "This is sentence two. ", "This is sentence three. "]
        text = "".join(sentences * 50)  # Repeat to exceed chunk size
        
        chunks = processor.create_chunks(text)
        
        # Check that chunks end with sentence boundaries when possible
        for chunk in chunks[:-1]:  # Exclude last chunk
            if len(chunk) < processor.chunk_size:
                continue
            # Should end with sentence punctuation
            assert chunk.rstrip().endswith(('.', '!', '?'))
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, processor):
        """Test embedding generation"""
        chunks = ["This is chunk one.", "This is chunk two.", "This is chunk three."]
        
        with patch.object(processor, 'initialize_embedding_model') as mock_init:
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]] * len(chunks)
            processor.embedding_model = mock_model
            
            embeddings = await processor.generate_embeddings(chunks)
            
            assert len(embeddings) == len(chunks)
            assert all(isinstance(emb, list) for emb in embeddings)
            mock_model.encode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_no_model(self, processor):
        """Test embedding generation when model is not initialized"""
        chunks = ["Test chunk"]
        
        with patch.object(processor, 'initialize_embedding_model') as mock_init:
            mock_init.side_effect = HTTPException(status_code=500, detail="Failed to initialize embedding model")
            
            with pytest.raises(HTTPException) as exc_info:
                await processor.generate_embeddings(chunks)
            assert exc_info.value.status_code == 500
    
    def test_generate_content_hash(self, processor):
        """Test content hash generation"""
        content = "This is test content"
        hash1 = processor._generate_content_hash(content)
        hash2 = processor._generate_content_hash(content)
        hash3 = processor._generate_content_hash("Different content")
        
        assert hash1 == hash2  # Same content should produce same hash
        assert hash1 != hash3  # Different content should produce different hash
        assert len(hash1) == 64  # SHA-256 produces 64-character hex string
    
    @pytest.mark.asyncio
    async def test_save_document_to_db(self, processor):
        """Test saving document to database"""
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        
        # Mock database session
        mock_db = Mock()
        mock_document = Mock()
        mock_document.id = "test-doc-id"
        
        with patch('services.document_service.Document') as mock_doc_class:
            mock_doc_class.return_value = mock_document
            
            result = await processor.save_document_to_db(
                user_id=1,
                filename="test.txt",
                content_type="text/plain",
                file_size=1000,
                chunks=chunks,
                db=mock_db
            )
            
            assert result == mock_document
            mock_db.add.assert_called()
            mock_db.flush.assert_called()
            mock_db.add_all.assert_called()
            mock_db.commit.assert_called()


class TestDocumentUploadIntegration:
    """Integration tests for document upload process"""
    
    @pytest.mark.asyncio
    async def test_process_document_upload_success(self):
        """Test successful document upload process"""
        # Create a mock file
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.txt"
        mock_file.content_type = "text/plain"
        mock_file.size = 100
        mock_file.read = AsyncMock(return_value=b"This is test content for processing.")
        mock_file.seek = AsyncMock()
        
        user_id = 1
        
        with patch('services.document_service.document_processor') as mock_processor:
            mock_processor.validate_file = Mock()
            mock_processor.extract_text = AsyncMock(return_value="This is test content for processing.")
            mock_processor.create_chunks = Mock(return_value=["This is test content for processing."])
            mock_processor.save_document_to_db = AsyncMock()
            mock_processor.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
            mock_processor.update_document_status = AsyncMock()
            
            # Mock document
            mock_document = Mock()
            mock_document.id = "test-doc-id"
            mock_processor.save_document_to_db.return_value = mock_document
            
            with patch('services.document_service.get_db_context'):
                result = await process_document_upload(mock_file, user_id)
                
                assert result.document_id == "test-doc-id"
                assert result.filename == "test.txt"
                assert result.chunks_created == 1
                assert result.embeddings_generated == 1
    
    @pytest.mark.asyncio
    async def test_process_document_upload_validation_error(self):
        """Test document upload with validation error"""
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.doc"
        mock_file.content_type = "application/msword"
        
        with patch('services.document_service.document_processor') as mock_processor:
            mock_processor.validate_file.side_effect = HTTPException(status_code=415, detail="Unsupported file type")
            
            with pytest.raises(HTTPException) as exc_info:
                await process_document_upload(mock_file, 1)
            assert exc_info.value.status_code == 415
    
    @pytest.mark.asyncio
    async def test_process_document_upload_no_content(self):
        """Test document upload with no extractable content"""
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "empty.txt"
        mock_file.content_type = "text/plain"
        mock_file.size = 0
        mock_file.read = AsyncMock(return_value=b"")
        mock_file.seek = AsyncMock()
        
        with patch('services.document_service.document_processor') as mock_processor:
            mock_processor.validate_file = Mock()
            mock_processor.extract_text = AsyncMock(return_value="")
            mock_processor.create_chunks = Mock(return_value=[])
            
            with pytest.raises(HTTPException) as exc_info:
                await process_document_upload(mock_file, 1)
            assert exc_info.value.status_code == 400
            assert "No content could be extracted" in str(exc_info.value.detail)


class TestPDFExtraction:
    """Test PDF text extraction functionality"""
    
    @pytest.mark.asyncio
    async def test_extract_pdf_text_success(self):
        """Test successful PDF text extraction"""
        processor = DocumentProcessor()
        
        # Create a simple PDF content (this is a mock - in real tests you'd use a real PDF)
        with patch('services.document_service.pypdf.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "This is page content"
            mock_reader.return_value.pages = [mock_page]
            
            result = await processor._extract_pdf_text(b"fake pdf content")
            assert result == "This is page content"
    
    @pytest.mark.asyncio
    async def test_extract_pdf_text_no_content(self):
        """Test PDF extraction with no text content"""
        processor = DocumentProcessor()
        
        with patch('services.document_service.pypdf.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = ""
            mock_reader.return_value.pages = [mock_page]
            
            with pytest.raises(HTTPException) as exc_info:
                await processor._extract_pdf_text(b"fake pdf content")
            assert exc_info.value.status_code == 500  # Error is caught and re-raised as 500
    
    @pytest.mark.asyncio
    async def test_extract_pdf_text_error(self):
        """Test PDF extraction with error"""
        processor = DocumentProcessor()
        
        with patch('services.document_service.pypdf.PdfReader') as mock_reader:
            mock_reader.side_effect = Exception("PDF parsing error")
            
            with pytest.raises(HTTPException) as exc_info:
                await processor._extract_pdf_text(b"fake pdf content")
            assert exc_info.value.status_code == 500


class TestTextExtraction:
    """Test text file extraction functionality"""
    
    @pytest.mark.asyncio
    async def test_extract_text_content_utf8(self):
        """Test text extraction with UTF-8 encoding"""
        processor = DocumentProcessor()
        content = "This is UTF-8 content with special chars: áéíóú".encode('utf-8')
        
        result = await processor._extract_text_content(content)
        assert "This is UTF-8 content with special chars: áéíóú" in result
    
    @pytest.mark.asyncio
    async def test_extract_text_content_latin1(self):
        """Test text extraction with Latin-1 encoding"""
        processor = DocumentProcessor()
        content = "This is Latin-1 content".encode('latin-1')
        
        result = await processor._extract_text_content(content)
        assert "This is Latin-1 content" in result
    
    @pytest.mark.asyncio
    async def test_extract_text_content_undecodable(self):
        """Test text extraction with empty content"""
        processor = DocumentProcessor()
        content = b''  # Empty content
        
        with pytest.raises(HTTPException) as exc_info:
            await processor._extract_text_content(content)
        assert exc_info.value.status_code == 500  # Error is caught and re-raised as 500