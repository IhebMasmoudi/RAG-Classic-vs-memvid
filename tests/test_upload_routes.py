"""
Unit tests for upload routes
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

from main import app
from models.database import User, Document
from models.schemas import UploadResponse


class TestUploadRoutes:
    """Test cases for upload routes"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user for testing"""
        user = Mock(spec=User)
        user.id = 1
        user.email = "test@example.com"
        user.is_active = True
        return user
    
    @pytest.fixture
    def mock_upload_response(self):
        """Create mock upload response"""
        return UploadResponse(
            document_id="test-doc-id",
            filename="test.txt",
            chunks_created=5,
            embeddings_generated=5,
            message="Document processed successfully"
        )
    
    def test_upload_document_success(self, client, mock_user, mock_upload_response):
        """Test successful document upload"""
        # Create test file
        file_content = b"This is test content for document upload."
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
        
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.process_document_upload', return_value=mock_upload_response):
                with patch('routes.upload.get_db_session'):
                    with patch('routes.upload.add_document_to_vector_store', return_value=["vec-1"]):
                        # Mock document and chunks
                        mock_document = Mock(spec=Document)
                        mock_document.id = "test-doc-id"
                        mock_document.chunks = [Mock()]
                        
                        with patch('routes.upload.Document') as mock_doc_class:
                            mock_query = Mock()
                            mock_query.filter.return_value.first.return_value = mock_document
                            
                            with patch('routes.upload.get_db_session') as mock_get_db:
                                mock_db = Mock()
                                mock_db.query.return_value = mock_query
                                mock_get_db.return_value.__enter__.return_value = mock_db
                                
                                response = client.post(
                                    "/upload/",
                                    files=files,
                                    headers={"Authorization": "Bearer test-token"}
                                )
        
        assert response.status_code == 201
        data = response.json()
        assert data["document_id"] == "test-doc-id"
        assert data["filename"] == "test.txt"
        assert data["chunks_created"] == 5
    
    def test_upload_document_no_auth(self, client):
        """Test document upload without authentication"""
        file_content = b"This is test content."
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
        
        response = client.post("/upload/", files=files)
        assert response.status_code == 403  # Forbidden
    
    def test_upload_document_invalid_file(self, client, mock_user):
        """Test document upload with invalid file"""
        file_content = b"This is test content."
        files = {"file": ("test.doc", io.BytesIO(file_content), "application/msword")}
        
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.process_document_upload') as mock_process:
                from fastapi import HTTPException
                mock_process.side_effect = HTTPException(status_code=415, detail="Unsupported file type")
                
                response = client.post(
                    "/upload/",
                    files=files,
                    headers={"Authorization": "Bearer test-token"}
                )
        
        assert response.status_code == 415
    
    def test_upload_document_processing_error(self, client, mock_user):
        """Test document upload with processing error"""
        file_content = b"This is test content."
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
        
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.process_document_upload') as mock_process:
                mock_process.side_effect = Exception("Processing failed")
                
                response = client.post(
                    "/upload/",
                    files=files,
                    headers={"Authorization": "Bearer test-token"}
                )
        
        assert response.status_code == 500
    
    def test_get_documents_success(self, client, mock_user):
        """Test successful retrieval of user documents"""
        mock_documents = [
            Mock(
                id="doc-1",
                filename="test1.txt",
                content_type="text/plain",
                upload_date="2023-01-01T00:00:00",
                chunk_count=5,
                embedding_model="all-MiniLM-L6-v2"
            ),
            Mock(
                id="doc-2",
                filename="test2.pdf",
                content_type="application/pdf",
                upload_date="2023-01-02T00:00:00",
                chunk_count=10,
                embedding_model="all-MiniLM-L6-v2"
            )
        ]
        
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.get_user_documents', return_value=mock_documents):
                response = client.get(
                    "/upload/documents",
                    headers={"Authorization": "Bearer test-token"}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "doc-1"
        assert data[1]["id"] == "doc-2"
    
    def test_get_documents_no_auth(self, client):
        """Test getting documents without authentication"""
        response = client.get("/upload/documents")
        assert response.status_code == 403
    
    def test_get_documents_error(self, client, mock_user):
        """Test getting documents with error"""
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.get_user_documents') as mock_get_docs:
                mock_get_docs.side_effect = Exception("Database error")
                
                response = client.get(
                    "/upload/documents",
                    headers={"Authorization": "Bearer test-token"}
                )
        
        assert response.status_code == 500
    
    def test_delete_document_success(self, client, mock_user):
        """Test successful document deletion"""
        document_id = "test-doc-id"
        
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.remove_document_from_vector_store') as mock_remove_vector:
                with patch('routes.upload.delete_document', return_value=True) as mock_delete:
                    response = client.delete(
                        f"/upload/documents/{document_id}",
                        headers={"Authorization": "Bearer test-token"}
                    )
        
        assert response.status_code == 204
    
    def test_delete_document_not_found(self, client, mock_user):
        """Test deleting non-existent document"""
        document_id = "non-existent-doc"
        
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.remove_document_from_vector_store'):
                with patch('routes.upload.delete_document', return_value=False):
                    response = client.delete(
                        f"/upload/documents/{document_id}",
                        headers={"Authorization": "Bearer test-token"}
                    )
        
        assert response.status_code == 404
    
    def test_delete_document_no_auth(self, client):
        """Test deleting document without authentication"""
        response = client.delete("/upload/documents/test-doc-id")
        assert response.status_code == 403
    
    def test_delete_document_error(self, client, mock_user):
        """Test document deletion with error"""
        document_id = "test-doc-id"
        
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.remove_document_from_vector_store'):
                with patch('routes.upload.delete_document') as mock_delete:
                    mock_delete.side_effect = Exception("Deletion failed")
                    
                    response = client.delete(
                        f"/upload/documents/{document_id}",
                        headers={"Authorization": "Bearer test-token"}
                    )
        
        assert response.status_code == 500
    
    def test_get_upload_stats_success(self, client, mock_user):
        """Test successful retrieval of upload statistics"""
        mock_documents = [
            Mock(chunk_count=5, processing_status="completed"),
            Mock(chunk_count=10, processing_status="completed"),
            Mock(chunk_count=3, processing_status="pending")
        ]
        
        mock_vector_stats = {
            'total_vectors': 100,
            'dimension': 384,
            'total_documents': 10
        }
        
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.get_user_documents', return_value=mock_documents):
                with patch('routes.upload.get_vector_store_stats', return_value=mock_vector_stats):
                    response = client.get(
                        "/upload/stats",
                        headers={"Authorization": "Bearer test-token"}
                    )
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_stats"]["user_documents"] == 3
        assert data["user_stats"]["total_chunks"] == 18
        assert data["vector_store_stats"]["total_vectors"] == 100
    
    def test_test_upload_success(self, client, mock_user):
        """Test successful file upload validation"""
        file_content = b"This is test content."
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
        
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.document_processor') as mock_processor:
                mock_processor.validate_file = Mock()  # No exception means validation passed
                
                response = client.post(
                    "/upload/test",
                    files=files,
                    headers={"Authorization": "Bearer test-token"}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["validation"] == "passed"
    
    def test_test_upload_validation_failed(self, client, mock_user):
        """Test file upload validation failure"""
        file_content = b"This is test content."
        files = {"file": ("test.doc", io.BytesIO(file_content), "application/msword")}
        
        with patch('routes.upload.get_current_user', return_value=mock_user):
            with patch('routes.upload.document_processor') as mock_processor:
                from fastapi import HTTPException
                mock_processor.validate_file.side_effect = HTTPException(
                    status_code=415, 
                    detail="Unsupported file type"
                )
                
                response = client.post(
                    "/upload/test",
                    files=files,
                    headers={"Authorization": "Bearer test-token"}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["validation"] == "failed"
        assert "Unsupported file type" in data["error"]


class TestUploadIntegration:
    """Integration tests for upload functionality"""
    
    @pytest.mark.asyncio
    async def test_full_upload_process(self):
        """Test the complete upload process integration"""
        # This would be a more comprehensive test that actually processes a file
        # through the entire pipeline in a test environment
        
        # Create a real UploadFile object
        file_content = b"This is a comprehensive test document with multiple sentences. It should be processed into chunks and embeddings should be generated."
        
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "integration_test.txt"
        mock_file.content_type = "text/plain"
        mock_file.size = len(file_content)
        mock_file.read = AsyncMock(return_value=file_content)
        mock_file.seek = AsyncMock()
        
        # Mock all the dependencies
        with patch('services.document_service.get_db_session'):
            with patch('services.document_service.document_processor') as mock_processor:
                # Configure the processor
                mock_processor.validate_file = Mock()
                mock_processor.extract_text = AsyncMock(return_value=file_content.decode())
                mock_processor.create_chunks = Mock(return_value=["Chunk 1", "Chunk 2"])
                mock_processor.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
                mock_processor.save_document_to_db = AsyncMock()
                mock_processor.update_document_status = AsyncMock()
                
                # Mock document
                mock_document = Mock()
                mock_document.id = "integration-test-doc"
                mock_processor.save_document_to_db.return_value = mock_document
                
                # Import and test the function
                from services.document_service import process_document_upload
                
                result = await process_document_upload(mock_file, user_id=1)
                
                assert result.document_id == "integration-test-doc"
                assert result.filename == "integration_test.txt"
                assert result.chunks_created == 2
                assert result.embeddings_generated == 2
                
                # Verify all steps were called
                mock_processor.validate_file.assert_called_once()
                mock_processor.extract_text.assert_called_once()
                mock_processor.create_chunks.assert_called_once()
                mock_processor.generate_embeddings.assert_called_once()
                mock_processor.save_document_to_db.assert_called_once()
                mock_processor.update_document_status.assert_called_once()