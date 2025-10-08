# RAG Comparison Platform - Backend

FastAPI-based backend server providing REST API endpoints for comparing different RAG (Retrieval-Augmented Generation) approaches.

## 🚀 Features

- **Three RAG Implementations**: Classic RAG, MemVid RAG, and LightRAG
- **Document Processing**: PDF/TXT upload and chunking
- **Vector Search**: FAISS-based similarity search
- **LLM Integration**: Google Gemini and OpenAI support
- **User Authentication**: JWT-based security
- **Performance Monitoring**: Response time tracking and analytics
- **Graph-based Knowledge**: Entity-relationship extraction with LightRAG

## 🛠️ Technology Stack

- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - Database ORM with PostgreSQL/SQLite support
- **FAISS** - Vector similarity search engine
- **LangChain** - LLM integration and document processing
- **LightRAG** - Graph-based RAG with knowledge graphs
- **Pydantic** - Data validation and serialization
- **JWT** - JSON Web Token authentication
- **Google Gemini** - Primary LLM provider
- **OpenAI** - Fallback LLM provider

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- Google Gemini API key (recommended)
- OpenAI API key (optional)

## 🚀 Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
cp .env.template .env
```

Edit the `.env` file with your configuration:
```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Database
DATABASE_URL=sqlite:///./rag_platform.db

# JWT Configuration
SECRET_KEY=your_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# LLM Configuration
GEMINI_MODEL=gemini-pro
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=all-MiniLM-L6-v2

# CORS
ALLOWED_ORIGINS=["http://localhost:4200"]
```

### 4. Initialize Database
```bash
python -c "from utils.database import create_tables; create_tables()"
```

### 5. Start the Server
```bash
python main.py
```

The server will start at http://localhost:8000

## 📚 API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoints

#### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user info
- `POST /auth/logout` - User logout

#### Document Management
- `POST /upload/` - Upload document (PDF/TXT)
- `GET /upload/documents` - List user documents
- `DELETE /upload/documents/{document_id}` - Delete document

#### Classic RAG
- `POST /classic_rag/query` - Execute Classic RAG query
- `GET /classic_rag/health` - Health check

#### MemVid RAG
- `POST /memvid_rag/query` - Execute MemVid RAG query
- `GET /memvid_rag/health` - Health check

#### LightRAG
- `POST /lightrag/query` - Execute LightRAG query
- `GET /lightrag/stats` - Get knowledge graph statistics
- `GET /lightrag/modes` - Get available query modes
- `DELETE /lightrag/documents/{document_id}` - Delete from graph

#### Comparison
- `POST /comparison/compare` - Compare all RAG methods
- `GET /comparison/capabilities` - Get method capabilities
- `GET /comparison/recommendations` - Get usage recommendations

#### Performance
- `GET /performance/stats` - Get performance statistics
- `GET /performance/history` - Get query history

## 🏗️ Project Structure

```
BackEnd/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env.template          # Environment variables template
├── routes/                # API route handlers
│   ├── auth.py           # Authentication endpoints
│   ├── upload.py         # Document upload endpoints
│   ├── classic_rag.py    # Classic RAG endpoints
│   ├── memvid_rag.py     # MemVid RAG endpoints
│   ├── lightrag.py       # LightRAG endpoints
│   ├── comparison.py     # Comparison endpoints
│   └── performance.py    # Performance endpoints
├── services/              # Business logic services
│   ├── user_service.py   # User management
│   ├── document_service.py # Document processing
│   ├── classic_rag.py    # Classic RAG implementation
│   ├── memvid_rag.py     # MemVid RAG implementation
│   ├── lightrag_service.py # LightRAG implementation
│   ├── rag_comparison_service.py # RAG comparison
│   ├── llm_service.py    # LLM integration
│   ├── embedding_service.py # Embedding generation
│   ├── vector_store.py   # Vector database operations
│   └── chunking_service.py # Document chunking
├── models/                # Data models
│   ├── database.py       # SQLAlchemy models
│   └── schemas.py        # Pydantic schemas
├── middleware/            # Custom middleware
│   ├── auth_middleware.py # JWT authentication
│   ├── error_handlers.py # Global error handling
│   └── request_logging.py # Request logging
├── utils/                 # Utility functions
│   ├── auth.py           # Authentication utilities
│   ├── database.py       # Database utilities
│   └── logging_config.py # Logging configuration
└── tests/                 # Test files
    ├── test_auth_routes.py
    ├── test_classic_rag.py
    ├── test_memvid_rag.py
    └── test_integration_simple.py
```

## 🔧 Configuration

### Database Configuration
The application supports both SQLite (development) and PostgreSQL (production):

```python
# SQLite (default)
DATABASE_URL=sqlite:///./rag_platform.db

# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost/rag_platform
```

### LLM Configuration
Configure your preferred LLM provider:

```python
# Google Gemini (recommended)
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-pro

# OpenAI (fallback)
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-3.5-turbo
```

### Vector Store Configuration
FAISS vector store settings:

```python
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Sentence transformer model
VECTOR_STORE_PATH=data/vector_store/
```

## 🧪 Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Authentication tests
pytest tests/test_auth_routes.py

# RAG pipeline tests
pytest tests/test_classic_rag.py tests/test_memvid_rag.py

# Integration tests
python test_integration_simple.py
```

### Test Coverage
```bash
pytest --cov=. --cov-report=html
```

## 📊 Performance Monitoring

The backend includes comprehensive performance monitoring:

- **Response Time Tracking**: All endpoints measure execution time
- **Query History**: User query patterns and performance
- **Error Logging**: Comprehensive error tracking and debugging
- **Resource Usage**: Memory and CPU monitoring for RAG operations

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt password hashing
- **CORS Protection**: Configurable cross-origin resource sharing
- **Input Validation**: Pydantic-based request validation
- **SQL Injection Prevention**: SQLAlchemy ORM protection
- **Rate Limiting**: Built-in request rate limiting

## 🚀 Deployment

### Development
```bash
python main.py
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

## 🐛 Troubleshooting

### Common Issues

1. **LightRAG Not Available**
   - Ensure `lightrag-hku` is properly installed
   - Check system resources (memory/CPU)
   - Verify LLM API keys are configured

2. **Database Connection Issues**
   - Check DATABASE_URL in .env file
   - Ensure database permissions
   - Run database initialization

3. **API Key Issues**
   - Verify Gemini/OpenAI API keys
   - Check API quotas and limits
   - Ensure proper environment variable loading

4. **Vector Store Issues**
   - Check FAISS installation
   - Verify data directory permissions
   - Clear vector store cache if corrupted

### Debug Mode
Enable debug logging:
```python
DEBUG=True
LOG_LEVEL=DEBUG
```

## 📈 Performance Optimization

- **Async Operations**: All I/O operations are asynchronous
- **Connection Pooling**: Database connection pooling
- **Caching**: LLM response caching for repeated queries
- **Batch Processing**: Efficient embedding generation
- **Memory Management**: Optimized vector store operations

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write comprehensive tests
4. Update documentation
5. Use meaningful commit messages

## 📝 License

This project is licensed under the MIT License.

---

For more information, see the main project README or visit the API documentation at http://localhost:8000/docs