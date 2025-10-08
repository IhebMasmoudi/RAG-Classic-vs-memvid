# PostgreSQL Setup Guide for RAG Comparison Platform

## Overview
The RAG Comparison Platform has been successfully configured to use PostgreSQL as the primary database instead of SQLite. This provides better performance, concurrent access, and production-ready features.

## Database Configuration

### Connection Details
- **Host**: localhost
- **Port**: 5432
- **Database**: rag_platform
- **Username**: postgres
- **Password**: admin

### Environment Variables
The following environment variables are configured in `.env`:

```bash
# Database Configuration - PostgreSQL
DATABASE_URL=postgresql://postgres:admin@localhost:5432/rag_platform
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=admin
POSTGRES_DB=rag_platform
```

## Setup Instructions

### 1. Install PostgreSQL Dependencies
```bash
pip install psycopg2-binary==2.9.9
```

### 2. Initialize Database
Run the initialization script to create the database and tables:
```bash
python scripts/init_postgres.py
```

This script will:
- Create the `rag_platform` database if it doesn't exist
- Test the connection to PostgreSQL
- Create all required tables using SQLAlchemy

### 3. Verify Setup
The application will automatically connect to PostgreSQL when started:
```bash
python main.py
```

## Database Schema

### Tables Created
1. **users** - User authentication and management
2. **documents** - Uploaded document metadata
3. **document_chunks** - Text chunks with embedding references
4. **query_history** - RAG query history and results
5. **system_logs** - Application logging and monitoring

### Key Features
- **UUID Primary Keys**: Documents and chunks use UUID for better distribution
- **Foreign Key Constraints**: Proper relationships between tables
- **Indexes**: Optimized for common query patterns
- **Connection Pooling**: Configured for concurrent access

## Configuration Changes Made

### 1. Database Engine (`utils/database.py`)
```python
# PostgreSQL configuration with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=300
)
```

### 2. Settings (`config.py`)
Added PostgreSQL-specific configuration:
```python
DATABASE_URL: str = "postgresql://postgres:admin@localhost:5432/rag_platform"
POSTGRES_HOST: str = "localhost"
POSTGRES_PORT: int = 5432
POSTGRES_USER: str = "postgres"
POSTGRES_PASSWORD: str = "admin"
POSTGRES_DB: str = "rag_platform"
```

### 3. Dependencies (`requirements.txt`)
Added PostgreSQL driver:
```
psycopg2-binary==2.9.9
```

## Performance Benefits

### Over SQLite
- **Concurrent Access**: Multiple users can access simultaneously
- **Better Performance**: Optimized for larger datasets
- **ACID Compliance**: Full transaction support
- **Scalability**: Can handle production workloads
- **Advanced Features**: Full-text search, JSON support, etc.

### Connection Pooling
- **pool_size=10**: Base connection pool size
- **max_overflow=20**: Additional connections when needed
- **pool_pre_ping=True**: Validates connections before use
- **pool_recycle=300**: Recycles connections every 5 minutes

## Production Considerations

### Security
- Change default password in production
- Use environment variables for sensitive data
- Enable SSL connections
- Configure proper user permissions

### Monitoring
- Enable PostgreSQL logging
- Monitor connection pool usage
- Set up database backups
- Configure performance monitoring

### Scaling
- Consider read replicas for heavy read workloads
- Implement database sharding if needed
- Use connection pooling at application level
- Monitor and optimize slow queries

## Troubleshooting

### Common Issues

#### Connection Refused
```bash
psycopg2.OperationalError: could not connect to server
```
**Solution**: Ensure PostgreSQL is running and accepting connections on port 5432.

#### Authentication Failed
```bash
psycopg2.OperationalError: FATAL: password authentication failed
```
**Solution**: Verify username/password in `.env` file matches PostgreSQL configuration.

#### Database Does Not Exist
```bash
psycopg2.OperationalError: FATAL: database "rag_platform" does not exist
```
**Solution**: Run the initialization script: `python scripts/init_postgres.py`

### Verification Commands

#### Test Connection
```python
python -c "from utils.database import engine; print(f'Connected to: {engine.url}')"
```

#### Check Tables
```python
python -c "from utils.database import get_db_context; from sqlalchemy import text; 
with get_db_context() as db: 
    result = db.execute(text('SELECT table_name FROM information_schema.tables WHERE table_schema = \\'public\\''));
    print([r[0] for r in result.fetchall()])"
```

## Migration from SQLite

If migrating from an existing SQLite database:

1. **Export Data**: Use SQLite dump or Python scripts to export data
2. **Initialize PostgreSQL**: Run the setup scripts
3. **Import Data**: Use PostgreSQL COPY or INSERT statements
4. **Update Configuration**: Change DATABASE_URL in `.env`
5. **Test Application**: Verify all functionality works

## Backup and Recovery

### Backup Database
```bash
pg_dump -h localhost -U postgres -d rag_platform > backup.sql
```

### Restore Database
```bash
psql -h localhost -U postgres -d rag_platform < backup.sql
```

## Development vs Production

### Development (Current)
- Local PostgreSQL instance
- Simple authentication
- Basic connection pooling

### Production Recommendations
- Managed PostgreSQL service (AWS RDS, Google Cloud SQL, etc.)
- SSL/TLS encryption
- Connection pooling with PgBouncer
- Read replicas for scaling
- Automated backups
- Monitoring and alerting

## Next Steps

The PostgreSQL database is now fully configured and ready for:
1. **Document Upload Processing** - All existing functionality works
2. **RAG Pipeline Implementation** - Optimized for query performance  
3. **User Management** - Concurrent user support
4. **Production Deployment** - Scalable database foundation

All existing API endpoints and functionality remain unchanged - only the underlying database has been upgraded from SQLite to PostgreSQL.