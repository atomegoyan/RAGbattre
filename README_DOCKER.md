# RAGbattre - Docker Setup Documentation

## Overview

This document provides comprehensive instructions for dockerizing and running the RAGbattre application - a Parliamentary Debate Analyzer using Retrieval-Augmented Generation (RAG) technology.

## Project Structure

```
RAGbattre/
├── retrieval_app/          # Main Streamlit application
│   ├── app.py              # Main application entry point
│   ├── core.py             # Core RAG functionality
│   ├── llm_utils.py        # LLM interaction utilities
│   └── ...
├── data/                   # Data directory
│   ├── corpus/             # Document corpus
│   ├── embeddings_cs1/     # Pre-computed embeddings
│   └── ...
├── scripts/                # Utility scripts
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
└── README_DOCKER.md        # This file
```

## Prerequisites

1. **Docker**: Install Docker Desktop or Docker Engine
   - [Docker Desktop for Windows/Mac](https://www.docker.com/products/docker-desktop)
   - [Docker Engine for Linux](https://docs.docker.com/engine/install/)

2. **Docker Compose**: Usually included with Docker Desktop
   - For standalone installation: [Docker Compose](https://docs.docker.com/compose/install/)

3. **System Requirements**:
   - Minimum 4GB RAM (8GB+ recommended for ML models)
   - 10GB+ free disk space
   - Internet connection for downloading dependencies

## Quick Start

### 1. Build and Run with Docker Compose (Recommended)

```bash
# Clone or navigate to the project directory
cd /path/to/RAGbattre

# Build and start the application
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d --build
```

The application will be available at: http://localhost:8501

### 2. Alternative: Using Docker directly

```bash
# Build the Docker image
docker build -t ragbattre-app .

# Run the container
docker run -p 8501:8501 ragbattre-app
```

## Configuration Options

### Environment Variables

You can customize the application behavior using environment variables:

```bash
# Example with custom configuration
docker run -p 8501:8501 \
  -e PYTHONPATH=/app \
  -e STREAMLIT_SERVER_PORT=8501 \
  ragbattre-app
```

### Volume Mounting for Development

For development purposes, you can mount local directories:

```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/retrieval_app:/app/retrieval_app \
  ragbattre-app
```

Or modify the `docker-compose.yml` file to uncomment the volume mounts.

## Application Features

The dockerized application provides three main modes:

1. **Chat with Ollama**: Direct interaction with local LLM models
2. **Document Retrieval**: Search and retrieve relevant documents
3. **RAG Mode**: Retrieval-Augmented Generation for enhanced responses

### Supported LLM Backends

- **Ollama** (local): Requires Ollama installation or service
- **Mistral AI** (cloud): Requires API key
- **Cohere** (cloud): Requires API key

## Dependencies

The application uses the following key dependencies (see `requirements.txt`):

- **streamlit**: Web application framework
- **ollama**: Local LLM interaction
- **chromadb**: Vector database for embeddings
- **transformers & torch**: ML models for reranking
- **mistralai & cohere**: External LLM APIs
- **pandas**: Data manipulation
- **regex**: Advanced text processing

## Data Management

### Included Data

The Docker image includes:
- Pre-processed document corpus
- Pre-computed ChromaDB embeddings
- Example questions and configurations

### Data Persistence

To persist data modifications or add new documents:

1. Use volume mounts in production:
```yaml
volumes:
  - ./data:/app/data
```

2. Or rebuild the image after adding new data to the `data/` directory.

## Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Use a different port
   docker run -p 8502:8501 ragbattre-app
   ```

2. **Memory Issues**:
   - Increase Docker's memory allocation (Docker Desktop settings)
   - Consider using lighter ML models

3. **Permission Issues**:
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   ```

4. **Missing Ollama Service**:
   - Install Ollama locally or
   - Use the cloud LLM backends (Mistral/Cohere)

### Logs and Debugging

```bash
# View application logs
docker-compose logs -f ragbattre-app

# Access container shell for debugging
docker-compose exec ragbattre-app /bin/bash
```

## Production Deployment

### Security Considerations

1. **API Keys**: Use environment variables or Docker secrets
   ```bash
   docker run -e MISTRAL_API_KEY=your_key ragbattre-app
   ```

2. **Network Security**: Use proper network configurations
3. **User Permissions**: The container runs as non-root user by default

### Scaling Options

1. **Horizontal Scaling**: Use multiple container instances
2. **Load Balancing**: Add nginx or similar reverse proxy
3. **Resource Limits**: Configure memory and CPU limits

```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2'
```

## Development

### Local Development with Docker

```bash
# Development mode with auto-reload
docker run -p 8501:8501 \
  -v $(pwd):/app \
  ragbattre-app streamlit run retrieval_app/app.py --server.fileWatcherType poll
```

### Building Custom Images

```bash
# Build with custom tag
docker build -t ragbattre-app:v1.0 .

# Build for different architectures
docker buildx build --platform linux/amd64,linux/arm64 -t ragbattre-app .
```

## Ollama Integration (Optional)

To run Ollama alongside the application:

1. Uncomment the Ollama service in `docker-compose.yml`
2. Download models after starting:

```bash
# Start services
docker-compose up -d

# Download a model
docker-compose exec ollama ollama pull llama3.2:1b
```

## Maintenance

### Updating Dependencies

1. Update `requirements.txt`
2. Rebuild the image:
   ```bash
   docker-compose build --no-cache
   ```

### Backup and Restore

```bash
# Backup data volume
docker run --rm -v ragbattre_data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .

# Restore data volume
docker run --rm -v ragbattre_data:/data -v $(pwd):/backup alpine tar xzf /backup/data-backup.tar.gz -C /data
```

## Support

For issues and questions:
1. Check the application logs
2. Verify all dependencies are correctly installed
3. Ensure sufficient system resources
4. Check network connectivity for external LLM services

## License

[Add your license information here]
