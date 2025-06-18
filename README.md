# RAGbattre - Parliamentary Debate Analyzer

A Retrieval-Augmented Generation (RAG) application for analyzing parliamentary debates with multiple LLM backends.

## 🚀 Quick Start

### Easy Way (Recommended)
```bash
# Use the management script
./ragbattre.sh start

# Check status
./ragbattre.sh status

# View logs
./ragbattre.sh logs
```

### Manual Way
```bash
# Clone or navigate to the project
cd RAGbattre

# Start the application
docker-compose up
```

Access the application at: **http://localhost:8501**

## 📋 Requirements

- Docker and Docker Compose
- Your data directory with corpus and embeddings
- (Optional) Ollama for local LLMs

## 🔧 Commands

### Using the Management Script (Recommended)
```bash
./ragbattre.sh start      # Start in background
./ragbattre.sh stop       # Stop the application
./ragbattre.sh restart    # Restart the application
./ragbattre.sh logs       # View logs in real-time
./ragbattre.sh status     # Detailed status check
./ragbattre.sh health     # Quick health check
./ragbattre.sh fix-perms  # Fix permissions
./ragbattre.sh setup      # Initial setup
./ragbattre.sh clean      # Clean up containers
```

### Manual Docker Commands
```bash
# Start in foreground (see logs)
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Rebuild and restart
docker-compose up --build
```

## 🛠️ Available Scripts

- **`./ragbattre.sh`** - Main management script with common commands
- **`./check-status.sh`** - Comprehensive status and health check
- **`./setup-external.sh`** - Initial setup for external directories
- **`./fix-permissions.sh`** - Fix file/directory permissions
- **`./start-docker.sh`** - Alternative startup script

## 🎯 Features

- **Multiple LLM Backends**: Ollama (local), Mistral AI, Cohere
- **Three Modes**: 
  - Chat with Ollama
  - Document Retrieval  
  - RAG Mode (retrieval + generation)
- **Lightweight Docker**: External data/model storage
- **Web Interface**: Streamlit-based UI

## 📚 Documentation

- **[Quick Start Guide](DOCKER_QUICKSTART.md)** - Get up and running fast
- **[External Setup Guide](EXTERNAL_SETUP.md)** - Detailed external storage setup
- **[Full Documentation](README_DOCKER.md)** - Comprehensive guide

## 🛠️ Troubleshooting

**Common issues:**

```bash
# Permission errors with ChromaDB
./fix-permissions.sh

# Test dependencies
./docker-run.sh test

# Check Ollama connection
./test-ollama.sh
```

## 📁 Project Structure

```
RAGbattre/
├── data/                   # Your corpus and embeddings (external)
├── retrieval_app/          # Main application code
├── docker-compose.yml      # Main Docker configuration
├── Dockerfile             # Container definition
└── scripts/               # Utility scripts
```

---

**Need help?** Check the documentation files or run `./docker-run.sh help` for management commands.
