# RAGbattre Docker Setup - Quick Start Guide

## ⚡ Fastest Start (TL;DR)

Just want to get it running quickly? Here's the minimal commands:

```bash
cd /home/atom/Bureau/Aurelien/RAGbattre

# Start the application (builds and runs everything)
docker-compose up
```

That's it! The application will be available at http://localhost:8501

*Note: Make sure your `data/` directory exists with your corpus and embeddings.*

---

## ✅ What's Been Created

Your RAGbattre Streamlit application has been successfully docke## 🐛 Troubleshooting

### Common Issues:
- **Docker not running**: Start Docker Desktop/Engine
- **Port in use**: Change port in docker-compose.yml
- **Memory issues**: Increase Docker memory allocation
- **Permissions**: Add user to docker group: `sudo usermod -aG docker $USER`
- **ChromaDB readonly error**: Run `./fix-permissions.sh` or `./docker-run.sh fix-perms`

### Getting Help:
- Check logs: `./docker-run.sh logs`
- Access container: `./docker-run.sh shell`
- Fix permissions: `./docker-run.sh fix-perms`
- Review documentation: `EXTERNAL_SETUP.md`he following files:

### 📁 Docker Files Created:
- **`Dockerfile`** - Lightweight Docker image (no data/models included)
- **`docker-compose.yml`** - Production with external volume mounts
- **`docker-compose.dev.yml`** - Development docker compose with live reload
- **`requirements.txt`** - Python dependencies with version constraints
- **`.dockerignore`** - Excludes heavy files (data/, models/, *.bin, etc.)
- **`.env.template`** - Environment variables template
- **`docker-run.sh`** - Easy management script (executable)
- **`setup-external.sh`** - External data/models setup script
- **`README_DOCKER.md`** - Comprehensive documentation
- **`EXTERNAL_SETUP.md`** - External storage guide

## 🚀 Quick Start Commands

### Option 1: Simple Docker Compose (Easiest)
```bash
# Make sure you're in the project directory
cd /home/atom/Bureau/Aurelien/RAGbattre

# Start the application (builds automatically if needed)
docker-compose up

# Or run in background
docker-compose up -d

# View logs (if running in background)
docker-compose logs -f

# Stop the application
docker-compose down
```

### Option 2: Automated Setup + Docker Compose
```bash
# Run external setup first (recommended for first time)
./setup-external.sh

# Start the application
docker-compose up -d
```

### Option 3: Using Management Script
```bash
# Test dependencies (optional but recommended)
./docker-run.sh test

# Start the application
./docker-run.sh start

# View logs
./docker-run.sh logs -f

# Stop the application
./docker-run.sh stop
```

### Option 3: Development Mode (with live reload)
```bash
# Use development compose file
docker-compose -f docker-compose.dev.yml up --build
```

## 🌐 Access the Application

Once running, access your application at:
- **Local**: http://localhost:8501
- **Network**: http://your-server-ip:8501

## ⚙️ Configuration

### Environment Variables
1. Copy the template: `cp .env.template .env`
2. Edit `.env` with your API keys and preferences
3. Restart the application

### Key Configuration Options:
- **LLM Backend**: Choose between Ollama (local), Mistral AI, or Cohere
- **API Keys**: Add your Mistral/Cohere API keys if using cloud services
- **Model Selection**: Configure which models to use

## 📊 Application Features

Your dockerized app includes:

1. **Chat with Ollama** - Direct LLM interaction
2. **Document Retrieval** - Search parliamentary documents  
3. **RAG Mode** - Enhanced responses using retrieved context

## 🔧 Dependencies Included

The lightweight Docker image includes:
- Streamlit (web framework)
- Ollama client (local LLM)
- ChromaDB (vector database) 
- Transformers + PyTorch (ML models)
- Mistral AI & Cohere (cloud LLMs)
- All Python dependencies

External (mounted as volumes):
- Document corpus and embeddings
- Model caches and downloaded models
- Configuration files and data

## 💾 Data & Model Storage

The setup uses external storage for heavy files:

**Lightweight Docker Image (~500MB):**
- Application code and Python dependencies only

**External Volumes:**
- `./data/` → Document corpus and pre-computed embeddings
- `./models/` → Custom models directory
- `huggingface_cache` → Downloaded transformer models
- `ollama_data` → Ollama models storage

**Benefits:**
- Fast image builds and deployments
- Persistent data across container restarts
- Easy data updates without rebuilding
- Shared model cache across containers

## 🛠️ Management Commands

Using `./docker-run.sh`:
- `start` - Build and start the application
- `stop` - Stop the application
- `restart` - Restart the application
- `build` - Rebuild the image
- `test` - Test dependencies without full build
- `logs` - View application logs
- `shell` - Access container shell
- `clean` - Remove all containers and images
- `status` - Show running status

## 📚 Next Steps

1. **Setup external storage**: `./setup-external.sh`
2. **Start the application**: `./docker-run.sh start`
3. **Open in browser**: http://localhost:8501
4. **Configure API keys** (if needed): Edit `.env` file
5. **Review documentation**: Read `EXTERNAL_SETUP.md` for detailed external storage info

## 🐛 Troubleshooting

### Common Issues:
- **Docker not running**: Start Docker Desktop/Engine
- **Port in use**: Change port in docker-compose.yml
- **Memory issues**: Increase Docker memory allocation
- **Permissions**: Add user to docker group: `sudo usermod -aG docker $USER`

### Getting Help:
- Check logs: `./docker-run.sh logs`
- Access container: `./docker-run.sh shell`
- Review documentation: `README_DOCKER.md`

## 🔄 Development Workflow

For active development:
1. Use `docker-compose.dev.yml` for live reload
2. Mount source code volumes
3. Use the Ollama service for local testing

---

Your RAGbattre application is now fully dockerized and ready to deploy! 🎉
