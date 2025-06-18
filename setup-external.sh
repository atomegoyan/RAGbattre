#!/bin/bash

# RAGbattre Setup Script for External Data and Models
# This script helps set up the required external directories for the Docker deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  RAGbattre External Setup${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

check_data_directory() {
    print_info "Checking data directory..."
    
    if [ ! -d "./data" ]; then
        print_error "Data directory not found!"
        print_info "The data directory should contain:"
        echo "  - corpus/ (document corpus)"
        echo "  - embeddings_cs1/ (pre-computed embeddings)"
        echo "  - *.jsonl (question files)"
        return 1
    fi
    
    # Check required subdirectories
    local missing_dirs=()
    
    if [ ! -d "./data/corpus" ]; then
        missing_dirs+=("data/corpus")
    fi
    
    if [ ! -d "./data/embeddings_cs1" ]; then
        missing_dirs+=("data/embeddings_cs1")
    fi
    
    if [ ${#missing_dirs[@]} -gt 0 ]; then
        print_warning "Missing required directories:"
        for dir in "${missing_dirs[@]}"; do
            echo "  - $dir"
        done
        print_info "These directories will be created, but you need to populate them."
        
        # Create missing directories
        for dir in "${missing_dirs[@]}"; do
            mkdir -p "$dir"
            print_info "Created: $dir"
        done
    fi
    
    # Fix permissions for Docker container (user ID 1000)
    print_info "Setting proper permissions for Docker container..."
    
    # Check if current user can modify the data directory
    if [ -w "./data" ]; then
        # Ensure the data directory is writable by the container user (UID 1000)
        # This is safe because we're only ensuring write permissions exist
        sudo chown -R 1000:1000 ./data 2>/dev/null || {
            print_warning "Could not change ownership to UID 1000. Trying alternative approach..."
            chmod -R 755 ./data
            print_info "Set directory permissions to 755"
        }
        print_success "Data directory permissions configured"
    else
        print_warning "Cannot modify data directory permissions"
        print_info "You may need to run: sudo chown -R 1000:1000 ./data"
    fi
    
    print_success "Data directory structure is ready"
    return 0
}

create_models_directory() {
    print_info "Setting up models directory..."
    
    if [ ! -d "./models" ]; then
        mkdir -p "./models"
        print_success "Created models directory: ./models"
    else
        print_info "Models directory already exists"
    fi
    
    # Create subdirectories for different model types
    mkdir -p "./models/transformers"
    mkdir -p "./models/custom"
    
    print_info "Model directory structure:"
    echo "  ./models/transformers/ - for HuggingFace models"
    echo "  ./models/custom/ - for custom models"
}

create_cache_directories() {
    print_info "Setting up cache directories..."
    
    # Create cache directory for HuggingFace models
    if [ ! -d "./.cache" ]; then
        mkdir -p "./.cache"
        print_success "Created cache directory: ./.cache"
    else
        print_info "Cache directory already exists"
    fi
    
    # Create Streamlit config directory
    if [ ! -d "./.streamlit" ]; then
        mkdir -p "./.streamlit"
        print_success "Created Streamlit config directory: ./.streamlit"
    else
        print_info "Streamlit config directory already exists"
    fi
    
    print_info "Cache directory structure:"
    echo "  ./.cache/ - for HuggingFace model cache"
    echo "  ./.streamlit/ - for Streamlit configuration"
}

setup_environment() {
    print_info "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.template" ]; then
            cp ".env.template" ".env"
            print_success "Created .env from template"
            print_warning "Please edit .env to add your API keys and preferences"
        else
            print_error ".env.template not found!"
            return 1
        fi
    else
        print_info ".env file already exists"
    fi
}

check_ollama_installation() {
    print_info "Checking Ollama installation..."
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama is installed"
        
        # Check if Ollama is running
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_success "Ollama is running on localhost:11434"
            
            # List available models
            local models=$(ollama list 2>/dev/null | tail -n +2 | wc -l)
            if [ $models -gt 0 ]; then
                print_info "Available models: $models"
                ollama list
            else
                print_warning "No models installed"
                print_info "Download a model with: ollama pull llama3.2:1b"
            fi
        else
            print_warning "Ollama is installed but not running"
            print_info "Start Ollama with: ollama serve"
        fi
    else
        print_warning "Ollama is not installed"
        print_info "Install Ollama from: https://ollama.ai/download"
        print_info "Or use cloud LLM backends (Mistral/Cohere) instead"
    fi
}

check_disk_space() {
    print_info "Checking available disk space..."
    
    # Check available space in current directory
    local available_space=$(df . | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    print_info "Available space: ${available_gb}GB"
    
    if [ $available_gb -lt 10 ]; then
        print_warning "Low disk space detected (${available_gb}GB available)"
        print_warning "Recommended: At least 10GB for models and data"
    else
        print_success "Sufficient disk space available"
    fi
}

display_summary() {
    echo ""
    print_info "Setup Summary:"
    echo "=============="
    echo ""
    
    if [ -d "./data" ]; then
        echo "✅ Data directory: ./data"
    else
        echo "❌ Data directory: Missing"
    fi
    
    if [ -d "./models" ]; then
        echo "✅ Models directory: ./models"
    else
        echo "❌ Models directory: Missing"
    fi
    
    if [ -f ".env" ]; then
        echo "✅ Environment config: .env"
    else
        echo "❌ Environment config: Missing"
    fi
    
    echo ""
    print_info "Volume mounts configured:"
    echo "  - ./data → /app/data (corpus, embeddings)"
    echo "  - ./models → /app/models (custom models)"
    echo "  - Docker volumes for model caches"
    echo ""
    
    print_info "Ollama setup:"
    echo "  - Uses local Ollama installation (not containerized)"
    echo "  - Connects to localhost:11434"
    echo "  - Install from: https://ollama.ai/download"
    echo ""
    
    print_info "Next steps:"
    echo "1. Review and edit .env file with your API keys"
    echo "2. Ensure your data directory contains required files"
    echo "3. Install and start Ollama locally (if using local LLMs)"
    echo "4. Run: ./docker-run.sh start"
    echo ""
}

# Main execution
main() {
    print_header
    
    check_disk_space
    check_data_directory
    create_models_directory
    create_cache_directories
    setup_environment
    check_ollama_installation
    
    display_summary
}

# Show help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "RAGbattre External Setup Script"
    echo ""
    echo "This script prepares external directories and configuration for the"
    echo "RAGbattre Docker deployment to keep the container image lightweight."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    echo ""
    echo "What this script does:"
    echo "  1. Checks data directory structure"
    echo "  2. Creates models directory"
    echo "  3. Sets up .env configuration"
    echo "  4. Optionally downloads Ollama models"
    echo "  5. Displays setup summary"
    echo ""
    exit 0
fi

# Run main function
main
