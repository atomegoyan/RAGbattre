#!/bin/bash

# Simple Docker Compose wrapper that handles permissions
# Use this if you get "permission denied" errors with Docker

echo "🚀 Starting RAGbattre with Docker (handling permissions)..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "✅ Running as root, starting directly..."
    docker-compose up "$@"
else
    echo "🔑 Running with sudo for Docker access..."
    sudo docker-compose up "$@"
fi
