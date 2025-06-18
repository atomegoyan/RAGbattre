#!/bin/bash

# Export current user's UID and GID for Docker Compose
export UID=$(id -u)
export GID=$(id -g)

# Start Docker Compose
docker-compose up "$@"
