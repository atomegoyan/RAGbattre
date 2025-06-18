#!/bin/bash

# RAGbattre Docker Management Script
# Simple wrapper for common Docker operations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
    echo "RAGbattre Docker Manager"
    echo "========================"
    echo
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  start          Start the application"
    echo "  stop           Stop the application"
    echo "  restart        Restart the application" 
    echo "  logs           Show application logs"
    echo "  status         Check application status"
    echo "  build          Rebuild the Docker image"
    echo "  clean          Stop and remove containers"
    echo "  fix-perms      Fix directory permissions"
    echo "  setup          Run initial setup"
    echo "  health         Check if app is healthy"
    echo "  open           Open app in browser"
    echo
    echo "Examples:"
    echo "  $0 start       # Start the application"
    echo "  $0 logs        # View logs in real-time"
    echo "  $0 status      # Check detailed status"
    echo
}

case "$1" in
    "start")
        echo "🚀 Starting RAGbattre..."
        sudo docker-compose up -d
        echo "✅ Started! Check status with: $0 status"
        echo "📱 Access at: http://localhost:8501"
        ;;
    "stop")
        echo "🛑 Stopping RAGbattre..."
        sudo docker-compose down
        echo "✅ Stopped!"
        ;;
    "restart")
        echo "🔄 Restarting RAGbattre..."
        sudo docker-compose restart
        echo "✅ Restarted! Check status with: $0 status"
        ;;
    "logs")
        echo "📋 Showing logs (Ctrl+C to exit)..."
        sudo docker-compose logs -f ragbattre-app
        ;;
    "status")
        echo "📊 Checking detailed status..."
        ./check-status.sh
        ;;
    "build")
        echo "🔨 Rebuilding RAGbattre..."
        sudo docker-compose build
        echo "✅ Build complete! Use '$0 start' to run."
        ;;
    "clean")
        echo "🧹 Cleaning up containers..."
        sudo docker-compose down -v
        echo "✅ Cleanup complete!"
        ;;
    "fix-perms")
        echo "🔧 Fixing permissions..."
        ./fix-permissions.sh
        echo "✅ Permissions fixed!"
        ;;
    "setup")
        echo "⚙️  Running initial setup..."
        ./setup-external.sh
        echo "✅ Setup complete! Use '$0 start' to run."
        ;;
    "health")
        if sudo docker-compose ps | grep -q "healthy"; then
            echo "✅ Application is healthy"
            exit 0
        else
            echo "❌ Application is not healthy"
            exit 1
        fi
        ;;
    "open")
        echo "🌐 Opening application in browser..."
        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8501
        elif command -v gnome-open &> /dev/null; then
            gnome-open http://localhost:8501
        else
            echo "Please open http://localhost:8501 in your browser"
        fi
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac
