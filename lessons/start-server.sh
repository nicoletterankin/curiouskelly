#!/bin/bash
# Simple HTTP Server for Calendar Page
# This script starts a local web server to avoid CORS issues

echo "Starting local web server for Calendar Page..."
echo ""
echo "Server will be available at: http://localhost:8000"
echo "Open: http://localhost:8000/calendar-page.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "Using Python HTTP server..."
    python3 -m http.server 8000
elif command -v python &> /dev/null; then
    echo "Using Python HTTP server..."
    python -m http.server 8000
elif command -v npx &> /dev/null; then
    echo "Using Node.js http-server..."
    npx --yes http-server -p 8000 -c-1
else
    echo "ERROR: Neither Python nor Node.js found!"
    echo ""
    echo "Please install one of the following:"
    echo "  - Python 3: https://www.python.org/downloads/"
    echo "  - Node.js: https://nodejs.org/"
    echo ""
    echo "Or use VS Code's Live Server extension"
    exit 1
fi

