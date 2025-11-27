#!/bin/bash

# Kelly WebGL Build Script
# Usage: ./build_kelly.sh [development|production]

MODE=${1:-production}
UNITY_PATH="/Applications/Unity/Hub/Editor/6000.2.10f1/Unity.app/Contents/MacOS/Unity"
PROJECT_PATH="$(pwd)"
BUILD_PATH="$PROJECT_PATH/Builds/WebGL"

echo "=== KELLY BUILD SCRIPT ==="
echo "Mode: $MODE"
echo "Project: $PROJECT_PATH"
echo ""

# Clean previous build
echo "Cleaning previous build..."
rm -rf "$BUILD_PATH"

# Build
echo "Starting Unity build..."
if [ "$MODE" = "development" ]; then
    "$UNITY_PATH" \
        -quit \
        -batchmode \
        -nographics \
        -projectPath "$PROJECT_PATH" \
        -executeMethod KellySetup.BuildWebGL.BuildFromCommandLine \
        -development \
        -logFile build.log
else
    "$UNITY_PATH" \
        -quit \
        -batchmode \
        -nographics \
        -projectPath "$PROJECT_PATH" \
        -executeMethod KellySetup.BuildWebGL.BuildFromCommandLine \
        -logFile build.log
fi

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "=== BUILD SUCCESS ==="
    echo "Output: $BUILD_PATH"
    
    # Get build size
    SIZE=$(du -sh "$BUILD_PATH" | cut -f1)
    echo "Size: $SIZE"
    
    # Optional: Deploy to hosting
    if [ "$MODE" = "production" ]; then
        echo ""
        echo "Deploying to Netlify..."
        netlify deploy --prod --dir="$BUILD_PATH"
    fi
else
    echo ""
    echo "=== BUILD FAILED ==="
    echo "Check build.log for errors"
    exit 1
fi

