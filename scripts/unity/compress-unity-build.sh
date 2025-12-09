#!/bin/bash
# ============================================================================
# Unity WebGL Build Compressor
# Compresses Unity WebGL build files using Brotli for optimal delivery
# ============================================================================

set -e

# Configuration
BUILD_NAME="${BUILD_NAME:-Kelly_Web_Build}"
BUILD_DIR="${1:-digital-kelly/engines/Kelly_Engine_V2/onlykelly/Builds/WebGL/Build}"
OUTPUT_DIR="${2:-$BUILD_DIR}"

echo "=============================================="
echo "Unity WebGL Build Compressor"
echo "=============================================="
echo ""
echo "Build Name: $BUILD_NAME"
echo "Input Dir:  $BUILD_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo ""

# Check if brotli is installed
if ! command -v brotli &> /dev/null; then
    echo "❌ Brotli is not installed."
    echo "   Install with: sudo apt install brotli (Linux) or brew install brotli (macOS)"
    exit 1
fi

# Create output directory if different from input
if [ "$OUTPUT_DIR" != "$BUILD_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

cd "$BUILD_DIR"

# Function to compress and rename
compress_file() {
    local src_pattern=$1
    local dst_name=$2
    
    # Try to find source file
    local src_file=""
    for pattern in $src_pattern; do
        if [ -f "$pattern" ]; then
            src_file="$pattern"
            break
        fi
    done
    
    if [ -z "$src_file" ]; then
        echo "⚠️  Not found: $src_pattern"
        return 1
    fi
    
    local src_size=$(stat -f%z "$src_file" 2>/dev/null || stat -c%s "$src_file" 2>/dev/null)
    
    echo -n "📦 Compressing $src_file..."
    brotli -q 11 -f "$src_file" -o "$OUTPUT_DIR/$dst_name"
    
    local dst_size=$(stat -f%z "$OUTPUT_DIR/$dst_name" 2>/dev/null || stat -c%s "$OUTPUT_DIR/$dst_name" 2>/dev/null)
    local ratio=$(echo "scale=1; 100 - ($dst_size * 100 / $src_size)" | bc)
    
    echo " ✅ (${ratio}% reduction)"
    return 0
}

# Copy loader (uncompressed - it's small and needs to load first)
copy_loader() {
    local src_pattern=$1
    local dst_name=$2
    
    for pattern in $src_pattern; do
        if [ -f "$pattern" ]; then
            cp "$pattern" "$OUTPUT_DIR/$dst_name"
            echo "📄 Copied $pattern → $dst_name"
            return 0
        fi
    done
    
    echo "⚠️  Loader not found: $src_pattern"
    return 1
}

echo "=== Compressing Unity Build Files ==="
echo ""

# Compress each file type
compress_file "WebGL.wasm ${BUILD_NAME}.wasm" "${BUILD_NAME}.wasm.br" || true
compress_file "WebGL.data ${BUILD_NAME}.data" "${BUILD_NAME}.data.br" || true
compress_file "WebGL.framework.js ${BUILD_NAME}.framework.js" "${BUILD_NAME}.framework.js.br" || true
copy_loader "WebGL.loader.js ${BUILD_NAME}.loader.js" "${BUILD_NAME}.loader.js" || true

echo ""
echo "=== Compression Summary ==="
echo ""

# List output files
cd "$OUTPUT_DIR"
ls -lh ${BUILD_NAME}.* 2>/dev/null || echo "No output files found"

echo ""
echo "✅ Compression complete!"
echo ""
echo "Files ready for upload to R2:"
for f in ${BUILD_NAME}.*.br ${BUILD_NAME}.loader.js; do
    if [ -f "$f" ]; then
        echo "  - $f"
    fi
done



