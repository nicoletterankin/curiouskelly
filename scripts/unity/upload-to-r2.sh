#!/bin/bash
# ============================================================================
# Upload Unity Build to Cloudflare R2
# Uploads Brotli-compressed Unity WebGL files to R2 bucket
# ============================================================================

set -e

# Configuration
BUILD_NAME="${BUILD_NAME:-Kelly_Web_Build}"
BUILD_DIR="${1:-digital-kelly/engines/Kelly_Engine_V2/onlykelly/Builds/WebGL/Build}"
R2_BUCKET="${R2_BUCKET:-curious-kelly-unity}"
VERSION="${2:-$(date +%Y%m%d-%H%M%S)}"

echo "=============================================="
echo "Unity R2 Upload"
echo "=============================================="
echo ""
echo "Build Name: $BUILD_NAME"
echo "Build Dir:  $BUILD_DIR"
echo "R2 Bucket:  $R2_BUCKET"
echo "Version:    $VERSION"
echo ""

# Check if wrangler is installed
if ! command -v wrangler &> /dev/null; then
    echo "❌ Wrangler CLI is not installed."
    echo "   Install with: npm install -g wrangler"
    exit 1
fi

# Check if authenticated
if ! wrangler whoami &> /dev/null; then
    echo "❌ Not authenticated with Cloudflare."
    echo "   Run: wrangler login"
    exit 1
fi

cd "$BUILD_DIR"

# Verify files exist
echo "=== Verifying files ==="
REQUIRED_FILES=(
    "${BUILD_NAME}.loader.js"
    "${BUILD_NAME}.data.br"
    "${BUILD_NAME}.framework.js.br"
    "${BUILD_NAME}.wasm.br"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ Found: $file"
    else
        echo "❌ Missing: $file"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "❌ $MISSING required files are missing."
    echo "   Run compress-unity-build.sh first to generate Brotli files."
    exit 1
fi

echo ""
echo "=== Uploading to R2 (root) ==="

# Upload to root path (current/latest)
for file in "${REQUIRED_FILES[@]}"; do
    echo "📤 Uploading $file..."
    wrangler r2 object put "$R2_BUCKET/$file" --file="$file"
done

echo ""
echo "=== Uploading to R2 (versioned: $VERSION) ==="

# Upload to versioned path
for file in "${REQUIRED_FILES[@]}"; do
    echo "📤 Uploading $VERSION/$file..."
    wrangler r2 object put "$R2_BUCKET/$VERSION/$file" --file="$file"
done

echo ""
echo "=============================================="
echo "✅ Upload Complete!"
echo "=============================================="
echo ""
echo "CDN URL: https://unity-cdn.nicoletterankin.workers.dev"
echo ""
echo "Latest files:"
echo "  https://unity-cdn.nicoletterankin.workers.dev/${BUILD_NAME}.loader.js"
echo ""
echo "Versioned files:"
echo "  https://unity-cdn.nicoletterankin.workers.dev/$VERSION/${BUILD_NAME}.loader.js"
echo ""
echo "Test with:"
echo "  curl -I https://unity-cdn.nicoletterankin.workers.dev/${BUILD_NAME}.loader.js"



