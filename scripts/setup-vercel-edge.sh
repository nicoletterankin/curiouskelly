#!/bin/bash
# Setup Vercel Edge Config and Blob Storage
# This script configures Edge Config and Blob Storage via Vercel CLI

set -e

echo "🚀 Setting up Vercel Edge Optimization..."

# Check if logged in
if ! vercel whoami > /dev/null 2>&1; then
    echo "❌ Not logged into Vercel. Please run: vercel login"
    exit 1
fi

echo "✅ Logged into Vercel"

# Generate a random secret for Edge Config sync
EDGE_CONFIG_SYNC_SECRET=$(openssl rand -hex 32)
echo "🔐 Generated Edge Config sync secret: $EDGE_CONFIG_SYNC_SECRET"

# Note: Vercel CLI doesn't have direct commands for Edge Config/Blob creation
# These need to be done via Dashboard or API
echo ""
echo "⚠️  Edge Config and Blob Storage must be created via Vercel Dashboard"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Go to: https://vercel.com/dashboard"
echo "2. Select your project: curiouskelly"
echo "3. Go to Storage → Edge Config → Create"
echo "   - Name: curious-kelly-lessons"
echo "   - Copy the connection string"
echo ""
echo "4. Go to Storage → Blob → Create buckets:"
echo "   - curious-kelly-videos"
echo "   - curious-kelly-audio"
echo "   - curious-kelly-visuals"
echo ""
echo "5. Add environment variables in Settings → Environment Variables:"
echo "   - EDGE_CONFIG=<connection-string-from-step-3>"
echo "   - EDGE_CONFIG_SYNC_SECRET=$EDGE_CONFIG_SYNC_SECRET"
echo ""
echo "✅ Setup script complete!"
echo ""
echo "After completing Dashboard setup, run:"
echo "  npm run sync-edge-config"

