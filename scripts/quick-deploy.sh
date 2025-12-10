#!/bin/bash
# Quick Deploy Script for Curious Kelly
# One-command deployment to Vercel

set -e

echo "🚀 Curious Kelly Quick Deploy"
echo "=============================="
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if logged in
if ! vercel whoami &> /dev/null; then
    echo "📝 Please log in to Vercel..."
    vercel login
fi

# Deploy
echo "📦 Deploying to Vercel..."
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your site should be live at: https://curiouskelly.com"
echo ""
echo "📋 Next steps:"
echo "1. Verify site is live"
echo "2. Test all features"
echo "3. Check email forms"
echo "4. Test payment flow (test mode)"












