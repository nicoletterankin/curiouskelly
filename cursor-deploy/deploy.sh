#!/bin/bash
# DEPLOY NOW — Cloudflare Pages Direct Upload
# Run this in Cursor terminal

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  LESSON OF THE DAY — CLOUDFLARE PAGES DEPLOYMENT              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Step 1: Check for API token
if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
    echo ""
    echo "⚠️  CLOUDFLARE_API_TOKEN not set."
    echo ""
    echo "Get your token:"
    echo "1. Go to: https://dash.cloudflare.com/profile/api-tokens"
    echo "2. Click: Create Token"
    echo "3. Use template: 'Edit Cloudflare Workers'"
    echo "4. Add permission: Zone > DNS > Edit"
    echo "5. Zone Resources: Include > thedailylesson.com"
    echo "6. Create Token > Copy"
    echo ""
    echo "Then run:"
    echo "export CLOUDFLARE_API_TOKEN='your-token-here'"
    echo ""
    exit 1
fi

ACCOUNT_ID="47ebb2a1adc311cb106acc89720e352c"

echo "✓ API Token found"
echo "✓ Account ID: $ACCOUNT_ID"
echo ""

# Step 2: Install wrangler if needed
if ! command -v wrangler &> /dev/null; then
    echo "Installing wrangler..."
    npm install -g wrangler
fi

echo "✓ Wrangler installed"
echo ""

# Step 3: Create invest directory if it doesn't exist
if [ ! -d "./invest" ]; then
    echo "Creating invest directory..."
    mkdir -p ./invest/videos
    
    # Download files from outputs or create placeholders
    echo "⚠️  Please copy invest.zip contents to ./invest/"
    echo "   - index.html"
    echo "   - videos/HOMEPAGE_HERO_WEB.mp4"
    echo "   - videos/HOMEPAGE_HERO_MOBILE.mp4"
    echo "   - videos/HOMEPAGE_HERO_POSTER.jpg"
    exit 1
fi

# Step 4: Create dallas directory if it doesn't exist
if [ ! -d "./dallas" ]; then
    echo "Creating dallas directory..."
    mkdir -p ./dallas/videos
    
    echo "⚠️  Please copy dallas.zip contents to ./dallas/"
    echo "   - index.html"
    echo "   - command.html"
    echo "   - videos/HOMEPAGE_HERO_WEB.mp4"
    echo "   - videos/HOMEPAGE_HERO_MOBILE.mp4"
    echo "   - videos/HOMEPAGE_HERO_POSTER.jpg"
    exit 1
fi

echo "✓ Source directories ready"
echo ""

# Step 5: Deploy invest site
echo "═══════════════════════════════════════════════════════════════"
echo "Deploying invest.thedailylesson.com..."
echo "═══════════════════════════════════════════════════════════════"

wrangler pages project create invest-dailylesson --production-branch=main 2>/dev/null || true
wrangler pages deploy ./invest --project-name=invest-dailylesson --branch=main

echo ""
echo "✓ invest-dailylesson deployed"
echo ""

# Step 6: Deploy dallas site
echo "═══════════════════════════════════════════════════════════════"
echo "Deploying dallas.thedailylesson.com..."
echo "═══════════════════════════════════════════════════════════════"

wrangler pages project create dallas-dailylesson --production-branch=main 2>/dev/null || true
wrangler pages deploy ./dallas --project-name=dallas-dailylesson --branch=main

echo ""
echo "✓ dallas-dailylesson deployed"
echo ""

# Step 7: Add custom domains
echo "═══════════════════════════════════════════════════════════════"
echo "Adding custom domains..."
echo "═══════════════════════════════════════════════════════════════"

# Add invest subdomain
curl -s -X POST "https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/pages/projects/invest-dailylesson/domains" \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"name":"invest.thedailylesson.com"}' | jq .

echo ""

# Add dallas subdomain
curl -s -X POST "https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/pages/projects/dallas-dailylesson/domains" \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"name":"dallas.thedailylesson.com"}' | jq .

echo ""

# Step 8: Verify
echo "═══════════════════════════════════════════════════════════════"
echo "Verifying deployments..."
echo "═══════════════════════════════════════════════════════════════"

sleep 5

echo "Testing invest-dailylesson.pages.dev..."
curl -s -o /dev/null -w "HTTP %{http_code}\n" https://invest-dailylesson.pages.dev

echo "Testing dallas-dailylesson.pages.dev..."
curl -s -o /dev/null -w "HTTP %{http_code}\n" https://dallas-dailylesson.pages.dev

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  DEPLOYMENT COMPLETE                                          ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║                                                               ║"
echo "║  invest-dailylesson.pages.dev      → invest.thedailylesson.com║"
echo "║  dallas-dailylesson.pages.dev      → dallas.thedailylesson.com║"
echo "║                                                               ║"
echo "║  Custom domains may take 1-2 minutes for SSL.                 ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
