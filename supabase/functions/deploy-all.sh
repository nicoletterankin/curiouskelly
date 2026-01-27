#!/bin/bash
# Deploy all Supabase Edge Functions
# Run from project root: ./supabase/functions/deploy-all.sh

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         🚀 DEPLOYING SUPABASE EDGE FUNCTIONS                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Check if Supabase CLI is installed
if ! command -v supabase &> /dev/null; then
    echo "❌ Supabase CLI not found. Install with: npm install -g supabase"
    exit 1
fi

# Check if logged in
if ! supabase projects list &> /dev/null; then
    echo "❌ Not logged in. Run: supabase login"
    exit 1
fi

# Deploy each function
FUNCTIONS=(
    "feedback-vote"
    "feedback-heartbeat" 
    "feedback-complete"
    "loop-analyze"
    "get-lesson"
    "get-progress"
)

for fn in "${FUNCTIONS[@]}"; do
    echo "Deploying $fn..."
    if supabase functions deploy $fn --no-verify-jwt; then
        echo "  ✅ $fn deployed"
    else
        echo "  ❌ $fn failed"
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    DEPLOYMENT COMPLETE                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Test endpoints:"
echo "  curl https://<project>.supabase.co/functions/v1/feedback-vote"
echo "  curl https://<project>.supabase.co/functions/v1/get-lesson?day=1&phase=hook"
