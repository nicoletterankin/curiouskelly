#!/bin/bash
# Secrets Setup Helper Script
# Helps you set up your .env file quickly

echo "🔐 Curious Kelly Secrets Setup Helper"
echo "====================================="
echo ""

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists!"
    read -p "Do you want to overwrite it? (y/N) " overwrite
    if [ "$overwrite" != "y" ] && [ "$overwrite" != "Y" ]; then
        echo "❌ Cancelled. Your existing .env file is safe."
        exit
    fi
fi

echo "📋 This script will help you create your .env file."
echo "   See SECRETS_MASTER_REFERENCE.md for where to get each secret."
echo ""

# Create .env file with template
cat > .env << 'EOF'
# ============================================
# Curious Kelly Platform - Environment Variables
# ============================================
# Created: $(date)
# See SECRETS_MASTER_REFERENCE.md for where to get each value
# ============================================

# ============================================
# SITE CONFIGURATION
# ============================================
PUBLIC_SITE_URL=https://curiouskelly.com

# ============================================
# STRIPE PAYMENT PROCESSING
# ============================================
# Get from: https://dashboard.stripe.com → Developers → API keys
STRIPE_SECRET_KEY=

# Get from: https://dashboard.stripe.com → Developers → Webhooks → [Endpoint] → Signing secret
STRIPE_WEBHOOK_SECRET=

# Get from: https://dashboard.stripe.com → Products → [Product] → Pricing tab
STRIPE_PRICE_MONTHLY=
STRIPE_PRICE_ANNUAL=
STRIPE_PRICE_FAMILY=
STRIPE_PRICE_GIFT=

# ============================================
# SUPABASE DATABASE
# ============================================
# Get from: https://app.supabase.com → Project → Settings → API
PUBLIC_SUPABASE_URL=
PUBLIC_SUPABASE_ANON_KEY=

# ⚠️ SECRET - Server-side only (get from same page, service_role key)
SUPABASE_SERVICE_ROLE_KEY=

# Get from: https://app.supabase.com → Project → Settings → Database → Connection string
SUPABASE_DB_URL=

# ============================================
# CLOUDFLARE R2 (BACKUPS)
# ============================================
# Get from: https://dash.cloudflare.com → R2 → Manage R2 API Tokens
CLOUDFLARE_R2_ENDPOINT=
CLOUDFLARE_R2_ACCESS_KEY=
CLOUDFLARE_R2_SECRET_KEY=
CLOUDFLARE_R2_BUCKET=

# ============================================
# CI/CD DEPLOYMENT TOKENS
# ============================================
# Get from: https://vercel.com/account → Tokens
VERCEL_TOKEN=

# Get from: https://app.netlify.com/user/applications → Personal access tokens
NETLIFY_AUTH_TOKEN=

# Get from: https://dash.cloudflare.com/profile/api-tokens
CLOUDFLARE_API_TOKEN=
CLOUDFLARE_ACCOUNT_ID=
CLOUDFLARE_PROJECT_NAME=

# ============================================
# ANALYTICS (GATED BY CONSENT)
# ============================================
PUBLIC_GTM_ID=
PUBLIC_GA4_ID=
PUBLIC_META_PIXEL_ID=
PUBLIC_TIKTOK_PIXEL_ID=
PUBLIC_TWITTER_PIXEL_ID=
PUBLIC_TABOOLA_ACCOUNT_ID=
PUBLIC_VWO_ID=
PUBLIC_HOTJAR_ID=
PUBLIC_CLARITY_ID=

# ============================================
# MONITORING
# ============================================
# Get from: https://sentry.io → Project Settings → Client Keys (DSN)
PUBLIC_SENTRY_DSN=

# Get from: https://sentry.io → Settings → Auth Tokens
SENTRY_AUTH_TOKEN=

# ============================================
# SECURITY / CAPTCHA
# ============================================
# Cloudflare Turnstile
# Get from: https://dash.cloudflare.com → Turnstile → Create Site Key
TURNSTILE_SITE_KEY=
TURNSTILE_SECRET_KEY=

# Google reCAPTCHA (alternative)
PUBLIC_RECAPTCHA_SITE_KEY=
RECAPTCHA_SECRET_KEY=

# ============================================
# CRM INTEGRATION (OPTIONAL)
# ============================================
CRM_WEBHOOK_URL=
CRM_AUTH_TOKEN=

# ============================================
# AUTHENTICATION (IF USING CLERK)
# ============================================
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
EOF

echo "✅ Created .env file!"
echo ""
echo "📝 Next steps:"
echo "   1. Open .env in your editor"
echo "   2. Fill in your actual values"
echo "   3. See SECRETS_MASTER_REFERENCE.md for where to get each secret"
echo ""
echo "🔗 Quick Links:"
echo "   • Stripe: https://dashboard.stripe.com"
echo "   • Supabase: https://app.supabase.com"
echo "   • Vercel: https://vercel.com/dashboard"
echo ""
echo "⚠️  Remember: Never commit .env to git!"














