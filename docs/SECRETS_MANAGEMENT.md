# Secrets Management Guide
**Curious Kelly Platform - Secure Secret Handling**

---

## 🎯 Overview

This guide covers how to securely manage secrets, API keys, and sensitive configuration for the Curious Kelly platform. Proper secret management is critical for security and compliance.

---

## 🔐 Secret Categories

### 1. **Public Secrets** (Safe to expose to browser)
- Prefixed with `PUBLIC_`
- Examples: `PUBLIC_SUPABASE_URL`, `PUBLIC_SITE_URL`
- ✅ Can be committed to git (in `.env.example`)
- ✅ Visible in client-side code

### 2. **Server Secrets** (Never expose to browser)
- No `PUBLIC_` prefix
- Examples: `STRIPE_SECRET_KEY`, `SUPABASE_DB_URL`
- ❌ Never commit to git
- ❌ Only used in server-side code

### 3. **CI/CD Secrets** (GitHub Actions only)
- Stored in GitHub Secrets
- Examples: `VERCEL_TOKEN`, `CLOUDFLARE_R2_SECRET_KEY`
- ❌ Never commit to git
- ✅ Only accessible in GitHub Actions

---

## 📋 Complete Secret Inventory

### Supabase
- `PUBLIC_SUPABASE_URL` - Public (safe to expose)
- `PUBLIC_SUPABASE_ANON_KEY` - Public (safe to expose)
- `SUPABASE_DB_URL` - **SECRET** (service role connection)

### Stripe
- `STRIPE_SECRET_KEY` - **SECRET** (starts with `sk_`)
- `STRIPE_WEBHOOK_SECRET` - **SECRET** (starts with `whsec_`)

### Authentication
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` - Public (if using Clerk)

### Cloudflare Turnstile
- `TURNSTILE_SITE_KEY` - Public
- `TURNSTILE_SECRET_KEY` - **SECRET**

### Google reCAPTCHA
- `PUBLIC_RECAPTCHA_SITE_KEY` - Public
- `RECAPTCHA_SECRET_KEY` - **SECRET**

### CRM Integration
- `CRM_WEBHOOK_URL` - Public URL
- `CRM_AUTH_TOKEN` - **SECRET**

### Analytics (Gated by Consent)
- `PUBLIC_GTM_ID` - Public
- `PUBLIC_GA4_ID` - Public
- `PUBLIC_META_PIXEL_ID` - Public

### Cloudflare R2 (Backups)
- `CLOUDFLARE_R2_ENDPOINT` - Public URL
- `CLOUDFLARE_R2_ACCESS_KEY` - **SECRET**
- `CLOUDFLARE_R2_SECRET_KEY` - **SECRET**
- `CLOUDFLARE_R2_BUCKET` - Public bucket name

### Deployment Platforms
- `VERCEL_TOKEN` - **SECRET** (CI/CD only)
- `VERCEL_ORG_ID` - Public
- `VERCEL_PROJECT_ID` - Public
- `NETLIFY_AUTH_TOKEN` - **SECRET** (CI/CD only)
- `NETLIFY_SITE_ID` - Public
- `CLOUDFLARE_API_TOKEN` - **SECRET** (CI/CD only)
- `CLOUDFLARE_ACCOUNT_ID` - Public
- `CLOUDFLARE_PROJECT_NAME` - Public
- `GOOGLE_CLOUD_PROJECT` - Public
- `GCP_SA_KEY` - **SECRET** (CI/CD only)

### Monitoring
- `PUBLIC_SENTRY_DSN` - Public (safe to expose)
- `SENTRY_AUTH_TOKEN` - **SECRET** (CI/CD only)

---

## 🛡️ Security Best Practices

### 1. Never Commit Secrets
- ✅ Use `.env` for local development (in `.gitignore`)
- ✅ Use `.env.example` for documentation (committed, no real values)
- ✅ Use GitHub Secrets for CI/CD
- ❌ Never commit `.env` files
- ❌ Never hardcode secrets in code
- ❌ Never commit secrets in comments

### 2. Secret Rotation Schedule

| Secret Type | Rotation Frequency | Last Rotated |
|------------|-------------------|--------------|
| Database passwords | Quarterly | - |
| API keys | Quarterly | - |
| OAuth tokens | Annually | - |
| SSH keys | Annually | - |
| Certificates | Annually | - |

**Action:** Set calendar reminders for rotations.

### 3. Access Control

**Who Can Access Secrets:**
- **Local `.env`:** Developers (local only)
- **GitHub Secrets:** Repository admins only
- **Production Secrets:** Platform owner only
- **CI/CD Secrets:** Automated workflows only

### 4. Secret Scanning

**Pre-commit Hook:**
```bash
# Install git-secrets
brew install git-secrets  # macOS
# or
git clone https://github.com/awslabs/git-secrets.git

# Configure
git secrets --install
git secrets --register-aws

# Test
git secrets --scan
```

**GitHub Actions Secret Scanning:**
- ✅ Enabled by default on GitHub
- Scans commits for exposed secrets
- Alerts automatically if found

---

## 📝 Setup Instructions

### Local Development

1. **Copy `.env.example` to `.env`:**
   ```bash
   cp .env.example .env
   ```

2. **Fill in actual values:**
   - Get Supabase keys from Supabase Dashboard
   - Get Stripe keys from Stripe Dashboard
   - Get other keys from respective services

3. **Verify `.env` is in `.gitignore`:**
   ```bash
   grep "^\.env$" .gitignore
   # Should output: .env
   ```

### CI/CD (GitHub Actions)

1. **Go to GitHub Repository Settings:**
   - Navigate to: Settings → Secrets and variables → Actions

2. **Add Required Secrets:**
   - Click "New repository secret"
   - Add each secret from the inventory above
   - Use descriptive names (match `.env.example`)

3. **Verify Secrets in Workflows:**
   - Check `.github/workflows/*.yml` files
   - Ensure secrets are referenced as `${{ secrets.SECRET_NAME }}`
   - Never hardcode values

### Production Deployment

**Vercel:**
1. Go to Vercel Dashboard → Project → Settings → Environment Variables
2. Add all required variables
3. Set environment (Production, Preview, Development)

**Netlify:**
1. Go to Netlify Dashboard → Site → Environment variables
2. Add all required variables
3. Set scopes (Build, Runtime, etc.)

**Cloudflare Pages:**
1. Go to Cloudflare Dashboard → Pages → Project → Settings → Environment Variables
2. Add all required variables
3. Set environment (Production, Preview)

---

## 🔄 Secret Rotation Procedure

### When to Rotate:
- Quarterly (scheduled)
- After security incident
- When team member leaves
- When secret is exposed

### Rotation Steps:

1. **Generate New Secret:**
   - Go to service dashboard (Stripe, Supabase, etc.)
   - Generate new key/token
   - **DO NOT DELETE OLD KEY YET**

2. **Update All Environments:**
   - Update local `.env` (test locally)
   - Update GitHub Secrets
   - Update production environment variables
   - Update staging/preview environments

3. **Deploy and Verify:**
   - Deploy to staging first
   - Verify functionality
   - Deploy to production
   - Monitor for errors

4. **Revoke Old Secret:**
   - Wait 24-48 hours (ensure no issues)
   - Revoke old key in service dashboard
   - Document rotation date

5. **Update Documentation:**
   - Update `SECRETS_MANAGEMENT.md` with rotation date
   - Update team on rotation

---

## 🚨 Incident Response

### If Secret is Exposed:

1. **Immediate Actions (Within 5 minutes):**
   - Rotate exposed secret immediately
   - Revoke old secret in service dashboard
   - Check GitHub commit history for exposure
   - Review access logs if available

2. **Investigation (Within 1 hour):**
   - Determine scope of exposure
   - Check if secret was used maliciously
   - Review audit logs
   - Document incident

3. **Communication (Within 4 hours):**
   - Notify team
   - Notify affected users (if PII exposed)
   - File security incident report
   - Update documentation

4. **Prevention (Within 1 week):**
   - Review how secret was exposed
   - Update procedures to prevent recurrence
   - Train team on best practices
   - Implement additional safeguards

---

## ✅ Security Checklist

### Before Committing Code:
- [ ] No secrets in code
- [ ] No secrets in comments
- [ ] `.env` is in `.gitignore`
- [ ] `.env.example` has placeholder values only
- [ ] No hardcoded API keys
- [ ] No hardcoded passwords

### Before Deployment:
- [ ] All secrets configured in production environment
- [ ] Secrets rotated within last quarter
- [ ] Access logs reviewed
- [ ] Team trained on secret management

### Monthly Review:
- [ ] Review secret access logs
- [ ] Check for exposed secrets (GitHub scanning)
- [ ] Verify `.gitignore` is up to date
- [ ] Review rotation schedule

---

## 📚 Resources

- [GitHub Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [OWASP Secret Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [12-Factor App: Config](https://12factor.net/config)

---

## 🆘 Support

For questions or security concerns:
1. Review this documentation
2. Check `CLAUDE.md` for operating rules
3. Review `RUNBOOK.md` for incident procedures
4. Contact platform owner for critical issues

---

**Last Updated:** December 2025  
**Next Secret Rotation:** Q1 2026




