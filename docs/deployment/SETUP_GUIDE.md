# CuriousKelly.com - Step-by-Step Deployment Setup Guide

**Target:** Deploy curiouskelly.com to production with GitHub, Cloudflare, Vercel, and Flutter integration.

---

## Prerequisites

- [ ] GitHub account: `nicoletterankin`
- [ ] Cloudflare account with domain `curiouskelly.com` registered
- [ ] Vercel account (team: `Lotd`)
- [ ] Node.js ≥ 18.17 installed
- [ ] Git installed
- [ ] PowerShell (Windows) or Bash (Mac/Linux)

---

## Phase 1: GitHub Repository Setup (15 minutes)

### Step 1.1: Initialize Git Repository

Open PowerShell in project root (`C:\Users\user\UI-TARS-desktop`):

```powershell
# Check if git is initialized
if (Test-Path .git) {
    Write-Host "Git already initialized"
} else {
    git init
    Write-Host "Git initialized"
}

# Check current status
git status
```

### Step 1.2: Create Initial Commit

```powershell
# Add all files (respecting .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: CuriousKelly.com production codebase

- Marketing site (Astro)
- Lesson player (static HTML/JS)
- Flutter mobile apps
- Backend services
- Deployment configurations"
```

### Step 1.3: Create GitHub Repository

1. Open browser: https://github.com/new
2. **Repository name:** `curiouskelly`
3. **Description:** `Striving to be the best digital teacher in the world.`
4. **Visibility:** Public
5. **DO NOT check:**
   - ❌ Add a README file
   - ❌ Add .gitignore
   - ❌ Choose a license
6. Click **"Create repository"**

### Step 1.4: Connect Local Repository to GitHub

```powershell
# Add remote (replace with your actual GitHub username if different)
git remote add origin https://github.com/nicoletterankin/curiouskelly.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Expected Output:**
```
Enumerating objects: X, done.
Counting objects: 100%, done.
Delta compression using up to X threads
Compressing objects: 100%, done.
Writing objects: 100%, done.
Total X (delta Y), reused Z (delta W)
remote: Resolving deltas: 100%, done.
To https://github.com/nicoletterankin/curiouskelly.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

### Step 1.5: Configure Branch Protection

1. Go to: https://github.com/nicoletterankin/curiouskelly/settings/branches
2. Click **"Add rule"**
3. **Branch name pattern:** `main`
4. Enable:
   - ✅ Require a pull request before merging
   - ✅ Require approvals: 1
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
5. Click **"Create"**

---

## Phase 2: Cloudflare Pages Setup (20 minutes)

### Step 2.1: Create Cloudflare API Token

1. Go to: https://dash.cloudflare.com/profile/api-tokens
2. Click **"Create Token"**
3. Click **"Get started"** on "Edit Cloudflare Workers" template
4. **Permissions:**
   - Account → Cloudflare Pages → Edit
   - Zone → Zone → Read
5. **Account Resources:**
   - Include → All accounts (or select specific account)
6. **Zone Resources:**
   - Include → Specific zone → `curiouskelly.com`
7. **Token name:** `curiouskelly-pages-deploy`
8. Click **"Continue to summary"** → **"Create Token"**
9. **COPY THE TOKEN** (you won't see it again!)
10. Save securely (password manager recommended)

### Step 2.2: Add Cloudflare Secrets to GitHub

1. Go to: https://github.com/nicoletterankin/curiouskelly/settings/secrets/actions
2. Click **"New repository secret"**
3. Add each secret:

**Secret 1:**
- **Name:** `CLOUDFLARE_API_TOKEN`
- **Value:** (paste the token from Step 2.1)
- Click **"Add secret"**

**Secret 2:**
- **Name:** `CLOUDFLARE_ACCOUNT_ID`
- **Value:** `47ebb2a1adc311cb106acc89720e352c`
- Click **"Add secret"**

**Secret 3:**
- **Name:** `CLOUDFLARE_ZONE_ID`
- **Value:** `510107ca53356bab42f8a8d1b2de1e59`
- Click **"Add secret"**

**Secret 4:**
- **Name:** `CLOUDFLARE_PROJECT_NAME`
- **Value:** `curiouskelly`
- Click **"Add secret"**

### Step 2.3: Create Cloudflare Pages Project

1. Go to: https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c/pages
2. Click **"Create a project"**
3. Click **"Connect to Git"**
4. Select **GitHub** → Authorize if needed
5. Select repository: **`nicoletterankin/curiouskelly`**
6. Click **"Begin setup"**
7. **Project name:** `curiouskelly`
8. **Production branch:** `main`
9. **Build settings:**
   - **Framework preset:** None (or Custom)
   - **Build command:** (leave empty for static files, or `npm run build` if needed)
   - **Build output directory:** `lesson-player` (for lesson player) or `dist` (for marketing)
   - **Root directory:** `/` (or specific subdirectory like `curiouskelly-marketing-site`)
10. Click **"Save and Deploy"**

### Step 2.4: Configure Custom Domain

1. In Pages project → **"Custom domains"** tab
2. Click **"Set up a custom domain"**
3. Enter: `curiouskelly.com`
4. Click **"Continue"**
5. Cloudflare will automatically configure DNS
6. Wait for SSL certificate (usually 1-5 minutes)
7. Verify: Visit `https://curiouskelly.com` (should show your site)

### Step 2.5: Set Environment Variables

1. In Pages project → **"Settings"** → **"Environment variables"**
2. Click **"Add variable"** for each:

**Production:**
- `ELEVENLABS_API_KEY` = (your ElevenLabs API key)
- `STRIPE_SECRET_KEY` = (your Stripe secret key)
- `ANALYTICS_ID` = (your analytics ID)
- `PUBLIC_SITE_URL` = `https://curiouskelly.com`

**Preview:**
- Same variables with test/development values

### Step 2.6: Enable Security Features

1. Go to: https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c/curiouskelly.com/security
2. **Page Shield:**
   - Toggle **ON**
   - Click **"Save"**
3. **Bot Fight Mode:**
   - Toggle **ON**
   - Click **"Save"**
4. **Leaked credentials detection:**
   - Click **"Activate"**
   - Configure rate limiting (default: 10 seconds)

### Step 2.7: Create R2 Storage Bucket (for assets)

1. Go to: https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c/r2
2. Click **"Create bucket"**
3. **Bucket name:** `curiouskelly-assets`
4. **Location:** (select closest to your users)
5. Click **"Create bucket"**
6. **Configure CORS:**
   - Go to bucket → **"Settings"** → **"CORS Policy"**
   - Add CORS policy:
   ```json
   [
     {
       "AllowedOrigins": ["https://curiouskelly.com"],
       "AllowedMethods": ["GET", "HEAD"],
       "AllowedHeaders": ["*"],
       "ExposeHeaders": [],
       "MaxAgeSeconds": 3600
     }
   ]
   ```

---

## Phase 3: Vercel Setup (20 minutes)

### Step 3.1: Install Vercel CLI

```powershell
npm install -g vercel
```

### Step 3.2: Login to Vercel

```powershell
vercel login
```

Follow the browser prompt to authenticate.

### Step 3.3: Create Vercel Project via Dashboard

1. Go to: https://vercel.com/lotd
2. Click **"Add New..."** → **"Project"**
3. Click **"Import Git Repository"**
4. Select **`nicoletterankin/curiouskelly`**
5. Click **"Import"**
6. **Configure Project:**
   - **Framework Preset:** Astro
   - **Root Directory:** `curiouskelly-marketing-site` (or `/` if single project)
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
   - **Install Command:** `npm install`
7. Click **"Deploy"**

### Step 3.4: Add Vercel Secrets to GitHub

1. Get Vercel credentials:
   ```powershell
   # In project root
   cd curiouskelly-marketing-site
   vercel link
   # Select team: Lotd
   # Project name: (will be shown or create new)
   ```

2. Get tokens from: https://vercel.com/account/tokens
   - Create new token: **"Create Token"**
   - Name: `curiouskelly-github-actions`
   - Scope: Full Account
   - Copy token

3. Add to GitHub Secrets:
   - Go to: https://github.com/nicoletterankin/curiouskelly/settings/secrets/actions
   - **`VERCEL_TOKEN`** = (token from step above)
   - **`VERCEL_ORG_ID`** = (get from `vercel whoami` or team settings)
   - **`VERCEL_PROJECT_ID`** = (get from project settings → General)

### Step 3.5: Add Custom Domain

1. In Vercel project → **"Settings"** → **"Domains"**
2. Click **"Add"**
3. Enter: `curiouskelly.com` (or `www.curiouskelly.com`)
4. Follow DNS configuration instructions
5. Wait for DNS propagation and SSL certificate

### Step 3.6: Set Environment Variables

1. In Vercel project → **"Settings"** → **"Environment Variables"**
2. Add each variable for **Production**, **Preview**, and **Development**:

```
PUBLIC_SITE_URL = https://curiouskelly.com
PUBLIC_DEFAULT_LOCALE = en-US
PUBLIC_AVAILABLE_LOCALES = en-US,es-ES,pt-BR
TURNSTILE_SITE_KEY = (your Turnstile site key)
TURNSTILE_SECRET_KEY = (your Turnstile secret key)
CRM_WEBHOOK_URL = (your CRM webhook URL)
CRM_AUTH_TOKEN = (optional)
ELEVENLABS_API_KEY = (your ElevenLabs API key)
STRIPE_SECRET_KEY = (your Stripe secret key)
ANALYTICS_ID = (your analytics ID)
```

---

## Phase 4: Flutter CI/CD Setup (30 minutes)

### Step 4.1: Create GitHub Actions Workflow

Create file: `.github/workflows/flutter-build.yml`

```yaml
name: Flutter Build & Release

on:
  push:
    branches: [main]
    paths:
      - 'curious-kellly/mobile/**'
      - 'digital-kelly/apps/kelly_app_flutter/**'
  workflow_dispatch:
    inputs:
      platform:
        description: 'Platform to build (ios/android/both)'
        required: true
        default: 'both'
      release:
        description: 'Create release?'
        type: boolean
        default: false

jobs:
  build-ios:
    if: github.event.inputs.platform == 'ios' || github.event.inputs.platform == 'both' || (github.event_name == 'push' && contains(github.event.head_commit.message, '[ios]'))
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: subosito/flutter-action@v2
        with:
          flutter-version: '3.24.0'
      - name: Install dependencies
        run: |
          cd curious-kellly/mobile
          flutter pub get
      - name: Build iOS
        run: |
          cd curious-kellly/mobile
          flutter build ios --release --no-codesign
      - name: Upload iOS artifact
        uses: actions/upload-artifact@v4
        with:
          name: ios-release
          path: curious-kellly/mobile/build/ios/iphoneos/Runner.app

  build-android:
    if: github.event.inputs.platform == 'android' || github.event.inputs.platform == 'both' || (github.event_name == 'push' && contains(github.event.head_commit.message, '[android]'))
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: subosito/flutter-action@v2
        with:
          flutter-version: '3.24.0'
      - name: Install dependencies
        run: |
          cd curious-kellly/mobile
          flutter pub get
      - name: Build Android APK
        run: |
          cd curious-kellly/mobile
          flutter build apk --release
      - name: Build Android AAB
        run: |
          cd curious-kellly/mobile
          flutter build appbundle --release
      - name: Upload Android artifacts
        uses: actions/upload-artifact@v4
        with:
          name: android-release
          path: |
            curious-kellly/mobile/build/app/outputs/flutter-apk/app-release.apk
            curious-kellly/mobile/build/app/outputs/bundle/release/*.aab
```

### Step 4.2: Configure App Store Connect (iOS)

1. Go to: https://appstoreconnect.apple.com
2. Navigate: **Users and Access** → **Keys** → **App Store Connect API**
3. Click **"Generate API Key"**
4. **Key name:** `curiouskelly-ci`
5. **Access:** App Manager
6. Click **"Generate"**
7. Download `.p8` key file
8. Note:
   - **Key ID** (e.g., `ABC123XYZ`)
   - **Issuer ID** (found in Keys page header)

9. Add to GitHub Secrets:
   - **`APPLE_KEY_ID`** = (Key ID from step 8)
   - **`APPLE_ISSUER_ID`** = (Issuer ID from step 8)
   - **`APPLE_KEY_CONTENT`** = (base64 encode the .p8 file):
     ```powershell
     [Convert]::ToBase64String([IO.File]::ReadAllBytes("path/to/key.p8"))
     ```

### Step 4.3: Configure Google Play (Android)

1. Go to: https://play.google.com/console
2. Navigate: **Setup** → **API access**
3. Click **"Create new service account"**
4. Follow Google Cloud Console link
5. Create service account:
   - **Name:** `curiouskelly-ci`
   - **Role:** Service Account User
6. Create JSON key:
   - Click service account → **Keys** → **Add Key** → **Create new key** → **JSON**
   - Download JSON file
7. Link to Play Console:
   - Back in Play Console → **Grant access**
   - Select service account
   - **Permissions:** Release apps, View app information
8. Add to GitHub Secrets:
   - **`GOOGLE_SERVICE_ACCOUNT_JSON`** = (paste entire JSON file content)

---

## Phase 5: Verification & Testing (15 minutes)

### Step 5.1: Test GitHub Workflows

1. Make a small change to trigger workflows:
   ```powershell
   # Create a test file
   echo "# Test" > test-deployment.md
   git add test-deployment.md
   git commit -m "Test: Trigger deployment workflows"
   git push origin main
   ```

2. Check workflow runs:
   - Go to: https://github.com/nicoletterankin/curiouskelly/actions
   - Verify workflows are running
   - Check for errors

### Step 5.2: Verify Cloudflare Deployment

1. Go to: https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c/pages
2. Click on `curiouskelly` project
3. Check **"Deployments"** tab
4. Verify latest deployment is successful
5. Visit: https://curiouskelly.com
6. Verify site loads correctly

### Step 5.3: Verify Vercel Deployment

1. Go to: https://vercel.com/lotd/curiouskelly-marketing
2. Check **"Deployments"** tab
3. Verify latest deployment is successful
4. Visit: https://curiouskelly.com (or Vercel preview URL)
5. Verify site loads correctly

### Step 5.4: Test Flutter Build

1. Go to: https://github.com/nicoletterankin/curiouskelly/actions
2. Click **"Flutter Build & Release"**
3. Click **"Run workflow"**
4. Select:
   - **Platform:** `android` (faster for testing)
   - **Release:** `false`
5. Click **"Run workflow"**
6. Wait for build to complete
7. Download artifacts to verify

---

## Troubleshooting

### GitHub Issues

**Problem:** "Repository not found"
- **Solution:** Verify repository name and permissions

**Problem:** "Permission denied"
- **Solution:** Check SSH keys or use HTTPS with personal access token

### Cloudflare Issues

**Problem:** "Invalid API token"
- **Solution:** Regenerate token with correct permissions

**Problem:** "DNS not resolving"
- **Solution:** Wait 24-48 hours for DNS propagation, check DNS records

**Problem:** "SSL certificate pending"
- **Solution:** Wait 5-10 minutes, check DNS records are correct

### Vercel Issues

**Problem:** "Build failed"
- **Solution:** Check build logs, verify `package.json` and dependencies

**Problem:** "Domain not connecting"
- **Solution:** Verify DNS records match Vercel instructions

### Flutter Issues

**Problem:** "iOS build fails"
- **Solution:** Verify Xcode version, check code signing setup

**Problem:** "Android build fails"
- **Solution:** Check `android/build.gradle`, verify Java version

---

## Next Steps

1. ✅ All deployments verified and working
2. 🔄 Set up monitoring and alerting
3. 🔄 Configure automated testing
4. 🔄 Set up staging environment
5. 🔄 Document API endpoints
6. 🔄 Create runbook for common operations

---

## Support & Resources

- **GitHub Docs:** https://docs.github.com
- **Cloudflare Pages Docs:** https://developers.cloudflare.com/pages
- **Vercel Docs:** https://vercel.com/docs
- **Flutter CI/CD:** https://docs.flutter.dev/deployment/ci-cd

---

**Last Updated:** 2025-01-11  
**Maintained By:** Senior Software Architect

