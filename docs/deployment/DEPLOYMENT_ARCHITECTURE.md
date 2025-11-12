# CuriousKelly.com - Complete Deployment Architecture

**Last Updated:** 2025-01-11  
**Status:** Production Deployment Plan  
**Domain:** curiouskelly.com  
**Primary Services:** GitHub, Cloudflare, Vercel, Flutter

---

## 📸 Screenshot Analysis & Current State

### 1. GitHub Repository Setup

**Repository Details (from screenshot):**
- **Repository Name:** `curiouskelly`
- **Owner:** `nicoletterankin`
- **Description:** "Striving to be the best digital teacher in the world."
- **Visibility:** Public
- **Initial Setup:** No README, .gitignore, or license (to be configured)

**Current Codebase Location:** `C:\Users\user\UI-TARS-desktop`

**Action Required:**
1. Initialize Git repository in current directory
2. Create GitHub repository `nicoletterankin/curiouskelly`
3. Push codebase to GitHub
4. Configure branch protection and workflows

---

### 2. Cloudflare Configuration

**Domain Details (from screenshot):**
- **Domain:** `curiouskelly.com`
- **Account:** `Nicoletterankin@gmail.com's Account`
- **Account ID:** `47ebb2a1adc311cb106acc89720e352c`
- **Zone ID:** `510107ca53356bab42f8a8d1b2de1e59`
- **Plan:** Free plan
- **Registrar:** Cloudflare
- **Expires:** November 11, 2026
- **Status:** Active

**API Tokens (from screenshot):**
1. **Cloudflare Tunnel API Token** (for ilearnhow.com)
   - Permissions: `Account.Cloudflare One Network...`
   - Resources: `1 Account, 1 Zone`

2. **dailylesson-foundry build token**
   - Permissions: `Account.Containers, Account.Se...`
   - Resources: `1 Account, nicoletterankin@gmail.com, All zones`

3. **dailylesson-r2-storage**
   - Permissions: `Account.Workers R2 Storage`
   - Resources: `All accounts`

4. **ilearn_how**
   - Permissions: `Account.Cloudflare Pages`
   - Resources: `All accounts`

5. **ilearnhow-pages-deploy**
   - Permissions: `Account.Cloudflare Pages`
   - Resources: `1 Account`

6. **mynextlesson-cache-purge** (2 tokens)
   - Permissions: `Zone.Zone`
   - Resources: `1 Zone`

7. **R2 User Token** (2 instances)
   - Permissions: `Account.Workers R2 Data Catalog...`
   - Resources: `1 Account`

8. **Workers AI**
   - Permissions: `Account.Workers AI, Account.W...`
   - Resources: `1 Account`

**Security Features (from screenshot):**
- **Page Shield:** OFF (needs activation)
- **Bot Fight Mode:** OFF (needs activation)
- **Leaked credentials detection:** Available (needs activation)

**Action Required:**
1. Create new API token for `curiouskelly.com` Pages deployment
2. Configure Cloudflare Pages project
3. Set up DNS records
4. Enable security features
5. Configure R2 storage for assets

---

### 3. Vercel Configuration

**Team Details (from screenshot):**
- **Team Name:** `Lotd`
- **Plan:** Hobby
- **Current State:** No projects deployed
- **Domain Modal:** Shows "No projects found" when trying to add domain

**Action Required:**
1. Create Vercel project for marketing site
2. Connect GitHub repository
3. Configure build settings
4. Add `curiouskelly.com` domain
5. Set environment variables

---

## 🏗️ Deployment Architecture

### Service Mapping

```
┌─────────────────────────────────────────────────────────────┐
│                    curiouskelly.com                         │
│                  (Cloudflare DNS)                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Marketing   │ │  Lesson      │ │  Mobile Apps │
│  Site        │ │  Player       │ │  (Flutter)   │
│              │ │              │ │              │
│  Vercel      │ │  Cloudflare  │ │  GitHub      │
│  Pages       │ │  Pages       │ │  Actions     │
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                ┌───────▼────────┐
                │   GitHub Repo  │
                │  nicoletterankin│
                │  /curiouskelly │
                └────────────────┘
```

### Component Breakdown

#### 1. Marketing Site (`curiouskelly-marketing-site/`)
- **Platform:** Vercel
- **Framework:** Astro
- **Domain:** `curiouskelly.com` (root)
- **Build Command:** `npm run build`
- **Output Directory:** `dist`
- **Environment:** Production

#### 2. Lesson Player (`lesson-player/`)
- **Platform:** Cloudflare Pages
- **Framework:** Static HTML/JS
- **Domain:** `curiouskelly.com/lesson-player`
- **Build Command:** None (static files)
- **Output Directory:** `lesson-player/`

#### 3. Backend API (`curious-kellly/backend/`)
- **Platform:** Cloudflare Workers or Vercel Functions
- **Framework:** Node.js/Express or Python/FastAPI
- **Domain:** `curiouskelly.com/api/*`
- **Functions:** API routes for lessons, analytics, etc.

#### 4. Mobile Apps (`curious-kellly/mobile/` & `digital-kelly/apps/kelly_app_flutter/`)
- **Platform:** GitHub Actions → App Store Connect / Google Play
- **Framework:** Flutter
- **CI/CD:** Automated builds and releases
- **Distribution:** iOS App Store, Google Play Store

---

## 🔧 Integration Setup Steps

### Phase 1: GitHub Repository Setup

#### Step 1.1: Initialize Git Repository
```powershell
# Navigate to project root
cd C:\Users\user\UI-TARS-desktop

# Initialize git (if not already)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: CuriousKelly.com production codebase"
```

#### Step 1.2: Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `curiouskelly`
3. Description: "Striving to be the best digital teacher in the world."
4. Visibility: Public
5. **DO NOT** initialize with README, .gitignore, or license (we have these)
6. Click "Create repository"

#### Step 1.3: Connect Local Repository
```powershell
# Add remote
git remote add origin https://github.com/nicoletterankin/curiouskelly.git

# Push to GitHub
git branch -M main
git push -u origin main
```

#### Step 1.4: Configure Branch Protection
1. Go to Settings → Branches
2. Add rule for `main` branch:
   - Require pull request reviews
   - Require status checks to pass
   - Require branches to be up to date
   - Include administrators

#### Step 1.5: Set Up GitHub Secrets
Go to Settings → Secrets and variables → Actions, add:

**For Vercel:**
- `VERCEL_TOKEN` - Vercel API token
- `VERCEL_ORG_ID` - Team ID (Lotd)
- `VERCEL_PROJECT_ID` - Project ID (after creating project)

**For Cloudflare:**
- `CLOUDFLARE_API_TOKEN` - API token for Pages deployment
- `CLOUDFLARE_ACCOUNT_ID` - `47ebb2a1adc311cb106acc89720e352c`
- `CLOUDFLARE_ZONE_ID` - `510107ca53356bab42f8a8d1b2de1e59`
- `CLOUDFLARE_PROJECT_NAME` - `curiouskelly` (to be created)

**For Flutter:**
- `APPLE_ID` - Apple Developer account email
- `APPLE_APP_SPECIFIC_PASSWORD` - App-specific password
- `APPLE_TEAM_ID` - Apple Developer Team ID
- `GOOGLE_SERVICE_ACCOUNT_JSON` - Google Play service account key

---

### Phase 2: Cloudflare Pages Setup

#### Step 2.1: Create API Token
1. Go to Cloudflare Dashboard → Profile → API Tokens
2. Click "Create Token"
3. Use "Edit Cloudflare Workers" template
4. Permissions:
   - Account: `Cloudflare Pages:Edit`
   - Zone: `Zone:Read`
5. Account Resources: `All accounts` or specific account
6. Zone Resources: `curiouskelly.com`
7. Name: `curiouskelly-pages-deploy`
8. Create token and copy (save securely)

#### Step 2.2: Create Cloudflare Pages Project
1. Go to Cloudflare Dashboard → Pages
2. Click "Create a project"
3. Connect to Git → GitHub → `nicoletterankin/curiouskelly`
4. Project name: `curiouskelly`
5. Production branch: `main`
6. Build configuration:
   - **Framework preset:** None (or Custom)
   - **Build command:** (leave empty for static, or `npm run build` if needed)
   - **Build output directory:** `lesson-player` (for lesson player) or `dist` (for marketing site)
   - **Root directory:** `/` (or specific subdirectory)

#### Step 2.3: Configure Custom Domain
1. In Pages project → Custom domains
2. Add domain: `curiouskelly.com`
3. Cloudflare will automatically configure DNS
4. SSL/TLS: Automatic (Full)

#### Step 2.4: Set Environment Variables
In Pages project → Settings → Environment variables, add:
- `ELEVENLABS_API_KEY`
- `STRIPE_SECRET_KEY`
- `ANALYTICS_ID`
- `PUBLIC_SITE_URL` = `https://curiouskelly.com`

#### Step 2.5: Enable Security Features
1. Go to Domain → Security
2. Enable **Page Shield** (monitor third-party scripts)
3. Enable **Bot Fight Mode** (challenge known bots)
4. Activate **Leaked credentials detection**

#### Step 2.6: Configure R2 Storage (for assets)
1. Go to R2 → Create bucket
2. Bucket name: `curiouskelly-assets`
3. Create API token with R2 read/write permissions
4. Configure CORS for web access

---

### Phase 3: Vercel Setup

#### Step 3.1: Install Vercel CLI
```powershell
npm install -g vercel
```

#### Step 3.2: Login to Vercel
```powershell
vercel login
```

#### Step 3.3: Link Project
```powershell
# Navigate to marketing site directory
cd curiouskelly-marketing-site
# Or if using root:
cd .

# Link to Vercel team
vercel link
# Select team: Lotd
# Project name: curiouskelly-marketing
```

#### Step 3.4: Create Vercel Project via Dashboard
1. Go to https://vercel.com/lotd
2. Click "Add New..." → "Project"
3. Import Git Repository → `nicoletterankin/curiouskelly`
4. Configure project:
   - **Framework Preset:** Astro
   - **Root Directory:** `curiouskelly-marketing-site` (or root if single project)
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
   - **Install Command:** `npm install`
5. Add environment variables (see below)
6. Deploy

#### Step 3.5: Add Custom Domain
1. Go to Project → Settings → Domains
2. Add domain: `curiouskelly.com` (or `www.curiouskelly.com`)
3. Configure DNS as instructed by Vercel
4. Wait for SSL certificate provisioning

#### Step 3.6: Set Environment Variables
In Project → Settings → Environment Variables, add:

**Production:**
- `PUBLIC_SITE_URL` = `https://curiouskelly.com`
- `PUBLIC_DEFAULT_LOCALE` = `en-US`
- `PUBLIC_AVAILABLE_LOCALES` = `en-US,es-ES,pt-BR`
- `TURNSTILE_SITE_KEY`
- `TURNSTILE_SECRET_KEY`
- `CRM_WEBHOOK_URL`
- `CRM_AUTH_TOKEN`
- `ELEVENLABS_API_KEY`
- `STRIPE_SECRET_KEY`
- `ANALYTICS_ID`

**Preview/Development:**
- Same variables with different values for testing

---

### Phase 4: Flutter Mobile Apps CI/CD

#### Step 4.1: Create GitHub Actions Workflow
Create `.github/workflows/flutter-build.yml`:

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
      - name: Archive IPA
        run: |
          cd curious-kellly/mobile/ios
          xcodebuild -workspace Runner.xcworkspace \
            -scheme Runner \
            -configuration Release \
            -archivePath build/Runner.xcarchive \
            archive
      - name: Export IPA
        run: |
          xcodebuild -exportArchive \
            -archivePath curious-kellly/mobile/ios/build/Runner.xcarchive \
            -exportPath build/ios \
            -exportOptionsPlist exportOptions.plist

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
      - name: Build Android APK/AAB
        run: |
          cd curious-kellly/mobile
          flutter build appbundle --release
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: android-release
          path: curious-kellly/mobile/build/app/outputs/bundle/release/*.aab
```

#### Step 4.2: Configure App Store Connect API
1. Go to App Store Connect → Users and Access → Keys
2. Create new key with "App Manager" role
3. Download `.p8` key file
4. Add to GitHub Secrets:
   - `APPLE_KEY_ID`
   - `APPLE_ISSUER_ID`
   - `APPLE_KEY_CONTENT` (base64 encoded .p8 file)

#### Step 4.3: Configure Google Play API
1. Go to Google Play Console → Setup → API access
2. Create service account
3. Download JSON key file
4. Add to GitHub Secrets as `GOOGLE_SERVICE_ACCOUNT_JSON`

---

## 🔄 Deployment Workflows

### Marketing Site (Vercel)
- **Trigger:** Push to `main` branch
- **Build:** `npm run build` in `curiouskelly-marketing-site/`
- **Deploy:** Automatic via Vercel GitHub integration
- **Domain:** `curiouskelly.com` (root)

### Lesson Player (Cloudflare Pages)
- **Trigger:** Push to `main` branch
- **Build:** Static files from `lesson-player/`
- **Deploy:** Automatic via Cloudflare Pages GitHub integration
- **Domain:** `curiouskelly.com/lesson-player`

### Backend API (Cloudflare Workers)
- **Trigger:** Push to `main` branch
- **Build:** `wrangler publish` or GitHub Actions
- **Deploy:** Cloudflare Workers
- **Domain:** `curiouskelly.com/api/*`

### Mobile Apps (GitHub Actions)
- **Trigger:** Manual workflow dispatch or tagged releases
- **Build:** Flutter build commands
- **Deploy:** Upload to App Store Connect / Google Play Console
- **Distribution:** App Store / Play Store

---

## 🔐 Security & Environment Variables

### Required Secrets

**GitHub Secrets:**
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`
- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_ZONE_ID`
- `CLOUDFLARE_PROJECT_NAME`
- `APPLE_KEY_ID`
- `APPLE_ISSUER_ID`
- `APPLE_KEY_CONTENT`
- `GOOGLE_SERVICE_ACCOUNT_JSON`

**Vercel Environment Variables:**
- `PUBLIC_SITE_URL`
- `TURNSTILE_SITE_KEY`
- `TURNSTILE_SECRET_KEY`
- `CRM_WEBHOOK_URL`
- `ELEVENLABS_API_KEY`
- `STRIPE_SECRET_KEY`
- `ANALYTICS_ID`

**Cloudflare Environment Variables:**
- `ELEVENLABS_API_KEY`
- `STRIPE_SECRET_KEY`
- `ANALYTICS_ID`
- `PUBLIC_SITE_URL`

---

## 📊 Monitoring & Analytics

### Cloudflare Analytics
- **RUM (Real User Monitoring):** Enable via `PUBLIC_RUM_ENABLED=true`
- **Web Analytics:** Available in Cloudflare dashboard
- **Page Shield:** Monitor third-party scripts

### Vercel Analytics
- **Web Vitals:** Automatic with Vercel deployment
- **Speed Insights:** Enable in project settings

### Application Monitoring
- **Error Tracking:** Firebase Crashlytics (Flutter apps)
- **Performance:** Firebase Performance Monitoring
- **Analytics:** Firebase Analytics / Google Analytics 4

---

## 🚀 Quick Start Deployment Checklist

### Pre-Deployment
- [ ] Git repository initialized and pushed to GitHub
- [ ] GitHub repository `nicoletterankin/curiouskelly` created
- [ ] Branch protection rules configured
- [ ] GitHub Secrets added

### Cloudflare Setup
- [ ] API token created for Pages deployment
- [ ] Cloudflare Pages project created and connected to GitHub
- [ ] Custom domain `curiouskelly.com` added
- [ ] DNS records configured
- [ ] SSL/TLS enabled (Automatic)
- [ ] Environment variables set
- [ ] Security features enabled (Page Shield, Bot Fight Mode)
- [ ] R2 bucket created for assets

### Vercel Setup
- [ ] Vercel CLI installed and logged in
- [ ] Project created and linked to GitHub
- [ ] Build configuration set
- [ ] Custom domain added
- [ ] Environment variables configured
- [ ] First deployment successful

### Flutter CI/CD
- [ ] GitHub Actions workflow created
- [ ] App Store Connect API key configured
- [ ] Google Play service account configured
- [ ] Test build successful
- [ ] Release process documented

### Post-Deployment Verification
- [ ] Marketing site accessible at `curiouskelly.com`
- [ ] Lesson player accessible at `curiouskelly.com/lesson-player`
- [ ] API endpoints responding
- [ ] SSL certificates valid
- [ ] Analytics tracking working
- [ ] Mobile apps building successfully
- [ ] All environment variables loaded correctly

---

## 📝 Next Steps

1. **Immediate:** Set up GitHub repository and initial push
2. **Short-term:** Configure Cloudflare Pages and Vercel projects
3. **Medium-term:** Set up Flutter CI/CD pipeline
4. **Long-term:** Implement monitoring, alerting, and automated testing

---

## 🔗 Important Links

- **GitHub Repository:** https://github.com/nicoletterankin/curiouskelly
- **Cloudflare Dashboard:** https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c/curiouskelly.com
- **Vercel Dashboard:** https://vercel.com/lotd
- **Domain:** https://curiouskelly.com

---

## 📚 Related Documentation

- `CLAUDE.md` - Operating rules and guidelines
- `CURIOUS_KELLLY_EXECUTION_PLAN.md` - Project roadmap
- `TECHNICAL_ALIGNMENT_MATRIX.md` - Technical specifications
- `BUILD_PLAN.md` - Build and deployment procedures
- `deployment/setup-cloud.sh` - Cloud setup script
- `.github/workflows/` - CI/CD workflows

---

**Document Maintained By:** Senior Software Architect  
**Last Review:** 2025-01-11  
**Next Review:** After initial deployment

