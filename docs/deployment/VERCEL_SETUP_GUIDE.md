# Vercel Setup Guide - Step by Step

**Complete guide to deploy the marketing site (`daily-lesson-marketing`) to Vercel.**

---

## 🎯 What We're Deploying

- **Project:** Marketing site (Astro)
- **Location:** `daily-lesson-marketing/`
- **Framework:** Astro
- **Domain:** `curiouskelly.com` (root domain)
- **Purpose:** Homepage, marketing pages, lead forms

---

## 📋 Prerequisites

Before starting, make sure you have:
- ✅ GitHub repository: `nicoletterankin/curiouskelly`
- ✅ Vercel account (free tier works)
- ✅ Access to `curiouskelly.com` domain DNS settings

---

## 🚀 Step 1: Create Vercel Account & Team

### 1.1 Sign Up / Log In
1. Go to **https://vercel.com**
2. Click **"Sign Up"** (or **"Log In"** if you have an account)
3. Sign up with **GitHub** (recommended for easy integration)

### 1.2 Create or Join Team
1. After logging in, you'll see your dashboard
2. If you need a team:
   - Click **"Add Team"** → **"Create Team"**
   - Name: `Lotd` (or your team name)
   - Click **"Create"**

---

## 🔗 Step 2: Connect GitHub Repository

### 2.1 Import Project from GitHub
1. In Vercel dashboard, click **"Add New..."** → **"Project"**
2. Click **"Import Git Repository"**
3. Find and select: **`nicoletterankin/curiouskelly`**
4. Click **"Import"**

### 2.2 Configure Project Settings
Vercel will auto-detect the project. Configure:

**Project Name:**
- Enter: `curiouskelly-marketing` (or your preferred name)

**Framework Preset:**
- Select: **Astro** (should auto-detect)

**Root Directory:**
- Click **"Edit"** next to Root Directory
- Select: **`daily-lesson-marketing`**
- Click **"Continue"**

**Build Settings:**
- **Build Command:** `npm run build` (should auto-fill)
- **Output Directory:** `dist` (should auto-fill)
- **Install Command:** `npm install` (should auto-fill)

**Environment Variables:**
- We'll add these in Step 4
- Click **"Deploy"** for now (we'll add env vars after)

---

## ⚙️ Step 3: Get Vercel API Credentials

You need these for GitHub Actions automation:

### 3.1 Get Vercel Token
1. Go to **Vercel Dashboard** → Click your **profile icon** (top right)
2. Click **"Settings"**
3. Go to **"Tokens"** tab
4. Click **"Create Token"**
5. Name: `GitHub Actions - CuriousKelly`
6. Expiration: **No expiration** (or set a long date)
7. Click **"Create"**
8. **COPY THE TOKEN** (you won't see it again!)
   - Example: `vercel_xxxxxxxxxxxxxxxxxxxxx`

### 3.2 Get Organization ID
1. In Vercel Dashboard, go to **Settings** → **General**
2. Find **"Team ID"** (or "Organization ID")
3. **COPY IT**
   - Example: `team_xxxxxxxxxxxxxxxxxxxxx`

### 3.3 Get Project ID
1. Go to your project: **`curiouskelly-marketing`**
2. Go to **Settings** → **General**
3. Find **"Project ID"**
4. **COPY IT**
   - Example: `prj_xxxxxxxxxxxxxxxxxxxxx`

---

## 🔐 Step 4: Add GitHub Secrets

Add the Vercel credentials to GitHub so GitHub Actions can deploy:

### 4.1 Go to GitHub Secrets
1. Go to: **https://github.com/nicoletterankin/curiouskelly**
2. Click **"Settings"** tab
3. Click **"Secrets and variables"** → **"Actions"**
4. Click **"New repository secret"**

### 4.2 Add Each Secret

**Secret 1: VERCEL_TOKEN**
- **Name:** `VERCEL_TOKEN`
- **Value:** Paste your Vercel token from Step 3.1
- Click **"Add secret"**

**Secret 2: VERCEL_ORG_ID**
- **Name:** `VERCEL_ORG_ID`
- **Value:** Paste your Team/Org ID from Step 3.2
- Click **"Add secret"**

**Secret 3: VERCEL_PROJECT_ID**
- **Name:** `VERCEL_PROJECT_ID`
- **Value:** Paste your Project ID from Step 3.3
- Click **"Add secret"**

---

## 🌍 Step 5: Configure Environment Variables

### 5.1 In Vercel Dashboard
1. Go to your project: **`curiouskelly-marketing`**
2. Click **"Settings"** tab
3. Click **"Environment Variables"**

### 5.2 Add Required Variables

Add these for **Production**, **Preview**, and **Development**:

**PUBLIC_SITE_URL**
- **Key:** `PUBLIC_SITE_URL`
- **Value:** `https://curiouskelly.com`
- **Environments:** ✅ Production ✅ Preview ✅ Development

**PUBLIC_DEFAULT_LOCALE**
- **Key:** `PUBLIC_DEFAULT_LOCALE`
- **Value:** `en-US`
- **Environments:** ✅ Production ✅ Preview ✅ Development

**PUBLIC_AVAILABLE_LOCALES**
- **Key:** `PUBLIC_AVAILABLE_LOCALES`
- **Value:** `en-US,es-ES,pt-BR`
- **Environments:** ✅ Production ✅ Preview ✅ Development

**Optional (if you have them):**
- `TURNSTILE_SITE_KEY` - Cloudflare Turnstile site key
- `TURNSTILE_SECRET_KEY` - Cloudflare Turnstile secret key
- `CRM_WEBHOOK_URL` - CRM webhook URL for lead forms
- `CRM_AUTH_TOKEN` - CRM authentication token
- `ELEVENLABS_API_KEY` - ElevenLabs API key
- `STRIPE_SECRET_KEY` - Stripe secret key
- `ANALYTICS_ID` - Analytics tracking ID

**For each variable:**
1. Click **"Add New"**
2. Enter **Key** and **Value**
3. Select **environments** (Production, Preview, Development)
4. Click **"Save"**

---

## 🌐 Step 6: Add Custom Domain

### 6.1 Add Domain in Vercel
1. In your project: **`curiouskelly-marketing`**
2. Go to **Settings** → **Domains**
3. Click **"Add Domain"**
4. Enter: `curiouskelly.com`
5. Click **"Add"**

### 6.2 Configure DNS Records

Vercel will show you DNS records to add. You need to add these in **Cloudflare DNS**:

**Option A: Root Domain (curiouskelly.com)**
- **Type:** `A`
- **Name:** `@` (or `curiouskelly.com`)
- **Value:** `76.76.21.21` (Vercel's IP - check Vercel dashboard for current IP)
- **Proxy:** ✅ Proxied (orange cloud)

**Option B: CNAME (Recommended)**
- **Type:** `CNAME`
- **Name:** `@` (or `curiouskelly.com`)
- **Value:** `cname.vercel-dns.com` (check Vercel dashboard for exact value)
- **Proxy:** ✅ Proxied (orange cloud)

**Option C: Subdomain (if using subdomain)**
- **Type:** `CNAME`
- **Name:** `www`
- **Value:** `cname.vercel-dns.com`
- **Proxy:** ✅ Proxied

### 6.3 Verify Domain
1. After adding DNS records, go back to Vercel
2. Click **"Refresh"** next to your domain
3. Wait 1-5 minutes for DNS propagation
4. Status should change to **"Valid Configuration"**
5. SSL certificate will be issued automatically

---

## 🚀 Step 7: Trigger First Deployment

### Option A: Manual Deploy
1. In Vercel dashboard, go to your project
2. Click **"Deployments"** tab
3. Click **"Redeploy"** → **"Redeploy"** (if there's an existing deployment)
4. Or push a commit to trigger auto-deploy

### Option B: Push to GitHub (Auto-Deploy)
1. Make a small change to `daily-lesson-marketing/`
2. Commit and push:
   ```bash
   git add daily-lesson-marketing/
   git commit -m "Trigger Vercel deployment"
   git push origin main
   ```
3. Vercel will automatically deploy

### Option C: Use GitHub Actions
1. Go to GitHub → **Actions** tab
2. Find **"Deploy to Vercel"** workflow
3. Click **"Run workflow"** → **"Run workflow"**
4. Select **environment:** `preview` or `production`
5. Click **"Run workflow"**

---

## ✅ Step 8: Verify Deployment

### 8.1 Check Deployment Status
1. In Vercel dashboard → **Deployments** tab
2. Find your latest deployment
3. Status should be **"Ready"** (green checkmark)
4. Click the deployment to see details

### 8.2 Test the Site
1. Click the **deployment URL** (e.g., `curiouskelly-marketing-xxx.vercel.app`)
2. Or visit: `https://curiouskelly.com` (if domain is configured)
3. Verify:
   - ✅ Homepage loads
   - ✅ No console errors
   - ✅ Forms work (if applicable)
   - ✅ Links work

### 8.3 Check Build Logs
1. In deployment details, click **"Build Logs"**
2. Verify:
   - ✅ Build completed successfully
   - ✅ No errors or warnings
   - ✅ Output directory `dist` was created

---

## 🔄 Step 9: Configure Auto-Deployments

### 9.1 GitHub Integration (Already Done)
- ✅ Repository is connected
- ✅ Auto-deploys on push to `main`
- ✅ Preview deployments for pull requests

### 9.2 Branch Settings
1. Go to **Settings** → **Git**
2. **Production Branch:** `main`
3. **Preview Deployments:** ✅ Enabled
4. **Pull Request Comments:** ✅ Enabled (optional)

---

## 🎨 Step 10: Test the Full Flow

### 10.1 Make a Test Change
1. Edit a file in `daily-lesson-marketing/src/pages/`
2. Commit and push:
   ```bash
   git add daily-lesson-marketing/
   git commit -m "Test: Vercel auto-deploy"
   git push origin main
   ```

### 10.2 Watch Deployment
1. Go to Vercel dashboard → **Deployments**
2. You should see a new deployment starting
3. Wait 1-3 minutes for build to complete
4. Status should be **"Ready"**

### 10.3 Verify Changes
1. Visit your site URL
2. Verify your changes are live
3. Check deployment logs if issues occur

---

## 🐛 Troubleshooting

### Issue: Build Fails
**Symptoms:** Deployment shows "Build Failed"

**Solutions:**
1. Check **Build Logs** in Vercel dashboard
2. Common issues:
   - Missing dependencies → Check `package.json`
   - Build command wrong → Verify in Settings → General
   - Environment variables missing → Check Settings → Environment Variables
   - Node version mismatch → Check `package.json` engines

### Issue: Domain Not Working
**Symptoms:** `curiouskelly.com` shows error or doesn't load

**Solutions:**
1. Check DNS records in Cloudflare:
   - Verify A/CNAME record points to Vercel
   - Check proxy status (orange cloud)
2. In Vercel → Settings → Domains:
   - Verify domain shows "Valid Configuration"
   - Check SSL certificate status
3. Wait 5-10 minutes for DNS propagation

### Issue: GitHub Actions Not Deploying
**Symptoms:** GitHub Actions workflow fails or doesn't run

**Solutions:**
1. Check GitHub Secrets:
   - Verify `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID` are set
2. Check workflow file:
   - Verify `.github/workflows/deploy-vercel.yml` exists
   - Check path filters match your directory structure
3. Check Vercel project settings:
   - Verify project name matches
   - Verify root directory is `daily-lesson-marketing`

### Issue: Environment Variables Not Loading
**Symptoms:** Site works but env vars are undefined

**Solutions:**
1. In Vercel → Settings → Environment Variables:
   - Verify variables are set for correct environments
   - Check variable names match code (case-sensitive)
   - Redeploy after adding variables
2. In code:
   - Use `import.meta.env.PUBLIC_*` for public vars
   - Use `process.env.*` for server-side vars

---

## 📊 Step 11: Monitor & Optimize

### 11.1 Enable Analytics
1. Go to **Settings** → **Analytics**
2. Enable **Web Vitals**
3. Enable **Speed Insights** (if available)

### 11.2 Review Performance
1. Go to **Analytics** tab
2. Check:
   - Page load times
   - Core Web Vitals
   - Error rates

### 11.3 Set Up Notifications
1. Go to **Settings** → **Notifications**
2. Enable:
   - ✅ Deployment notifications (email/Slack)
   - ✅ Error notifications
   - ✅ Performance alerts

---

## ✅ Completion Checklist

- [ ] Vercel account created
- [ ] Team created (if needed)
- [ ] GitHub repository connected
- [ ] Project configured (root directory: `daily-lesson-marketing`)
- [ ] Vercel token created
- [ ] Organization ID noted
- [ ] Project ID noted
- [ ] GitHub Secrets added (`VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`)
- [ ] Environment variables configured
- [ ] Custom domain added (`curiouskelly.com`)
- [ ] DNS records configured in Cloudflare
- [ ] Domain verified in Vercel
- [ ] First deployment successful
- [ ] Site accessible at `https://curiouskelly.com`
- [ ] Auto-deployments working
- [ ] Build logs clean
- [ ] Analytics enabled (optional)

---

## 🎉 Success!

Your marketing site is now deployed to Vercel! 

**Next Steps:**
1. Test all pages and forms
2. Configure Cloudflare DNS to route `curiouskelly.com` → Vercel
3. Set up monitoring and alerts
4. Review deployment logs regularly

---

## 📚 Additional Resources

- **Vercel Docs:** https://vercel.com/docs
- **Astro + Vercel:** https://docs.astro.build/en/guides/integrations-guide/vercel/
- **GitHub Actions:** https://docs.github.com/actions
- **Cloudflare DNS:** https://developers.cloudflare.com/dns/

---

**Last Updated:** 2025-01-13  
**Status:** ✅ Ready for deployment

