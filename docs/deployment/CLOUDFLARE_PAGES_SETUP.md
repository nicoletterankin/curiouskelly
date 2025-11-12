# Cloudflare Pages Project Setup - Step-by-Step

**Current Status:** All GitHub Secrets Added ✅  
**Next Step:** Create Cloudflare Pages Project

---

## Step-by-Step Instructions

### Step 1: Navigate to Create Application

1. You're currently on: https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c/workers-and-pages
2. Click the blue **"Create application"** button (top right of the main content area)

### Step 2: Select Pages

1. You'll see options for "Workers" and "Pages"
2. Click on **"Pages"** tab or section
3. Click **"Connect to Git"** button

### Step 3: Connect GitHub Repository

1. **Select Git provider:** Choose **"GitHub"**
2. If prompted, authorize Cloudflare to access your GitHub account
3. **Select repository:** Find and select **`nicoletterankin/curiouskelly`**
4. Click **"Begin setup"** or **"Continue"**

### Step 4: Configure Project Settings

1. **Project name:** `curiouskelly`
   - This should auto-populate from the repository name

2. **Production branch:** `main`
   - This should be the default

3. **Build settings:**
   
   **Option A: For Lesson Player (Static Files)**
   - **Framework preset:** None
   - **Build command:** (leave empty)
   - **Build output directory:** `lesson-player`
   - **Root directory:** `/` (or leave empty)

   **Option B: For Marketing Site (Astro)**
   - **Framework preset:** Astro
   - **Build command:** `npm run build` (auto-detected)
   - **Build output directory:** `dist` (auto-detected)
   - **Root directory:** `curiouskelly-marketing-site` (if marketing site is in subdirectory)

   **Option C: Root Project Build**
   - **Framework preset:** None or Custom
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
   - **Root directory:** `/` (or leave empty)

4. **Environment variables (optional):**
   - You can add these now or later in project settings
   - `ELEVENLABS_API_KEY`
   - `STRIPE_SECRET_KEY`
   - `ANALYTICS_ID`
   - `PUBLIC_SITE_URL` = `https://curiouskelly.com`

### Step 5: Save and Deploy

1. Review all settings
2. Click **"Save and Deploy"** button
3. Cloudflare will:
   - Clone your repository
   - Run the build (if configured)
   - Deploy to a `.pages.dev` subdomain
   - Show deployment progress

### Step 6: Wait for Initial Deployment

1. You'll see a deployment in progress
2. This may take 2-5 minutes
3. Once complete, you'll see:
   - Deployment status (Success/Failed)
   - Preview URL (e.g., `curiouskelly.pages.dev`)
   - Deployment logs

---

## Post-Deployment: Configure Custom Domain

### Step 7: Add Custom Domain

1. In the Pages project, go to **"Custom domains"** tab
2. Click **"Set up a custom domain"**
3. Enter: `curiouskelly.com`
4. Click **"Continue"**
5. Cloudflare will automatically:
   - Configure DNS records
   - Provision SSL certificate
   - Set up routing

### Step 8: Verify DNS Configuration

1. Cloudflare will show DNS records to add (if not automatic)
2. Usually these are auto-configured
3. Wait 1-5 minutes for DNS propagation
4. SSL certificate will be issued automatically (usually 1-5 minutes)

### Step 9: Test Deployment

1. Visit: `https://curiouskelly.com` (or the `.pages.dev` URL)
2. Verify the site loads correctly
3. Check deployment logs for any errors

---

## Troubleshooting

### Issue: Build Fails
**Solution:**
- Check build logs in Cloudflare dashboard
- Verify build command and output directory
- Ensure `package.json` exists if using npm build
- Check for missing dependencies

### Issue: "Repository not found"
**Solution:**
- Verify GitHub authorization
- Check repository visibility (should be public or Cloudflare has access)
- Re-authorize GitHub connection if needed

### Issue: DNS Not Resolving
**Solution:**
- Wait 5-10 minutes for DNS propagation
- Check DNS records in Cloudflare dashboard
- Verify domain is using Cloudflare nameservers

### Issue: SSL Certificate Pending
**Solution:**
- Wait 5-10 minutes
- Check DNS records are correct
- Verify domain is active in Cloudflare

---

## Next Steps After Pages Project Creation

1. ✅ Cloudflare Pages project created
2. ✅ Custom domain configured
3. ⏭️ Set environment variables in Pages project
4. ⏭️ Enable Cloudflare security features
5. ⏭️ Create Vercel project for marketing site
6. ⏭️ Add Vercel secrets to GitHub

---

## Quick Reference

- **Cloudflare Pages Dashboard:** https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c/pages
- **Account ID:** `47ebb2a1adc311cb106acc89720e352c`
- **Zone ID:** `510107ca53356bab42f8a8d1b2de1e59`
- **Project Name:** `curiouskelly`
- **Domain:** `curiouskelly.com`

---

**Last Updated:** 2025-01-11

