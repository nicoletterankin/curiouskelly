# Cloudflare Pages Git LFS Fix - Environment Variable

**Current Issue:** Build command has `GIT_LFS_SKIP_SMUDGE=1`, but LFS error happens during **clone phase** (before build runs).

**Solution:** Set `GIT_LFS_SKIP_SMUDGE` as an **environment variable** so it applies during git clone.

---

## Step-by-Step Fix

### Step 1: Add Environment Variable

1. In Cloudflare Pages project → **Settings** tab
2. Click **"Variables and Secrets"** in the left sub-navigation
3. Click **"Add variable"** button
4. Configure:
   - **Variable name:** `GIT_LFS_SKIP_SMUDGE`
   - **Value:** `1`
   - **Type:** Plain text (or Secret if you prefer)
   - **Environment:** 
     - ✅ Production
     - ✅ Preview
     - ✅ Branch previews
5. Click **"Save"**

### Step 2: Verify Build Command

1. Still in **Settings** → **Build** tab
2. Verify **Build command** is: `npm run build`
   - You can remove `GIT_LFS_SKIP_SMUDGE=1` from build command since it's now an environment variable
   - OR keep it as extra safety: `GIT_LFS_SKIP_SMUDGE=1 npm run build`

### Step 3: Force New Deployment

1. Go to **Deployments** tab
2. Click **"Retry deployment"** on the failed deployment
   - OR trigger a new deployment by pushing a commit
3. The environment variable will be available during the clone phase

---

## Why This Works

- **Environment variables** are available during ALL build phases (clone, install, build)
- **Build command variables** only apply during the build phase
- Setting `GIT_LFS_SKIP_SMUDGE=1` as an environment variable tells Git to skip LFS during clone

---

## Alternative: Update Build Command Format

If environment variables don't work, try updating the build command to:

```
GIT_LFS_SKIP_SMUDGE=1 git lfs install --skip-smudge && npm run build
```

This explicitly skips LFS before the build.

---

## Current Status

- ✅ Build command updated: `GIT_LFS_SKIP_SMUDGE=1 npm run build`
- ⚠️ Still failing because LFS error happens during clone (before build command runs)
- 🔧 Need to set `GIT_LFS_SKIP_SMUDGE` as environment variable
- ⚠️ Cloudflare cloning old commit `27b8385` instead of latest `5d0cf95`

---

## Quick Action

1. Go to: **Settings** → **Variables and Secrets**
2. Add variable: `GIT_LFS_SKIP_SMUDGE` = `1`
3. Apply to: Production, Preview, Branch previews
4. Click **"Save"**
5. Go to **Deployments** tab
6. Click **"Retry deployment"**

---

**Note:** The deployment is still using commit `27b8385` (old). After adding the environment variable and retrying, it should use the latest commit `5d0cf95` which doesn't have the video file.

