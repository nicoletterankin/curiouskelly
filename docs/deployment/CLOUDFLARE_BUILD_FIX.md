# Cloudflare Pages Build Fix - Git LFS Issue

**Problem:** Build fails because Git LFS objects don't exist on server.

**Root Cause:** `.gitattributes` file tracks many file types (wav, mp4, png) with Git LFS, but LFS objects aren't uploaded to GitHub.

**Solution:** Configure Cloudflare Pages to skip Git LFS during clone/build.

---

## Immediate Fix: Update Build Command

### In Cloudflare Pages Project Settings:

1. Go to your Pages project → **Settings** → **Builds & deployments**
2. Find **Build configuration**
3. Update **Build command** to:
   ```
   GIT_LFS_SKIP_SMUDGE=1 npm run build
   ```
   This tells Git to skip LFS file downloads during the build.

4. Click **Save**

5. Click **"Retry deployment"** or wait for automatic retry

---

## Alternative: Add Environment Variable

### In Cloudflare Pages Project Settings:

1. Go to **Settings** → **Environment variables**
2. Add new variable:
   - **Variable name:** `GIT_LFS_SKIP_SMUDGE`
   - **Value:** `1`
   - **Environment:** Production, Preview, Branch previews
3. Click **Save**

---

## Why This Works

- `GIT_LFS_SKIP_SMUDGE=1` tells Git to skip downloading LFS files
- LFS files are replaced with pointer files (small text files)
- Build process doesn't need the actual media files
- Only the code and assets needed for build are used

---

## Current Status

- ✅ Latest commit (`5d0cf95`) removes problematic video file
- ✅ Video files added to `.gitignore`
- ⚠️ Many other LFS files still tracked (images, audio)
- 🔧 Need to configure Cloudflare to skip LFS

---

## Next Steps

1. **Update build command** in Cloudflare Pages settings
2. **Retry deployment**
3. Build should succeed without LFS errors

---

**Quick Action:** 
Go to Cloudflare Pages project → Settings → Builds & deployments → Update build command to `GIT_LFS_SKIP_SMUDGE=1 npm run build`

