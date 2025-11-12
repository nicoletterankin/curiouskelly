# Fixing Git LFS Issues for Cloudflare Pages

**Problem:** Cloudflare Pages build fails because Git LFS objects don't exist on the server.

**Solution:** Configure Cloudflare Pages to skip Git LFS files during build.

---

## Option 1: Configure Build Environment Variable (Recommended)

### In Cloudflare Pages Project Settings:

1. Go to your Pages project → **Settings** → **Environment variables**
2. Add a new variable:
   - **Variable name:** `GIT_LFS_ENABLED`
   - **Value:** `false`
   - **Environment:** Production, Preview, Branch previews

This tells Cloudflare to skip Git LFS during the clone process.

---

## Option 2: Update Build Settings

### In Cloudflare Pages Project Settings:

1. Go to **Settings** → **Builds & deployments**
2. Under **Build configuration**, you can add:
   - **Build command:** `GIT_LFS_SKIP_SMUDGE=1 npm run build`
   - This skips LFS smudge filter during build

---

## Option 3: Remove Git LFS from Repository (Most Reliable)

If LFS files aren't needed for the build, we can remove LFS tracking entirely:

1. Remove `.gitattributes` file (or update it to not use LFS)
2. Add all LFS file types to `.gitignore`
3. Remove LFS files from repository
4. Commit and push

---

## Quick Fix: Retry with Latest Commit

The deployment might be using an old commit. Try:

1. **In Cloudflare Pages dashboard:**
   - Click **"Retry deployment"**
   - OR go to **Settings** → **Builds & deployments**
   - Make sure **Production branch** is set to `main`
   - Cloudflare should pick up the latest commit (`5d0cf95`)

2. **Verify latest commit:**
   - Latest commit on GitHub: `5d0cf95` (removed video file)
   - Old commit that failed: `27b8385` (had video file)

---

## Recommended Action

**Immediate:** Click "Retry deployment" in Cloudflare - it should use the new commit.

**If still failing:** Add environment variable `GIT_LFS_ENABLED=false` in Pages project settings.

---

**Current Status:**
- ✅ Latest commit (`5d0cf95`) removes the problematic video file
- ✅ Video files added to `.gitignore`
- ⏳ Cloudflare needs to retry with new commit
- ⚠️ Many other LFS files exist (images, audio) - may need to address if they cause issues

