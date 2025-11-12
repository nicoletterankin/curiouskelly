# Cloudflare Pages WAV File LFS Fix

**Issue:** Cloudflare Pages build fails on `kelly25_audio.wav` with Git LFS error (404 - Object does not exist on server).

**Root Cause:** WAV files are tracked by Git LFS, but the lesson player doesn't use them. The lesson player uses MP3 files in `lesson-player/videos/audio/`, not WAV files from `projects/Kelly/Audio/`.

---

## ✅ Fixes Applied

### 1. Updated `.gitattributes`
- Removed WAV files from Git LFS tracking
- WAV files are now stored directly in Git (not LFS)
- Commit: `cab1803`

### 2. Updated GitHub Actions Workflow
- Added `lfs: false` to checkout step in `.github/workflows/deploy-cloudflare.yml`
- This ensures GitHub Actions doesn't download LFS files

---

## 🔧 Cloudflare Pages Configuration

### Current Status
- ✅ Environment variable `GIT_LFS_SKIP_SMUDGE=1` is set in Cloudflare Pages
- ✅ Build command: (empty - static files, no build needed)
- ✅ Build output directory: `/lesson-player`
- ⚠️ Still failing because existing WAV files in Git history are LFS pointers

### Next Steps

1. **Retry Deployment**
   - Go to Cloudflare Pages → Deployments tab
   - Click "Retry deployment" on the failed deployment
   - The new commit (`cab1803`) should work because WAV files are no longer tracked by LFS

2. **If Still Failing: Update Build Command**
   
   If the environment variable isn't working, try adding this to the build command:
   
   ```
   GIT_LFS_SKIP_SMUDGE=1 git lfs install --skip-smudge || true
   ```
   
   This explicitly tells Git LFS to skip downloading objects.

3. **Alternative: Shallow Clone**
   
   In Cloudflare Pages settings, you can configure a shallow clone to avoid downloading LFS history:
   - Go to Settings → Build
   - Add build command: `git config --global lfs.fetchexclude "*" && npm run build`
   - This tells Git LFS to exclude all files from fetching

---

## 📋 Files Changed

- `.gitattributes` - Removed WAV from LFS tracking
- `.github/workflows/deploy-cloudflare.yml` - Added `lfs: false` to checkout

---

## 🎯 Why This Works

1. **WAV files not needed:** The lesson player uses MP3 files, not WAV
2. **Future commits:** New WAV files won't be tracked by LFS
3. **Existing files:** The environment variable `GIT_LFS_SKIP_SMUDGE=1` should skip downloading existing LFS pointers

---

## ⚠️ Important Notes

- **WAV files in `projects/Kelly/Audio/`** are used for Audio2Face/Unity integration, not the web lesson player
- **MP3 files in `lesson-player/videos/audio/`** are what the lesson player actually uses
- If you need WAV files for other builds (Unity, Audio2Face), they can be stored outside the repository or in a separate branch

---

## 🚀 Quick Action

1. Go to Cloudflare Pages → Deployments
2. Click "Retry deployment"
3. The build should succeed with commit `cab1803`

If it still fails, update the build command as described above.

