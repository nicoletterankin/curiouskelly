# Cloudflare Pages - Force Latest Commit & Skip LFS

**Current Issue:** Cloudflare Pages is deploying old commit `fbf8545` instead of latest `cab1803` (which has WAV fix).

**Root Cause:** Cloudflare is caching or using an old deployment. The environment variable `GIT_LFS_SKIP_SMUDGE=1` isn't preventing LFS downloads during clone.

---

## 🚀 Immediate Fix: Update Build Command

### Step 1: Update Build Command in Cloudflare Pages

1. Go to **Cloudflare Pages** → Your project → **Settings** → **Builds & deployments**
2. Find **Build configuration**
3. Update **Build command** to:
   ```
   git config --global lfs.fetchexclude "*" && git lfs install --skip-smudge || true
   ```
   This tells Git LFS to:
   - Exclude ALL files from fetching (`lfs.fetchexclude "*"`)
   - Skip smudge filter during install
   - Continue even if LFS isn't installed (`|| true`)

4. **Build output directory:** `/lesson-player` (keep as is)
5. Click **"Save and Deploy"**

### Step 2: Force New Deployment

1. Go to **Deployments** tab
2. Click **"Retry deployment"** on the failed deployment
   - This should trigger a new deployment with the latest commit
3. OR manually trigger:
   - Go to **Settings** → **Builds & deployments**
   - Click **"Trigger deployment"** → Select branch `main`
   - This forces Cloudflare to fetch the latest commit

---

## 🔧 Alternative: Update Build Command (More Explicit)

If the above doesn't work, use this more explicit build command:

```
GIT_LFS_SKIP_SMUDGE=1 git config --global lfs.fetchexclude "*" && git lfs install --skip-smudge || true && echo "LFS skipped successfully"
```

This ensures:
- Environment variable is set
- Git LFS excludes all files
- LFS install skips smudge
- Build continues even if LFS fails

---

## 📋 Verify Latest Commit

Before retrying, verify the latest commit on GitHub:

1. Go to: https://github.com/nicoletterankin/curiouskelly/commits/main
2. Latest commit should be: `cab1803` - "Fix: Remove WAV files from Git LFS tracking"
3. Check that `.gitattributes` in that commit excludes WAV files

---

## 🎯 Why This Works

1. **`git config --global lfs.fetchexclude "*"`**
   - Tells Git LFS to exclude ALL files from fetching
   - Applied globally before any git operations

2. **`git lfs install --skip-smudge`**
   - Installs Git LFS with skip-smudge flag
   - Prevents LFS from downloading objects during checkout

3. **`|| true`**
   - Ensures build continues even if LFS commands fail
   - Prevents build from stopping on LFS errors

4. **Force new deployment**
   - Triggers Cloudflare to fetch latest commit from GitHub
   - Uses commit `cab1803` which has WAV files removed from LFS

---

## ⚠️ Current Status

- ✅ Latest commit `cab1803` on GitHub (WAV removed from LFS)
- ✅ Environment variable `GIT_LFS_SKIP_SMUDGE=1` set
- ⚠️ Cloudflare deploying old commit `fbf8545`
- 🔧 Need to update build command and force new deployment

---

## 🚨 If Still Failing

### Option 1: Disable Git LFS Entirely

Add this to build command:
```
git config --global filter.lfs.required false && git config --global filter.lfs.clean "" && git config --global filter.lfs.smudge ""
```

This completely disables LFS filters.

### Option 2: Use Shallow Clone

In Cloudflare Pages settings, configure:
- **Root directory:** Leave empty (uses repo root)
- **Build command:** `git config --global lfs.fetchexclude "*" && echo "Build complete"`

Since lesson-player is static, you might not need a build command at all.

### Option 3: Contact Cloudflare Support

If none of the above works, the issue might be:
- Cloudflare caching old commits
- Git LFS configuration not being respected
- Need to disconnect/reconnect GitHub integration

---

## ✅ Success Criteria

After applying the fix, deployment should:
1. ✅ Use commit `cab1803` (latest)
2. ✅ Skip Git LFS downloads
3. ✅ Complete successfully
4. ✅ Deploy lesson-player files

---

## 📝 Quick Reference

**Build Command (Recommended):**
```
git config --global lfs.fetchexclude "*" && git lfs install --skip-smudge || true
```

**Build Output Directory:**
```
/lesson-player
```

**Production Branch:**
```
main
```

**Environment Variable (Already Set):**
```
GIT_LFS_SKIP_SMUDGE=1
```

---

**Last Updated:** 2025-01-11  
**Status:** Ready to apply  
**Next Action:** Update build command in Cloudflare Pages and retry deployment

