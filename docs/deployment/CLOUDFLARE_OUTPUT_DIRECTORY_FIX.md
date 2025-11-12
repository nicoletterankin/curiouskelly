# Cloudflare Pages - Output Directory Fix

**Issue:** Build fails with error:
```
Error: Output directory "dist" not found.
Failed: build output directory not found
```

**Root Cause:** `wrangler.toml` specifies `pages_build_output_dir = "dist"`, but the lesson-player is static files that don't have a build step. The output directory should be `lesson-player` (the directory itself).

---

## ✅ Fix Applied

### Updated `wrangler.toml`

Changed:
- `pages_build_output_dir = "dist"` → `pages_build_output_dir = "lesson-player"`
- `name = "curiouskelly-marketing"` → `name = "curiouskelly-lessons-v2"` (to match Cloudflare project name)

---

## 🔧 Alternative: Cloudflare Pages Settings Override

If `wrangler.toml` still causes issues, you can override it in Cloudflare Pages settings:

1. Go to **Cloudflare Pages** → Your project → **Settings** → **Builds & deployments**
2. Find **Build configuration**
3. Set **Build output directory** to: `lesson-player`
4. This will override `wrangler.toml` settings

---

## 📋 Why This Works

1. **Lesson-player is static:** No build step needed - files are already ready to deploy
2. **Output directory:** Should point directly to the `lesson-player` folder
3. **No dist folder:** Since there's no build, there's no `dist` directory created

---

## 🎯 Current Status

- ✅ Git clone: Working
- ✅ Dependencies: Installing successfully
- ✅ Build command: Running successfully
- ✅ Output directory: Updated to `lesson-player`
- ⏳ Next deployment: Should succeed

---

## 🚀 Next Steps

1. Commit and push the `wrangler.toml` change
2. Retry deployment in Cloudflare Pages
3. Build should succeed and deploy the lesson-player files

---

**Last Updated:** 2025-01-11  
**Status:** Ready to deploy  
**Next Action:** Commit `wrangler.toml` change and retry deployment

