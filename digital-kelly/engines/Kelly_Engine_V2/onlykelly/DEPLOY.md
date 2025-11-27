# 🚀 KELLY V2 - DEPLOYMENT GUIDE

## Current Deployment Status

| Item | Status |
|------|--------|
| **Live URL** | https://effervescent-stroopwafel-4cd21d.netlify.app |
| **Platform** | Netlify |
| **Last Deploy** | November 26, 2025 |
| **Build Size** | 248 MB |
| **Status** | ✅ LIVE (Kelly renders and loads) |

### Known Issues (Post-Launch Fixes)
- ⚠️ "Trial Version" watermark (requires $199 license)
- ⚠️ Hair transparency (fix in Unity, rebuild)
- ⚠️ Camera too far (fix in Unity, rebuild)

---

## Quick Deploy (One Command)

### Netlify (Recommended - Currently Working)

**Option 1: Drag & Drop (Easiest)**
1. Go to https://app.netlify.com/drop
2. Drag the `Builds/WebGL` folder onto the page
3. Done! Get instant URL

**Option 2: Script Deploy**
```powershell
.\deploy-kelly.ps1
```

### GitHub Pages (Not Recommended)
GitHub Pages doesn't work well with Git LFS + Brotli compression.

### Vercel (Requires Auth)
```powershell
vercel --prod
```
Note: May require authentication setup.

---

## 📋 Pre-Deployment Checklist

Before deploying, verify:

- [x] Unity build completed successfully
- [x] `Builds/WebGL/` folder exists
- [x] `Builds/WebGL/index.html` exists
- [x] `Builds/WebGL/Build/` contains `.unityweb` files
- [x] Kelly loads in local test

---

## 🧪 Local Testing

**Always test locally before deploying!**

```powershell
.\test-kelly-local.ps1
```

This will:
1. Start a local server on http://localhost:8000
2. Open your browser automatically
3. Let you test Kelly before going live

### Manual Local Test

```powershell
cd Builds/WebGL
python -m http.server 8000
```

Then open http://localhost:8000

---

## 📁 File Structure

```
onlykelly/
├── Assets/
│   ├── Editor/KellySetup/      ← Automation scripts
│   ├── Scripts/                ← Runtime scripts
│   └── Scenes/
│       └── KellyMain.unity     ← Main scene (12 KB)
├── Builds/
│   └── WebGL/                  ← Deployment files (248 MB)
│       ├── index.html
│       ├── Build/
│       │   ├── WebGL.data.unityweb (238 MB)
│       │   ├── WebGL.wasm.unityweb (9 MB)
│       │   ├── WebGL.framework.js.unityweb
│       │   └── WebGL.loader.js
│       ├── StreamingAssets/
│       └── TemplateData/
├── vercel.json                 ← Vercel config
├── deploy-kelly.ps1            ← Deploy script
├── deploy-kelly-github.ps1     ← GitHub Pages script
├── test-kelly-local.ps1        ← Local test script
├── LICENSE_APPLICATION.md      ← How to remove watermark
├── REEXPORT_KELLY.md          ← How to re-export from CC5
└── DEPLOY.md                   ← This file
```

---

## 🔧 Custom Domain Setup (curiouskelly.com)

### For Netlify:

1. Create Netlify account (if not done)
2. Claim your deployed site
3. Go to **Site Settings > Domain Management**
4. Click **Add custom domain**
5. Enter: `curiouskelly.com`
6. Follow DNS configuration:

```
Type: A
Name: @
Value: 75.2.60.5

Type: CNAME
Name: www
Value: [your-site-name].netlify.app
```

### For Vercel:

```
Type: A
Name: @
Value: 76.76.21.21

Type: CNAME
Name: www
Value: cname.vercel-dns.com
```

---

## ⚠️ Troubleshooting

### "Build folder not found"
Run Unity build first:
- Open Unity
- Kelly > Build > 🚀 Build WebGL (Production)

### "Unable to parse Build/WebGL.framework.js.unityweb"
This is a Content-Encoding header issue. Use Netlify instead of GitHub Pages.

### "Kelly appears black/no textures"
- Ensure URP is configured correctly
- Check that materials are included in build
- Verify lighting in scene

### "Hair is see-through"
Fix in Unity:
1. Select hair material
2. Change Surface Type: Transparent → Opaque
3. Enable Alpha Clipping, Threshold: 0.5
4. Rebuild and redeploy

### "Trial Version watermark"
Purchase CC/iC Unity Tools license ($199). See `LICENSE_APPLICATION.md`.

---

## 📊 Build Size Reference

Current Kelly V2 Build:
| File | Size |
|------|------|
| WebGL.data.unityweb | 238 MB |
| WebGL.wasm.unityweb | 9 MB |
| WebGL.framework.js.unityweb | 78 KB |
| WebGL.loader.js | 117 KB |
| **Total** | **~248 MB** |

---

## 🎯 Success Criteria

Your deployment is successful when:

1. ✅ Kelly loads in browser
2. ✅ No "Unable to parse" errors
3. ⬜ Hair appears solid (not transparent)
4. ⬜ Kelly properly framed in viewport
5. ⬜ No "Trial Version" watermark (post-license)

---

## 📅 Launch Timeline

### Completed (November 26, 2025)
- [x] Unity project automation
- [x] WebGL build pipeline
- [x] First successful deployment
- [x] Kelly renders in browser

### Week 2 (Dec 4-10)
- [ ] Fix hair material
- [ ] Fix camera positioning
- [ ] Add idle animation
- [ ] Integrate ElevenLabs TTS

### Week 3 (Dec 11-17)
- [ ] Connect curiouskelly.com domain
- [ ] Purchase CC/iC license (optional)
- [ ] Full QA testing
- [ ] **LAUNCH: December 17, 2025** 🚀

---

## 📞 Quick Commands Reference

| Action | Command |
|--------|---------|
| Test locally | `.\test-kelly-local.ps1` |
| Deploy to Netlify | Drag `Builds/WebGL` to netlify.com/drop |
| Deploy via script | `.\deploy-kelly.ps1` |
| Rebuild in Unity | Kelly > Build > 🚀 Build WebGL (Production) |

---

## 🎉 December 17, 2025 Launch Day

1. Final local test: `.\test-kelly-local.ps1`
2. Deploy to Netlify
3. Verify at live URL
4. Connect curiouskelly.com domain
5. Announce to the world! 🚀

---

**Current Live URL:** https://effervescent-stroopwafel-4cd21d.netlify.app
**Password:** My-Drop-Site (temporary, remove for production)

**Kelly V2 is LIVE!** ✨

*Last Updated: November 26, 2025*
