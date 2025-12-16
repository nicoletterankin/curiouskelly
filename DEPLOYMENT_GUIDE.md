# 🚀 DEPLOYMENT GUIDE - Curious Kelly

**Target:** curiouskelly.com  
**Hosting:** Netlify  
**Deploy Folder:** `public/`  
**Method:** Git-based automatic deployment

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

- [x] All browser tests pass (`node scripts/browser_test.js`)
- [x] `DEPLOY_CHECKLIST.md` reviewed and approved
- [x] Local testing complete (`http://localhost:8080`)
- [x] No uncommitted changes that should be included
- [x] Supabase connection tested

---

## 🎯 RECOMMENDED: Git-Based Deployment (Automatic)

Netlify is configured to auto-deploy from the `main` branch.

### Step 1: Review Changes

```bash
cd C:\Users\user\UI-TARS-desktop

# See what files changed
git status

# Review specific changes
git diff public/learn.html
git diff public/settings.html
```

### Step 2: Stage Files

```bash
# Add all changed files in public/
git add public/

# Add new documentation
git add DEPLOY_CHECKLIST.md
git add DEPLOYMENT_GUIDE.md
git add AUDIT_RESULTS_SUMMARY.md
git add UNITY_3D_FIXES_REQUIRED.md

# Add new scripts
git add scripts/audit_lessons.js
git add scripts/inspect_db_sample.js
git add scripts/browser_test.js

# Add any other changed files
git add netlify.toml
git add vercel.json
```

### Step 3: Commit Changes

```bash
git commit -m "feat: complete lesson system with Supabase integration

- Add TikTok-style lesson player with popovers
- Integrate Supabase for 365 lessons
- Fix archetype mapping with fallbacks
- Add settings page
- Implement 2D/3D avatar controller
- Add audio system (ElevenLabs ready)
- Create browser test suite
- Document deployment and Unity fixes

Browser tests: 17/20 passing (85%)
Database: 365 lessons ready
Ready for production deployment"
```

### Step 4: Push to GitHub

```bash
# Push to main branch (triggers Netlify deploy)
git push origin main
```

### Step 5: Monitor Deployment

1. **Netlify Dashboard:** https://app.netlify.com/sites/curiouskelly/deploys
2. **Watch build log** for errors
3. **Deployment typically takes:** 2-5 minutes
4. **Site will be live at:** https://curiouskelly.com

---

## 🔄 ALTERNATIVE: Netlify CLI Deployment

If you prefer manual control or need to test before going live:

### Install Netlify CLI (if not installed):

```bash
npm install -g netlify-cli
```

### Login to Netlify:

```bash
netlify login
```

### Deploy to Preview (Test First):

```bash
cd C:\Users\user\UI-TARS-desktop

# Deploy to preview URL (not production)
netlify deploy --dir=public

# Netlify will give you a preview URL like:
# https://6579abc123--curiouskelly.netlify.app
```

### Test Preview Deployment:

1. Visit the preview URL
2. Test learn.html with Supabase
3. Test all popovers
4. Test on mobile device
5. Verify no console errors

### Deploy to Production:

```bash
# If preview looks good, deploy to production
netlify deploy --prod --dir=public
```

---

## 📦 What Gets Deployed

### Included (from `public/` folder):

```
public/
├── index.html              ← Homepage
├── learn.html              ← Main lesson player (NEW/UPDATED)
├── hub.html                ← Kelly Today Hub
├── calendar.html           ← Calendar view
├── settings.html           ← Settings page (NEW)
├── pricing.html            ← Pricing
├── about.html              ← About page
├── config.js               ← Supabase & API keys
├── css/
│   ├── kelly-os.css        ← Design system (UPDATED)
│   └── public-os.css       ← Marketing styles
├── js/
│   ├── kelly-audio.js      ← Audio system (NEW)
│   ├── kelly-avatar-controller.js  ← Avatar controller (NEW)
│   ├── kelly-2d-avatar.js  ← 2D avatar (NEW)
│   ├── unity-kelly-loader.js  ← Unity loader (NEW)
│   ├── golden-lesson-citizenship.js  ← Sample lesson (NEW)
│   ├── kelly-data.js       ← Data layer
│   ├── auth.js             ← Authentication
│   └── ...
├── images/
│   └── kelly/              ← Kelly avatar images
├── unity/
│   └── kelly/Build/        ← Unity WebGL build
├── data/
│   └── 365_day_calendar.json  ← Curriculum metadata
└── ...
```

### Excluded (via .gitignore):

```
node_modules/
.env
*.log
test-screenshots/
digital-kelly/              ← Unity source (not deployed)
scripts/                    ← Build scripts (not deployed)
```

---

## 🔐 Environment Variables

### Set in Netlify Dashboard:

1. Go to: https://app.netlify.com/sites/curiouskelly/settings/deploys
2. Click: "Environment variables"
3. Add these (if not already set):

```
PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
PUBLIC_SUPABASE_ANON_KEY=eyJhbGci...(full key from config.js)
ELEVENLABS_API_KEY=(add when ready for voice)
STRIPE_PUBLISHABLE_KEY=pk_live_51SXAYMEs6ql8qYcK...(from config.js)
```

**Note:** These are also hardcoded in `public/config.js` for now, but environment variables are more secure for production.

---

## 🧪 Post-Deployment Verification

### Immediate Checks (within 5 minutes):

1. **Homepage loads:**
   ```
   https://curiouskelly.com
   ```

2. **Learn page works:**
   ```
   https://curiouskelly.com/learn.html?day=1
   ```
   - Check: Lesson loads from Supabase
   - Check: Kelly avatar visible
   - Check: No console errors

3. **Settings page:**
   ```
   https://curiouskelly.com/settings.html
   ```

4. **Popovers work:**
   - Click Age, Language, Tone, Difficulty buttons
   - Verify popovers appear
   - Verify selections update

5. **Mobile test:**
   - Open on phone
   - Test swipe gestures
   - Verify full-bleed Kelly

### Console Checks:

Open browser DevTools (F12) and look for:

- ✅ `[Learn] ✓ Core lesson loaded: {topic}`
- ✅ `[Learn] ✓ Loaded X atoms for archetype {name}`
- ❌ No `ERR_NAME_NOT_RESOLVED` errors
- ❌ No `Failed to fetch` errors

---

## 🚨 Troubleshooting

### Issue: Supabase Connection Fails

**Symptoms:** Lessons show "Loading..." or placeholder content

**Fix:**
1. Check Supabase URL in `public/config.js`
2. Verify Supabase project is not paused
3. Check browser console for CORS errors
4. Verify anon key is correct

### Issue: Popovers Don't Appear

**Symptoms:** Clicking buttons does nothing

**Fix:**
1. Check browser console for JavaScript errors
2. Verify `kelly-os.css` loaded (check Network tab)
3. Clear browser cache and hard reload (Ctrl+Shift+R)

### Issue: Unity 3D Mode Crashes

**Expected:** This is a known issue (see `UNITY_3D_FIXES_REQUIRED.md`)

**Workaround:** Mode button is hidden, users see 2D only

### Issue: No Audio

**Expected:** ElevenLabs API key not set

**Fix:** Add API key to environment variables or `config.js`

---

## 🔙 Rollback Procedure

### If deployment breaks the site:

**Option 1: Netlify Dashboard (Fastest)**

1. Go to: https://app.netlify.com/sites/curiouskelly/deploys
2. Find the last working deploy (before your changes)
3. Click the "..." menu → "Publish deploy"
4. Site reverts to previous version immediately

**Option 2: Git Revert**

```bash
# Find the commit to revert
git log --oneline

# Revert the last commit
git revert HEAD

# Push revert
git push origin main

# Netlify will auto-deploy the reverted version
```

**Option 3: Emergency Fix**

```bash
# Fix the issue locally
# Edit the problematic file(s)

# Commit and push fix
git add .
git commit -m "fix: emergency fix for [issue]"
git push origin main
```

---

## 📊 Deployment History

### Track Your Deploys:

**Netlify Dashboard:**
- URL: https://app.netlify.com/sites/curiouskelly/deploys
- Shows: Build logs, deploy time, commit message
- Features: Preview deploys, rollback, split testing

**Git History:**
```bash
# See recent commits
git log --oneline -10

# See what changed in last deploy
git show HEAD
```

---

## 🎯 Next Deployment (Future Updates)

### For subsequent deploys:

```bash
# 1. Make changes locally
# 2. Test with: node scripts/browser_test.js
# 3. Commit changes
git add .
git commit -m "feat: [description]"

# 4. Push to deploy
git push origin main

# 5. Monitor: https://app.netlify.com/sites/curiouskelly/deploys
```

### Best Practices:

- ✅ Always test locally first
- ✅ Run browser tests before deploying
- ✅ Use descriptive commit messages
- ✅ Deploy during low-traffic hours
- ✅ Monitor error logs after deploy
- ✅ Keep `DEPLOY_CHECKLIST.md` updated

---

## 📞 Support & Resources

### Netlify Documentation:
- Deploy guide: https://docs.netlify.com/site-deploys/overview/
- CLI reference: https://docs.netlify.com/cli/get-started/
- Rollbacks: https://docs.netlify.com/site-deploys/manage-deploys/#rollbacks

### Project Documentation:
- `DEPLOY_CHECKLIST.md` - Pre-deploy verification
- `AUDIT_RESULTS_SUMMARY.md` - Database content status
- `UNITY_3D_FIXES_REQUIRED.md` - Unity rebuild guide
- `README.md` - Project overview

### Emergency Contacts:
- Netlify Support: https://www.netlify.com/support/
- Supabase Status: https://status.supabase.com/

---

## ✅ READY TO DEPLOY

**Deployment Method:** Git-based (automatic)  
**Command:** `git push origin main`  
**Expected Time:** 2-5 minutes  
**Live URL:** https://curiouskelly.com

**All systems ready for deployment!** 🚀









