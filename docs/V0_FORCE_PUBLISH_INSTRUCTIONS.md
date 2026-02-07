# V0 FORCE PUBLISH INSTRUCTIONS
**Date:** February 3, 2026  
**Priority:** CRITICAL  
**Goal:** Fix broken production deployment

---

## THE PROBLEM

Production at thedailylesson.com is running **stale code**.

- Database columns `video_url` and `subtitle` **DO EXIST**
- But production throws "column does not exist" errors
- v0's code is correct, but changes aren't reaching Vercel

---

## V0.APP: FORCE PUBLISH (Click by Click)

### Step 1: Open v0.app
1. Go to https://v0.app
2. Open the chat: "Full audit and evals"

### Step 2: Click Publish
1. In the top-right corner, find the **"Publish"** button
2. Click it
3. Wait for "Deployment started..."

### Step 3: Verify Deployment
1. Check the Production Deployment panel (right side)
2. Wait for status: **"Ready"** (green)
3. Note the "Updated" time should be fresh

### Step 4: Test Production
1. Click **"Visit Site"** button
2. Go to any lesson (e.g., Day 1)
3. Verify:
   - Kelly video PLAYS (not static image)
   - Audio syncs with video
   - No console errors about "column does not exist"

---

## IF PUBLISH BUTTON DOESN'T WORK

### Alternative: Trigger via GitHub

1. Open GitHub: https://github.com/nicoletterankin/v0-curious-kelly-app
2. Make a small change (add a comment or whitespace)
3. Commit and push to `main`
4. Vercel should auto-deploy

### Manual Deploy from Vercel

1. Go to https://vercel.com/lotd/v0-the-dl (or your project)
2. Click **Deployments** tab
3. Find the latest deployment
4. Click **"..."** menu → **"Redeploy"**
5. Select "Use existing Build Cache" = OFF
6. Click **"Redeploy"**

---

## AFTER DEPLOYMENT: VERIFY THESE ITEMS

| Item | Expected | How to Check |
|------|----------|--------------|
| Video plays | Kelly video, not static | Go to any lesson |
| Audio works | Audio syncs with video | Listen to Kelly speak |
| No errors | No "column does not exist" | Open browser DevTools |
| Copy says "Universal" | Not "Personalized" | Check About page |
| Kelly face (not K logo) | Real Kelly photo | Check UI elements |

---

## WHAT V0 SHOULD SAY AFTER PUBLISH

Ask v0 to confirm:

```
Please confirm:
1. Deployment completed successfully
2. Production URL is responding
3. Video API returns video URLs (not null)
4. No database column errors in logs
```

---

## CURSOR WILL VERIFY

After v0 publishes, I will:
1. Test the production API endpoint
2. Verify video playback
3. Confirm the fix

---

**DO THIS NOW. The lip-sync pipeline is ready but useless if production is broken.**
