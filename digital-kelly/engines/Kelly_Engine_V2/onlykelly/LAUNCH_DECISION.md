# KELLY V2 - LAUNCH DECISION GUIDE

## Current Situation (November 26, 2025)

| Item | Status |
|------|--------|
| Kelly deployed | ✅ https://effervescent-stroopwafel-4cd21d.netlify.app |
| Kelly renders | ✅ Working in browser |
| Hair issue | ⚠️ Transparent (needs Unity fix) |
| Camera issue | ⚠️ Too far (needs Unity fix) |
| Watermark | ⚠️ "Trial Version" visible |
| Days to launch | **21 days** (December 17) |

---

## YOUR DECISION: Choose a Path

### 🚀 PATH A: QUICK LAUNCH (Recommended)
**Timeline:** Today → December 17
**Cost:** $0 now, $199 later (optional)

```
TODAY:        Fix hair + camera → Deploy
THIS WEEK:    Add animations, test features  
DEC 17:       LAUNCH with small watermark
POST-LAUNCH:  Buy license, update, remove watermark
```

**Pros:**
- ✅ Hits December 17 deadline GUARANTEED
- ✅ Kelly functional and looks good
- ✅ Can generate revenue immediately
- ✅ Buy license with revenue later

**Cons:**
- ⚠️ Small "Trial Version" watermark visible
- ⚠️ Need post-launch update

---

### 💎 PATH B: PERFECT LAUNCH
**Timeline:** This week → December 17
**Cost:** $199 now

```
TODAY:        Fix hair + camera → Deploy (temporary)
TOMORROW:     Purchase CC/iC Unity Tools ($199)
THIS WEEK:    Set up pipeline, re-export Kelly
NEXT WEEK:    Add animations, test
DEC 17:       LAUNCH perfect version (no watermark)
```

**Pros:**
- ✅ No watermark on launch day
- ✅ Professional appearance from day 1
- ✅ Proper pipeline for future updates

**Cons:**
- ⚠️ Tighter timeline (riskier)
- ⚠️ $199 upfront cost
- ⚠️ Learning curve for pipeline

---

### ⭐ PATH C: HYBRID (Best of Both)
**Timeline:** Today → December 17
**Cost:** $199 this week

```
TODAY (Nov 26):
  ✓ Fix hair + camera in Unity
  ✓ Deploy to Netlify (safety net)
  
THIS WEEK (Nov 27-30):
  → Purchase CC/iC Unity Tools ($199)
  → Watch tutorial video (8 min)
  → Set up pipeline
  → Re-export Kelly (no watermark)
  → Rebuild and deploy

WEEK 2 (Dec 1-7):
  → Add idle animation
  → Integrate ElevenLabs TTS
  → Test lip sync

WEEK 3 (Dec 8-17):
  → Connect curiouskelly.com
  → Full QA testing
  → Final polish
  → LAUNCH! 🚀
```

**Pros:**
- ✅ Safety net deployed TODAY
- ✅ Watermark removed BEFORE launch
- ✅ Buffer time for testing
- ✅ Proper pipeline for future

**Cons:**
- ⚠️ $199 cost this week
- ⚠️ More work than Path A

---

## Recommendation: PATH C (Hybrid)

**Why:**
1. You have 21 days - enough time for proper setup
2. $199 is reasonable investment for professional launch
3. Having safety net TODAY reduces stress
4. Proper pipeline makes future updates easier

---

## Immediate Actions (Do Today)

### Action 1: Fix Hair in Unity (30 min)
1. Open Unity project
2. Find hair material
3. Change: Surface Type → Opaque
4. Enable: Alpha Clipping, Threshold 0.5
5. Save

### Action 2: Fix Camera in Unity (15 min)
1. Open KellyMain.unity
2. Select Main Camera
3. Position: (0, 1.5, 2)
4. Rotation: (0, 180, 0)
5. Field of View: 40
6. Save scene

### Action 3: Rebuild WebGL (30 min)
1. Kelly → Build → Build WebGL (Production)
2. Wait for completion

### Action 4: Deploy to Netlify (5 min)
1. Drag Builds/WebGL to netlify.com/drop
2. Verify Kelly loads with fixes

---

## This Week Actions

### Action 5: Purchase License
- URL: https://www.reallusion.com/auto-setup/unity/default.html
- Cost: ~$199 USD
- Delivery: Instant (email)

### Action 6: Set Up Pipeline
- Watch: https://www.youtube.com/watch?v=hyX8MG5ZIpk (8 min)
- Follow: PIPELINE_SETUP.md
- Result: One-button export, no watermark

### Action 7: Re-Export Kelly
- In iClone: Plugins → Send to Unity
- In Unity: Add new Kelly to scene
- Rebuild WebGL
- Deploy to Netlify

---

## Documentation Created

| File | Purpose |
|------|---------|
| `PIPELINE_SETUP.md` | Step-by-step pipeline setup |
| `PIPELINE_TROUBLESHOOTING.md` | Common issues & solutions |
| `CHECK_LICENSE.md` | How to verify license status |
| `LICENSE_APPLICATION.md` | How to apply license |
| `REEXPORT_KELLY.md` | How to re-export from CC5 |
| `DEPLOY.md` | Deployment guide |
| `LAUNCH_DECISION.md` | This file |

---

## Success Metrics

### Today's Goal:
- [ ] Hair solid (not transparent)
- [ ] Kelly properly framed
- [ ] Deployed to Netlify

### This Week's Goal:
- [ ] License purchased
- [ ] Pipeline set up
- [ ] Kelly re-exported (no watermark)

### Launch Day (Dec 17):
- [ ] Kelly live on curiouskelly.com
- [ ] No watermark
- [ ] Animations working
- [ ] TTS integrated

---

## Your Next Step

**RIGHT NOW:** Open Unity and fix the hair material.

That's it. One step at a time. Kelly is already live - we're just making her look better.

🚀 **Let's go!**

---

*Decision made: November 26, 2025*
*Launch target: December 17, 2025*
*Days remaining: 21*

