# 🎨 DAY 351 VISUAL LAUNCH GUIDE

**Topic:** "Practicing in Your Mind" — Visualization  
**Launch Day:** December 17, 2025  
**Status:** READY TO GENERATE

---

## 🚀 QUICK START - ONE COMMAND

```powershell
# Generate ALL Day 351 visuals (Kelly + Infographics + Social)
npx tsx scripts/generate-day-351-visuals.ts
```

---

## 📊 WHAT GETS GENERATED

### Kelly Phase Images (7 images)
| Phase | Pose | Emotion |
|-------|------|---------|
| hook | Eyes closed, holding invisible sphere | Peaceful meditation, wonder |
| cliff | Finger on temple, painting vision | Curiosity, revelation |
| q1 | Animated explaining, "mind blown" | Excitement about 90% overlap |
| q2 | Playing invisible piano keys | Storytelling the piano study |
| q3 | Fist raised, champion energy | Olympic athlete inspiration |
| wisdom | Warm open hands, giving gesture | Wise mentor sharing truth |
| outro | Wave goodbye, hand on heart | Encouragement, warm farewell |

### Educational Infographics (5 images)
| ID | Visual | Purpose |
|----|--------|---------|
| brain-scan-90-percent | Split brain comparison doing vs imagining | Show 90% neural overlap |
| piano-study-comparison | 3-group experiment comparison | Harvard piano study results |
| elite-athletes-50-percent | Olympic athlete split figure | 50% mental rehearsal stat |
| visualization-guide-steps | 5-step how-to process | Actionable technique guide |
| cosmic-mind-background | Galaxy inside head silhouette | Hero background for lesson |

### Social Media Visuals (5 images)
| Platform | Visual | Purpose |
|----------|--------|---------|
| Instagram Carousel 1 | Athlete meditating hook | "50% of training..." teaser |
| Instagram Carousel 2 | Brain scan comparison | 90% overlap educational |
| Quote Card | Cosmic wisdom background | Shareable wisdom quote |
| Twitter Header | Galaxy eye reflection | Thread header image |
| TikTok Thumbnail | Split face concept | Vertical video thumbnail |

---

## 🎯 TARGETED GENERATION

```powershell
# Preview what will be generated (no API calls)
npx tsx scripts/generate-day-351-visuals.ts --dry-run

# Generate ONLY Kelly phase images
npx tsx scripts/generate-day-351-visuals.ts --kelly-only

# Generate ONLY educational infographics  
npx tsx scripts/generate-day-351-visuals.ts --infographics-only

# Generate ONLY social media visuals
npx tsx scripts/generate-day-351-visuals.ts --social-only
```

---

## 📁 OUTPUT LOCATIONS

```
public/kelly/phases/351/
├── hook.png        ← Kelly peaceful visualization pose
├── cliff.png       ← Kelly discovery pose
├── q1.png          ← Kelly explaining 90% overlap
├── q2.png          ← Kelly piano study storytelling
├── q3.png          ← Kelly Olympic champion energy
├── wisdom.png      ← Kelly wise mentor pose
├── outro.png       ← Kelly warm farewell
└── visual-plan.json ← Full prompt definitions

public/kelly/infographics/351/
├── infographic-brain-scan.png       ← 90% neural overlap
├── infographic-piano-study.png      ← Harvard experiment
├── infographic-olympic-athletes.png ← 50% mental rehearsal
├── infographic-how-to-visualize.png ← 5-step guide
└── background-cosmic-mind.png       ← Hero background

public/kelly/social/351/
├── social-ig-carousel-1.png   ← Instagram hook slide
├── social-ig-carousel-2.png   ← Instagram brain scan
├── social-quote-card.png      ← Wisdom quote card
├── social-twitter-header.png  ← Twitter thread header
└── social-tiktok-thumb.png    ← TikTok thumbnail
```

---

## 🔧 PREREQUISITES

### Required API Keys
```env
# .env.local
REPLICATE_API_TOKEN=r8_...    # Required for Flux Pro + Kelly LoRA
```

### Verify Setup
```powershell
# Check Replicate token is set
echo $env:REPLICATE_API_TOKEN

# Or check in .env.local
Get-Content .env.local | Select-String "REPLICATE"
```

---

## 💰 COST ESTIMATE

| Asset Type | Count | Cost/Image | Subtotal |
|------------|-------|------------|----------|
| Kelly LoRA (Flux-dev-lora) | 7 | ~$0.05 | $0.35 |
| Infographics (Flux Pro) | 5 | ~$0.04 | $0.20 |
| Social Media (Flux Pro) | 5 | ~$0.04 | $0.20 |
| **TOTAL** | **17** | | **~$0.75** |

---

## 🎨 VISUAL PROMPT PHILOSOPHY

### For This Lesson (Visualization):

1. **Kelly poses are meditation-focused**
   - Eyes closed, peaceful expressions
   - Hands in contemplative positions
   - Cosmic/neural backgrounds

2. **Infographics are science-backed**
   - Real research: 90% overlap, piano study, Olympic stats
   - Clean medical/scientific aesthetic
   - Data-driven but beautiful

3. **Social media is scroll-stopping**
   - Bold statistics front and center
   - Mystery hooks ("doing THIS...")
   - Premium aesthetic, not clickbait

---

## 📋 GENERATION CHECKLIST

### Before Generation
- [ ] REPLICATE_API_TOKEN is set
- [ ] Run `--dry-run` to preview
- [ ] Verify Replicate account has credits

### Generation
- [ ] Run full generation command
- [ ] Watch for errors in console
- [ ] Takes ~5-10 minutes for all 17 images

### After Generation
- [ ] Check output directories exist
- [ ] Visually verify each image
- [ ] Kelly images show consistent character
- [ ] Infographics are educational and clear
- [ ] Social images are platform-appropriate

### Quality Verification
```powershell
# Check all files generated
Get-ChildItem public/kelly/phases/351/*.png | Measure-Object
Get-ChildItem public/kelly/infographics/351/*.png | Measure-Object
Get-ChildItem public/kelly/social/351/*.png | Measure-Object
```

---

## 🔄 REGENERATION (if needed)

To regenerate specific images, delete them first:

```powershell
# Delete one image to regenerate it
Remove-Item public/kelly/phases/351/hook.png

# Then run generator (skips existing files)
npx tsx scripts/generate-day-351-visuals.ts --kelly-only
```

---

## 🚀 LAUNCH SEQUENCE

```powershell
# 1. Generate all visuals
npx tsx scripts/generate-day-351-visuals.ts

# 2. Generate audio (if not done)
npx tsx scripts/generate-day-351-audio.ts

# 3. Verify lesson data exists
Test-Path public/data/day-351-complete.js

# 4. Test locally
# Open http://localhost:4321/learn.html?day=351
```

---

## 📊 INFOGRAPHIC SPECIFICATIONS

### Brain Scan (90% Neural Overlap)
- **Key Message:** Your brain can't tell the difference between doing and imagining
- **Visual:** Split-screen brain comparison with identical neural activation
- **Statistic:** 90% prominently displayed
- **Style:** Medical illustration meets cinematic lighting

### Piano Study
- **Key Message:** Mental practice creates real neural changes
- **Visual:** 3-lane experiment comparison (physical, mental, control)
- **Research:** Harvard/Pascual-Leone study
- **Style:** Clean scientific infographic

### Olympic Athletes (50% Mental Rehearsal)
- **Key Message:** Elite performers spend half their training visualizing
- **Visual:** Split athlete (physical/ethereal) with 50% statistic
- **Supporting:** Surgeon, pianist, basketball player icons
- **Style:** Inspirational sports documentary

### How To Visualize (5 Steps)
- **Key Message:** Actionable technique anyone can use tonight
- **Visual:** Flowing 5-step process with icons
- **Steps:** Relax → See Details → Feel Movements → Hear Sounds → Practice Daily
- **Style:** Warm, inviting, encouraging

### Cosmic Mind (Hero Background)
- **Key Message:** Your imagination is infinite
- **Visual:** Galaxy/universe inside human head silhouette
- **Purpose:** Lesson hero image and background
- **Style:** Cosmic digital art, awe-inspiring

---

## ✨ READY FOR LAUNCH DAY

**December 17, 2025**

*"The mind that rehearses grows stronger than the mind that merely waits."*

---

*Generated: December 16, 2025*  
*For: Curious Kelly Launch Day*
