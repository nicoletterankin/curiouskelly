# 🎯 HAPPY LEARNERS DIRECTIVE

**Mission:** Every learner gets a perfect lesson, every day, forever.

**Generated:** December 21, 2025  
**Updated:** December 21, 2025  
**Status:** ACTIVE

---

## 📊 Current State

### Lesson Content (THE FOUNDATION)

| Content Layer | Files | English | Spanish | Portuguese |
|---------------|-------|---------|---------|------------|
| **Lesson JSONs** (`public/lessons/`) | 366 | ✅ 100% | ⚠️ ~50 days | ❌ 0 days |
| **Curriculum Titles** (`public/data/curriculum/`) | 365 | ✅ 100% | N/A | N/A |
| **Summary Manifests** (`content/email-summary-video/`) | ~290 | ✅ | N/A | N/A |
| **Gold Standard Examples** | 1 (Day 355) | ✅ | N/A | N/A |

**Translation Gaps:** 43,950 `[NEEDS TRANSLATION]` placeholders across ES/PT.

### Player & Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| Lesson Player (`learn.html`) | ✅ Working | Loads all 365 days |
| Lesson Loader (`kelly-lesson-loader.js`) | ✅ Working | Fetches from `/lessons/day-{N}.json` |
| Curriculum-based fallback | ✅ Working | Bulletproof title loading |
| Pixi Compositor | ✅ Fixed | Duplicate removed |
| Pricing Page Copy | ✅ Fixed | No longer says "yours" |
| TTS Worker | ✅ Ready | Cloudflare + ElevenLabs |
| Deployment | ⚠️ Pending | Fixes ready, need push |

### Media Assets (SEPARATE FROM JSON CONTENT)

| Asset Type | Status | Notes |
|------------|--------|-------|
| Day 1 HD Videos | ✅ Complete | 15 lipsync videos |
| Days 2-365 HD Videos | Not generated | Future work |
| Visual Plans | 2 complete | Day 1, Day 5 |
| Audio (TTS) | Runtime | ElevenLabs generates on demand |

---

## 🗂️ Repository Structure (Lesson JSONs)

### THE LESSON FILES (Complete for EN)
```
public/lessons/
├── day-1.json                      # Day 1 - Starting Fresh
├── day-2.json                      # Day 2 - The Three Lives of Water
├── ...
└── day-365.json                    # Day 365 - complete

Each file contains:
├── meta                            # day, topic, emoji, category, version
├── headline                        # {en, es, pt}
├── universal_truth                 # {en, es, pt}
├── fun_facts[]                     # {en, es, pt}
├── discussion_questions[]          # {en, es, pt}
└── phases
    ├── hook                        # Welcome phase
    │   ├── title, script, duration # {en, es, pt}
    │   ├── prompt                  # {en, es, pt}
    │   └── options[]               # A, B, C with text, quality, response
    ├── question1                   # Fact 1 phase
    ├── question2                   # Fact 2 phase  
    ├── question3                   # Fact 3 phase
    └── wisdom                      # Closing phase
```

### Curriculum Titles (Backup/Metadata)
```
public/data/curriculum/
├── year1-foundations/              # Learn track (365 days)
│   ├── january_curriculum.json     # Days 1-31 titles
│   └── ... (12 months)
└── year2-ai-fluency/               # Grow track (365 days)
    └── ... (12 months)
```

### Player Files
```
public/
├── learn.html                      # Main lesson player
├── pricing.html                    # Pricing page
└── js/
    ├── kelly-lesson-loader.js      # Loads from /lessons/day-{N}.json
    └── kelly-pixi-compositor.js    # WebGL overlays (FIXED)
```

---

## 📋 TASK 1: Deploy Current Fixes

### What Changed
- `public/js/kelly-pixi-compositor.js` - Removed duplicate implementation (353 lines deleted)
- `public/pricing.html` - Changed "Today's lesson is yours" → "One subscription, 365 daily lessons"
- `public/learn.html` - Improved curriculum-based topic loading, fixed week lessons

### Deploy Command
```powershell
git add -A
git commit -m "fix: remove pixi duplicate, fix pricing copy, bulletproof curriculum loader"
git push origin main
# Vercel auto-deploys from main branch
```

### Verification
1. Open https://curiouskelly.com/pricing.html - confirm badge text
2. Open https://curiouskelly.com/learn.html - load any day
3. DevTools Console - no duplicate PIXI warnings
4. Week view - shows correct titles for all days

---

## 📋 TASK 2: Verify All 365 Lesson JSONs Load Correctly

### What to Check
The lesson JSONs are **100% complete for English**. Verify the player loads them correctly:

```powershell
# Quick validation - all files exist and have content
Get-ChildItem public\lessons\day-*.json | 
  ForEach-Object { 
    $json = Get-Content $_.FullName | ConvertFrom-Json
    [PSCustomObject]@{
      Day = $json.meta.day
      Topic = $json.meta.topic.en
      HasPhases = ($json.phases -ne $null)
    }
  } | Format-Table
```

### Manual Spot Check
1. Load Day 1 - verify topic "Starting Fresh"
2. Load Day 100 - verify topic "Splitting Things Fairly"  
3. Load Day 200 - check phases work
4. Load Day 365 - verify end of year works

### Success Criteria
- [ ] All 365 days load in player
- [ ] Each phase displays script
- [ ] Options are clickable
- [ ] Responses show after selection

---

## 📋 TASK 3: Translation Completion (ES/PT)

### Current Gap
- **43,950** instances of `[NEEDS TRANSLATION]` 
- Spanish: ~50 days complete, ~315 days need translation
- Portuguese: All days need translation

### Translation Structure
Each lesson JSON has i18n objects:
```json
{
  "topic": { "en": "Starting Fresh", "es": "Comenzando de Nuevo", "pt": "[NEEDS TRANSLATION]" },
  "script": { "en": "Welcome...", "es": "Bienvenidos...", "pt": "[NEEDS TRANSLATION]" }
}
```

### Translation Process
1. Extract all `[NEEDS TRANSLATION]` entries with context
2. Batch translate (Gemini/GPT-4 with Kelly voice guidelines)
3. Replace placeholders with translations
4. Validate no empty strings remain

### Existing Translations
```
content/translations/
├── es/day-1.json through day-50.json    # Spanish (50 days)
├── pt/day-1.json through day-50.json    # Portuguese (50 days)
└── PROGRESS.json                         # Tracking file
```

### Priority
1. Spanish Days 1-31 (January) - most likely first users
2. Spanish Days 32-365
3. Portuguese Days 1-365

---

## 📋 TASK 4: Production Verification

### Test Checklist
```
□ Age Gate
  - Visit curiouskelly.com
  - Age selector appears
  - Selecting age proceeds to lesson

□ Lesson Loading (try Days 1, 50, 150, 300, 365)
  - Correct topic title displays
  - Phase scripts load
  - Kelly image/avatar shows

□ Interaction
  - Options A/B/C are clickable
  - Kelly response text displays
  - Phase progression works (all 5 phases)

□ Journey/Week View
  - Shows correct titles for visible days
  - Navigation to other days works
  - Day numbers are correct

□ Settings Panel
  - Opens without covering content
  - Billing/Support sections work

□ Pricing
  - All 4 tiers display correctly
  - Stripe checkout initiates
  - No console errors
```

### Browser Test Command
```powershell
# Open browser tools
npx playwright test tests/lesson-flow.spec.ts --headed
```

---

## 🚦 Execution Order

```
1. DEPLOY ────────────────────────────────────────────────────
   └── Push fixes to main, verify on curiouskelly.com
   └── Time: ~5 minutes
   └── Blocks: Nothing (do first)

2. VERIFY ENGLISH ────────────────────────────────────────────
   └── Spot-check 10+ days load correctly
   └── Time: ~30 minutes
   └── Blocks: Confidence in EN content

3. SPANISH TRANSLATION ───────────────────────────────────────
   └── Complete ES for remaining ~315 days
   └── Time: ~4-8 hours (batch API + validation)
   └── Blocks: Spanish-speaking learners

4. PORTUGUESE TRANSLATION ────────────────────────────────────
   └── Complete PT for all 365 days
   └── Time: ~8-12 hours (batch API + validation)
   └── Blocks: Portuguese-speaking learners

5. (FUTURE) HD VIDEO GENERATION ──────────────────────────────
   └── Generate lipsync videos for enhanced experience
   └── Time: 48-72 hours
   └── NOT blocking - TTS works without pre-rendered video
```

---

## ⚠️ Critical Constraints

### From CLAUDE.md (Non-negotiable)
- **Never use browser TTS** - Only ElevenLabs
- **60+ minutes training audio per voice** - Already trained (Kelly/Kyle)
- **Languages precomputed** - EN + ES/FR in every DNA file
- **60 FPS target** - GPU-accelerated where applicable
- **No shortcuts** - Every frame must be perfect [[memory:12016734]]

### API Rate Limits
| Service | Limit | Strategy |
|---------|-------|----------|
| ElevenLabs | 100 req/min | Batch with 1s delay |
| Sync Labs | 10 concurrent | Queue with polling |
| MiniMax | 50 req/min | Exponential backoff |
| Supabase | 1000 req/min | Bulk uploads |

### Cost Awareness
- ElevenLabs: ~$0.30/minute audio
- Sync Labs: ~$0.10/second video
- MiniMax: ~$0.05/video
- **Estimated total for 365 days: $2,000-3,000**
- Budget is pre-approved [[memory:12051774]]

---

## 📈 Success Metrics

**Happy Learner = All of these true:**
1. ✓ Lesson loads in <3 seconds
2. ✓ Correct topic title displays
3. ✓ All 5 phases have scripts
4. ✓ Options are interactive
5. ✓ Kelly responses match selections
6. ✓ No console errors
7. ✓ Works on mobile

**Content Coverage:**
| Language | Lessons Complete | Status |
|----------|------------------|--------|
| English | 365/365 | ✅ READY |
| Spanish | ~50/365 | ⚠️ 315 need translation |
| Portuguese | ~0/365 | ❌ All need translation |

**Media Status (Enhancement, not blocking):**
- TTS Audio: Runtime via ElevenLabs (works for all 365)
- HD Videos: Day 1 complete, Days 2-365 future work
- Infographics: 2 complete, rest future work

---

## 🔄 Daily Operations (Post-Launch)

Once all content is generated:
1. Monitor Supabase storage usage
2. Check video playback analytics
3. Review learner completion rates
4. Address any reported issues same-day

---

**This directive is the source of truth for achieving happy daily learners.**

*Last updated: December 21, 2025*

