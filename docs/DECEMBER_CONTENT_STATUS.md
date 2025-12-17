# December Content Status — Launch Week

## Current Status (December 17, 2025)

### ✅ Day 351 (Today - December 17) — COMPLETE

| Component | Status | Details |
|-----------|--------|---------|
| Lesson JSON | ✅ Ready | `public/lessons/day-351.json` |
| Complete Day Pack | ✅ Ready | `public/data/day-351-complete.js` |
| Kelly Phase Images | ✅ Ready | `public/kelly/phases/351/` (7 images) |
| LEARN Videos | ✅ Generated | 6 archetypes in `generated-videos/day-351-manifest.json` |
| GROW Content | ✅ Ready | "Learning Accountability - Staying on Track" |
| Summary Video | 🔄 Processing | Video ID: `1216925cc46f42e29c0322e12fc504f5` |
| Email | ✅ Ready | `generated-emails/day-351-email.html` |

### Content Details

**LEARN Track:**
- 🔮 Practicing in Your Mind (Visualization)
- Headline: "Your brain can't tell the difference between doing and imagining"
- Wisdom: "The mind that rehearses grows stronger than the mind that merely waits"

**GROW Track:**
- 🎯 Learning Accountability - Staying on Track
- Activity: "Choose one person you trust and tell them about your learning goal for the week. Ask them to check in with you in 7 days."

---

## December Days (335-365) — Content Gap

### What Exists

| Location | Content Type | Days Covered |
|----------|--------------|--------------|
| `public/data/day-XXX-complete.js` | Full day packs with scripts | Days 240-365 ✅ |
| `lessons/year1-foundations/december_curriculum.json` | Curriculum titles/objectives | Days 335-365 ✅ |
| `public/lessons/day-XXX.json` | Lesson JSON for video gen | Only Day 351 ⚠️ |
| `public/data/curriculum/year2-ai-fluency/december_curriculum.json` | GROW track curriculum | Days 335-365 ✅ |

### What's Missing

**Lesson JSON files (`public/lessons/day-XXX.json`)** for Days 335-365 (except 351)

These are needed for the summary video generator which expects:
- `meta.day`, `meta.topic`, `meta.emoji`
- `phases.hook.script`, `phases.fact1.script`, etc.
- `growTrack.title`, `growTrack.activity`, `growTrack.learning_objective`

---

## Path Forward

### Option A: Extract from Day Packs (Recommended)

The complete data already exists in `public/data/day-XXX-complete.js`. 
Create a script to extract and generate lesson JSONs:

```bash
npx tsx scripts/extract-lessons-from-packs.ts --range 335-365
```

### Option B: Generate Fresh

Use the curriculum as input to generate new lesson scripts via AI.

---

## December Schedule

| Day | Date | Learn Topic | Grow Topic |
|-----|------|-------------|------------|
| 335 | Dec 1 | How Groups Make Rules | Year in Review Part 1 |
| 336 | Dec 2 | Rules Everyone Agrees On | Year in Review Part 2 |
| 337 | Dec 3 | How People Trade | Year in Review Part 3 |
| ... | ... | ... | ... |
| 351 | Dec 17 | 🔮 Practicing in Your Mind ✅ | 🎯 Learning Accountability ✅ |
| 352 | Dec 18 | Quieting Your Thoughts | Learning Joy |
| ... | ... | ... | ... |
| 365 | Dec 31 | 365 Days of Growing | The Next Chapter |

---

## Immediate Priority (Launch Day)

1. ✅ Day 351 content complete
2. ✅ Day 351 summary video generating
3. ✅ Day 351 email ready
4. ⏳ Wait for summary video to complete (~5-10 min)
5. 🔜 Upload summary video to CDN
6. 🔜 Send test email with video thumbnail

---

## Commands

```bash
# Check Day 351 summary video status
npx tsx scripts/heygen-check-status.ts 1216925cc46f42e29c0322e12fc504f5

# Generate summary video for any day with lesson JSON
npx tsx scripts/generate-email-summary-video.ts --day 351

# Dry run to preview script
npx tsx scripts/generate-email-summary-video.ts --day 351 --dry-run
```

---

*Status updated: December 17, 2025*
