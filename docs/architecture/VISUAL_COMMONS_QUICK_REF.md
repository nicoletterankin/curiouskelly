# 🎨 Visual Commons Quick Reference

**The One-Page Implementation Guide**

---

## The Loop (5 Steps)

```
1. HASH   →  SHA-256(day + phase + age + type) = content_hash
2. CHECK  →  SELECT FROM visual_commons WHERE content_hash = ?
3. HIT?   →  Return cached URL (99% of cases)
4. MISS?  →  Generate via Gemini/Imagen, upload to storage
5. CACHE  →  INSERT into visual_commons, serve forever
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/visual/check` | GET | Check cache for existing visual |
| `/api/visual/generate` | POST | Create new visual |
| `/api/visual/stats` | GET | User contribution stats |

---

## Database Tables

```sql
visual_commons          -- The main cache (content_hash unique)
visual_generation_queue -- Background generation queue
user_visual_contributions -- Gamification stats
```

---

## Key Files

```
lib/visual-prompts.ts           -- Prompt generation library
api/visual/check.ts             -- Cache lookup endpoint
api/visual/generate.ts          -- Generation endpoint
public/js/visual-commons.js     -- Frontend controller
docs/architecture/AGENTIC_VISUAL_COMMONS_PRD.md -- Full spec
```

---

## Hash Function

```typescript
const hash = SHA256(JSON.stringify({
  d: dayNumber,      // 1-365
  p: phase,          // hook, cliff, fact1...
  t: topic.lower().trim(),
  v: visualType,     // infographic, diagram, scene
  a: ageGroup,       // 2-5, 6-12, 13-17, 18+, all
  s: style,          // default
  ver: '1'           // schema version
}));
```

---

## Visual Types by Phase

| Phase | Primary Type | Template | Purpose |
|-------|--------------|----------|---------|
| Hook | infographic | radial | Create curiosity |
| Cliff | diagram | compare | Deepen mystery |
| Fact1 | infographic | process_flow | Teach foundation |
| Fact2 | diagram | cross_section | Show depth |
| Fact3 | scene | - | Wow moment |
| Wisdom | infographic | radial | Life application |
| Complete | infographic | process_flow | Summary |

---

## BYOK (Bring Your Own Key)

```javascript
// Get user's key from localStorage
const userKey = localStorage.getItem('kelly_google_api_key');

// Test key validity
const isValid = await testGeminiKey(userKey);

// Use for generation
const source = userKey ? 'byok' : 'platform';
```

---

## Prompt Layers

```
Layer 1: SYSTEM_CONTEXT    ← Brand, safety, format
Layer 2: TYPE_TEMPLATE     ← Infographic schema, scene rules
Layer 3: PHASE_CONTEXT     ← Hook question, facts, wisdom
```

---

## Infographic Brief Schema

```json
{
  "template": "cross_section|process_flow|compare|timeline|radial",
  "headline": "8 words max",
  "subhead": "16 words max",
  "callouts": [
    { "label": "4 words", "detail": "18 words", "icon": "atom" }
  ]
}
```

---

## Icons

`atom` `spark` `arrow` `leaf` `heart` `wave` `dot` `star` `bulb`

---

## Rate Limits

| Source | Daily Limit | Notes |
|--------|-------------|-------|
| BYOK | 500 | User's Google free tier |
| Platform | 100 | Per anonymous user |
| Staff | Unlimited | Admin batch generation |

---

## Error Codes

| Code | Meaning |
|------|---------|
| VC001 | Hash collision |
| VC002 | Storage upload failed |
| VC003 | API key invalid |
| VC004 | User rate limited |
| VC005 | Platform rate limited |
| VC006 | Content flagged |

---

## Badges

| Badge | Requirement |
|-------|-------------|
| 💡 First Light | 1 visual |
| 🎨 Visual Pioneer | 10 visuals |
| ✨ Illuminator | 50 visuals |
| 🌟 Master Illuminator | 100 visuals |
| 🤝 Helper | 100 learners helped |
| 🏗️ Community Builder | 1,000 learners helped |
| 🏆 Legend | 10,000 learners helped |

---

## UI States

```html
<div class="visual-slot">
  .visual-loading     <!-- Checking cache -->
  .visual-cached      <!-- Showing existing -->
  .visual-generate    <!-- CTA button -->
  .visual-generating  <!-- In progress -->
  .visual-complete    <!-- Just created -->
  .visual-error       <!-- Failed -->
</div>
```

---

## Cost Model

| Item | Cost |
|------|------|
| Gemini Infographic | $0 (text model + SVG) |
| Imagen Fast | $0.02/image |
| Imagen Ultra | $0.06/image |
| BYOK Generation | $0 (user's credits) |
| Storage (110GB) | $25/mo |

---

## Implementation Checklist

**Database (Day 1)**
- [ ] Create `visual_commons` table
- [ ] Create `visual_generation_queue` table
- [ ] Create `user_visual_contributions` table
- [ ] Set up storage bucket

**API (Day 2)**
- [ ] `/api/visual/check` endpoint
- [ ] `/api/visual/generate` endpoint
- [ ] `/api/visual/stats` endpoint

**Frontend (Day 3-4)**
- [ ] Visual slot component
- [ ] States: loading, cached, generate, generating, complete, error
- [ ] Settings UI for BYOK

**Prompts (Day 5-6)**
- [ ] Test all 7 phase prompts
- [ ] Test all 5 age adaptations
- [ ] Test all visual types

**Polish (Day 7)**
- [ ] Attribution display
- [ ] Badges/gamification
- [ ] Error handling
- [ ] Mobile responsive

---

## Key Insight

> **One generation. Infinite learners.**
> 
> The first learner to explore a concept creates the visual.
> Every learner after benefits instantly.
> That's the power of the Visual Commons.

---

*Quick Reference v1.0 | December 17, 2025*
