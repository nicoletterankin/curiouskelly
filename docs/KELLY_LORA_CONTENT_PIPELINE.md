# 🎨 Kelly LoRA + Content Population Pipeline

> **Status:** Ready to wire  
> **LoRA Model:** https://huggingface.co/CuriousKellycom/curious-kelly-lora  
> **Goal:** Generate visuals for all 730 lessons using community BYOK credits

---

## 📊 Current Content Status

### Core Lessons
| Track | Lessons | Topics | Truths | Facts | Reflections |
|-------|---------|--------|--------|-------|-------------|
| **Learn** | 365 | ✅ 365 | ✅ 365 | ✅ 365 | ✅ 365 |
| **Grow** | 365 | ✅ 365 | ✅ 365 | ❌ 0 | ❌ 7 |

### Visual Assets
| Asset Type | Count | Target | Gap |
|------------|-------|--------|-----|
| Visual Commons | 330 | 730 | 400 |
| Kelly Videos (validated) | 241 | 5,110 | 4,869 |
| Lesson Atoms | 20,481 | ~51,000 | ~30,500 |
| Lesson Shards | 60 | ? | ? |

---

## 🎯 Kelly LoRA Configuration

```typescript
const KELLY_LORA = {
  // Hugging Face location
  url: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  
  // Replicate model (for API generation)
  replicateModel: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
  
  // LoRA settings
  scale: 0.85,
  trigger: 'kelly',
  
  // Base prompt (LOCKED)
  basePrompt: `kelly, photorealistic woman named Kelly, late 20s to early 30s, 
brown wavy shoulder-length hair with caramel and honey highlights center-parted, 
hazel-brown almond-shaped eyes, soft symmetrical features with natural makeup, 
light-medium warm skin tone with healthy glow, 
wearing soft powder blue cashmere crewneck sweater, 
warm but professional expression, intelligent curious eyes`,
};
```

---

## 🔗 BYOK Integration for LoRA Generation

### Provider Mapping

| Provider | Use Case | BYOK Status |
|----------|----------|-------------|
| **Replicate** | Run FLUX + LoRA | ✅ Added |
| **Fal.ai** | Fast inference, SadTalker | ✅ Added |
| **Stability AI** | SD fallback | ✅ Added |
| **Together AI** | Image models | ✅ Added |

### Generation Flow

```
User BYOK Key (Replicate/Fal) 
        ↓
Kelly LoRA loaded from HF
        ↓
Phase-specific prompt generated
        ↓
Image generated with LoRA
        ↓
Uploaded to Supabase
        ↓
logContribution() called
        ↓
Dashboard updated
```

---

## 📝 Content Population Tasks

### 1. Grow Track: Fun Facts (0 → 365)

Each Grow track lesson needs 3-5 fun facts about the AI/meta-learning topic.

```sql
-- Example structure
UPDATE core_lessons
SET fun_facts = '["Fact 1 about AI topic", "Fact 2", "Fact 3"]'::jsonb
WHERE track = 'grow' AND day_number = 1;
```

**Source:** `lessons/year2-ai-fluency/*.json` files contain detailed lesson content.

### 2. Grow Track: Reflection Prompts (7 → 365)

Each lesson needs 2-3 reflection prompts.

```sql
UPDATE core_lessons
SET reflection_prompts = '["How might this change your understanding?", "What questions does this raise?"]'::jsonb
WHERE track = 'grow' AND day_number = 1;
```

### 3. Visual Commons (330 → 730)

Need 400 more visuals to cover both tracks.

**Generation priority:**
1. Today's lesson (Day 354)
2. Next 7 days
3. Backfill remaining

---

## 🔧 Scripts to Wire

### 1. Add Replicate to BYOK Frontend

Already done in `public/learn.html`:
- Input for Replicate key (`r8_...`)
- Connect button
- Link to get API key

### 2. Visual Generation with BYOK

File: `scripts/byok-visual-generator.ts`

```typescript
async function generateVisualWithBYOK(
  dayNumber: number,
  phase: string,
  track: 'learn' | 'grow',
  provider: 'replicate' | 'fal' | 'stability'
) {
  // Get lesson data
  const lesson = await getLesson(dayNumber, track);
  
  // Build prompt
  const prompt = buildKellyPrompt(lesson.topic, phase);
  
  // Generate with user's BYOK key
  const imageUrl = await generateWithProvider(provider, prompt, KELLY_LORA);
  
  // Upload to Supabase
  const assetUrl = await uploadToSupabase(imageUrl, dayNumber, phase, track);
  
  // Log contribution
  await logContribution(provider, 'image', dayNumber, phase, 5); // ~$0.05/image
  
  return assetUrl;
}
```

### 3. Batch Content Population

File: `scripts/populate-grow-track-content.ts`

```typescript
async function populateGrowTrackContent() {
  // Read all year2-ai-fluency JSON files
  const months = [
    'month-01-foundations',
    'month-02-questioning', 
    // ... all 12 months
  ];
  
  for (const month of months) {
    const data = await readFile(`lessons/year2-ai-fluency/${month}.json`);
    
    for (const day of data.days) {
      await supabase.from('core_lessons').update({
        fun_facts: generateFunFacts(day),
        reflection_prompts: generateReflectionPrompts(day)
      }).eq('day_number', day.day).eq('track', 'grow');
    }
  }
}
```

---

## 🎬 Phase-to-Template Mapping

Each lesson phase uses a specific Kelly template:

| Phase | Template | Kelly Pose | LoRA Prompt Suffix |
|-------|----------|------------|-------------------|
| Hook | `excited` | Arms up, big smile | "excited expression, arms raised, welcoming" |
| Cliff | `curious` | Head tilt, questioning | "curious expression, head tilted, questioning look" |
| Fact 1-3 | `explain` | Gesturing, teaching | "explaining gesture, engaged, teaching" |
| Wisdom | `heartfelt` | Hand on heart | "sincere expression, hand on heart, warm" |
| Outro | `welcome` | Arms open | "welcoming pose, arms open, proud smile" |

---

## 🚀 Immediate Action Items

### This Session
1. [ ] Create `scripts/populate-grow-content.ts` to fill fun_facts + reflection_prompts
2. [ ] Wire Replicate LoRA generation to BYOK flow
3. [ ] Test visual generation with community key

### This Week
4. [ ] Generate visuals for days 354-365 (current + upcoming)
5. [ ] Backfill visuals for days 1-30
6. [ ] Test full pipeline: BYOK key → LoRA → visual → contribution logged

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `lora_url.txt` | LoRA model URL |
| `scripts/fill-supabase-with-assets.ts` | Existing asset generator |
| `scripts/kelly-lora-asset-factory.ts` | LoRA-specific factory |
| `lessons/year2-ai-fluency/` | Grow track content source |
| `public/js/byok-manager.js` | BYOK key management |

---

*"Kelly's face is consistent because of the LoRA. Every visual is recognizably HER."*
