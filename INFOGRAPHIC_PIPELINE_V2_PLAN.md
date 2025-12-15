## Infographic Pipeline v2 (Production Quality)

### Problem
- Image-generation models frequently produce **garbled/incorrect text** inside images.
- Infographics require **precise typography, correct labels, and consistent brand layout**.
- Therefore: **no raw image-model “infographics” go into Supabase.**

### Core approach (Gemini + deterministic renderer)
- **Gemini does not “draw the infographic.”**
- Gemini produces a **structured infographic brief** (JSON): headline, subhead, callouts/steps/compare bullets, icons.
- We render a **Curious Kelly brand-locked SVG**:
  - Crisp real text (no hallucinated glyphs)
  - Locked palette + typography
  - Deterministic layout templates
- Optional later: rasterize SVG → PNG/WebP for runtime compatibility.

### Quality gates (hard)
- **Default local-only** generation.
- Uploads require explicit opt-in:
  - `--allow-upload` or `CK_ALLOW_SUPABASE_UPLOADS=1`
- DB mutations require explicit opt-in:
  - `--allow-db-mutation` or `CK_ALLOW_DB_MUTATIONS=1`
- Human review required on a proof pack (10–20 lessons) before any bulk run.

### Current safety state
- Visual generator uploads stopped.
- Bad visuals quarantined: `lesson_atoms.visual_url` cleared for Days 1–50.
- UI infographic button disabled via config until approval.

### Proof pack workflow
1. Generate proofs (local-only):

```bash
npx tsx scripts/gemini-infographic-proof.ts --day=6  --template=process_flow
npx tsx scripts/gemini-infographic-proof.ts --day=7  --template=cross_section
npx tsx scripts/gemini-infographic-proof.ts --day=10 --template=compare
npx tsx scripts/gemini-infographic-proof.ts --day=33 --template=cross_section
```

2. Review in browser:
- `public/infographic-proof.html`

3. Iterate on:
- brief schema constraints (length, allowed icons, required fields)
- renderer templates (spacing, hierarchy, iconography)
- brand palette/typography lock

### Scale plan (after approval)
- Generate one infographic SVG per lesson phase (Hook/Fact1/Fact2/Fact3/Wisdom) using templates.
- Run automated checks:
  - no external images
  - text length bounds
  - SVG size limits
  - deterministic template usage
- Only then enable controlled upload to Supabase storage + DB linking.
