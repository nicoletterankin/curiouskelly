# DAY 17 - IMAGE GENERATION CHECKLIST

**Status:** Interactive system deployed, images needed  
**Priority:** CRITICAL - 2 days to launch  
**Total Images Needed:** 42 (7 phases × 6 images each)

---

## IMAGES NEEDED PER PHASE

### Hook Phase (6 images)
- [ ] `hook_option_a.svg` - "Feel It First" - person stretching
- [ ] `hook_option_b.svg` - "Understand Why" - brain/science icon
- [ ] `hook_success_a.svg` - Energy surge visualization
- [ ] `hook_success_b.svg` - Blood flow diagram
- [ ] `hook_alt_a.svg` - Small movements count
- [ ] `hook_alt_b.svg` - Fidgeting burns calories

### Cliff Phase (6 images)
- [ ] `cliff_option_a.svg` - "Data & Research" - charts/graphs
- [ ] `cliff_option_b.svg` - "Personal Experiment" - experiment icon
- [ ] `cliff_success_a.svg` - Statistics visualization
- [ ] `cliff_success_b.svg` - Experiment results
- [ ] `cliff_alt_a.svg` - Longitudinal studies
- [ ] `cliff_alt_b.svg` - Body truth visualization

### Fact1 Phase (6 images)
- [ ] `fact1_option_a.svg` - "See the System" - muscle map
- [ ] `fact1_option_b.svg` - "Learn the Mechanics" - mechanics diagram
- [ ] `fact1_success_a.svg` - Synchronized muscle patterns
- [ ] `fact1_success_b.svg` - Muscle-tendon-bone system
- [ ] `fact1_alt_a.svg` - Muscle group coordination
- [ ] `fact1_alt_b.svg` - ATP energy in muscles

### Fact2 Phase (6 images)
- [ ] `fact2_option_a.svg` - "Prevent Disease" - shield/protection
- [ ] `fact2_option_b.svg` - "Optimize Health" - upward arrow/growth
- [ ] `fact2_success_a.svg` - Disease risk reduction
- [ ] `fact2_success_b.svg` - Performance boost
- [ ] `fact2_alt_a.svg` - Cellular anti-inflammation
- [ ] `fact2_alt_b.svg` - Resilient systems

### Fact3 Phase (6 images)
- [ ] `fact3_option_a.svg` - "See the Damage" - weakening visualization
- [ ] `fact3_option_b.svg` - "Learn Recovery" - recovery path
- [ ] `fact3_success_a.svg` - Sitting vs active comparison
- [ ] `fact3_success_b.svg` - 5-minute breaks effect
- [ ] `fact3_alt_a.svg` - Timeline of muscle loss
- [ ] `fact3_alt_b.svg` - Strength rebuilding curve

### Wisdom Phase (6 images)
- [ ] `wisdom_option_a.svg` - "The Metaphor" - engine icon
- [ ] `wisdom_option_b.svg` - "The Practical Truth" - real-world icon
- [ ] `wisdom_success_a.svg` - Engine needs fuel/motion
- [ ] `wisdom_success_b.svg` - Movement as medicine
- [ ] `wisdom_alt_a.svg` - Body maintains through use
- [ ] `wisdom_alt_b.svg` - Investment in longevity

### Outro Phase (6 images)
- [ ] `outro_option_a.svg` - "Commit to Action" - action icon
- [ ] `outro_option_b.svg` - "Reflect First" - reflection icon
- [ ] `outro_success_a.svg` - Momentum visualization
- [ ] `outro_success_b.svg` - Understanding deepens
- [ ] `outro_alt_a.svg` - Small steps concept
- [ ] `outro_alt_b.svg` - Return anytime

---

## GENERATION STRATEGY

### Option 1: Gemini App (Recommended)
1. Use existing Gemini infographic app
2. Input: Day 17 data (see DAY_17_VISUAL_ASSET_MATRIX.md)
3. Generate 42 SVG files
4. Upload to Supabase: `lesson-visuals/day_017/{filename}.svg`

### Option 2: Placeholder SVGs (Temporary)
Use simple SVG placeholders until real images ready:
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300">
  <rect fill="#18181b" width="400" height="300"/>
  <text x="50%" y="50%" text-anchor="middle" fill="#2563eb" font-size="20">
    Option A
  </text>
</svg>
```

### Option 3: Batch via Replicate
Use Flux Pro to generate all 42 images:
```bash
npx tsx scripts/generate-day17-choice-images.ts
```

---

## UPLOAD SCRIPT

```bash
# After generating images locally
node -e "
const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function uploadImages() {
  const imageDir = './generated-images/day-017';
  const files = fs.readdirSync(imageDir);
  
  console.log('Uploading', files.length, 'images...');
  
  for (const file of files) {
    const filePath = path.join(imageDir, file);
    const buffer = fs.readFileSync(filePath);
    
    const storagePath = \`day_017/\${file}\`;
    
    const { error } = await supabase.storage
      .from('lesson-visuals')
      .upload(storagePath, buffer, {
        contentType: 'image/svg+xml',
        upsert: true
      });
    
    if (error) {
      console.log('❌', file, error.message);
    } else {
      const { data } = supabase.storage
        .from('lesson-visuals')
        .getPublicUrl(storagePath);
      console.log('✅', file, '→', data.publicUrl);
    }
  }
}

uploadImages();
"
```

---

## BRAND REQUIREMENTS FOR IMAGES

### Colors
- Primary: Kelly Blue (#2563eb)
- Background: Dark (#09090b - #18181b)
- Text: White (#fafafa)
- Success: Green (#10b981)
- Wisdom: Gold (#f59e0b)

### Style
- Clean, modern, educational
- Icons and diagrams (not photos)
- High contrast for readability
- Mobile-friendly (works at 320px width)

### Dimensions
- 400×300px (4:3 aspect ratio)
- SVG format preferred (scalable)
- PNG acceptable (optimize for web)

---

## PRIORITY ORDER

### Phase 1: Critical Path (Launch Day)
1. **Hook images** (first impression) - HIGHEST PRIORITY
2. **Cliff images** (choice point) - HIGH PRIORITY
3. **Wisdom images** (memorable close) - HIGH PRIORITY

### Phase 2: Complete Experience
4. Fact1 images
5. Fact2 images
6. Fact3 images
7. Outro images

### Phase 3: Polish
- Replace any placeholder images
- A/B test different visual styles
- Optimize file sizes

---

## TESTING CHECKLIST

After images uploaded:

### Desktop
- [ ] Images load on all phases
- [ ] Hover effects work
- [ ] Click triggers response
- [ ] No broken images

### Mobile
- [ ] Options stack vertically
- [ ] Touch interactions work
- [ ] Images scale properly
- [ ] Text is readable

### Performance
- [ ] Images load < 1s
- [ ] No layout shift
- [ ] Smooth transitions
- [ ] No memory leaks

---

## FALLBACK PLAN

If images not ready by launch:
1. ✅ System works with placeholder SVGs
2. ✅ Text-only mode still functional
3. ✅ Can add images post-launch
4. ✅ No breaking changes required

---

## TIMELINE

**Now:** Interactive system deployed  
**+2 hours:** Generate 42 images  
**+4 hours:** Upload and test  
**Dec 17:** Launch with complete interactive experience

---

## COST ESTIMATE

### Using Flux Pro (Replicate)
- 42 images × $0.04 = **$1.68**

### Using Gemini (Free Tier)
- 42 images × $0 = **$0** (within daily limit)

### Recommendation
Use Gemini free tier for Day 17, then scale with Flux Pro for remaining 364 days.


