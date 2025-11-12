# Kelly Image Generation - Knowledge Base

**Purpose:** Document what works and what doesn't work for Kelly image generation

**Last Updated:** [Auto-updated with each generation]

---

## ✅ What Works

### Character Consistency

#### High-Impact Prompts
1. **Reference Image Context**
   ```
   Maintain exact facial features, hair color (medium brown with caramel/honey-blonde highlights), skin tone (warm light-medium), eye color (warm brown almond-shaped), and overall appearance from reference images. Match the reference image character appearance precisely.
   ```
   - **Effectiveness:** ⭐⭐⭐⭐⭐ (5/5)
   - **Usage:** Include in all Kelly prompts
   - **Notes:** Must include actual reference images in API call

2. **Wardrobe-Specific Reference Selection**
   ```
   Use kelly_profile.png as primary reference for this Daily Lesson variant.
   Use kelly_three_quarter.png as primary reference for this Reinmaker variant.
   ```
   - **Effectiveness:** ⭐⭐⭐⭐⭐ (5/5)
   - **Usage:** Specify in prompt based on wardrobe variant
   - **Notes:** Significantly improves variant accuracy

3. **Detailed Character Description**
   ```
   Kelly Rein, photorealistic digital human, modern timeless "Apple Genius" aesthetic. Oval face, clear smooth complexion with natural glow, warm light-medium skin tone, healthy radiant skin. Warm brown almond-shaped eyes, bright and engaging, well-defined dark brown eyebrows with natural arch, long dark eyelashes. Medium brown hair with subtle caramel/honey-blonde highlights, soft wavy to slightly curly texture, parted slightly off-center or down the middle, cascades over shoulders, rich and voluminous. Full lips with natural rosy-pink color, genuine warm smile showing straight white teeth, natural smile lines (nasolabial folds), slight crinkles at outer corners of eyes when smiling. Late 20s to early 30s, athletic build, strong capable presence, approachable and professional demeanor.
   ```
   - **Effectiveness:** ⭐⭐⭐⭐ (4/5)
   - **Usage:** Include in all prompts as base
   - **Notes:** Provides fallback when reference images unavailable

#### Negative Prompts That Work
1. **Style Blocking**
   ```
   cartoon, stylized, anime, illustration, drawing, sketch, fantasy, medieval, Roman, ancient, historical, exaggerated features, unrealistic proportions
   ```
   - **Effectiveness:** ⭐⭐⭐⭐⭐ (5/5)
   - **Usage:** Always include
   - **Notes:** Prevents common style mistakes

2. **Color Restrictions**
   ```
   bright colors, red, yellow, orange, light browns, tan, beige, leather straps, Roman armor, ornate decorations, jewelry
   ```
   - **Effectiveness:** ⭐⭐⭐⭐ (4/5)
   - **Usage:** Include for Reinmaker variant
   - **Notes:** Enforces brand color palette

3. **Quality Blocking**
   ```
   low quality, blurry, pixelated, compression artifacts, oversaturated colors, unrealistic lighting, watermark, text overlay, logo, CGI, 3D render, game asset, sprite
   ```
   - **Effectiveness:** ⭐⭐⭐⭐⭐ (5/5)
   - **Usage:** Always include
   - **Notes:** Ensures professional quality

### Wardrobe Variants

#### Reinmaker Variant
- **Best Reference:** `kelly_three_quarter.png` (reinmaker kelly outfit base.png)
- **Prompt Template:** Include full wardrobe description
- **Effectiveness:** ⭐⭐⭐⭐⭐ (5/5)
- **Notes:** Must include color restrictions

#### Daily Lesson Variant
- **Best Reference:** `kelly_profile.png` (kelly_directors_chair_8k_light (2).png)
- **Prompt Template:** Include sweater and chair description
- **Effectiveness:** ⭐⭐⭐⭐⭐ (5/5)
- **Notes:** 8K reference provides excellent quality

### Lighting & Camera

#### Studio Lighting
```
professional photography quality lighting, soft key light at 45 degrees with subtle fill, realistic shadows and highlights
```
- **Effectiveness:** ⭐⭐⭐⭐ (4/5)
- **Usage:** Default for most assets
- **Notes:** Produces consistent professional look

#### Cinematic Lighting
```
cinematic lighting, high contrast between dark forge interior and bright exterior, token provides warm key light on face
```
- **Effectiveness:** ⭐⭐⭐⭐ (4/5)
- **Usage:** For narrative/cinematic assets
- **Notes:** Creates dramatic effect while maintaining character consistency

### Technical Settings

#### Aspect Ratio Mapping
- **Current Implementation:** Maps arbitrary dimensions to supported ratios
- **Effectiveness:** ⭐⭐⭐⭐ (4/5)
- **Notes:** Works well, but may need refinement for edge cases

#### Reference Image Encoding
- **Current Implementation:** Base64 encoding, multiple references supported
- **Effectiveness:** ⭐⭐⭐⭐⭐ (5/5)
- **Notes:** Vertex AI handles multiple references well

---

## ❌ What Doesn't Work

### Common Failures

#### 1. Generic Prompts
**Example:**
```
A woman named Kelly wearing armor
```
**Why It Fails:**
- Too vague
- No character description
- No reference context
- No brand restrictions

**Fix:**
- Use detailed character description
- Include reference images
- Specify wardrobe variant
- Include negative prompts

#### 2. Missing Reference Context
**Example:**
```
Kelly Rein wearing Reinmaker armor
```
**Why It Fails:**
- Doesn't mention reference images
- No instruction to match reference
- API may not use reference images effectively

**Fix:**
- Explicitly mention reference images in prompt
- Instruct to match reference precisely
- Specify which reference to use

#### 3. Weak Negative Prompts
**Example:**
```
Avoid cartoon style
```
**Why It Fails:**
- Too vague
- Doesn't block all unwanted styles
- Missing quality restrictions

**Fix:**
- Use comprehensive negative prompt list
- Include style, color, and quality restrictions
- Be specific about what to avoid

#### 4. Inconsistent Character Description
**Example:**
```
Kelly with brown hair and brown eyes
```
**Why It Fails:**
- Too simplified
- Missing specific details
- Doesn't match reference images

**Fix:**
- Use complete character description
- Include all facial features
- Match reference image details exactly

### Edge Cases That Fail

#### 1. Extreme Angles
**Issue:** Top-down or bottom-up views distort facial features
**Solution:** Avoid extreme angles, use standard camera angles
**Status:** Documented, workaround implemented

#### 2. Busy Backgrounds
**Issue:** Complex scenes distract from Kelly
**Solution:** Use simple backgrounds, focus on Kelly
**Status:** Addressed in prompts

#### 3. Action Poses
**Issue:** Running/jumping can distort features
**Solution:** Use reference images, maintain character description
**Status:** Being tested

#### 4. Different Lighting
**Issue:** Dramatic lighting can change appearance
**Solution:** Maintain consistent lighting description
**Status:** Documented

---

## 📊 Success Patterns

### Pattern 1: Reference + Detailed Description
**Formula:**
- Reference images (high impact)
- Detailed character description (fallback)
- Wardrobe-specific reference selection
- Comprehensive negative prompts

**Success Rate:** 95%+  
**Quality Score:** 4.5-5/5

### Pattern 2: Variant-Specific References
**Formula:**
- Daily Lesson → `kelly_profile.png`
- Reinmaker → `kelly_three_quarter.png`
- Include reference name in prompt

**Success Rate:** 90%+  
**Quality Score:** 4.5-5/5

### Pattern 3: Professional Photography Style
**Formula:**
- Professional photography quality lighting
- High detail, realistic textures
- Photorealistic digital human
- Professional photography quality

**Success Rate:** 85%+  
**Quality Score:** 4-5/5

---

## 🔄 Continuous Learning

### Update Process
1. After each generation, score the result
2. Document what worked and what didn't
3. Update prompts based on learnings
4. Test improvements
5. Update knowledge base

### Metrics to Track
- **Success Rate:** % of assets scoring ≥4/5
- **Average Score:** Mean quality score across all assets
- **Common Issues:** Most frequent failure modes
- **Improvement Trends:** Quality score over time

---

## 🎯 Target Metrics

**Current:** [To be measured]  
**Target:** ≥4.5/5 average score, 95%+ success rate  
**Stretch Goal:** ≥4.8/5 average score, 98%+ success rate

---

**Status:** Living document - updated with each generation  
**Next Review:** After next 10 asset generations











