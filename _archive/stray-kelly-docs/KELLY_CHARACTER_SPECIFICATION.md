# Kelly Character Specification - Complete Reference

**Version:** 1.0  
**Last Updated:** November 1, 2025  
**Source:** Kelly Reference Assets + Uncanny Blueprint + Visual Analysis

---

## 🎯 Character Overview

**Kelly Rein** is a photorealistic digital human designed for "The Daily Lesson" project and "The Rein Maker's Daughter" game. She serves as the primary teaching avatar and game protagonist, requiring perfect character consistency across all media.

**Key Characteristics:**
- **Age Range:** Late 20s to early 30s
- **Build:** Athletic, strong capable presence
- **Personality:** Approachable, professional, warm, engaging
- **Aesthetic:** Modern timeless "Apple Genius" aesthetic

---

## 👤 Facial Features

### Face Structure
- **Shape:** Oval face
- **Complexion:** Clear, smooth complexion with natural glow
- **Skin Tone:** Warm light-medium skin tone
- **Skin Quality:** Healthy, radiant skin
- **Texture:** Photorealistic pores, natural texture (see `close up pores .png`)

### Eyes
- **Shape:** Almond-shaped
- **Color:** Warm brown
- **Expression:** Bright and engaging
- **Eyebrows:** Well-defined dark brown eyebrows with natural arch
- **Eyelashes:** Long dark eyelashes
- **Catchlight:** Natural catchlight in eyes (added in iClone)

### Hair
- **Base Color:** Medium brown
- **Highlights:** Subtle caramel/honey-blonde highlights
- **Texture:** Soft waves that fall together cohesively in unified sections
- **Style:** Hair strands stay together (NOT frizzy, NOT whispy, NOT curly)
- **Finish:** Smooth polished texture, professional appearance
- **Cohesion:** Hair moves as cohesive wave sections - no separate curls, no frizz, no flyaways
- **Parting:** Slightly off-center or down the middle
- **Length:** Cascades over shoulders
- **Volume:** Rich and voluminous
- **Movement:** Natural, smooth wave pattern with hair staying together

### Nose
- **Bridge:** Slight angle (approximately -5 to -10 degrees from photos)
- **Shape:** Natural, proportionate

### Mouth & Expression
- **Lips:** Full lips with natural rosy-pink color
- **Smile:** Genuine warm smile showing straight white teeth
- **Smile Lines:** Natural smile lines (nasolabial folds)
- **Eye Crinkles:** Slight crinkles at outer corners of eyes when smiling
- **Expression:** Approachable, professional demeanor

---

## 👕 Wardrobe Variants

### Reinmaker Variant (Game Assets)
**Primary Reference:** `reinmaker kelly outfit base.png`

**Base Layer:**
- Dark gray ribbed turtleneck base layer

**Upper Garment:**
- Form-fitting dark charcoal-gray tactical garment
- Structured seams and panels
- Form-fitting sleeves

**Armor Elements:**
- Metallic dark steel-colored shoulder pauldrons
  - Multi-layered design
  - Riveted construction
  - Curved protective design
- Wide dark gray fabric sash draped diagonally from left shoulder to right hip
- Sash secured by dark metallic straps
- Wide dark metallic horizontal strap across chest

**Accessories:**
- Multiple dark utilitarian belts around waist
- Rectangular metallic buckle
- Fingerless glove-like covering on left hand
- Textured wrapped detailing on right forearm

**Lower Garment:**
- Dark gray tactical pants matching upper garment

**Color Palette:**
- Dark grays
- Charcoal
- Metallic steel
- Dark browns

**Restrictions:**
- ❌ NO bright colors
- ❌ NO reds
- ❌ NO yellows
- ❌ NO light browns
- ❌ NO Roman/ancient elements

### Daily Lesson Variant (Teaching Avatar)
**Primary Reference:** `kelly_directors_chair_8k_light (2).png`

**Upper Garment:**
- Light blue ribbed knit sweater with crew neck
- Soft muted blue color
- Clean contemporary wardrobe

**Seating:**
- Classic director's chair with dark brown wooden frame
- Black canvas seat/backrest
- Visible armrests

**Environment:**
- Clean bright white or very light gray studio background
- Plain uncluttered environment
- Soft even studio lighting with subtle shadows
- Professional photography setup

**Character Consistency:**
- Same facial features and hair as Reinmaker variant
- Warm brown eyes
- Medium brown hair with caramel highlights
- Oval face
- Genuine smile

---

## 🎨 Art Style Requirements

### Photorealistic Quality
- **Style:** Photorealistic digital human
- **Quality:** Professional photography quality
- **Detail Level:** High detail, realistic textures
- **Skin:** Realistic skin textures with natural pores
- **Fabric:** Realistic fabric textures
- **Metallic:** Realistic metallic surfaces (for armor)

### Lighting
- **Type:** Professional photography quality lighting
- **Setup:** Soft key light at 45 degrees with subtle fill
- **Shadows:** Realistic shadows and highlights
- **Studio:** Clean, even lighting for Daily Lesson variant
- **Cinematic:** High contrast for Reinmaker variant

### Camera
- **Portrait:** 85mm focal length (per Uncanny Blueprint)
- **Eye-Level:** Camera at eye level
- **DOF:** Depth of field on irises
- **Format:** Professional photography setup

---

## ❌ Mandatory Negative Prompts

**Style Restrictions:**
- cartoon, stylized, anime, illustration, drawing, sketch
- fantasy, medieval, Roman, ancient, historical
- exaggerated features, unrealistic proportions
- memes, internet humor, casual style

**Character Restrictions:**
- second person
- extra people, multiple faces
- bright colors, red, yellow, orange
- light browns, tan, beige
- leather straps (Roman style), Roman armor
- ornate decorations, jewelry

**Quality Restrictions:**
- low quality, blurry, pixelated, compression artifacts
- oversaturated colors, unrealistic lighting
- watermark, text overlay, logo
- CGI, 3D render, game asset, sprite

**Reinmaker-Specific:**
- Avoid: painterly texture, vector style, stylized illustration
- Prefer: Photorealistic rendering with realistic materials

---

## 📸 Reference Images

### Primary References
1. **`kelly_front.png`** (from `headshot2-kelly-base169 101225.png`)
   - Primary Headshot 2 input photo
   - Front-facing, high resolution
   - Studio-lit, professional quality
   - Use for: General character consistency

2. **`kelly_profile.png`** (from `kelly_directors_chair_8k_light (2).png`)
   - Director's chair variant
   - 8K resolution
   - Daily Lesson wardrobe
   - Use for: Daily Lesson variant assets

3. **`kelly_three_quarter.png`** (from `reinmaker kelly outfit base.png`)
   - Reinmaker armor variant
   - Full outfit reference
   - Use for: Reinmaker variant assets

### Additional References
- **Numbered headshots:** `8.png`, `9.png`, `12.png`, `24.png`, `32.png` (multiple angles)
- **Generated images:** October 12-13, 2025 iterations (consistency patterns)
- **Bald reference:** `bald-kelly reference.png` (3D modeling reference)
- **Texture reference:** `close up pores .png` (skin texture detail)

---

## 🎬 Animation & Performance (From Uncanny Blueprint)

### Lip-Sync
- **Tool:** AccuLips visemes from WAV
- **Accuracy:** Custom dictionary, coarticulation smoothing 0.2-0.4
- **QA Target:** Forced alignment drift ≤ ±3 frames

### Facial Nuance
- **Tool:** AccuFACE VIDEO driving brows/lids/cheeks/head
- **Mouth/Jaw:** Disabled (AccuLips drives these)
- **Source:** HeyGen video reference

### Eye Behavior
- **Blinks:** 12-18/min, 120-200ms closure
- **Rest Blinks:** Occasional longer rest blink
- **Saccades:** Small gaze shifts every 1-3s
- **Head-Nod Coupling:** Micro head-nod coupling 1-2°
- **Pupil Dilation:** Slight dilation on excitement (+3-5%)

### Breathing
- **Rate:** 5-9 cycles/min
- **Motion:** Subtle chest/shoulder motion

---

## 🔧 Technical Pipeline Requirements

### 3D Model
- **Base:** CC5 HD with HS2 wrap
- **Render SubD:** Level 2 (Level 3 for hero)
- **Topology:** CC5 HD topology
- **FACS Rig:** 45-60 primary expressions + ≥20 corrective shapes

### Materials & Shading
- **Skin Shader:** Digital Human Shader (DHS)
- **Maps:** Base, normal, micro-normal, roughness map with T-zone variation
- **SSS:** ~0.28 ±0.05
- **Roughness:** 0.38-0.42 (T-zone variation)
- **Wrinkle Normals:** Tied to expression curves

### Eyes
- **Cornea:** Separate cornea with bulge
- **Tear-Line:** Tear-line mesh
- **Iris Parallax:** Yes
- **Sclera Veins:** 2-4K resolution

### Hair
- **Style:** Multi-card style + baby hairs
- **Physics:** Simmer ≤ 0.3cm amplitude
- **Specular:** 0.25 for natural look

### Rendering
- **Engine:** Realtime PBR (iClone 8.62) with TAA
- **Color:** Rec.709 export
- **Hero Option:** Iray plug-in for stills (if licensed)
- **Motion Blur:** On for head turns, shutter 180° equiv.

---

## ✅ QA Checklist

### Likeness
- [ ] 20-landmark average px error (4K) ≤ 3px
- [ ] Facial features match reference images
- [ ] Hair color matches (medium brown with caramel highlights)
- [ ] Skin tone matches (warm light-medium)

### Lip-Sync
- [ ] Forced alignment drift ≤ ±3 frames
- [ ] Natural mouth movement
- [ ] No viseme popping

### Eye Realism
- [ ] Blink rate: 12-18/min + randomness
- [ ] No sclera pop-through
- [ ] Natural catchlight
- [ ] Saccades every 1-3s

### Skin Quality
- [ ] No pore swimming under expression
- [ ] Realistic texture
- [ ] Natural SSS
- [ ] No plastic appearance

### Wardrobe Consistency
- [ ] Reinmaker variant matches `reinmaker kelly outfit base.png`
- [ ] Daily Lesson variant matches `kelly_directors_chair_8k_light (2).png`
- [ ] Color palette correct (no bright colors, no reds/yellows)

### Overall Quality
- [ ] Photorealistic (not cartoon/stylized)
- [ ] Professional photography quality
- [ ] No artifacts, compression, or quality issues
- [ ] Character consistency across all assets

---

## 📊 Character Consistency Scorecard

**Target:** Mean "looks/feels real" score ≥ 4.2/5

**Testing Method:**
- 30-viewer blind A/B test
- Real video vs Kelly
- Randomized 15-second clips
- Record comments for failure analysis

---

## 📝 Notes

### From Reference Analysis
- Primary reference (`headshot2-kelly-base169 101225.png`) is the canonical source
- Director's Chair variant (`kelly_directors_chair_8k_light (2).png`) is 8K quality
- Reinmaker outfit reference shows full tactical armor design
- Generated images from October 12-13 show consistency patterns

### From Documentation
- Pipeline uses Headshot 2 → Photo to 3D workflow
- Requires G3+ base character (not CC5 HD initially)
- Convert to CC5 HD after Headshot 2 processing
- AccuLips + AccuFACE hybrid workflow for facial animation

### Best Practices
1. Always use primary reference images when available
2. Select appropriate reference based on wardrobe variant
3. Maintain photorealistic style (no stylization)
4. Enforce color palette restrictions
5. Verify consistency across all generated assets

---

**Document Maintained By:** AI Assistant (Kelly Expert)  
**Last Review:** November 1, 2025  
**Status:** Active - Ready for Production Use

