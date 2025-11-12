# Kelly Character Reference - Locked Canonical Appearance

**Purpose:** This document defines Kelly's exact appearance and art style to ensure character consistency across all asset generation.

## Base Character Description (ALWAYS INCLUDE)

**Character Identity:** Kelly Rein, "The Rein Maker's Daughter" - a photorealistic digital human with modern, timeless aesthetic. NOT a cartoon, NOT stylized, NOT fantasy. Photorealistic only.

**Physical Appearance:**
- **Face:** Oval face shape, clear skin, warm approachable expression with subtle gentle smile
- **Eyes:** Dark brown eyes, direct and engaging gaze
- **Hair:** Long, wavy dark brown hair, parted slightly off-center, falls over shoulders
- **Age:** Late 20s to early 30s
- **Build:** Athletic, capable, strong presence

**Wardrobe - Reinmaker Armor Variant (Blue-Gray Steel Consistency):**
- **Base Layer:** Dark gray ribbed turtleneck top
- **Main Garment:** Form-fitting dark charcoal-gray garment (durable textured fabric, heavy canvas or synthetic with subtle weave)
- **Structure:** Tailored, tactical appearance with structured seams and panels
- **Shoulder Pauldrons:** Metallic dark steel-colored pauldrons on both shoulders, multi-layered, riveted pieces, slightly worn but sturdy finish, curved protective design
- **Chest Straps:** Wide dark gray fabric sash draped diagonally from left shoulder to right hip, secured by dark metallic-accented straps. Wide dark metallic horizontal strap across chest below sash
- **Belts:** Multiple dark utilitarian belts around waist. Main belt with rectangular metallic buckle. Secondary belt with loops and pouches for utility
- **Sleeves:** Long form-fitting sleeves. Left sleeve extends into fingerless glove-like covering. Right sleeve has textured/wrapped detailing on forearm
- **Pants:** Dark gray pants matching upper garment in color and texture

**Wardrobe - Daily Lesson Variant (White Studio Background):**
- Modern, clean, professional attire
- Soft, approachable clothing (sweater, blouse)
- Director's chair setting
- White studio background
- Same facial features and hair as Reinmaker variant

**Color Palette (STRICT):**
- Dark grays (#333333, #2a2a2a)
- Charcoal (#1a1a1a, #1e1e1e)
- Metallic steel (#495057, #6c757d)
- Dark browns (#1b1b1b, #2d2d2d)
- NO bright colors, NO reds, NO yellows, NO light browns
- NO Roman/ancient elements
- NO leather straps, NO brass/gold (unless specified as circuit-rein accents)

## Art Style Requirements (MANDATORY)

**Style:** Photorealistic, modern timeless "Apple Genius" digital human aesthetic
- High detail rendering of skin, hair, fabric textures, metallic surfaces
- Professional photography quality
- Realistic lighting and shadows
- Natural proportions
- No stylization, no cartoon elements, no meme aesthetics

**Lighting:** Studio-quality lighting
- Soft key light at 45 degrees
- Subtle fill light
- Realistic shadows and highlights
- No harsh contrast unless specified

**Background:** 
- For Reinmaker: Dark forge/workshop environment, blue-gray steel tones
- For Daily Lesson: Clean white studio background, director's chair
- Neutral/solid backgrounds unless specified otherwise

## Negative Prompts (ALWAYS INCLUDE)

**Character:**
- cartoon, stylized, anime, illustration, drawing, sketch
- fantasy, medieval, Roman, ancient, historical
- exaggerated features, unrealistic proportions
- memes, internet humor, casual style
- second person, extra people, multiple faces

**Wardrobe:**
- bright colors, red, yellow, orange
- light browns, tan, beige
- leather straps, Roman armor, ancient elements
- casual clothing, streetwear, modern fashion
- ornate decorations, jewelry (unless specified)

**Art Style:**
- cartoon, illustration, painting, drawing
- low quality, blurry, pixelated, compression artifacts
- oversaturated colors, unrealistic lighting
- watermark, text overlay, logo

## Prompt Template Usage

When generating ANY asset featuring Kelly, use this structure:

```
[SCENE DESCRIPTION], featuring Kelly Rein, [CHARACTER BASE DESCRIPTION from above], 
wearing [WARDROBE VARIANT], [POSE/ACTION], [LIGHTING], 
photorealistic digital human, modern timeless aesthetic, 
professional photography quality, high detail, 
realistic skin textures, realistic fabric textures, realistic metallic surfaces.

Negative: [NEGATIVE PROMPTS from above], [additional asset-specific negatives]
```

## Reference Image Locations

**Primary Base Images:**
- `iLearnStudio/projects/Kelly/ref/front.png` - Front view reference
- `iLearnStudio/projects/Kelly/ref/three_quarter.png` - Three-quarter view reference  
- `iLearnStudio/projects/Kelly/ref/profile.png` - Profile view reference

**Note:** When using reference images via API, encode as base64 and include in request payload.












