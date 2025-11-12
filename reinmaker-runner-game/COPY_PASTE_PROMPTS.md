# COPY-PASTE READY: Kelly's Running Animation Prompts (Based on Reference Images)

**Reference Style:** Real person in casual clothing (light blue sweatshirt, jeans, white sneakers) on white background
**Game Style:** Stylized game sprite, cel-shaded, painterly texture
**Target:** Match the reference poses but adapt to game art style

---

## 🎨 Frame 0: Contact Frame (Current player.png - needs transparency fix)

**File Name:** `player_run_0.png` (rename from `player.png` after transparency fix)
**Reference:** Left foot forward, right leg back, weight on left foot
**Size:** 1024x1280px (master, will be scaled down in game)

### Prompt (Copy This):
```
Stylized game sprite of Kelly, the Rein Maker's Daughter, full-body running pose (CONTACT FRAME - LEFT FOOT DOWN), left leg extended forward with foot flat on ground bearing weight, right leg extended backward with heel lifted and only ball of foot touching ground, body slightly angled forward, arms swinging naturally in opposition (left arm forward bent at elbow, right arm back bent at elbow), light blue crew-neck sweatshirt, blue jeans cuffed at ankles, white sneakers, long wavy brown hair flowing, determined expression looking forward-right, light linen tunic fused with delicate brass/circuit accents (adapt reference clothing to fantasy game style), small glowing circuit-rein brooch on chest, readable silhouette, facing right, orthographic, cel-shaded, painterly texture pass, soft forge key light, cool rim light, transparent background, master size 1024x1280 for downscaling, no background, no watermark. 

Style: Game sprite, side-scrolling runner character, clean edges, consistent art style. Based on reference photo of woman in casual clothing mid-stride. Adapt the pose exactly but stylize the clothing to match game's fantasy aesthetic (light linen tunic with brass/circuit accents instead of sweatshirt, keep jeans/sneakers aesthetic but make them game-appropriate).

Negative: photo, blurry, watermark, static pose, facing left, complex background, realistic photo style.
```

---

## 🎨 Frame 1: Passing/Mid-Stride Position

**File Name:** `player_run_1.png`
**Reference:** Right leg down, left leg crossing mid-stride, body facing forward
**Size:** 1024x1280px (master, will be scaled down in game)

### Prompt (Copy This):
```
Stylized game sprite of Kelly, the Rein Maker's Daughter, full-body running pose (PASSING / MID-STRIDE FRAME), right leg planted firmly on ground with knee slightly bent bearing weight, left leg bent at knee and lifted crossing past the standing leg mid-stride with foot off ground, body more upright and facing forward, torso rotated slightly, arms swinging in opposition (right arm forward bent at elbow, left arm back bent at elbow), light blue crew-neck sweatshirt, blue jeans cuffed at ankles, white sneakers, long wavy brown hair flowing, determined expression looking forward-right, light linen tunic fused with delicate brass/circuit accents (adapt reference clothing to fantasy game style), small glowing circuit-rein brooch on chest, readable silhouette, facing right, orthographic, cel-shaded, painterly texture pass, soft forge key light, cool rim light, transparent background, master size 1024x1280 for downscaling, no background, no watermark.

Style: Game sprite, side-scrolling runner character, clean edges, consistent art style. Based on reference photo of woman in mid-stride passing position - right leg down, left leg crossing. This is the "passing" frame where legs are crossing mid-stride. Adapt the pose exactly from reference but stylize clothing to match game's fantasy aesthetic.

Negative: photo, blurry, watermark, static pose, both feet on ground, no motion, facing left, complex background, realistic photo style.
```

---

## 🎨 Frame 2: Air/Maximum Extension Position

**File Name:** `player_run_2.png`
**Reference:** Left leg forward extended, right leg back extended, both feet off ground
**Size:** 1024x1280px (master, will be scaled down in game)

### Prompt (Copy This):
```
Stylized game sprite of Kelly, the Rein Maker's Daughter, full-body running pose (AIR / MAXIMUM EXTENSION FRAME), left leg forward fully extended with foot slightly off ground preparing to land, right leg backward fully extended with foot completely off ground, both feet clearly off the ground at maximum extension, body stretched forward in full running stride, arms at maximum swing (left arm forward fully extended, right arm back fully extended), light blue crew-neck sweatshirt, blue jeans cuffed at ankles, white sneakers, long wavy brown hair flowing dramatically with motion, determined expression looking forward-right, light linen tunic fused with delicate brass/circuit accents flowing dramatically with motion (adapt reference clothing to fantasy game style), small glowing circuit-rein brooch on chest, maximum forward momentum and speed, readable silhouette, facing right, orthographic, cel-shaded, painterly texture pass, soft forge key light, cool rim light, transparent background, master size 1024x1280 for downscaling, no background, no watermark.

Style: Game sprite, side-scrolling runner character, clean edges, consistent art style. Based on reference photo of woman at maximum extension - left leg forward, right leg back, both feet off ground. This is the "peak" frame where she's fully extended and airborne. Adapt the pose exactly from reference but stylize clothing to match game's fantasy aesthetic.

Negative: photo, blurry, watermark, static pose, feet on ground, no motion, facing left, complex background, legs together, compact pose, realistic photo style.
```

---

## 📋 Key Pose Details from Reference Images

### Frame 0 (Contact):
- **Left foot:** Flat on ground, bearing weight
- **Right foot:** Ball of foot touching, heel lifted
- **Arms:** Left forward, right back (opposite to legs)
- **Body:** Slightly angled forward, head looking forward-right

### Frame 1 (Passing):
- **Right leg:** Firmly planted, knee bent, bearing weight
- **Left leg:** Bent at knee, crossing past right leg, foot off ground
- **Arms:** Right forward, left back (opposite to legs)
- **Body:** More upright, torso facing forward

### Frame 2 (Air/Maximum Extension):
- **Left leg:** Forward fully extended, foot off ground
- **Right leg:** Back fully extended, foot off ground
- **Arms:** Left forward extended, right back extended
- **Body:** Fully stretched forward, maximum extension

---

## 🎨 Art Style Adaptation Notes

**From Reference:** Casual modern clothing (sweatshirt, jeans, sneakers)
**To Game Style:** Fantasy aesthetic (light linen tunic, brass/circuit accents, game-appropriate footwear)

**Key Adaptations:**
- Sweatshirt → Light linen tunic with brass/circuit decorative elements
- Keep the casual, practical feel
- Add small glowing circuit-rein brooch on chest
- Maintain the same poses and proportions
- Keep hair style (long wavy brown)
- Keep expression (determined, focused)
- Maintain the clean, readable silhouette

---

## 📋 Checklist After Generation

- [ ] All 3 frames have transparent backgrounds (PNG-24 with alpha)
- [ ] All 3 frames are 1024x1280px
- [ ] All 3 frames match the art style (stylized, cel-shaded, painterly)
- [ ] All 3 frames face right
- [ ] All 3 frames match the reference poses accurately
- [ ] Clothing adapted to fantasy style but maintains reference proportions
- [ ] Clear running motion progression across all 3 frames
- [ ] Consistent character design across all frames

---

## ⚠️ IMPORTANT: Also Fix Existing Files

Before using these new frames, you MUST fix the existing `player.png`:
1. Remove opaque background (make transparent)
2. Export as PNG-24 with transparency
3. Rename to `player_run_0.png`

**Same fixes needed for:**
- `obstacle.png` - Remove background
- All 7 stone files - Remove backgrounds

See `PNG_ASSET_ANALYSIS.md` for full list of fixes needed.
