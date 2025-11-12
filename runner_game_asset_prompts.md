# Runner Game Asset Prompts: The Rein Maker's Daughter
**Version:** 1.0
**Source Brief:** User-provided asset list.

This document contains the final, copy-paste ready prompts for generating all 2D game assets.

---
## A. Core Gameplay Sprites
---

### A1. Player: Kelly (Runner)
- **File:** `assets/player.png`
- **Master Size:** 1024x1280px (for downscaling)
- **Final Specs:** ~52x64px, pivot bottom-center, transparent BG.
- **Prompt:**
```
Stylized game sprite of Kelly, the Rein Maker’s Daughter, full-body, neutral run pose, light linen tunic fused with delicate brass/circuit accents, short practical boots, small glowing circuit-rein brooch on chest, readable silhouette, facing right, orthographic, cel-shaded, painterly texture pass, soft forge key light, cool rim light, transparent background, master size 1024x1280 for downscaling, no background, no watermark. Negative: photo, blurry, watermark.
```

---

### A2. Obstacle — Knowledge Shards
- **File:** `assets/obstacle.png`
- **Master Size:** 512px tall
- **Final Specs:** Height variants 40/60/80px, pivot bottom-center, transparent BG.
- **Prompt:**
```
Stylized game obstacle: vertical circuit-rein shard totem, forged brass and steel with a glowing teal inlay, simple sharp geometry, clean edges, orthographic, cel-shaded, painterly texture, warm forge key light, cool rim light, transparent background, 512px tall master, no base shadow, no background. Negative: complex, detailed, rounded, photo.
```

---

### A3. Ground Stripe
- **File:** `assets/ground_stripe.png`
- **Final Specs:** 60x6px, opaque, transparent BG.
- **Prompt:**
```
Minimal road dash sprite, 60x6 px, rounded ends, off-white (#F2F7FA), flat fill, orthographic, transparent background, no shadow.
```

---
## B. Background & Environment
---

### B1. Parallax Skyline Strip
- **File:** `assets/bg.png`
- **Final Specs:** 1024x256px, seamless horizontally, opaque.
- **Prompt:**
```
Tiling horizontal skyline for a side-scrolling runner game, silhouettes of a mythic academy / Hall of the Seven Tribes in the far distance, faint vertical banners in seven colors (red, blue, green, yellow, purple, orange, white), subtle painterly clouds, dusk palette matching a #0B1020 night steel sky with ember-colored accents on the horizon, seamless horizontally, 1024x256px, soft parallax readability, no foreground details, no text.
```
---

### B2. Ground Texture
- **File:** `assets/ground_tex.png`
- **Final Specs:** 512x64px tile, seamless horizontally.
- **Prompt:**
```
Seamless ground strip tile, dark steel-stone texture with very faint forge specks, low contrast, 512x64px, seamless horizontally, orthographic, transparent background. Negative: busy, high contrast.
```

---
## C. UI & Meta
---

### C1. Logo / Title Card
- **Files:** `marketing/cover-1280x720.png`, `marketing/square-600.png`
- **Final Specs:** 1280x720px and 600x600px, transparent BG.
- **Prompt:**
```
Key art logo cinematic title card, text reads 'The Rein Maker’s Daughter', elegant serif combined with a clean geometric sans-serif font, features a central emblem of a broken leather rein seamlessly reshaping into a glowing circuit loop, brass and ember (#D8A24A) accents on the emblem, deep blue steel (#495057) background vignette, cinematic composition, crisp, high contrast, transparent background, export at 1280x720.
```

---

### C2. Favicon / App Icon
- **File:** `assets/favicon.png`
- **Master Size:** 256x256px
- **Final Specs:** Export to 32x32/64x64.
- **Prompt:**
```
Icon only: a glowing circuit-rein emblem, simple 2-tone (teal #0BB39C and graphite #1B1E22), centered, flat background, clean and readable silhouette, 256x256px, no background.
```

---
## D. Lore Collectibles
---

### D1. Knowledge Stones (7)
- **Files:** `assets/stones/stone_{tribe}.png`
- **Final Specs:** 64x64px each, transparent BG.

- **Light:** `Stylized gem icon, for the Light Tribe, color #F2F7FA, subtle etched symbol of an eye, soft inner glow, clean silhouette, 64x64, transparent background.`
- **Stone:** `Stylized gem icon, for the Stone Tribe, color #8E9BA7, subtle etched symbol of a mountain, soft inner glow, clean silhouette, 64x64, transparent background.`
- **Metal:** `Stylized gem icon, for the Metal Tribe, color #adb5bd, subtle etched symbol of a gear, soft inner glow, clean silhouette, 64x64, transparent background.`
- **Code:** `Stylized gem icon, for the Code Tribe, color #0BB39C, subtle etched symbol of code brackets '<>', soft inner glow, clean silhouette, 64x64, transparent background.`
- **Air:** `Stylized gem icon, for the Air Tribe, color #aed9e0, subtle etched symbol of a feather, soft inner glow, clean silhouette, 64x64, transparent background.`
- **Water:** `Stylized gem icon, for the Water Tribe, color #4dabf7, subtle etched symbol of a wave, soft inner glow, clean silhouette, 64x64, transparent background.`
- **Fire:** `Stylized gem icon, for the Fire Tribe, color #F25F5C, subtle etched symbol of a flame, soft inner glow, clean silhouette, 64x64, transparent background.`

---

### D2. Tribe Banners (7)
- **Files:** `assets/banners/banner_{tribe}.png`
- **Final Specs:** 128x256px, tileable vertically, transparent BG.

- **Light:** `Vertical banner, fabric weave texture, bold symbol for the Light Tribe (eye), base color #F2F7FA, subtle gold trim, orthographic, transparent background, 128x256px.`
- **Stone:** `Vertical banner, fabric weave texture, bold symbol for the Stone Tribe (mountain), base color #8E9BA7, subtle gold trim, orthographic, transparent background, 128x256px.`
- **Metal:** `Vertical banner, fabric weave texture, bold symbol for the Metal Tribe (gear), base color #adb5bd, subtle gold trim, orthographic, transparent background, 128x256px.`
- **Code:** `Vertical banner, fabric weave texture, bold symbol for the Code Tribe (code brackets '<>'), base color #0BB39C, subtle gold trim, orthographic, transparent background, 128x256px.`
- **Air:** `Vertical banner, fabric weave texture, bold symbol for the Air Tribe (feather), base color #aed9e0, subtle gold trim, orthographic, transparent background, 128x256px.`
- **Water:** `Vertical banner, fabric weave texture, bold symbol for the Water Tribe (wave), base color #4dabf7, subtle gold trim, orthographic, transparent background, 128x256px.`
- **Fire:** `Vertical banner, fabric weave texture, bold symbol for the Fire Tribe (flame), base color #F25F5C, subtle gold trim, orthographic, transparent background, 128x256px.`

---
## E. Narrative Inserts
---

### E1. Opening Panel (Splash)
- **File:** `marketing/splash_intro.png`
- **Final Specs:** 1280x720px.
- **Prompt:**
```
Cinematic splash art: Kelly, the Rein Maker’s Daughter, is stepping out from a grand, dark forge into the light; on the dark side, old leather reins hang like chains; on the light side, the air is clean. She carries a small, glowing circuit-rein token that illuminates her determined face. Painterly-meets-vector style, high contrast between shadow and light, dusk palette, 1280x720px. The glowing token should be the focal point.
```

---

### E2. Game-Over Panel
- **File:** `assets/gameover_bg.png`
- **Final Specs:** 960x540px, low contrast.
- **Prompt:**
```
Soft, dark, low-contrast vignette background panel, suggesting shattered rein fragments slowly reforming into a whole circuit loop in the center, 960x540px, for background usage behind text. The imagery should be very subtle and not distracting.
```

---
## F. Marketing & Storefront
---

### F1. Itch.io Banner (Wide)
- **File:** `marketing/itch-banner-1920x480.png`
- **Final Specs:** 1920x480px.
- **Prompt:**
```
Wide marketing banner montage: on the right, the stylized sprite of Kelly is in a full run; on the left, the game's bold logo ('The Rein Maker's Daughter'); in the background is the parallax skyline of the Hall of the Seven Tribes at dusk. Clean, dynamic composition, 1920x480px.
```

---
## G. Stretch Goals
---

### G1. Coin Pickup
- **File:** `assets/coin.png`
- **Final Specs:** 24x24px, transparent BG.
- **Prompt:**
```
Small glowing glyph coin for a 2D runner game, circular, abstract circuit pattern inside, teal and amber (#D8A24A) color blend, 24x24px, transparent background, clean silhouette.
```

### G2. Run Animation (3-frame)
- **Files:** `player_run_0.png`, `player_run_1.png`, `player_run_2.png`
- **Prompt Snippets:**
  - **Contact:** `...full-body, running pose (contact frame, one foot on ground)...`
  - **Air:** `...full-body, running pose (air frame, both feet off ground)...`
  - **Passing:** `...full-body, running pose (passing frame, legs crossing)...`
- **Note:** Use the main player prompt as a base and modify the pose description for each frame.











