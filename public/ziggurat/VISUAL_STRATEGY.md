# Ziggurat Visual Strategy
## Updated: 2026-02-28

---

## Current Direction: SLATE BLUE (No LED)

Per Nicolette's Feb 15, 2026 directive:
- **NO LED screens** — budget goes to operations, not spectacle
- **Slate blue paint** — elastomeric coating transforms yellow concrete
- **Refreshed grounds** — native landscaping, clean campus
- **Capital to operations** — printing, manufacturing, broadcast

---

## Asset Hierarchy

### Primary (Use These)

| Asset | Location | Purpose |
|-------|----------|---------|
| `slate/front-comparison.jpg` | Before/after hero | Landing pages, pitch decks |
| `slate/aerial-comparison.jpg` | Before/after aerial | Investor materials |
| `slate/front-slate.jpg` | Clean "after" shot | Social, headers |
| `slate/aerial-slate.jpg` | Clean aerial "after" | Maps, overview |
| `kelly/hero-composite.jpg` | Kelly + Ziggurat | /bren page, one-pager |
| `kelly/portrait.png` | Kelly portrait | About Kelly sections |
| `laguna-ridge/LAGUNA_RIDGE.mp4` | 79s institutional video | Primary video asset |
| `kelly/demo.mp4` | 56s Kelly demo | AI teacher showcase |

### Secondary (Context)

| Asset | Purpose |
|-------|---------|
| `slate/commons-interior.png` | Interior vision |
| `slate/levels-diagram.png` | Floor concept |
| `slate/future-2028.png` | Grand opening vision |
| `slate/founder.png` | Founder portrait |
| `kelly/architect.png` | Kelly architect variant |

### Archived (DO NOT USE)

| Folder | Reason |
|--------|--------|
| `precision/rainbow-*` | LED concepts deprecated |
| `precision/cool-*` | LED concepts deprecated |
| `variants/` | Old LED renders |

---

## Page Strategy

### /bren (Investor One-Pager)
- Hero: `kelly/hero-composite.jpg`
- Comparison: `slate/aerial-comparison.jpg`
- Kelly: `kelly/portrait.png`
- Video: Link to LAGUNA_RIDGE.mp4
- **Purpose:** Print-ready, 5-minute pitch for Bren meeting

### /ziggurat/slate (Vision Gallery)
- Full slate image gallery with navigation
- Nicolette quote about direction
- Download links for all assets
- **Purpose:** Public-facing current vision

### /zig (Redirect)
- Now redirects to `/ziggurat/slate`
- Old LED page at `/ziggurat` still exists for reference

### /ziggurat/investors (Financial Model)
- Interactive model, projections
- Should reference slate images, not LED
- **TODO:** Update hero images to slate

---

## Video Assets

### LAGUNA_RIDGE.mp4 (79.5s)
- **Format:** 1080p, H.264, AAC audio
- **Content:** Building footage + Kelly voiceover
- **Use:** Primary pitch video, /bren page, investor presentations
- **Location:** `/ziggurat/laguna-ridge/LAGUNA_RIDGE.mp4`

### Kelly Demo (56s)
- **Format:** HeyGen-generated, Architect archetype
- **Content:** Kelly teaching a lesson
- **Use:** AI demo, /bren page Kelly section
- **Location:** `/ziggurat/kelly/demo.mp4`

### R2 CDN (Existing)
- `THE_ZIGGURAT_FINAL.mp4` — Full production (LED era)
- `twitter_60sec.mp4` — Twitter clip
- **Note:** These are LED-era; still work but less aligned with current direction

---

## Social Media Assets

### LinkedIn/Twitter Headers (1200x628)
- Use `slate/aerial-comparison.jpg` cropped
- Or `kelly/hero-composite.jpg`

### Instagram (1080x1080)
- Crop `slate/front-comparison.jpg` to square
- Or use `kelly/portrait.png`

### Story/Reels (1080x1920)
- Use Kelly demo video
- Or LAGUNA_RIDGE.mp4 with captions

---

## Color Palette

### Brand Colors
- **Slate Blue-Gray:** `#7A8B9A` (building)
- **Kelly Blue:** `#3B82F6` (accent)
- **Background:** `#0A0A0C` (dark mode)
- **Text:** `#FFFFFF` / `#71717A` (zinc-500)

### Avoid
- Rainbow LED colors (deprecated)
- Bright yellow (old building color)
- Neon/spectacle aesthetics

---

## Key Messages (Visual)

1. **Transformation** — Before/after comparisons show the vision
2. **Dignity** — Slate blue is professional, not flashy
3. **Mission-first** — Money goes to learners, not LEDs
4. **Scale** — 92 acres, 7 floors, global reach
5. **Kelly** — Human-like AI teacher, approachable

---

## Deployment Checklist

- [x] Slate images in `/public/ziggurat/slate/`
- [x] Kelly assets in `/public/ziggurat/kelly/`
- [x] Video in `/public/ziggurat/laguna-ridge/`
- [x] /bren page created
- [x] /ziggurat/slate gallery created
- [x] /zig redirects to slate vision
- [ ] Update /ziggurat/investors to use slate hero
- [ ] Upload LAGUNA_RIDGE.mp4 to R2 CDN for faster delivery
- [ ] Update social media assets
- [ ] Run Vercel deploy
