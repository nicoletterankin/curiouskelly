# Coalition Page Assets for v0.app

## Ready-to-Use Files

All assets are in `/public/coalition/`:

| File | Size | Purpose |
|------|------|---------|
| `og-coalition.png` | 4.3 MB | Open Graph social sharing image (1200x630) |
| `hero-ziggurat-gradient.png` | 1.8 MB | Hero background with bottom gradient fade |
| `ziggurat-twilight.jpg` | 278 KB | Warm twilight building shot |
| `ziggurat-night.jpg` | 234 KB | Gold night building shot |
| `ziggurat-aerial.jpg` | 1.3 MB | Aerial/overhead view |
| `ziggurat-showcase.mp4` | 37.9 MB | 55-second cinematic video (optional embed) |
| `coalition-data.json` | 5 KB | All data pre-structured for fetch/import |

---

## Asset Usage in Sections

### Hero Section
```jsx
// Background image with gradient already baked in
<div style={{ backgroundImage: 'url(/coalition/hero-ziggurat-gradient.png)' }}>
```

### OG Meta Tags
```html
<meta property="og:image" content="/coalition/og-coalition.png" />
```

### The Ziggurat Section
```jsx
// Options:
<img src="/coalition/ziggurat-aerial.jpg" /> // Aerial overview
<img src="/coalition/ziggurat-twilight.jpg" /> // Warm dramatic
<img src="/coalition/ziggurat-night.jpg" /> // Night with gold lights
```

### Optional Video Embed
```jsx
<video 
  src="/coalition/ziggurat-showcase.mp4" 
  autoPlay 
  muted 
  loop 
  playsInline
/>
```

---

## Data Import

The `coalition-data.json` file contains all structured data matching the prompt's TypeScript constants. v0 can fetch and render directly:

```typescript
// Example usage
const data = await fetch('/coalition/coalition-data.json').then(r => r.json())

// Access:
data.coalition          // Partner array with amount, type, status
data.financials         // 5-year projection by year
data.milestones         // Journey to 8B learners
data.ziggurat           // Building stats
data.timeline           // Key dates
data.pillars            // Four business pillars
```

---

## Image Specifications

| Asset | Dimensions | Format | Notes |
|-------|-----------|--------|-------|
| og-coalition.png | 1200x630 | PNG | Social sharing optimized |
| hero-ziggurat-gradient.png | 1920x1055 | PNG | Has alpha gradient at bottom |
| ziggurat-twilight.jpg | 1920x1055 | JPEG | Warm color, no overlay |
| ziggurat-night.jpg | 1920x1055 | JPEG | Gold lights, dramatic |
| ziggurat-aerial.jpg | Variable | JPEG | Overhead perspective |

---

## Video Specifications

| Property | Value |
|----------|-------|
| Duration | 55 seconds |
| Resolution | 1920x1080 |
| Frame rate | 24 fps (cinematic) |
| Codec | H.264 + AAC audio |
| File size | 37.9 MB |

The video includes:
- Ken Burns zoom effects
- 12 different transitions
- Title overlays
- Ambient soundtrack
- Fade in/out

---

## Notes for v0

1. **Hero gradient**: The PNG has transparency baked in - bottom fades to transparent, designed to sit over #0a0a0a background

2. **Video is optional**: Page works without it, but video could be a powerful "View the Building" CTA

3. **Data is denormalized**: coalition-data.json has pre-calculated totals so you don't need to sum in components

4. **Images are real**: These are actual renders of the Ziggurat building, not placeholders
