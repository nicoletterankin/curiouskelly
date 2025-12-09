# Kelly Brand Asset Production Specification

**Version:** 1.0  
**Date:** November 29, 2025  
**Agency:** Curious Kelly Creative  
**Contact:** hello@curiouskelly.com

---

## 📊 Raw Asset Inventory

### High-Resolution Source Files (3072x5504 portrait / 6000x3375 landscape)

| Current Name | Dimensions | Size | Recommended Name | Use Case |
|--------------|------------|------|------------------|----------|
| `1.jpg` | 3072×5504 | 7.3MB | `kelly-portrait-neutral-01` | Hero, Full-body |
| `2.jpeg` | 3072×5504 | 11.5MB | `kelly-portrait-thoughtful-01` | Hero, Full-body |
| `3.jpeg` | 3072×5504 | 5.7MB | `kelly-portrait-neutral-02` | Hero, Full-body |
| `4.jpg` | 3072×5504 | 6.1MB | `kelly-portrait-engaged-01` | Hero, Full-body |
| `6.jpeg` | 3072×5504 | 6.2MB | `kelly-portrait-warm-01` | Hero, Full-body |
| `7.jpeg` | 3072×5504 | 6.0MB | `kelly-portrait-curious-01` | Hero, Full-body |
| `blink.jpeg` | 3072×5504 | 5.8MB | `kelly-portrait-blink-01` | Animation frame |
| `2 (1).jpeg` | 6000×3375 | 3.9MB | `kelly-landscape-wide-01` | Banner, OG Image |
| `4 (1).jpeg` | 6000×3375 | 4.7MB | `kelly-landscape-engaged-01` | Banner |
| `kelly2-directors-chair.jpeg` | 6000×3375 | 4.1MB | `kelly-landscape-chair-01` | Banner |

### Medium-Resolution Assets (1024x1024 square)

| Current Name | Size | Quality | Recommended Name |
|--------------|------|---------|------------------|
| `hero/neutral.jpeg` | 512KB | Good | `kelly-square-neutral` |
| `hero/looking-at-us.jpeg` | 518KB | Good | `kelly-square-explaining` |
| `hero/big-smile.jpeg` | 3.5MB | Excellent | `kelly-square-celebrating` |
| `hero/blink.jpeg` | 1.4MB | Good | `kelly-square-blink` |
| `hero/flicking-hand.jpg` | 827KB | Good | `kelly-square-gesture-wave` |
| `hero/raise-wrist.jpg` | 1.5MB | Good | `kelly-square-gesture-wrist` |
| `hero/lookaway.png` | 1.7MB | Good | `kelly-square-lookaway` |

---

## 🎯 Production Asset Specification Matrix

### Format Strategy

| Format | Use Case | Browser Support | Fallback |
|--------|----------|-----------------|----------|
| **WebP** | Primary web delivery | 97%+ modern browsers | JPEG |
| **AVIF** | Future-proof, smallest | 85%+ (growing) | WebP → JPEG |
| **JPEG** | Universal fallback | 100% | — |
| **PNG** | Transparency needed | 100% | — |

### Size Variants Required

| Variant Name | Dimensions | Use Case | Target Size |
|--------------|------------|----------|-------------|
| `hero-4k` | 3840×2160 | 4K displays, hero backgrounds | <500KB |
| `hero-desktop` | 1920×1080 | Desktop hero sections | <200KB |
| `hero-tablet` | 1280×720 | Tablet hero sections | <150KB |
| `hero-mobile` | 640×360 | Mobile hero sections | <80KB |
| `avatar-lg` | 512×512 | Large avatar displays | <60KB |
| `avatar-md` | 256×256 | Medium avatars, cards | <30KB |
| `avatar-sm` | 128×128 | Small avatars, thumbnails | <15KB |
| `avatar-xs` | 64×64 | Tiny avatars, favicons | <8KB |
| `og-image` | 1200×630 | Social sharing (OG) | <100KB |
| `twitter-card` | 1200×600 | Twitter cards | <100KB |

### Quality Settings

| Asset Type | WebP Quality | JPEG Quality | Notes |
|------------|--------------|--------------|-------|
| Hero images | 85 | 90 | Balance quality/size |
| Avatars | 80 | 85 | Smaller file priority |
| OG/Social | 85 | 90 | Quality for sharing |
| Thumbnails | 75 | 80 | Speed priority |

---

## 📁 Recommended Folder Structure

```
/public/assets/kelly/
├── source/                          # Original high-res files (not deployed)
│   ├── portraits/
│   │   ├── kelly-portrait-neutral-01.jpg
│   │   ├── kelly-portrait-thoughtful-01.jpg
│   │   └── ...
│   └── landscapes/
│       ├── kelly-landscape-wide-01.jpg
│       └── ...
│
├── production/                      # Optimized web assets
│   ├── hero/
│   │   ├── kelly-hero-4k.webp
│   │   ├── kelly-hero-4k.jpg        # Fallback
│   │   ├── kelly-hero-desktop.webp
│   │   ├── kelly-hero-desktop.jpg
│   │   ├── kelly-hero-tablet.webp
│   │   ├── kelly-hero-tablet.jpg
│   │   ├── kelly-hero-mobile.webp
│   │   └── kelly-hero-mobile.jpg
│   │
│   ├── avatars/
│   │   ├── curious/
│   │   │   ├── kelly-curious-512.webp
│   │   │   ├── kelly-curious-256.webp
│   │   │   ├── kelly-curious-128.webp
│   │   │   └── kelly-curious-64.webp
│   │   ├── explaining/
│   │   ├── celebrating/
│   │   ├── listening/
│   │   └── wisdom/
│   │
│   ├── social/
│   │   ├── og-image.jpg             # 1200x630
│   │   ├── twitter-card.jpg         # 1200x600
│   │   └── linkedin-banner.jpg      # 1584x396
│   │
│   └── animations/
│       ├── kelly-blink-sequence/
│       └── kelly-gesture-wave/
│
└── manifest.json                    # Asset manifest for programmatic access
```

---

## 🏷️ Naming Convention

### Pattern
```
kelly-[context]-[expression]-[size].[format]
```

### Components

| Component | Values | Example |
|-----------|--------|---------|
| `context` | `hero`, `avatar`, `card`, `og`, `banner` | `kelly-hero-...` |
| `expression` | `neutral`, `curious`, `explaining`, `celebrating`, `listening`, `wisdom`, `blink`, `gesture` | `kelly-avatar-curious-...` |
| `size` | `4k`, `desktop`, `tablet`, `mobile`, `512`, `256`, `128`, `64` | `kelly-avatar-curious-256` |
| `format` | `.webp`, `.jpg`, `.png`, `.avif` | `kelly-avatar-curious-256.webp` |

### Examples
- `kelly-hero-neutral-desktop.webp`
- `kelly-avatar-curious-256.webp`
- `kelly-og-celebrating.jpg`
- `kelly-banner-landscape-tablet.webp`

---

## 🔧 Production Pipeline

### Step 1: Source Organization
1. Rename raw files with semantic names
2. Organize into `source/portraits/` and `source/landscapes/`
3. Document original dimensions and metadata

### Step 2: Image Processing
Using Sharp.js or ImageMagick:

```javascript
// Example Sharp.js pipeline
const sharp = require('sharp');

const variants = [
  { name: 'hero-4k', width: 3840, height: 2160 },
  { name: 'hero-desktop', width: 1920, height: 1080 },
  { name: 'hero-tablet', width: 1280, height: 720 },
  { name: 'hero-mobile', width: 640, height: 360 },
];

async function processHeroImage(sourcePath, outputDir) {
  for (const variant of variants) {
    // WebP (primary)
    await sharp(sourcePath)
      .resize(variant.width, variant.height, { fit: 'cover' })
      .webp({ quality: 85 })
      .toFile(`${outputDir}/kelly-hero-${variant.name}.webp`);
    
    // JPEG (fallback)
    await sharp(sourcePath)
      .resize(variant.width, variant.height, { fit: 'cover' })
      .jpeg({ quality: 90, progressive: true })
      .toFile(`${outputDir}/kelly-hero-${variant.name}.jpg`);
  }
}
```

### Step 3: Responsive Image Implementation

```html
<!-- Hero Section -->
<picture>
  <source 
    type="image/webp"
    srcset="
      /assets/kelly/production/hero/kelly-hero-mobile.webp 640w,
      /assets/kelly/production/hero/kelly-hero-tablet.webp 1280w,
      /assets/kelly/production/hero/kelly-hero-desktop.webp 1920w,
      /assets/kelly/production/hero/kelly-hero-4k.webp 3840w
    "
    sizes="100vw"
  />
  <source 
    type="image/jpeg"
    srcset="
      /assets/kelly/production/hero/kelly-hero-mobile.jpg 640w,
      /assets/kelly/production/hero/kelly-hero-tablet.jpg 1280w,
      /assets/kelly/production/hero/kelly-hero-desktop.jpg 1920w,
      /assets/kelly/production/hero/kelly-hero-4k.jpg 3840w
    "
    sizes="100vw"
  />
  <img 
    src="/assets/kelly/production/hero/kelly-hero-desktop.jpg"
    alt="Kelly - Your AI learning companion"
    loading="eager"
    fetchpriority="high"
  />
</picture>
```

### Step 4: Avatar State System

```javascript
// Avatar manifest for lesson player
const kellyAvatars = {
  curious: {
    512: '/assets/kelly/production/avatars/curious/kelly-curious-512.webp',
    256: '/assets/kelly/production/avatars/curious/kelly-curious-256.webp',
    128: '/assets/kelly/production/avatars/curious/kelly-curious-128.webp',
  },
  explaining: { /* ... */ },
  celebrating: { /* ... */ },
  listening: { /* ... */ },
  wisdom: { /* ... */ },
};

// Usage
function getKellyAvatar(state, size = 256) {
  return kellyAvatars[state]?.[size] || kellyAvatars.curious[size];
}
```

---

## 📋 Production Checklist

### Pre-Production
- [ ] Review all source files for quality
- [ ] Select best shot for each expression/pose
- [ ] Remove duplicates and low-quality variants
- [ ] Document selected source files

### Processing
- [ ] Create folder structure
- [ ] Generate all hero variants (4 sizes × 2 formats = 8 files)
- [ ] Generate all avatar variants (5 states × 4 sizes × 2 formats = 40 files)
- [ ] Generate social media assets (OG, Twitter, LinkedIn)
- [ ] Optimize all files for web delivery

### Quality Assurance
- [ ] Verify all files load correctly
- [ ] Test responsive images on multiple devices
- [ ] Validate file sizes meet targets
- [ ] Check WebP fallbacks work
- [ ] Run Lighthouse audit for image optimization

### Deployment
- [ ] Update all code references
- [ ] Configure CDN caching headers
- [ ] Set up lazy loading for below-fold images
- [ ] Add preload hints for critical images

---

## 🎨 Expression/State Mapping

| Emotional State | Best Source File | Use Context |
|-----------------|------------------|-------------|
| **Curious** | `hero/neutral.jpeg` or `1.jpg` | Default state, welcome |
| **Explaining** | `hero/looking-at-us.jpeg` | Teaching, Q&A |
| **Celebrating** | `hero/big-smile.jpeg` | Correct answers, achievements |
| **Listening** | `hero/neutral.jpeg` | Waiting for input |
| **Wisdom** | `3.jpeg` (thoughtful pose) | Closing, insights |
| **Thinking** | `hero/lookaway.png` | Processing, transitions |

---

## 📊 Target Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Hero LCP | <2.5s | TBD |
| Hero image size (mobile) | <80KB | ~500KB |
| Avatar load time | <100ms | TBD |
| Total image weight (landing) | <500KB | ~3MB+ |
| WebP adoption | 100% primary | 0% |

---

## 🚀 Implementation Priority

### Phase 1: Critical (Week 1)
1. Process hero images for landing page
2. Create OG/social sharing images
3. Update `index.astro` with responsive images

### Phase 2: Core (Week 2)
1. Process all avatar state images
2. Update lesson player with optimized avatars
3. Implement lazy loading

### Phase 3: Polish (Week 3)
1. Add AVIF support
2. Implement blur-up placeholders
3. Fine-tune compression settings

---

**Document Status:** SPECIFICATION COMPLETE  
**Next Step:** Approve spec and begin asset processing  
**Estimated Production Time:** 4-6 hours for full asset package










