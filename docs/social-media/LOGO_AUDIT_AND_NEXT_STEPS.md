# Logo & Branding Assets Audit - Curious Kelly
## Current State & Social Media Requirements

> **🔒 LOGO DECISION LOCKED (Nov 21, 2025):** Using ✨ Sparkles. See `LOGO_DECISION.md` for details.

**Audit Date:** November 21, 2025  
**Purpose:** Identify existing logo assets and create action plan for social media launch

---

## 🔍 What We Found

### ✅ **Assets That Exist**

#### 1. **Kelly Avatar Images** (72 total)
**Location:** `assets/kelly_canonical/marketing/age_variants/`

**Complete Age Progression Set:**
- Ages: 3, 9, 15, 27, 48, 82
- Shots: Closeup, Front Lean, Upper Body, Full Body
- Ratios: 16:9, 1:1, 3:4

**Quality:** ✅ High resolution, professional, multiple variants  
**Usable for Social:** ✅ YES - Perfect for profile pictures and posts

**Example Files:**
```
kelly-age27-closeup-1x1.png (Perfect for profile pics)
kelly-age27-upperbody-3x4.png (Great for Stories/vertical posts)
kelly-age27-front-lean-16x9.png (Good for headers/banners)
```

---

#### 2. **Kelly Chair Poses** (5 total)
**Location:** `assets/kelly_canonical/core/chair/`

**Available:**
- kelly-chair-celebrating.png
- kelly-chair-curious.png
- kelly-chair-explaining.png
- kelly-chair-listening.png
- kelly-chair-wisdom.png

**Also in:** `public/images/kelly/` (with directors-chair variants)

**Quality:** ✅ High resolution, teaching context  
**Usable for Social:** ✅ YES - Great for educational posts, lesson previews

---

#### 3. **Kelly Identity References** (10 total)
**Location:** `assets/kelly_canonical/reference/identity/`

**Available:**
- kelly-headshot-4k.png
- kelly-identity-front-studio-neutral.png
- kelly-identity-profile-studio-neutral.png
- kelly-identity-three-quarter-studio-neutral.png
- kelly-profile.png
- kelly-three-quarter.png
- kelly-ref-contact-sheet.png

**Quality:** ✅ 4K resolution, reference quality  
**Usable for Social:** ✅ YES - Can be cropped for profile pictures

---

#### 4. **Current Favicon**
**Location:** `public/favicons/favicon.svg`

**Current Design:** Purple gradient abstract "K" shape  
**Issue:** ❌ Not aligned with brand colors (#d97757 orange)  
**Usable for Social:** ⚠️ NO - Wrong brand color

---

#### 5. **Text-Based Logo (UPDATED)**
**Current Implementation:** `✨ Curious Kelly` (sparkles + text)

**Found in:** All HTML files (social.html, index.html, etc.)  
**Format:** Plain text with starburst emoji  
**Colors:** Orange (#d97757) starburst, white text on dark backgrounds

**Usable for Social:** ⚠️ PARTIAL - Works for web, but need proper image files for social media

---

### ❌ **What's Missing for Social Media**

#### 1. **Proper Logo Files** ⚠️ CRITICAL
Missing formats needed for social media:
- [ ] PNG logo (transparent background)
- [ ] PNG logo (with background)
- [ ] SVG logo (scalable vector)
- [ ] Square logo (1:1 ratio for profile pics)
- [ ] Horizontal logo (for headers/banners)
- [ ] Vertical logo (for Stories)
- [ ] Icon-only version (just starburst)
- [ ] Wordmark-only version (just "Curious Kelly")

---

#### 2. **Sized Profile Pictures** ⚠️ HIGH PRIORITY
Platform-specific profile picture requirements:
- [ ] Twitter: 400x400px minimum (recommended 800x800px)
- [ ] Instagram: 320x320px minimum (recommended 640x640px)
- [ ] YouTube: 800x800px minimum
- [ ] LinkedIn: 300x300px minimum (recommended 600x600px)
- [ ] TikTok: 200x200px minimum (recommended 400x400px)
- [ ] Discord: 512x512px minimum

**Current Status:** We have Kelly avatars, but need to:
1. Decide which age variant to use (recommend age 27)
2. Decide which shot (recommend closeup-1x1)
3. Add branded background or border
4. Resize for each platform

---

#### 3. **Header/Banner Images** ⚠️ HIGH PRIORITY
Platform-specific header requirements:
- [ ] Twitter Header: 1500x500px
- [ ] YouTube Channel Art: 2560x1440px
- [ ] LinkedIn Cover: 1584x396px
- [ ] Facebook Cover: 820x312px (for Instagram Business)

**Current Status:** We have 16:9 Kelly images, but need:
1. Add branding elements (logo, tagline)
2. Add website URL
3. Add "Launching Dec 17, 2025" badge
4. Design consistent template across platforms

---

#### 4. **Brand Kit** ⚠️ MEDIUM PRIORITY
For Canva and design consistency:
- [ ] Logo variations (all sizes, formats)
- [ ] Color palette swatches
- [ ] Typography samples
- [ ] Icon set (starburst variations)
- [ ] Pattern library (backgrounds, textures)

---

## 🎨 Brand Assets Analysis

### Current Brand Identity

**Logo:** `✨ Curious Kelly`
- ✨ = Sparkles (emoji U+2728)
- Color: Orange (#d97757)
- Font: SF Pro (sans-serif) + Times New Roman (serif for headlines)

**Colors:**
- Primary: Orange #d97757
- Background: Dark #0f0f11
- Text: Off-white #f4f4f5
- Secondary: Gray #a1a1aa
- Card: Charcoal #18181b

**Typography:**
- Body: SF Pro (Apple system font)
- Headlines: Times New Roman (classic serif)

**Kelly's Visual Identity:**
- Age: Late 20s-early 30s (age 27 variant)
- Style: Photorealistic digital human
- Aesthetic: Modern, timeless, "Apple Genius"
- Expression: Warm, approachable, intelligent

---

## 🚀 Action Plan: Create Missing Assets

### **Phase 1: Profile Pictures (URGENT - 2 hours)**

**Decision Point:** Which Kelly to use?
- **Recommended:** Age 27 (adult, professional, relatable)
- **Alternative:** Age 15 (younger, more energetic) for TikTok only

**Tasks:**
1. Select base image: `kelly-age27-closeup-1x1.png`
2. Create variations:
   - **Option A:** Kelly on solid orange background
   - **Option B:** Kelly on dark gradient with subtle orange glow
   - **Option C:** Kelly with orange border/frame
3. Resize for each platform (Twitter, Instagram, YouTube, LinkedIn, TikTok, Discord)
4. Export as PNG (high quality)

**Tools Needed:**
- Photoshop / GIMP / Photopea
- Canva Pro (once account created)

**Output:**
```
assets/social-media/profile-pictures/
├── kelly-profile-twitter-800x800.png
├── kelly-profile-instagram-640x640.png
├── kelly-profile-youtube-800x800.png
├── kelly-profile-linkedin-600x600.png
├── kelly-profile-tiktok-400x400.png
└── kelly-profile-discord-512x512.png
```

---

### **Phase 2: Logo Files (URGENT - 3 hours)**

**Task:** Create proper logo files in multiple formats

#### 2A. Design Logo Variations

**Full Logo:** `✨ Curious Kelly`
- Horizontal layout (icon + wordmark side-by-side)
- Vertical layout (icon above wordmark)
- Square layout (balanced for profile pics)

**Icon Only:** `✨`
- Just the sparkles
- Orange (#d97757) on transparent
- Orange on dark background
- White on orange background

**Wordmark Only:** `Curious Kelly`
- SF Pro Semibold
- With proper spacing and kerning

**Color Variations:**
- Light mode (dark text on light background)
- Dark mode (light text on dark background)
- Monochrome (all one color)

#### 2B. Export Formats

**For Each Variation, Export:**
- [ ] PNG @ 1x, 2x, 3x (72dpi, 150dpi, 300dpi)
- [ ] SVG (scalable vector)
- [ ] PDF (print quality)

**Sizes Needed:**
- Small: 120px width
- Medium: 240px width
- Large: 480px width
- Extra Large: 960px width

**Output Structure:**
```
assets/social-media/logos/
├── full-logo/
│   ├── horizontal/
│   │   ├── full-logo-horizontal-light-1x.png
│   │   ├── full-logo-horizontal-light-2x.png
│   │   ├── full-logo-horizontal-light-3x.png
│   │   ├── full-logo-horizontal-dark-1x.png
│   │   ├── full-logo-horizontal-dark-2x.png
│   │   ├── full-logo-horizontal-dark-3x.png
│   │   └── full-logo-horizontal.svg
│   ├── vertical/
│   │   └── (same as horizontal)
│   └── square/
│       └── (same as horizontal)
├── icon-only/
│   ├── icon-orange-transparent.png (multiple sizes)
│   ├── icon-white-transparent.png
│   ├── icon-orange-dark-bg.png
│   └── icon.svg
└── wordmark-only/
    ├── wordmark-light.png (multiple sizes)
    ├── wordmark-dark.png
    └── wordmark.svg
```

---

### **Phase 3: Header/Banner Images (HIGH PRIORITY - 4 hours)**

**Task:** Create platform-specific header images

**Design Template:**
- Background: Dark gradient (#0f0f11 to #18181b)
- Kelly: Age 27, upperbody or front-lean shot (left or right third)
- Logo: Top or bottom corner
- Tagline: "8-minute daily lessons for ages 2-102"
- Launch badge: "Launching December 17, 2025"
- Website: curiouskelly.com

**Platform-Specific Sizes:**

#### Twitter Header (1500x500px)
```
[Kelly left third] | [Empty space] | [Logo + tagline right]
```

#### YouTube Channel Art (2560x1440px)
```
Safe area (center 1546x423px) contains:
[Logo] [Kelly center] [Tagline + Launch badge]
Full canvas: Gradient background extending to edges
```

#### LinkedIn Cover (1584x396px)
```
[Kelly left] | [Company name + tagline center-right]
```

#### Facebook Cover (820x312px)
```
[Kelly left] | [Logo + tagline right]
```

**Output:**
```
assets/social-media/headers/
├── twitter-header-1500x500.png
├── youtube-channel-art-2560x1440.png
├── linkedin-cover-1584x396.png
└── facebook-cover-820x312.png
```

---

### **Phase 4: Brand Kit for Canva (MEDIUM PRIORITY - 2 hours)**

**Task:** Prepare assets for Canva Pro upload

**What to Include:**
1. **Logo Set:**
   - All logo variations (PNG + SVG)
   - Minimum 3 color variations each

2. **Color Palette:**
   - Export as swatches
   - Include HEX codes
   - Name each color

3. **Typography:**
   - Upload SF Pro font (if available)
   - Document Times New Roman usage
   - Include font pairing examples

4. **Kelly Avatar Library:**
   - Age 27 shots (all angles)
   - Pre-sized for common posts:
     - Instagram post (1080x1080)
     - Instagram story (1080x1920)
     - Twitter post (1200x675)
     - LinkedIn post (1200x627)

5. **Templates:**
   - Fact post template
   - Quote post template
   - Lesson highlight template
   - Story template

**Output:**
Upload to Canva Pro "Brand Kit" section once account is created.

---

### **Phase 5: Fix Favicon (LOW PRIORITY - 30 min)**

**Task:** Update favicon to match brand colors

**Current Issue:** Favicon is purple, should be orange

**Solution:**
1. Create new favicon.svg with orange (#d97757) color
2. Design options:
   - Option A: Orange starburst on dark background
   - Option B: Orange "K" lettermark
   - Option C: Kelly's face (simplified icon)
3. Export as:
   - favicon.svg
   - favicon.ico (16x16, 32x32, 48x48)
   - apple-touch-icon.png (180x180)
   - favicon-192.png (Android)
   - favicon-512.png (Android)

**Output:**
```
public/favicons/
├── favicon.svg (new, orange)
├── favicon.ico
├── apple-touch-icon.png
├── favicon-192.png
└── favicon-512.png
```

---

## ⏱️ Time Estimate Summary

| Phase | Task | Time | Priority |
|-------|------|------|----------|
| **Phase 1** | Profile pictures (6 platforms) | 2 hours | 🔴 URGENT |
| **Phase 2** | Logo files (all formats) | 3 hours | 🔴 URGENT |
| **Phase 3** | Header/banner images | 4 hours | 🟠 HIGH |
| **Phase 4** | Brand kit for Canva | 2 hours | 🟡 MEDIUM |
| **Phase 5** | Favicon update | 30 min | 🟢 LOW |
| **TOTAL** | - | **11.5 hours** | - |

**Recommendation:** Phases 1-2 (5 hours) must be done before account creation. Phases 3-5 can be done during Week 2 of launch prep.

---

## 🛠️ Tools Needed

### Design Software (Choose One)
- **Option A:** Adobe Photoshop (professional, paid)
- **Option B:** GIMP (free, open-source)
- **Option C:** Photopea (free, browser-based, Photoshop-like)
- **Option D:** Canva Pro (after account setup)

### Vector Graphics (for logos)
- **Option A:** Adobe Illustrator (professional, paid)
- **Option B:** Inkscape (free, open-source)
- **Option C:** Figma (free for personal use)

### Automation (optional)
```bash
# Batch resize images
python scripts/resize_profile_pics.py

# Generate all logo variations
python scripts/generate_logos.py
```

---

## ✅ Checklist: Ready for Social Media?

### Profile Pictures
- [ ] Age 27 Kelly selected
- [ ] Closeup-1x1 base image chosen
- [ ] Background/border design decided
- [ ] Resized for Twitter (800x800)
- [ ] Resized for Instagram (640x640)
- [ ] Resized for YouTube (800x800)
- [ ] Resized for LinkedIn (600x600)
- [ ] Resized for TikTok (400x400)
- [ ] Resized for Discord (512x512)

### Logo Files
- [ ] Full logo horizontal (light + dark)
- [ ] Full logo vertical (light + dark)
- [ ] Full logo square (light + dark)
- [ ] Icon only (orange, white, transparent)
- [ ] Wordmark only (light + dark)
- [ ] All exported as PNG (1x, 2x, 3x)
- [ ] All exported as SVG
- [ ] Logo usage guidelines documented

### Headers/Banners
- [ ] Twitter header (1500x500)
- [ ] YouTube channel art (2560x1440)
- [ ] LinkedIn cover (1584x396)
- [ ] Facebook cover (820x312)

### Brand Kit
- [ ] Canva Pro account created
- [ ] All logo variations uploaded
- [ ] Color palette uploaded
- [ ] Kelly avatar library uploaded
- [ ] Font preferences set

### Bonus
- [ ] Favicon updated to orange
- [ ] All assets organized in /assets/social-media/
- [ ] Asset manifest/inventory created
- [ ] Usage guidelines documented

---

## 📂 Recommended Folder Structure

```
assets/social-media/
├── profile-pictures/
│   ├── kelly-profile-twitter-800x800.png
│   ├── kelly-profile-instagram-640x640.png
│   ├── kelly-profile-youtube-800x800.png
│   ├── kelly-profile-linkedin-600x600.png
│   ├── kelly-profile-tiktok-400x400.png
│   └── kelly-profile-discord-512x512.png
├── logos/
│   ├── full-logo/
│   │   ├── horizontal/
│   │   ├── vertical/
│   │   └── square/
│   ├── icon-only/
│   └── wordmark-only/
├── headers/
│   ├── twitter-header-1500x500.png
│   ├── youtube-channel-art-2560x1440.png
│   ├── linkedin-cover-1584x396.png
│   └── facebook-cover-820x312.png
├── posts/
│   ├── instagram-templates/
│   ├── twitter-templates/
│   └── tiktok-templates/
└── brand-kit/
    ├── colors.ase (Adobe swatch file)
    ├── typography-guide.pdf
    └── usage-guidelines.pdf
```

---

## 🎯 Quick Win: Use What We Have Now

**While waiting for new assets, you can START with:**

1. **Profile Pictures:**
   - Use: `kelly-age27-closeup-1x1.png`
   - Quick edit: Crop to square, add subtle orange border in any photo editor
   - Export at required sizes

2. **Official Logo:**
   - Use: ✨ Curious Kelly (emoji + text)
   - Consistent across all platforms
   - Replace with proper logo files when ready

3. **Headers:**
   - Use: `kelly-age27-front-lean-16x9.png`
   - Add text overlay with branding
   - Quick template in Canva (even free version)

**Estimated Time for Quick Win:** 1 hour  
**Quality:** Good enough to launch, upgrade later

---

## 📞 Questions & Next Steps

**Questions to Answer:**
1. Which Kelly age variant for profile pictures? (Recommend: 27)
2. Which background style? (Solid orange, gradient, or border?)
3. Logo design direction? (Keep minimal starburst + text, or create something more elaborate?)
4. Who will create these assets? (Designer, or DIY with tools?)

**Next Steps:**
1. **Review this document** and make decisions
2. **Phase 1 (URGENT):** Create profile pictures (2 hours)
3. **Phase 2 (URGENT):** Create logo files (3 hours)
4. **Use assets** for social media account creation
5. **Phase 3-5:** Complete remaining assets during Week 2

---

**Document Owner:** Social Media Lead  
**Created:** November 21, 2025  
**Priority:** HIGH - Blocks social media account setup

🎨 **Let's create some beautiful, on-brand assets!**

