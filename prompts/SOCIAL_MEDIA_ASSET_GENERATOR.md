# Social Media Asset Generator (Zero-Shot Prompt)

**System Role:** You are the **Curious Kelly Brand Automation Engineer**. Your goal is to generate a complete suite of production-ready social media assets from our canonical source files, strictly adhering to platform-specific dimensions and naming conventions.

**Context:** We have high-resolution 3D renders of Kelly (the AI teacher). We need to package these into profile pictures, headers, and logo files for Twitter, Instagram, YouTube, LinkedIn, and TikTok.

---

## 1. Input Resources (Assume available in environment)
*   **Source Avatar:** `kelly-age27-closeup-1x1.png` (Base 4K image)
*   **Source Logo Icon:** `sparkles.png` (Or standard emoji '✨')
*   **Brand Color:** Orange `#D97757`
*   **Background Color:** Dark `#0F0F11`

---

## 2. Output Schema (Directory Structure)
You must generate files matching this EXACT structure. Do not deviate from file names or dimensions.

```text
assets/social-media/
├── profile-pictures/
│   ├── kelly-profile-twitter-800x800.png      (Target: 800x800)
│   ├── kelly-profile-instagram-640x640.png    (Target: 640x640)
│   ├── kelly-profile-youtube-800x800.png      (Target: 800x800)
│   ├── kelly-profile-linkedin-600x600.png     (Target: 600x600)
│   ├── kelly-profile-tiktok-400x400.png       (Target: 400x400)
│   └── kelly-profile-discord-512x512.png      (Target: 512x512)
├── logos/
│   ├── icon-only/
│   │   ├── icon-orange-transparent-512.png    (Target: 512x512, Transparent BG)
│   │   └── icon-orange-dark-bg-512.png        (Target: 512x512, Hex #0F0F11 BG)
│   └── full-logo/
│       └── logo-horizontal-dark-1200x400.png  (Target: 1200x400)
└── headers/
    ├── twitter-header-1500x500.png            (Target: 1500x500)
    └── linkedin-cover-1584x396.png            (Target: 1584x396)
```

---

## 3. Processing Requirements (Python/PIL Specification)

### A. Profile Pictures
1.  **Crop:** Center-crop the source image to a square (1:1).
2.  **Border:** Apply a **20px solid border** in Brand Orange `#D97757` (inside stroke).
3.  **Resize:** Lanczos resampling to specific target dimensions.
4.  **Format:** PNG (Optimized).

### B. Logo Generation
1.  **Icon:** Use the Sparkles emoji or shape. Color: `#D97757`.
2.  **Text:** "Curious Kelly". Font: San Francisco (or Arial/Helvetica Bold). Color: White `#FFFFFF`.
3.  **Layout:** 
    *   *Horizontal:* Icon [Space] Text. Vertically centered.
    *   *Icon Only:* Just the sparkle shape, centered.

### C. Header Generation
1.  **Background:** Solid `#0F0F11` or subtle gradient.
2.  **Composition:**
    *   Left: Kelly Avatar (Fade right edge to transparent).
    *   Right: Logo + Tagline "Daily Lessons for Ages 2-102".
    *   Bottom Right: "Launching Dec 17".

---

## 4. Execution Prompt (Copy-Paste to Code Interpreter)

"Write a Python script using `PIL` (Pillow) that:
1. Creates the directory structure defined in the Output Schema.
2. Generates the Profile Pictures by loading `kelly-age27-closeup-1x1.png`, adding the orange border, and resizing.
3. Generates simple placeholder Logos using `ImageDraw` text (since we might not have the vector file yet).
4. Saves all files to the `assets/social-media` folder.
5. Verify the output paths."

**Rationale:** This approach ensures pixel-perfect consistency across all 6 platforms without manual editing, allowing us to update the base asset later and regenerate everything instantly.
















