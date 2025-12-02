# Kelly Animation Quick Reference

**TL;DR for December 17 Launch**

---

## THE TWO ANIMATIONS YOU NEED

### 🎯 #1: Homepage Hero
**Folder:** `our-girl-too-excited/`  
**Files:** `our-girl.jpeg` + `too-excited.jpeg`  
**What it does:** Thoughtful Kelly → Excited Kelly  
**Why it works:** Same pose, only facial expression changes  
**Technique:** Optical flow morph, 1.5s duration, loop  
**Score:** 5/5 ✅

---

### 🎯 #2: Lesson Pointing  
**Folder:** `top-bottom/`  
**Files:** `top-choice.jpeg` + `bottom-choice.jpeg`  
**What it does:** Points up or down to indicate choices  
**Why it works:** Smooth arm movement, body stays still  
**Technique:** Optical flow morph, 1s duration, on-demand  
**Score:** 5/5 ✅

---

## BONUS ANIMATIONS (If Time Allows)

### 🌟 #3: Blink/Attention
**Folder:** `open-close/`  
**Files:** `close.jpeg` + `open.png`  
**What it does:** Eyes close → open (natural blink)  
**Why it works:** ONLY eyes change, everything else identical  
**Technique:** Optical flow loop, 0.5s  
**Score:** 5/5 ✅  
**Note:** Convert PNG to JPEG first

---

### 🌟 #4: Listen/Reflect
**Folder:** `shh-listen/`  
**Files:** `lips.jpeg` + `rest-chair.jpeg`  
**What it does:** "Shh, listen..." → peaceful reflection  
**Why it works:** Gentle hand movement, same seated pose  
**Technique:** Optical flow, 1.2s  
**Score:** 4.5/5 ✅

---

### 🌟 #5: Full-Body Blink
**Root files:** `open-chair.jpeg` + `closed-chair.jpeg`  
**What it does:** Same as #3 but full-body framing  
**Why it works:** Only eyes change  
**Technique:** Optical flow or crossfade  
**Score:** 4.5/5 ✅

---

## SKIP THESE

### ❌ Walk Animation
**Folder:** `walk/`  
**Why skip:** You already tested this - it failed. Too much body movement creates artifacts.  
**Score:** 1/5 🔴

### ⏸️ Expression Sets
**Folders:** `yay-but/`, `yay-pray-huh/`, `square-chair/`  
**Why skip for launch:** Hand movements too large for smooth optical flow. Use as separate stills instead.  
**Score:** 2-3/5 🟡

---

## File Specifications

### High-Res Full-Body (3072×5504)
- our-girl-too-excited/
- top-bottom/
- shh-listen/
- Root open/closed-chair
- walk/ (deprecated)

### Medium-Res Headshots (1024×1024)
- open-close/
- yay-but/
- yay-pray-huh/

---

## Animation Techniques Explained

**Optical Flow:** Smooth morphing between images (best for minimal changes)  
**Crossfade:** Simple fade between images (fallback for larger changes)  
**Sprite Sheet:** Frame-by-frame animation (for distinct poses)

---

## Launch Day Priorities

```
[Must Have]
✅ Hero animation (our-girl → excited)
✅ Pointing system (top → bottom)

[Nice to Have]  
⭐ Blink animation (eyes only)

[Skip]
❌ Walk animation
❌ Expression morphing
```

---

**Questions?** See full report: `KELLY_ANIMATION_INVENTORY.md`






