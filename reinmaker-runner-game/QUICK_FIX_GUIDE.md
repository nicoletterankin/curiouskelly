# Quick Fix Summary: PNG Assets

## 🔴 CRITICAL FIXES NEEDED

### 1. Add Transparency to These Files (MUST DO):
- `player.png` - Remove opaque background
- `obstacle.png` - Remove opaque background  
- All 7 stone files - Remove opaque backgrounds
- `favicon.png` - Add transparency (optional)

### 2. Optimize File Sizes (SHOULD DO):
- `player.png`: 1139 KB → Target: <300 KB
- `gameover_bg.png`: 1101 KB → Target: <500 KB
- `bg.png`: 905 KB → Target: <500 KB
- Large stone files: ~500-600 KB → Target: <200 KB each

### 3. Fix Banner Inconsistencies:
- Some banners are 128x256px (~7 KB) ✅ Good
- Some banners are 768x1408px (~1.7 MB) ❌ Too large
- Regenerate oversized banners to match correct size

---

## ✅ FILES THAT ARE CORRECT

- `ground_tex.png` - Perfect ✅
- `ground_stripe.png` - Perfect ✅
- `banner_code.png`, `banner_fire.png`, `banner_metal.png` - Perfect ✅

---

## 📝 NEW FILES NEEDED

### Running Animation Frames:
1. **`player_run_1.png`** - Passing/Mid-stride frame
2. **`player_run_2.png`** - Air/Maximum extension frame

**See:** `KELLY_RUNNING_ANIMATION_PROMPTS.md` for detailed prompts.

**Note:** Current `player.png` will become `player_run_0.png` after transparency fix.

---

## 🛠️ QUICK FIX STEPS

### Step 1: Fix Transparency (Do First)
1. Open each file in Photoshop/GIMP/Photopea
2. Select background (Magic Wand tool)
3. Delete or make transparent
4. Export as PNG-24 with transparency checked

### Step 2: Optimize Sizes
1. Use TinyPNG.com or ImageOptim
2. Upload files
3. Download optimized versions
4. Replace originals

### Step 3: Generate Animation Frames
1. Use prompts in `KELLY_RUNNING_ANIMATION_PROMPTS.md`
2. Generate Frame 1 and Frame 2
3. Ensure transparency on all frames
4. Save with correct naming

### Step 4: Update Code
Once files are fixed, update:
- `MenuScene.ts` - Load animation frames
- `GameScene.ts` - Create and play animation

---

## 📊 Current Status

**Total Issues:** 25
- **Critical (transparency):** 10 files
- **Size optimization:** 8 files  
- **Missing frames:** 2 files needed

**Time Estimate:**
- Fix transparency: 30-60 minutes
- Optimize sizes: 15 minutes
- Generate animation frames: 30-60 minutes
- **Total:** ~2 hours








