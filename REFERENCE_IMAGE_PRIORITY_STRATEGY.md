# Reference Image Priority Strategy

**Date:** November 1, 2025  
**Status:** ✅ IMPLEMENTED - Reference Images are Primary

---

## 🎯 CORE PRINCIPLE

**Reference Images > Text Descriptions**

When reference images are available:
- ✅ **PRIMARY:** Reference images handle ALL character likeness
- ❌ **SECONDARY:** Text descriptions are minimal (scene/wardrobe only)
- ❌ **REMOVED:** Detailed character descriptions from text

---

## ✅ UPDATED STRATEGY

### With Reference Images (PRIMARY METHOD)
```
Prompt: "[Scene Description], featuring Kelly Rein, [Wardrobe]"
Reference Images: [Primary character references loaded]
Result: Reference images handle face shape, hair length, features
```

**Benefits:**
- Reference images show exact face shape (oval, soft contours)
- Reference images show exact hair length (long, past shoulders)
- Reference images show exact features (eyes, skin tone, etc.)
- No ambiguity from text descriptions

### Without Reference Images (FALLBACK ONLY)
```
Prompt: "[Scene Description], featuring [Detailed Character Base], [Wardrobe]"
Reference Images: None
Result: Text descriptions attempt character likeness (less accurate)
```

**Only used when:** Reference images unavailable or API format issue

---

## 🔧 IMPLEMENTATION CHANGES

### Build-KellyPrompt Function
**Before:**
- Always included detailed character base text
- Added text instructions even with reference images

**After:**
- **With references:** Minimal prompt (scene + wardrobe only)
- **Without references:** Full character base text (fallback)

### Reference Image Format
**Updated:** Wrapped format to fix API error
```json
{
  "referenceImages": [
    {
      "image": {
        "bytesBase64Encoded": "...",
        "mimeType": "image/png"
      }
    }
  ]
}
```

---

## 📋 REFERENCE IMAGE SELECTION

### Primary References (Best Quality)
1. `headshot2-kelly-base169 101225.png` - Primary Headshot 2
2. `kelly_directors_chair_8k_light (2).png` - 8K quality

### Usage Strategy
- **Start with 1-2 best references** (don't overload API)
- **If format works:** Use best 1-2 primary references
- **If format fails:** Try alternative formats or GCS URIs

---

## 🧪 TESTING PLAN

1. **Test Format:** Wrapped "image" object format
2. **If succeeds:** Use reference images for all Kelly assets
3. **If fails:** Try alternative formats (parameters section, GCS URIs, etc.)
4. **Goal:** Get reference images working, then regenerate all Kelly assets

---

## 🎯 EXPECTED RESULTS

**With Working Reference Images:**
- ✅ Perfect character likeness (matches reference exactly)
- ✅ Correct face shape (oval, soft contours from reference)
- ✅ Correct hair length (long, past shoulders from reference)
- ✅ Consistent features across all assets

**This is the ONLY way to achieve perfect character consistency.**

---

**Status:** ✅ Strategy Updated - Reference Images are Primary  
**Next:** Test format fix, then regenerate all Kelly assets with references












