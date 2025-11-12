# Kelly's Hair - Deep Dive Specification

**Critical Requirement:** Hair consistency is NON-NEGOTIABLE. Kelly's hair must be EXACTLY as specified.

---

## 🔍 HAIR ANALYSIS FROM REFERENCES

### Current Description Issues
**Current:** "soft wavy to slightly curly texture"
**Problem:** "Slightly curly" suggests separate curls or frizziness
**User Feedback:** "Keep curls and hair together... not whispy or frizzy hair. Not curly."

### What This Means
- ✅ **Wavy** - YES (has movement, not straight)
- ❌ **Curly** - NO (not tight curls, not separate curls)
- ❌ **Frizzy** - NO (not separate strands, not flyaways)
- ❌ **Whispy** - NO (hair stays together, not separate strands)
- ✅ **Cohesive** - YES (waves stay together, unified)

---

## ✅ CORRECT HAIR DESCRIPTION

### Kelly's Hair Specifications

#### Texture & Style
- **Pattern:** Soft waves (NOT curly, NOT straight)
- **Cohesion:** Waves stay together in cohesive sections (NOT whispy, NOT frizzy)
- **Movement:** Natural, smooth wave pattern
- **Separation:** Minimal separation between strands - hair moves as unified sections
- **Finish:** Smooth, polished appearance (NOT frizzy, NOT unkempt)

#### Visual Characteristics
- **Wave Pattern:** Soft, flowing waves (like 2A-2B hair type)
- **Strand Cohesion:** Hair strands stay together in wave sections
- **No Flyaways:** Hair stays together, no whispy separate strands
- **No Frizz:** Smooth texture, not frizzy or unkempt
- **Natural Movement:** Waves flow naturally but stay cohesive

#### Specific Descriptions to Use

**✅ CORRECT Descriptions:**
- "soft waves that fall together cohesively"
- "smooth waves, hair stays together in cohesive sections"
- "soft wavy texture with hair strands staying together"
- "natural waves, cohesive and unified"
- "smooth flowing waves, no separation or frizz"
- "soft waves, hair moves together as unified sections"
- "cohesive soft waves, not frizzy or whispy"

**❌ INCORRECT Descriptions (AVOID):**
- "slightly curly" ❌ (implies separate curls)
- "curly" ❌ (not curly at all)
- "frizzy" ❌ (hair stays together)
- "whispy" ❌ (hair stays together)
- "separate curls" ❌ (waves stay together)
- "loose curls" ❌ (not curly, it's wavy)
- "flyaways" ❌ (no separate strands)

---

## 📋 COMPLETE HAIR SPECIFICATION

### Hair Color
- **Base Color:** Medium brown
- **Highlights:** Subtle caramel/honey-blonde highlights
- **Overall:** Rich, natural-looking color with subtle dimension

### Hair Texture & Style
- **Pattern:** Soft waves (NOT curly, NOT straight)
- **Cohesion:** Waves stay together in cohesive sections
- **Movement:** Natural, smooth wave pattern
- **Finish:** Smooth, polished, professional appearance
- **NO Characteristics:** NOT frizzy, NOT whispy, NOT curly, NOT separate strands

### Hair Structure
- **Parting:** Slightly off-center or down the middle
- **Length:** Cascades over shoulders
- **Volume:** Rich and voluminous
- **Layering:** Natural layers that create movement without separation

### Hair Behavior
- **Motion:** Hair moves together as unified sections
- **Strand Cohesion:** Strands stay together, minimal separation
- **No Flyaways:** Hair stays together, no whispy separate strands
- **Smooth Texture:** Polished, smooth appearance

---

## 🎯 UPDATED PROMPT LANGUAGE

### Correct Hair Description for Prompts

**Full Hair Description:**
```
Medium brown hair with subtle caramel/honey-blonde highlights, soft waves that fall together cohesively in unified sections, hair strands stay together (not frizzy, not whispy, not curly), smooth polished texture, parted slightly off-center or down the middle, cascades over shoulders, rich and voluminous. Hair moves as cohesive wave sections - no separate curls, no frizz, no flyaways.
```

**Shorter Version:**
```
Medium brown hair with caramel/honey-blonde highlights, soft cohesive waves (hair stays together, not frizzy or whispy), smooth polished texture, cascades over shoulders.
```

**Negative Prompts for Hair:**
```
frizzy hair, whispy hair, separate curls, tight curls, curly hair, flyaways, unkempt hair, messy hair, separate strands, hair strands flying apart
```

---

## 🔧 UPDATED CHARACTER BASE DESCRIPTION

### Before (INCORRECT):
```
Medium brown hair with subtle caramel/honey-blonde highlights, soft wavy to slightly curly texture, parted slightly off-center or down the middle, cascades over shoulders, rich and voluminous.
```

### After (CORRECT):
```
Medium brown hair with subtle caramel/honey-blonde highlights, soft waves that fall together cohesively in unified sections (hair strands stay together, NOT frizzy, NOT whispy, NOT curly), smooth polished texture, parted slightly off-center or down the middle, cascades over shoulders, rich and voluminous. Hair moves as cohesive wave sections - no separate curls, no frizz, no flyaways.
```

---

## 📝 REFERENCE IMAGE ANALYSIS

### Expected Hair Appearance in References
Based on reference images (`headshot2-kelly-base169 101225.png`, `kelly_directors_chair_8k_light (2).png`):

**Key Observations:**
- Hair has natural wave pattern
- Waves are cohesive and unified
- No visible frizz or flyaways
- Hair falls together smoothly
- Professional, polished appearance
- Rich volume without separation

**Visual Description:**
- Like well-styled 2A-2B hair type
- Smooth, flowing waves
- Hair moves together as sections
- No individual strands separating
- Polished, professional finish

---

## ✅ VALIDATION CHECKLIST

When reviewing generated assets, verify hair:

- [ ] **Pattern:** Soft waves (NOT curly, NOT straight)
- [ ] **Cohesion:** Hair stays together in unified sections
- [ ] **NO Frizz:** Smooth texture, no frizzy appearance
- [ ] **NO Whispy:** Hair stays together, no separate strands
- [ ] **NO Curly:** Not tight curls or separate curls
- [ ] **Color:** Medium brown with caramel/honey-blonde highlights
- [ ] **Texture:** Smooth, polished, professional
- [ ] **Movement:** Waves flow naturally but stay cohesive
- [ ] **Volume:** Rich and voluminous
- [ ] **Finish:** Professional, polished appearance

**All must pass for character consistency.**

---

## 🚀 IMPLEMENTATION

### Update Character Base
Replace hair description with corrected version that emphasizes:
- Soft waves (NOT curly)
- Cohesive sections (NOT frizzy/whispy)
- Hair stays together
- Smooth polished texture

### Update Prompts
Include explicit instruction:
- "hair strands stay together"
- "not frizzy, not whispy, not curly"
- "cohesive wave sections"
- "smooth polished texture"

### Update Negative Prompts
Add hair-specific negatives:
- frizzy hair, whispy hair, separate curls, tight curls, curly hair, flyaways

---

**Status:** ✅ SPECIFICATION COMPLETE  
**Next:** Update all prompts and character descriptions with corrected hair specification











