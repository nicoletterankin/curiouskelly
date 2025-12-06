# KELLY PRODUCTION FACTORY

**Status:** READY FOR PRISTINE GENERATION  
**Last Audit:** December 6, 2025  
**Quarantined:** 831 imposter/suspicious files  

---

## CANONICAL KELLY SPECIFICATION

### Physical Characteristics (MUST MATCH)

| Attribute | Specification | CRITICAL |
|-----------|---------------|----------|
| **Eyes** | Brown, warm, expressive | YES |
| **Hair** | Long wavy brown with subtle blonde highlights | YES |
| **Hair Length** | Below shoulders | YES |
| **Face** | Soft features, distinctive nose shape | YES |
| **Skin** | Fair/medium with warm undertones | YES |
| **Age** | Late 20s (27-29) | YES |
| **Outfit** | Light blue/teal ribbed crew-neck sweater | YES |
| **Style** | Photorealistic 3D render (iClone/CC quality) | YES |

### Reference Location

```
C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference\
├── close up of face.jpeg           <- PRIMARY FACE REFERENCE
├── head and shoulders without chair.png
├── neutral face with hair.png      <- Has transparent background
├── Curious Kelly in final pose in Chair...png
├── facing to the left.png
├── slightly turning her head.png
├── profile of kelly.png
├── close up of kellys eyes.png
└── CANONICAL_CHECKSUMS.sha256      <- Integrity verification
```

---

## PRE-GENERATION CHECKLIST

Before generating ANY Kelly images, verify:

- [ ] Reference images loaded into generation tool
- [ ] Correct model/checkpoint selected
- [ ] Prompt includes ALL canonical characteristics
- [ ] Blue/teal sweater specified
- [ ] Brown eyes specified (NOT green/hazel)
- [ ] Long wavy brown hair with blonde highlights specified
- [ ] Output resolution set (minimum 1344x768)
- [ ] Background set (white studio or transparent)

---

## GENERATION PROMPTS

### Base Prompt (ALWAYS USE)

```
A photorealistic 3D rendered portrait of a young woman in her late twenties,
long wavy brown hair with subtle blonde highlights flowing past her shoulders,
warm brown eyes, fair-medium skin tone with warm undertones, athletic healthy build,
wearing a blue-teal ribbed crew-neck sweater, soft professional studio lighting,
clean white background, 16:9 aspect ratio, ultra-high quality 3D render,
Character Creator or iClone style, film-grade photorealistic quality,
natural beauty, approachable and friendly demeanor
```

### Negative Prompt (ALWAYS USE)

```
green eyes, hazel eyes, blue eyes, short hair, straight hair, blonde hair,
different sweater, different outfit, cartoon, anime, illustration, painting,
low quality, blurry, distorted, extra limbs, deformed face
```

---

## POST-GENERATION VALIDATION

After generating, run Kelly Cop face audit:

```powershell
cd C:\Users\user\UI-TARS-desktop\tools\kelly-cop
python kelly_face_audit.py --limit [NUMBER_OF_NEW_FILES] --html
```

### Acceptance Criteria

| Status | Action |
|--------|--------|
| MATCH (distance < 0.385) | APPROVED - Use in production |
| SUSPICIOUS (0.385-0.55) | REVIEW - Manual inspection required |
| NO_MATCH (> 0.55) | REJECT - Regenerate |

---

## QUALITY GATES

### Gate 1: Visual Inspection
- [ ] Eyes are brown (not green/hazel)
- [ ] Hair matches reference (long, wavy, brown with blonde highlights)
- [ ] Sweater is correct blue/teal color
- [ ] Face structure matches reference

### Gate 2: Face Recognition
- [ ] Run `kelly_face_audit.py`
- [ ] All images score MATCH or SUSPICIOUS
- [ ] Zero NO_MATCH files

### Gate 3: Consistency Check
- [ ] All images in batch look like the same person
- [ ] No outliers or style drift
- [ ] Lighting is consistent

---

## FOLDER STRUCTURE

### Production Assets (APPROVED)

```
public/kelly/
├── poses/          <- Kelly pose images (VERIFIED)
├── phases/         <- Lesson phase images (NEEDS REGENERATION)
├── lessons/        <- Lesson-specific images (NEEDS REGENERATION)
├── thumbnails/     <- Thumbnail images
└── choices/        <- Choice UI images
```

### Quarantine (DO NOT USE)

```
_quarantine/kelly-imposters/
├── generated-poses-final/      <- 12 imposters (different person)
├── no-match-batch-20251206/    <- 440 failed face verification
└── suspicious-batch-20251206/  <- 391 needs manual review
```

### Archive (Reference Only)

```
_archive/
├── test-files-20251206/       <- Old test files
└── stray-kelly-docs/          <- Scattered documentation
```

---

## REGENERATION PRIORITY

### Phase 1: Lesson Assets (HIGHEST)
- 365 lessons × 5 images each = 1,825 images needed
- Use batch generation with strict quality gates

### Phase 2: Phase Assets
- `phases/001` through `phases/365`
- hook.png, q1.png, q2.png, q3.png, wisdom.png per lesson

### Phase 3: UI Assets
- Thumbnails
- Choice cards
- Marketing materials

---

## KELLY COP TOOLS

### Location
```
C:\Users\user\UI-TARS-desktop\tools\kelly-cop\
```

### Available Commands

```powershell
# Face recognition audit (recommended)
python kelly_face_audit.py --html

# Perceptual hash audit (whole image)
python kelly_audit.py --html

# Batch quarantine
python quarantine_batch.py
```

### Reports Location
```
tools/kelly-cop/face_audit_report/
tools/kelly-cop/audit_report/
```

---

## EMERGENCY PROCEDURES

### If Imposters Detected in Production

1. Stop generation immediately
2. Run full face audit:
   ```
   python kelly_face_audit.py --html
   ```
3. Review NO_MATCH files
4. Quarantine imposters:
   ```
   python quarantine_batch.py
   ```
5. Fix generation prompts/references
6. Regenerate affected files

### If Reference Images Corrupted

1. Verify checksums:
   ```
   cd "C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference"
   Get-FileHash *.png,*.jpeg | Format-Table
   ```
2. Compare against `CANONICAL_CHECKSUMS.sha256`
3. If mismatch, restore from backup

---

## CONTACTS

**Reference Documentation:**
- `C:\iLearnStudio\projects\Kelly\Ref\GENERATION_PROMPTS.md`
- `C:\iLearnStudio\projects\Kelly\Ref\KELLY_ASSET_CATALOG.md`
- `C:\iLearnStudio\projects\Kelly\Ref\QUICK_REFERENCE.md`

---

**Factory Status:** PRISTINE  
**Ready for Generation:** YES  
**Quality System:** ACTIVE  

