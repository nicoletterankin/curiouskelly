# KELLY LESSON ASSETS - REGENERATION MANIFEST

**Created:** December 6, 2025  
**Status:** READY FOR GENERATION  
**Total Lessons:** 365  

---

## FOLDER STRUCTURE

```
public/kelly/
├── poses/          <- VERIFIED (10 images - keep as-is)
├── phases/         <- EMPTY (ready for generation)
│   ├── 001/
│   ├── 002/
│   └── ... (through 365/)
├── lessons/        <- EMPTY (ready for generation)
│   ├── 001/
│   ├── 002/
│   └── ... (through 365/)
└── thumbnails/     <- EMPTY (ready for generation)
```

---

## REQUIRED ASSETS PER LESSON

### Phase Assets (per lesson folder)
Each `phases/XXX/` folder needs:

| File | Description | Resolution |
|------|-------------|------------|
| `hook.png` | Welcome/intro image | 1344×768 |
| `q1.png` | Question 1 image | 1344×768 |
| `q2.png` | Question 2 image | 1344×768 |
| `q3.png` | Question 3 image | 1344×768 |
| `wisdom.png` | Wisdom/conclusion image | 1344×768 |

**Total per lesson:** 5 images  
**Total for 365 lessons:** 1,825 images

### Lesson Assets (per lesson folder)
Each `lessons/XXX/` folder needs:

| File | Description | Resolution |
|------|-------------|------------|
| `lesson-X-hero.png` | Hero/main image | 1344×768 |
| `lesson-X-bg.png` | Background image | 1344×768 |
| `lesson-X-guide-point.png` | Guide/pointer image | 768×1344 |
| `lesson-X-reaction.png` | Reaction image | 1024×1024 |
| `lesson-X-prop.png` | Prop/accessory image | 1344×768 |

**Total per lesson:** 5 images  
**Total for 365 lessons:** 1,825 images

### Thumbnails
Each lesson needs a thumbnail:

| File | Resolution |
|------|------------|
| `lesson-XXX.png` | 512×512 |

**Total:** 365 thumbnails

---

## GRAND TOTAL

| Category | Count |
|----------|-------|
| Phase images | 1,825 |
| Lesson images | 1,825 |
| Thumbnails | 365 |
| **TOTAL** | **4,015 images** |

---

## GENERATION REQUIREMENTS

### Must Include
- Brown eyes
- Long wavy brown hair with blonde highlights
- Blue/teal ribbed crew-neck sweater
- Soft features, late 20s appearance
- Photorealistic 3D render style

### Reference Images
```
C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference\
```

### Generation Prompts
```
C:\iLearnStudio\projects\Kelly\Ref\GENERATION_PROMPTS.md
```

---

## VALIDATION PROCESS

After each batch:

```powershell
cd C:\Users\user\UI-TARS-desktop\tools\kelly-cop
python kelly_face_audit.py --html
```

**Acceptance:** Only MATCH status files approved for production.

---

## VERIFIED ASSETS (DO NOT REGENERATE)

The following are already verified and should be kept:

```
public/kelly/poses/
├── kelly_idle.png
├── kelly_welcome.png
├── kelly_hint.png
├── kelly_hint_flip.png
├── kelly_clasp.png
├── kelly_listening.png
├── kelly_choice_left.png
├── kelly_choice_right.png
└── (others in this folder)
```

---

## NOTES

- All previous lesson assets have been backed up to quarantine
- Clean folder structure created for 365 lessons
- Quality control system active via Kelly Cop
- Reference images verified and checksummed

---

**LET'S MAKE PRISTINE KELLY CONTENT!**

