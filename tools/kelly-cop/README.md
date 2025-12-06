# KELLY COP - Visual Identity Protection System

**Purpose:** Ensure all Kelly images match the canonical "Our Kelly" character.

---

## Quick Start

```powershell
cd C:\Users\user\UI-TARS-desktop\tools\kelly-cop

# Face recognition audit (RECOMMENDED)
python kelly_face_audit.py --html

# Test on limited files first
python kelly_face_audit.py --limit 50 --html

# Batch quarantine flagged files
python quarantine_batch.py
```

---

## Tools

| Tool | Purpose | Output |
|------|---------|--------|
| `kelly_face_audit.py` | Face recognition comparison | JSON, CSV, HTML |
| `kelly_audit.py` | Perceptual hash comparison | JSON, CSV, HTML |
| `quarantine_batch.py` | Move flagged files | Manifest |

---

## Canonical Kelly Reference

**Location:** `C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference\`

**Key Characteristics:**
- Brown eyes (NOT green/hazel)
- Long wavy brown hair with blonde highlights
- Blue/teal ribbed crew-neck sweater
- Late 20s age
- Photorealistic 3D render style

---

## Face Audit Thresholds

| Status | Distance | Meaning |
|--------|----------|---------|
| MATCH | < 0.385 | Confirmed Our Kelly |
| SUSPICIOUS | 0.385-0.55 | Needs manual review |
| NO_MATCH | > 0.55 | Different person |

---

## Production Workflow

1. **Before Generation:** Load canonical references
2. **During Generation:** Use approved prompts from `GENERATION_PROMPTS.md`
3. **After Generation:** Run `kelly_face_audit.py --html`
4. **Quality Gate:** Only MATCH files go to production

See `KELLY_PRODUCTION_FACTORY.md` for full details.

---

## Reports

```
face_audit_report/
├── face_audit_[timestamp].json      # Summary stats
├── face_audit_details_[timestamp].csv # Per-image results
├── face_no_matches_[timestamp].txt  # Imposter file paths
└── face_visual_report_[timestamp].html # Visual gallery
```

---

## Quarantine Location

```
C:\Users\user\UI-TARS-desktop\_quarantine\kelly-imposters\
```

---

## Emergency Commands

```powershell
# Full audit
python kelly_face_audit.py --html

# Quarantine all flagged
python quarantine_batch.py

# View visual comparison
start kelly_comparison.html
```

---

**Version:** 2.0  
**Updated:** December 6, 2025  
**Status:** Production Ready
