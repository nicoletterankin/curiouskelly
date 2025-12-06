# 🚔 Kelly Cop - Visual Identity Audit Tool

**Purpose:** Scan all Kelly avatar images and verify they match "Our Kelly" using perceptual hashing.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the audit
python kelly_audit.py

# 3. Generate HTML visual report
python kelly_audit.py --html

# 4. Quarantine imposters (moves files!)
python kelly_audit.py --quarantine
```

## What It Does

1. **Loads canonical references** from `C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference`
2. **Computes perceptual hashes** (pHash, aHash, dHash) for each image
3. **Compares all Kelly images** against the canonical references
4. **Classifies each image** as:
   - ✅ **GOOD** - Matches Our Kelly (distance ≤ 15)
   - ⚠️ **SUSPICIOUS** - Might be off, needs review (distance 16-25)
   - 🚨 **IMPOSTER** - Different person (distance > 25)
5. **Generates reports** (JSON, CSV, HTML)

## Output

```
tools/kelly-cop/audit_report/
├── audit_summary_YYYYMMDD_HHMMSS.json    # Summary statistics
├── audit_details_YYYYMMDD_HHMMSS.csv     # Full results for each image
├── imposters_YYYYMMDD_HHMMSS.txt         # List of imposter file paths
└── visual_report_YYYYMMDD_HHMMSS.html    # Visual gallery (with --html)
```

## Canonical "Our Kelly" Specs

| Attribute | Specification |
|-----------|---------------|
| **Eyes** | Brown (NOT green/hazel) |
| **Hair** | Long wavy brown with blonde highlights |
| **Age** | Late 20s (27-29) |
| **Outfit** | Light blue/teal ribbed crew-neck sweater |
| **Setting** | Black canvas director's chair |
| **Style** | Photorealistic 3D render (iClone/CC quality) |

## Folders Scanned

- `public/kelly/` (poses, phases, lessons, thumbnails)
- `generated-poses-final/`
- `generated-poses-production/`
- `generated-poses-presenter/`
- `daily-lesson-marketing/public/assets/kelly/`
- Root-level Kelly test images

## Commands

```bash
# Basic scan with console output
python kelly_audit.py

# Scan and generate HTML visual report
python kelly_audit.py --html

# Scan and move imposters to quarantine
python kelly_audit.py --quarantine

# All options
python kelly_audit.py --html --quarantine --workers 16

# Scan only (no file output)
python kelly_audit.py --scan-only
```

## How Perceptual Hashing Works

Unlike cryptographic hashes (MD5, SHA), **perceptual hashes** generate similar values for visually similar images.

We use three hash types:
- **pHash** (perceptual hash) - Best for facial recognition
- **aHash** (average hash) - Fast overall similarity
- **dHash** (difference hash) - Good for detecting modifications

The tool computes a **weighted combination** (pHash 50%, aHash 25%, dHash 25%) and calculates the **Hamming distance** to reference images.

## Thresholds

| Status | Hamming Distance | Interpretation |
|--------|------------------|----------------|
| GOOD | 0-15 | Very similar to Our Kelly |
| SUSPICIOUS | 16-25 | Might have issues, review needed |
| IMPOSTER | 26+ | Definitely a different person |

## Known Issues

Based on initial audits:
- **`generated-poses-final/`** - Contains imposters (different person)
- AI-generated images in Ref folder may have inconsistent quality

## Maintenance

After quarantining imposters:
1. Review quarantine folder manually
2. Delete confirmed imposters or regenerate with correct prompts
3. Update generation prompts to ensure consistency
4. Re-run audit to verify cleanup

---

**Created:** December 2025  
**Author:** Kelly Cop 🚔

