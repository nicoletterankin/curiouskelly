# Kelly Age-Progressive Images - Quick Start

## TL;DR - Run This Command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate_kelly_batch_ages.ps1
```

This generates 72 images (6 ages × 4 poses × 3 formats) in ~15-20 minutes.

---

## What You Get

- **Age 3** (Toddler) - 12 images
- **Age 9** (Child) - 12 images  
- **Age 15** (Teen) - 12 images
- **Age 27** (Adult) - 12 images
- **Age 48** (Mature) - 12 images
- **Age 82** (Elder) - 12 images

Each age has 4 poses × 3 formats:
- 16:9 (lesson player)
- 1:1 (3D reference)
- 3:4 (portrait)

---

## Step-by-Step

1. **Check environment** (optional, usually already set):
```powershell
$env:GOOGLE_CLOUD_PROJECT = "gen-lang-client-0005524332"
```

2. **Run generation**:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate_kelly_batch_ages.ps1
```

3. **Review results**:
```powershell
start projects\Kelly\assets\age_progressive\review.html
```

4. **Validate**:
```powershell
python scripts\validate_age_consistency.py
```

---

## Test First (Single Image)

To test before running all 72:

```powershell
python scripts\generate_kelly_multiformat.py presets\age_progressive\kelly_age27_pose1_v001.yaml
```

This generates 3 images for age 27, pose 1 to verify everything works.

---

## Outputs

All images saved to:
```
projects/Kelly/assets/age_progressive/renders/
```

View them in the interactive gallery:
```
projects/Kelly/assets/age_progressive/review.html
```

---

## Cost & Time

- **Time**: ~15-20 minutes (3-10 seconds per image)
- **Cost**: ~$3 (72 images × ~$0.04 each)
- **Size**: ~150-200MB total

---

## Full Documentation

See: `docs/AGE_PROGRESSIVE_GENERATION_SETUP.md`



