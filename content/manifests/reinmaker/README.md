# Reinmaker Overlay & FX Manifest Schema (Draft)

> Applies to all game-specific overlays, particle effects, UI frames, and audio stingers that sit on top of shared Lesson of the Day assets.

## Directory Layout
```
content/
  manifests/
    reinmaker/
      2025-10-30-gamepass.json
      2025-11-15-holiday-event.json
      README.md (this file)
assets/
  reinmaker/
    overlays/
    fx/
    audio/
```

## Naming Convention
- `YYYY-MM-DD-<slug>.json` for each manifest release.  
- Slug should match campaign or feature (`gamepass`, `holiday-event`, `battle-pass`).

## Manifest Schema (v0.1)

```json
{
  "$schema": "https://schemas.lessonoftheday.com/reinmaker/manifest-v0.1.json",
  "manifestId": "2025-10-30-gamepass",
  "version": "2025.10.30",
  "product": "reinmaker",
  "campaign": {
    "name": "Game Pass Launch",
    "startAt": "2025-11-01T00:00:00Z",
    "endAt": "2025-12-31T23:59:59Z"
  },
  "sourcePhaseDNA": {
    "id": "kelly-solar-system-v3",
    "locale": "en-US",
    "agePersona": "age8_adventurer"
  },
  "assets": [
    {
      "id": "overlay-levelup-banner",
      "type": "overlay",
      "url": "s3://lotd-reinmaker-assets/overlays/levelup-banner.png",
      "hash": "<sha256>",
      "bytes": 48213,
      "metadata": {
        "resolution": "3840x2160",
        "format": "png",
        "safeZone": "80%"
      }
    },
    {
      "id": "fx-streak-sparkle",
      "type": "fx",
      "url": "s3://lotd-reinmaker-assets/fx/streak-sparkle.vfx",
      "hash": "<sha256>",
      "bytes": 12904,
      "metadata": {
        "engine": "unity_vfxgraph",
        "durationMs": 1200
      }
    }
  ],
  "dependencies": [
    "kelly-vo-2025.10.15",
    "kelly-a2f-2025.10.15"
  ]
}
```

### Field Notes
- `manifestId` = filename without extension.  
- `sourcePhaseDNA` references the shared lesson DNA that these overlays accompany.  
- `dependencies` list shared asset package IDs (voice, animation) to guarantee compatibility.

## Storage & Hosting
- Primary bucket: `s3://lotd-reinmaker-assets` (per-region replicas coming in Phase 2).  
- Assets must be immutable once published; publish updates via new manifest version.  
- Include SHA-256 hash for verification; CI will compare against actual file hash.

## Validation Pipeline
1. Run `tools/validate_manifest.py --path content/manifests/reinmaker/<file>`.
2. Validator checks JSON schema compliance, hash presence, and S3 accessibility (HEAD request).
3. On success, pipeline uploads manifest to Reinmaker API (`/v1/content/manifests`).
4. Contract tests confirm Reinmaker acknowledges `manifestId` and status becomes `active`.

## Version Control Policy
- Commit manifests alongside related PhaseDNA updates.  
- Use PR labels `reinmaker-assets` + `content-update`.  
- Require at least one reviewer from Game Experience + one from Core Assets.

## Future Enhancements
- Add localization overrides for overlays (text in ES/FR).  
- Embed animation timeline metadata to sync FX with VO cues.  
- Automate dependency resolution by querying asset registry service.













