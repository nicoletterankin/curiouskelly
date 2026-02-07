# Kelly Video Asset Inventory

**Created:** 2026-02-03
**Purpose:** Map all existing video assets before generating more

## The Real Math

| Dimension | Count | Values |
|-----------|-------|--------|
| **Ages** | 3 | Kid, Adult, Elder |
| **Languages** | 6 | EN, ES, FR, DE, PT, ZH |
| **Archetypes** | 12 | scientist, explorer, rebel, architect, diplomat, empath, macgyver, mystic, provider, storyteller, strategist, survivor |
| **Phases** | 5 | hook, story, wonder, action, wisdom |
| **Days** | 365 | |
| **TOTAL FINAL VIDEOS** | **394,200** | 3 × 6 × 12 × 5 × 365 |

---

## Current Status: INVENTORY NEEDED

### Known Video Locations

#### 1. Vercel Blob Storage
- URL pattern: `https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly-base-videos/...`
- **STATUS:** Need to list all files in this bucket

#### 2. Local: `C:\Users\user\kelly-pipeline\videos\`
```
base/
├── adult/
│   ├── curious/    (4 videos, UUID named)
│   ├── excited/    (20+ videos, UUID named)
│   └── talking/    (80+ videos, UUID named)
└── [kid/, elder/ - NEED TO CHECK]

lipsync/
├── 2026/en/day-020/  (test lip-syncs)
└── [age-range tests]
```

#### 3. Local: `C:\Users\user\heygen-batch\output\`
- `base/` folder
- `day-021/` folder (reported 913 videos, need to verify)

#### 4. Local: Other locations
- `C:\Users\user\template-forge\lesson-videos\` (239 videos, Days 1-5)
- `C:\Users\user\kelly-sovereign-training\videos\` (134 raw HeyGen)
- `C:\Users\user\UI-TARS-desktop\generated-videos\heygen-production\` (49 videos)

#### 5. Supabase Storage
- Bucket: `kelly-videos`
- Some production videos exist (day_001)
- Bucket: `kelly-templates` (images, audio)

---

## The 12 Archetypes

| ID | Name | Icon | Color |
|----|------|------|-------|
| scientist | The Scientist | 🔬 | #3b82f6 |
| explorer | The Explorer | 🧭 | #eab308 |
| rebel | The Rebel | ⚡ | #ef4444 |
| architect | The Architect | 🏛️ | #6b7280 |
| diplomat | The Diplomat | 🤝 | #22c55e |
| empath | The Empath | 💗 | #ec4899 |
| macgyver | The MacGyver | 🔧 | #f97316 |
| mystic | The Mystic | ✨ | #a855f7 |
| provider | The Provider | 🛡️ | #14b8a6 |
| storyteller | The Storyteller | 📖 | #f472b6 |
| strategist | The Strategist | 🎯 | #6366f1 |
| survivor | The Survivor | 🏕️ | #84cc16 |

---

## Critical Questions to Answer

1. **UUID → Archetype Mapping**
   - HeyGen videos have UUID filenames
   - Need to find the mapping (likely in HeyGen dashboard or a manifest file)

2. **What's in Vercel Blob?**
   - Need to list `kelly-base-videos/` bucket
   - This requires BLOB_READ_WRITE_TOKEN

3. **Age variants exist?**
   - Manifest shows: kid, teen, adult, elder, super_elder
   - User says: Kid, Adult, Elder (3)
   - Where are Kid and Elder videos?

4. **Language-specific bases?**
   - Base videos are lip-sync targets (no audio)
   - Same base can be used for all languages
   - BUT expression might vary by culture

---

## Next Steps

### BEFORE generating anything:

- [ ] List Vercel Blob contents (`kelly-base-videos/`)
- [ ] Map UUID → Archetype for existing videos
- [ ] Verify what ages exist as base videos
- [ ] Check HeyGen dashboard for video inventory
- [ ] Create unified asset registry

### DO NOT:
- Run batch lip-sync with single base video
- Generate new HeyGen videos without checking existing
- Spend money on duplicate work

---

## v0.app Integration Notes

v0.app has direct access to:
- Vercel deployments
- Production database (possibly different schema)
- Possibly different storage buckets

**Risk:** v0 may have created assets/schema that we can't see from Cursor.

---

*This document should be updated as we discover assets.*
