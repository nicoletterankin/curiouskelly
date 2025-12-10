# HeyGen Kelly Video Pipeline - READY

## 🎯 Quick Commands

### Check video status
```bash
npx tsx scripts/heygen-check-video.ts <video_id>
```

### Download completed video + upload to Supabase
```bash
npx tsx scripts/heygen-download-upload.ts <video_id> [day] [phase] [archetype]
# Example:
npx tsx scripts/heygen-download-upload.ts abc123 1 Hook "The Scientist"
```

### Generate video with Kelly avatar
```bash
npx tsx scripts/heygen-kelly-pipeline.ts
# Or for Day 1:
npx tsx scripts/heygen-kelly-pipeline.ts --day1
```

### List all talking photos (to find avatar IDs)
```bash
npx tsx scripts/heygen-list-avatars.ts
```

---

## 📋 Avatar ID Mapping

Update `scripts/heygen-kelly-pipeline.ts` with your Kelly talking photo IDs:

```typescript
const KELLY_AVATARS: Record<string, string> = {
  "The Scientist": "paste_id_here",
  "The Explorer": "paste_id_here",
  "The Rebel": "paste_id_here",
  "The Architect": "paste_id_here",
  "The Diplomat": "paste_id_here",
  "The Empath": "paste_id_here",
  "The MacGyver": "paste_id_here",
  "The Mystic": "paste_id_here",
  "The Provider": "paste_id_here",
  "The Storyteller": "paste_id_here",
  "The Strategist": "paste_id_here",
  "The Survivor": "paste_id_here",
};
```

---

## 🎬 Video Output Locations

- **Local**: `generated-videos/heygen-kelly/`
- **Supabase**: `kelly-videos/production/heygen/`
- **Database**: `lesson_atoms.hd_video_url`

---

## 📝 Motion Prompts

See `docs/HEYGEN_12_MOTION_PROMPTS.md` for archetype-specific motion prompts.

---

## ✅ Ready Scripts

| Script | Purpose |
|--------|---------|
| `heygen-kelly-pipeline.ts` | Full: Audio → HeyGen → Supabase → DB |
| `heygen-download-upload.ts` | Download + upload individual videos |
| `heygen-check-video.ts` | Check video generation status |
| `heygen-list-avatars.ts` | List all talking photo IDs |
| `heygen-v1-test.ts` | API testing |

---

*Created: December 10, 2025*

