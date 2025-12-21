# 🔒 KELLY LESSON ARCHITECTURE - LOCKED

> **STATUS:** LOCKED - THIS IS THE PRODUCTION STANDARD  
> **LOCKED DATE:** 2024-12-19  
> **LOCKED BY:** Nicolette  

---

## 📐 THE INTERACTION MODEL

Every lesson phase follows this EXACT flow:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. KELLY SPEAKS [INTRO VIDEO/AUDIO]                            │
│     └─ Main script introducing the phase content                │
├─────────────────────────────────────────────────────────────────┤
│  2. UI REVEALS CHOICES                                          │
│     └─ 2-3 clickable options appear                             │
├─────────────────────────────────────────────────────────────────┤
│  3. LEARNER CLICKS ONE                                          │
│     └─ Choice recorded, option highlighted                      │
├─────────────────────────────────────────────────────────────────┤
│  4. KELLY RESPONDS [RESPONSE VIDEO/AUDIO]                       │
│     └─ Specific response to their choice                        │
├─────────────────────────────────────────────────────────────────┤
│  5. TRANSITION TO NEXT PHASE                                    │
│     └─ Automatic progression                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎬 CLIPS PER PHASE

| Phase | Intro | Resp A | Resp B | Resp C | Total |
|-------|-------|--------|--------|--------|-------|
| Hook | 1 | 1 | 1 | 1 | **4** |
| Cliff | 1 | 1 | 1 | - | **3** |
| Fact1 | 1 | 1 | 1 | 1 | **4** |
| Fact2 | 1 | 1 | 1 | 1 | **4** |
| Fact3 | 1 | 1 | 1 | 1 | **4** |
| Wisdom | 1 | 1 | 1 | 1 | **4** |
| Outro | 1 | - | - | - | **1** |
| **TOTAL** | **7** | **6** | **6** | **5** | **24** |

---

## 📁 FILE NAMING CONVENTION

```
kelly-{DAY}-{ARCHETYPE}-{PHASE}-{TYPE}

Examples:
  kelly-355-explorer-hook-intro      (intro for Hook phase)
  kelly-355-explorer-hook-resp-a     (response A for Hook)
  kelly-355-explorer-hook-resp-b     (response B for Hook)
  kelly-355-explorer-cliff-intro     (intro for Cliff phase)
  kelly-355-explorer-cliff-resp-a    (response A for Cliff)
```

---

## 📋 DATABASE SCHEMA

### lesson_atoms.content Structure

```json
{
  "script": "The intro script Kelly speaks...",
  "clipId": "kelly-355-explorer-hook-intro",
  "kellyPose": "curious",
  "kellyEmotion": "engaged",
  "cliffPrompt": "Which calls to you?",  // Cliff phase only
  "options": [
    {
      "letter": "A",
      "text": "Button text learner sees",
      "icon": "🗺️",                       // Optional
      "quality": "good|best|redirect",
      "response": "What Kelly says when they pick this",
      "responseClipId": "kelly-355-explorer-hook-resp-a"
    }
  ]
}
```

---

## 🎙️ KELLY'S VOICE RULES (MANDATORY)

### ✅ ALWAYS:
- Use contractions (that's, it's, you're, don't, won't)
- Use em-dashes for natural pauses (—)
- Vary openers (never start every script the same way)
- Maintain intellectual authority
- Be warm without being unprofessional
- State facts with quiet confidence

### ❌ NEVER:
- "Hey!" as an opener
- "Get this—" repeatedly
- "Crazy, right?"
- "Wild, right?"
- "How cool is that?"
- "Super cool"
- "Pretty amazing"
- Valley Girl inflections
- Teenage enthusiasm
- Dumbed-down vocabulary

---

## 🗂️ GOLD STANDARD FILES

Each day requires TWO locked files:

1. **Markdown** (human-readable):
   ```
   content/gold-standard/DAY-{XXX}-{ARCHETYPE}-LOCKED.md
   ```

2. **JSON** (machine-readable):
   ```
   content/gold-standard/DAY-{XXX}-{ARCHETYPE}-LOCKED.json
   ```

Both must contain the COMPLETE scripts for all 24 clips.

---

## 🔄 GENERATION PIPELINE

```
┌──────────────────────┐
│  1. WRITE SCRIPTS    │  Human creates/approves gold standard
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  2. LOCK CONTENT     │  Save as LOCKED.md and LOCKED.json
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  3. UPDATE DATABASE  │  Push content to lesson_atoms
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  4. GENERATE AUDIO   │  ElevenLabs: 24 MP3 files
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  5. GENERATE VIDEO   │  SadTalker: 24 MP4 files
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  6. UPLOAD ASSETS    │  Supabase Storage + kelly_video_assets
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  7. VERIFY PLAYBACK  │  Test in player, confirm all clips work
└──────────────────────┘
```

---

## 📊 QUALITY GATES

Before any day goes to production:

1. ✅ All 24 scripts reviewed and approved
2. ✅ Voice compliance verified (no forbidden phrases)
3. ✅ All 24 audio files generated
4. ✅ All 24 video files generated
5. ✅ All assets uploaded to Supabase Storage
6. ✅ All assets registered in kelly_video_assets
7. ✅ Player tested with all interaction paths
8. ✅ Duration validated (no clips under 3s or over 60s)

---

## 🚨 ARCHETYPES (7 VARIANTS PER DAY)

Each day has 7 archetype variants, each needing 24 clips:

| Archetype | Total Clips |
|-----------|-------------|
| The Explorer | 24 |
| The Sage | 24 |
| The Hero | 24 |
| The Creator | 24 |
| The Caregiver | 24 |
| The Jester | 24 |
| The Rebel | 24 |
| **DAILY TOTAL** | **168 clips** |

For 365 days: **61,320 total clips**

---

## 🎯 CURRENT STATUS

### Day 355 (December 20, 2024)

| Archetype | Scripts | Audio | Video | Status |
|-----------|---------|-------|-------|--------|
| The Explorer | ✅ LOCKED | ⬜ | ⬜ | Ready for generation |
| The Sage | ⬜ | ⬜ | ⬜ | Pending |
| The Hero | ⬜ | ⬜ | ⬜ | Pending |
| The Creator | ⬜ | ⬜ | ⬜ | Pending |
| The Caregiver | ⬜ | ⬜ | ⬜ | Pending |
| The Jester | ⬜ | ⬜ | ⬜ | Pending |
| The Rebel | ⬜ | ⬜ | ⬜ | Pending |

---

## 📜 COMMANDS

### Generate audio from gold standard:
```bash
npx tsx scripts/generate-full-lesson-audio.ts --from-gold-standard=content/gold-standard/DAY-355-EXPLORER-LOCKED.json
```

### Generate audio from database:
```bash
npx tsx scripts/generate-full-lesson-audio.ts --day=355 --archetype="The Explorer"
```

### Validate gold standard:
```bash
npx tsx scripts/validate-gold-standard.ts content/gold-standard/DAY-355-EXPLORER-LOCKED.json
```

---

## 🔒 LOCK CERTIFICATE

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   THIS ARCHITECTURE IS LOCKED                                    ║
║                                                                  ║
║   - 24 clips per archetype per day                               ║
║   - 7 archetypes per day                                         ║
║   - 168 clips per day total                                      ║
║   - Full interaction coverage                                    ║
║   - Kelly speaks EVERYTHING                                      ║
║                                                                  ║
║   Locked: 2024-12-19                                             ║
║   Approved by: Nicolette                                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**DO NOT MODIFY THIS ARCHITECTURE WITHOUT EXPLICIT APPROVAL**
