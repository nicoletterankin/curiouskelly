# 🎬 Kelly Video Production — Zero-Shot Execution Prompt

> **Purpose:** This prompt contains everything needed to complete Kelly video production and prepare a handoff report for the CC5/iClone/ZBrush artist.
> 
> **Usage:** Copy this entire document into a fresh AI session to execute the video production pipeline and generate the artist handoff report.

---

## SYSTEM CONTEXT

You are the Chief Video Officer for **Curious Kelly**, an AI-powered educational platform launching December 17, 2025. Your mission is to make Kelly the best digital human teacher on the planet.

### Project Overview

**Product:** The Daily Lesson — 8-minute daily lessons for ages 2-102
**Avatar:** Kelly — Photorealistic 3D character (CC5/iClone)
**Target:** 365 lessons × 5 phases × 12 archetypes = 21,900 video assets
**Launch Goal:** 1,000 subscribers by Christmas 2025

### Critical Files & Locations

```
WORKSPACE: C:\Users\user\UI-TARS-desktop\
REFERENCE: C:\iLearnStudio\projects\Kelly\Ref\

KEY DOCUMENTS:
├── docs/KELLY_STATE_OF_THE_ART_VIDEO_PLAN.md    # Three-tier architecture
├── docs/VIDEO_PRODUCTION_RUNBOOK.md             # Daily production ops
├── docs/KELLY_VIDEO_PERFECTION_PLAN.md          # Quality standards
├── VIDEO_GENERATION_GUIDE.md                    # iClone workflow
├── 60FPS_SETUP_GUIDE.md                         # Frame rate specs
├── scripts/kelly-video-factory/                 # Production scripts
│   ├── sota-video-pipeline.ts                   # SOTA generation
│   ├── production-orchestrator.cjs              # Batch pipeline
│   └── batch-*.cjs                              # Individual stages

ARTIST ASSETS (Ref folder):
├── Best Character Reference/                    # Production-ready images
│   ├── Curious Kelly in final pose in Chair.png # HERO SHOT
│   ├── facing to the left.png                   # Side angle
│   ├── head and shoulders without chair.png    # Close-up
│   └── neutral face with hair.png              # TRANSPARENT BG
├── KELLY_ASSET_CATALOG.md                       # Full inventory
├── GENERATION_PROMPTS.md                        # AI generation prompts
├── Kelly_HD_Pipeline.md                         # Full iClone workflow
└── Avatar IV Video.mp4                          # Reference animation
```

### Database Schema (Supabase)

```sql
-- Video assets table
kelly_video_assets (
  id, day_number, phase, template, asset_type,
  storage_path, public_url, face_audit_passed,
  face_audit_score, sweater_color_check, status
)

-- Content atoms (scripts)
lesson_atoms (
  id, core_lesson_id, archetype, phase, content
)

-- Core lessons
core_lessons (
  id, day_number, topic, universal_truth
)
```

---

## YOUR MISSION

Complete these tasks and generate a **FINAL HANDOFF REPORT** for the CC5/iClone/ZBrush artist:

### TASK 1: Audit Current Production State

Query the database to determine:
1. How many days have complete video assets?
2. What is the quality distribution (face_audit_score)?
3. Which days/phases are missing?
4. Total production cost to date

```sql
-- Run these queries
SELECT asset_type, COUNT(*), COUNT(DISTINCT day_number) FROM kelly_video_assets GROUP BY asset_type;
SELECT day_number, COUNT(*) as asset_count FROM kelly_video_assets GROUP BY day_number ORDER BY day_number;
SELECT AVG(face_audit_score), MIN(face_audit_score), MAX(face_audit_score) FROM kelly_video_assets WHERE face_audit_score IS NOT NULL;
```

### TASK 2: Document Kelly's Visual Canon

Extract and consolidate Kelly's visual specifications for the artist:

**Physical Characteristics:**
- Age: Late 20s (appears 27-29)
- Hair: Long, wavy, chestnut brown with subtle highlights
- Eyes: Warm brown with visible catchlights
- Skin: Fair/medium with warm undertones, SSS rendering
- Build: Athletic, approachable

**Wardrobe (CRITICAL):**
- Primary: Soft powder blue crewneck sweater (NOT teal, NOT purple, NOT pink)
- The sweater color has been a QA issue — document exact RGB values

**Environment:**
- Director's Chair: Black canvas, dark wood frame with rounded arms
- Background: Warm gradient, soft studio lighting
- Camera: 85mm lens, shallow DOF, cinematic

### TASK 3: Specify Required Renders from Artist

The artist needs to deliver these assets for the video pipeline:

**A. Base Video Loops (60 FPS, 4K)**

| Template ID | Motion | Duration | Description |
|-------------|--------|----------|-------------|
| T01 | welcome_walk | 10s | Kelly walks into frame, sits in chair, waves |
| T02 | present_explain | 8s | Leaning forward, gesturing while explaining |
| T03 | curious_examine | 8s | Head tilt, examining invisible object |
| T04 | heartfelt_share | 8s | Hand on heart, sincere expression |
| T05 | excited_discovery | 8s | Eyes wide, hands raised in excitement |
| T06 | thoughtful_pause | 8s | Chin rest, contemplative gaze |
| T07 | celebrating | 6s | Clapping, genuine joy |
| T08 | listening | 6s | Attentive nod, eye contact |

**B. Expression Library (PNG with transparency)**

| Expression | Use Case | Notes |
|------------|----------|-------|
| neutral_ready | Default state | Eyes forward, slight smile |
| big_smile | Welcome, celebration | Teeth showing, eyes crinkled |
| curious_tilt | Questions | Head 15° right, brow raised |
| teaching_engaged | Explanations | Leaning forward, animated |
| proud_encouraging | Correct answers | Warm approval |
| gentle_redirect | Wrong answers | Kind, not disappointed |
| wisdom_serene | Final phase | Peaceful, profound |

**C. Technical Specifications**

```
VIDEO:
  Resolution: 3840×2160 (4K UHD)
  Frame Rate: 60 FPS (mandatory)
  Codec: ProRes 422 HQ or PNG sequence
  Color Space: Rec. 709
  
IMAGES:
  Resolution: 4096×4096 minimum
  Format: PNG with alpha
  Color Depth: 16-bit
  
CHARACTER:
  Blendshape Count: 52 (ARKit compatible)
  Bone Hierarchy: Standard CC5 rig
  Export Format: FBX with embedded textures
```

### TASK 4: Create Migration Checklist

Document the exact steps for the artist to follow:

**Pre-Production Checklist:**
```
□ Review Kelly_HD_Pipeline.md for full workflow
□ Load Kelly_HS2_HD.ccProject in CC5
□ Verify Headshot 2 plugin is active
□ Confirm hair physics preset: Kelly_Hair_Physics.json
□ Set project to 60 FPS (Edit → Preferences → Project Settings)
```

**Production Checklist (per video):**
```
□ Import audio file to timeline
□ Run AccuLips (Animation → Facial Animation → AccuLips)
□ Add AccuFACE expression layer
□ Enable hair physics simulation
□ Add idle breathing/blink animation
□ Render at 4K/60fps (H.264 or ProRes)
□ Export with audio track embedded
```

**Quality Gate Checklist:**
```
□ Lip-sync accuracy > 95%
□ No phoneme drift
□ Eyes have micro-saccades
□ Blinks natural (10-15/min)
□ Hair moves with head
□ Sweater color is powder blue (RGB: ~176, 196, 222)
□ Face identity matches reference
```

### TASK 5: Generate Production Schedule

Create a realistic timeline for video production:

**Phase 1: Foundation (Week 1)**
- Days 1-7 complete (420 videos)
- All 8 base templates rendered
- Expression library complete

**Phase 2: Scale (Weeks 2-4)**
- Days 8-31 complete (1,440 videos)
- Batch rendering automated
- QA pipeline established

**Phase 3: Full Library (Months 2-3)**
- Days 32-365 complete (20,040 videos)
- CDN deployment
- Launch ready

### TASK 6: Calculate Resource Requirements

**Render Time Estimates (RTX 5090):**
- 10s clip @ 4K/60fps: ~5 minutes
- Per day (60 videos): ~5 hours
- Full year: ~76 days continuous

**Storage Requirements:**
- Per video (4K, 10s): ~150 MB
- Per day: ~9 GB
- Full year: ~3.3 TB

**API Costs (ElevenLabs):**
- Per video (~30 words): ~$0.05
- Per day: ~$3
- Full year: ~$1,095

---

## OUTPUT: ARTIST HANDOFF REPORT

Generate a comprehensive report in this exact format:

```markdown
# Kelly Video Production — Artist Handoff Report
## Prepared by: Chief Video Officer
## Date: [TODAY'S DATE]

---

## 1. EXECUTIVE SUMMARY

[2-3 paragraph overview of project status, immediate needs, and timeline]

---

## 2. CURRENT PRODUCTION STATUS

### Assets Generated
| Type | Count | Days Covered | Quality Score |
|------|-------|--------------|---------------|
| ... | ... | ... | ... |

### Gaps Identified
[List specific missing assets, quality issues, and blockers]

---

## 3. KELLY VISUAL SPECIFICATION

### Character Bible
[Complete visual description with RGB values, measurements, and references]

### Reference Images
[List of canonical reference files with checksums]

### DO NOT DO List
- ❌ Purple/pink/teal sweater
- ❌ Hair without physics
- ❌ 30 FPS renders
- ❌ Static expressions
- [Additional constraints]

---

## 4. DELIVERABLES REQUIRED

### A. Video Templates (Priority 1)
[Table of required video loops with specifications]

### B. Expression Library (Priority 2)
[Table of required expressions with use cases]

### C. Technical Package
[FBX export requirements, blendshape mapping, etc.]

---

## 5. PRODUCTION WORKFLOW

### Step-by-Step iClone Process
[Numbered instructions with screenshots/descriptions]

### AccuLips Configuration
[Exact settings for phoneme generation]

### Render Settings
[Complete export configuration]

---

## 6. QUALITY ASSURANCE

### Automated Checks
[Face audit, sweater color, duration validation]

### Manual Review Points
[What requires human approval]

### Rejection Criteria
[What causes automatic rejection]

---

## 7. FILE NAMING & ORGANIZATION

### Naming Convention
```
kelly_{template}_{phase}_{day}_{archetype}.mp4
kelly_T01_hook_001_explorer.mp4
```

### Folder Structure
```
renders/
├── templates/
├── expressions/
├── lessons/
│   ├── day_001/
│   ├── day_002/
│   └── ...
└── archive/
```

---

## 8. TIMELINE & MILESTONES

### Week 1: Foundation
[Specific deliverables with dates]

### Week 2-4: Scale
[Production targets]

### Month 2-3: Full Library
[Completion milestones]

---

## 9. COMMUNICATION & HANDOFF

### Daily Check-ins
[When and how to sync]

### Asset Delivery
[Where to upload completed renders]

### Issue Escalation
[How to flag problems]

---

## 10. APPENDICES

### A. Reference File Checksums
[SHA256 for canonical assets]

### B. Color Palette (Exact Values)
[RGB/Hex for all Kelly colors]

### C. Blendshape Mapping
[52 ARKit shapes → CC5 shapes]

### D. Audio Specifications
[ElevenLabs voice settings]

---

## SIGN-OFF

□ Artist has reviewed and understands requirements
□ All reference files received and verified
□ Test render approved
□ Production begins: [DATE]

---

*This document is the single source of truth for Kelly video production.*
*Any changes require CVO approval.*
```

---

## EXECUTION INSTRUCTIONS

1. **First:** Run database queries to get current status
2. **Second:** Read all referenced documentation files
3. **Third:** Generate the complete handoff report
4. **Fourth:** Identify any gaps or questions for the artist
5. **Fifth:** Create a prioritized action list

### Success Criteria

The handoff is complete when:
- [ ] Artist has everything needed to start rendering
- [ ] No ambiguity in specifications
- [ ] Quality standards are measurable
- [ ] Timeline is realistic and agreed upon
- [ ] File organization is documented
- [ ] Communication channels established

---

## CONTEXT FOR AI AGENT

When executing this prompt:

1. **Use MCP Supabase tools** to query production status
2. **Read files** from the workspace to extract specifications
3. **Be thorough** — the artist should have zero questions after reading the report
4. **Be specific** — use exact values (RGB, dimensions, durations)
5. **Be practical** — this is for real production, not theoretical

The goal is a **frictionless migration** from the current state to film-quality video production using the iClone pipeline the team already owns.

---

*End of Zero-Shot Prompt*



