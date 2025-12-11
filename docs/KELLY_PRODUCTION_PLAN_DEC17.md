# 🚀 KELLY PRODUCTION PLAN: DEC 17 LAUNCH

**Objective**: Generate 12-variant daily lessons for December 17 launch.
**Status**: 12 Archetypes uploading now.
**Constraint**: 2-hour window for setup; production run follows.

---

## 1. CAPACITY & CREDITS MATH

To generate **ONE Daily Lesson** (which consists of 12 variants):

*   **Average Lesson Audio**: ~3 minutes (split into Intro, Concept, Outro)
*   **Variants**: 12 Archetypes
*   **Total Generation**: 3 mins × 12 = **36 minutes** of video.
*   **Cost**: **36 Credits per Daily Lesson**.

| Scale | Total Video | Credits Required | Time (Sequential) | Time (Parallel 3x) |
| :--- | :--- | :--- | :--- | :--- |
| **1 Day** | 36 mins | 36 | ~45 mins | ~15 mins |
| **7 Days** | 4.2 hours | 252 | ~5 hours | ~1.5 hours |
| **30 Days** | 18 hours | 1,080 | ~24 hours | ~8 hours |

**Recommendation**: Ensure your HeyGen plan has at least **2,000 credits** available for the initial Dec 17 push (launch buffer).

---

## 2. QUALITY OPTIMIZATION CHECKLIST

To ensure "Perfection" quality:

1.  **✅ Super Resolution**: Must enable `super_resolution: true` in API payload.
2.  **✅ Lossless Audio**: Use ElevenLabs `pcm_44100` format (highest quality) to avoid compression artifacts in lipsync.
3.  **✅ Motion Prompts**: Use the **Accessory-Aware Motion Prompts** (created in `docs/HEYGEN_12_HEAD_ACCESSORY_MOTION_PROMPTS.md`) to prevent floating goggles/bandanas.
4.  **✅ Facial Calibration**: Use "Fine-tune" mode on HeyGen if available for Talking Photos (upscaling step you are doing now).

---

## 3. TECHNICAL ARCHITECTURE

We will upgrade `scripts/heygen-kelly-pipeline.ts` to a robust **Production Engine**:

### A. The "Group Avatar" Logic
Since you are grouping them, we need to map Archetypes to their specific IDs.
*   **Input**: `day_number` (e.g., 1)
*   **Process**:
    1.  Fetch all 12 scripts for Day 1 (from `lesson_atoms` table).
    2.  Generate 12 distinct audios (using specific Voice Settings if varied, or standard Kelly voice).
    3.  **Parallel Dispatch**: Send 3-5 generation requests to HeyGen simultaneously (respecting API rate limits).
    4.  **Polling**: Robust loop checking status every 10s.
    5.  **Download & Upload**: Save to Supabase `production/day_{N}/{archetype}.mp4`.

### B. Failure Handling
*   **Quota Limits**: If we hit a limit, pause for 60s and retry (Exponential Backoff).
*   **Timeouts**: If a video hangs >10 mins, cancel and retry.

### C. Dynamic Compilation (NEW)
**Post-processing Step**: Convert static talking heads into dynamic educational content.
*   **Script**: `scripts/compile_lesson_day1.py`
*   **Logic**:
    *   **Hook**: Keep Full Screen Kelly.
    *   **Facts/Wisdom**: PiP (Picture-in-Picture) with Phase Infographic as background + Kelly in bottom-right (35% scale).
*   **Upload**: `scripts/upload_day1_dynamic.ts` uploads to Supabase (Primary) + Cloudflare R2 (Backup) and updates `lesson_atoms`.

---

## 4. IMMEDIATE ACTION ITEMS

1.  **Retrieve IDs**: Once your upload finishes (in ~2 hours), we need the 12 `talking_photo_id`s (or `avatar_id` + `style_id`).
2.  **Update Config**: Place these IDs into `scripts/heygen-kelly-pipeline.ts`.
3.  **Run Day 1 Test**: Generate the full 12-variant suite for Day 1.
4.  **Compile Dynamic**: Run `scripts/compile_lesson_day1.py` to add infographics.
5.  **Upload & Wire**: Run `scripts/upload_day1_dynamic.ts` to deploy.
6.  **Review**: Check `learn.html` for final quality.

**Ready to build the upgraded pipeline script.**
