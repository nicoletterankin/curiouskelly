# ✅ Day 1 Variant Readiness Audit  
**Date:** December 11, 2025  
**Scope:** Lesson Day 1 (learn.html + marketing learn.html)  

This audit captures the current database and asset readiness before we scale to “all languages, all tones, all phases, all responses” for Day 1. Counts are pulled directly from Supabase after the December 11 translation run.

---

## 1. Data Inventory

| Asset / Table | Current Count | Notes |
| --- | --- | --- |
| `lesson_atoms` | **60** (12 archetypes × 5 phases) | English-only content; each atom includes `content.script` plus 3 options with embedded `response` strings. |
| `lesson_atoms.hd_video_url` | **60 populated** | 57 unique dynamic PiP renders + 3 patched fallbacks (Provider Fact2, Strategist Fact2/Wisdom). English-only. |
| `lesson_atoms.visual_url` | ✅ Set via infographic uploader | Points to Day 1 infographics used inside the PiP composite. |
| `lesson_shards` (Day 1) | **54** rows | 6 age buckets × 3 languages (EN/ES/FR) × 3 tones (curious/playful/serious). Each shard holds `script_content` (title, script, vocab, teaching moments). No per-choice responses yet. |
| `kelly_video_assets` (Day 1) | **0** rows | Table exists (see `supabase/migrations/004_kelly_video_assets.sql`), but no assets registered for Day 1. This means the frontend cannot fetch multi-language/tone variants yet. |
| Response scripts (inside `lesson_atoms.content.options[].response`) | ✅ Present for EN across all 27 combinations | Needs tone rewrites + ES/FR translation before video generation. |
| Response videos | **0** generated | Script `scripts/kelly-video-factory/generate-response-videos.ts` has not been run since raw responses were added. |

---

## 2. Gap Summary vs. “All Languages / All Tones / All Responses” Goal

| Requirement | Current State | Gap |
| --- | --- | --- |
| **Main phase videos for every tone** (UI exposes Curious, Playful, Serious → Scientist, Explorer, Rebel) | Only English videos exist. | Need 45 renders (5 phases × 3 tones × 3 languages) + Supabase wiring. |
| **Response videos for every option** | None rendered. | Need 81 renders (3 phases × 3 options × 3 tones × 3 languages). |
| **Localized scripts** | Phase scripts now covered for EN/ES/FR across Curious/Playful/Serious. | Need to carry the same tone + language variants into choice responses during response-video generation. |
| **Variant storage / lookup** | `kelly_video_assets` empty. | Must insert per-variant metadata (lesson_day, phase mapped to welcome/q1/q2/q3/wisdom, age_bucket, language, tone/archetype, URLs). |
| **Frontend wiring** | `public/learn.html` currently reads only `lesson_atoms.hd_video_url`. | Needs logic to fetch `kelly_video_assets` by language/tone before falling back to English. |

---

## 3. Tables & Schemas to Leverage

1. **`kelly_video_assets`** — canonical home for per-variant videos. Ensure inserts include:
   - `lesson_day`, `phase` *(use migration’s enum: Hook → welcome, Fact1 → q1, Fact2 → q2, Fact3 → q3, Wisdom → wisdom)*
   - `age_bucket` *(pick canonical bucket for each render; still need 6 × languages later, but we can start with `young_adult` = 18‑35 for video assets)*
   - `language`, `archetype`, `script_text`, `video_public_url`, `status='completed'`.

2. **`lesson_shards`** — already wired to personalization UI. We will:
   - Duplicate Day 1 English shards for tones `playful` + `serious` (tone rewrites).
   - Translate each tone to ES/FR, yielding 54 shards (6 ages × 3 languages × 3 tones).

3. **`lesson_atoms.content.options`** — authoritative source for choice text + response copy. We will export, translate, and feed into the response-generation pipeline.

---

## 4. Ready for Step 2

The audit confirms:
- Schema support exists (`kelly_video_assets`, `lesson_shards`, response scripts inside atoms).
- No conflicting legacy data (Day 1 rows = 0 in `kelly_video_assets`).
- Clear quantitative targets for the upcoming steps.

Next action per execution plan: **“Expand shard generation for tones and translations.”**


