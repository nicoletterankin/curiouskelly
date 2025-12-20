# ARCHITECTURE_TRUTH

## Part 1 – Content Source of Truth

### 1. Sequence when `learn.html?day=1` loads

```mermaid
flowchart TD
  browser[Browser loads learn.html?day=1]
  initFlow[init() in public/learn.html]
  stateInit[loadState() + DEV param overrides]
  loaderCall[loadLessonRuntime(1)]
  kellyLoader[KellyLessonLoader.loadLesson(1)]
  applyLesson[applyLoadedLesson()]
  render[updateLessonUI + playPhaseMedia]

  browser --> initFlow --> stateInit --> loaderCall
  loaderCall --> kellyLoader --> applyLesson --> render
  loaderCall --> apiFallback[loadLessonFromApi()] --> applyLesson
  loaderCall --> fallbackLesson[loadFallbackLesson()] --> render
```

* `init()` is called once the DOM loads and applies URL overrides. When `?day=1` is present, `DEV_START_DAY` is parsed (`public/learn.html` lines 8901‑8907) and `state.currentDay` is set during init (lines 18407‑18505).
* The same `init()` call wires up Supabase by invoking `window.KellyLessonLoader.init(_db)` (lines 18422‑18425).
* `loadLessonRuntime(dayNumber)` (lines 11147‑11180) performs the cascade: it first calls `window.KellyLessonLoader.loadLesson` with the current day, then falls back to `/api/lessons/<day>` via `loadLessonFromApi` (lines 11046‑11060), and finally to the hard‑coded `loadFallbackLesson` (lines 11204‑11228) if everything else fails.
* Successful payloads are normalized by `applyLoadedLesson` (lines 11063‑11145), which populates `currentLesson`, `lessonAtoms`, and dispatches phase thumbnails before the UI renders.

### 2. Where lesson metadata and phase content originate

`KellyLessonLoader` (see `public/js/kelly-lesson-loader.js`) is the only component that decides which data source to use:

1. **URL / Local storage overrides** – `loadLesson` first checks `getOnDemandTopic` for user‑supplied prompts (lines 200‑213 and 771‑809). If present, it generates lesson metadata on the fly.
2. **Local packs** – It then checks `window.CURIOUS_KELLY.LOCAL_PACKS` (lines 215‑243). These packs are built by the 365 files under `public/data/day-XXX-complete.js`, each registering deterministic lesson/atom JSON for offline use.
3. **Seed JSON** – `getLesson` prefers `trySeedLessons` (lines 602‑618) which fetches `/lessons/day-<N>.json`. There are 366 JSON files under `public/lessons/` (counted via `Get-ChildItem`). These seed files include `meta`, `phases`, and bilingual text, and `seedToLesson/seedToAtoms` (lines 623‑708) convert them into Supabase-shaped payloads.
4. **Supabase** – If seeds aren’t preferred or available, `fetchFromSupabaseWithTimeout` (lines 457‑506) queries `core_lessons`, `lesson_atoms`, and `lesson_shards` via the anon key defined in `public/config.js`. This is the primary source of truth in production.
5. **Cloudflare D1 & Local API** – When no Supabase client is available the loader calls `tryCloudflareD1` and `tryLocalApi` (lines 334‑368).
6. **Emergency fallback** – As a last resort `getFallback(dayNum)` (line 449) provides generic scripts.

Once a payload arrives, `applyLoadedLesson` (lines 11063‑11145) maps:

```11063:11113:public/learn.html
function applyLoadedLesson(dayNumber, payload) {
  const lesson = payload.lesson || payload;
  const atoms = payload.atoms || [];
  currentLesson = {
    dayNumber,
    topic: lesson.topic || lesson.title || 'Daily Discovery',
    universalTruth: lesson.universal_truth || '',
    ...
  };
  lessonAtoms = PHASES.map(phase => {
    const atom = atoms.find(a => a.phase === phase.dbName);
    const content = atom?.content || {};
    return {
      phase: phase.name,
      script: content.script || content.text || atom?.script,
      videoUrl: atom?.hd_video_url,
      visualUrl: atom?.visual_url,
      choice_intro: content.prompt || content.choice_intro,
      option_a: (content.options?.[0] || {}).text,
      option_b: (content.options?.[1] || {}).text,
      ...
    };
  });
}
```

* **Lesson metadata** (`topic`, `headline`, `universal_truth`, etc.) therefore comes straight from the object returned by the loader chain (Supabase, seed JSON, local pack, etc.).
* **Phase content** (Hook/Cliff/Q1/Q2/Q3/Wisdom/Outro scripts and choice text) comes from the `lesson_atoms` array attached to the payload, regardless of whether it originated in Supabase or a local seed.

### 3. Where each phase’s video URL comes from

1. **Primary pointer** – Each `lesson_atoms` row has an `hd_video_url` column. When Supabase or seed data includes a value, `lessonAtoms[i].videoUrl` is initialized with it (line 11102). Scripts such as `scripts/heygen-day1-full-production.ts` upload HeyGen renders to the `kelly-videos` storage bucket and write the resulting public URL back to `lesson_atoms.hd_video_url` (see lines 214‑239 of that script).
2. **Variant registry** – During playback `playPhaseMedia` (lines 11656‑11788) first tries the `videoUrl` that came from `lesson_atoms`. If it’s missing it calls `getVideoUrl` (lines 11530‑11545), which queries the `kelly_video_assets` table for a `template`/phase/age match. The table schema is defined in `supabase/migrations/004_kelly_video_assets.sql` and stores `public_url` values that point to `kelly-videos/...` assets.
3. **Motion library** – When neither of the above returns a URL, `getMotionClipUrl` (lines 11564‑11641) queries `/api/motion-clip`, which in turn reads the `kelly_motion_library` table (`api/motion-clip.ts` lines 28‑72). If there’s still nothing, it falls back to the Explorer/adult clip defined by `MVP_MOTION_ARCHETYPE` (line 11556).
4. **Text-only fallback** – Finally, if no video exists, `playPhaseMedia` calls `kellyAudio.speak` (lines 11778‑11781) so the lesson still executes with captions and TTS.

## Part 2 – Video Architecture & Fallbacks

### `lesson_atoms.hd_video_url`
* **Purpose:** Direct link to the HeyGen video that matches the archetype + phase loaded from Supabase.
* **Population:** `scripts/heygen-day1-full-production.ts` generates each clip, uploads it to `kelly-videos/production/...`, and updates `lesson_atoms` via Supabase (`scripts/heygen-day1-full-production.ts` lines 214‑239).
* **Usage:** `applyLoadedLesson` attaches the URL to `lessonAtoms[..].videoUrl`, so `playPhaseMedia` treats it as the authoritative per-phase video.

### `kelly_video_assets`
* **Schema:** Added by `supabase/migrations/004_kelly_video_assets.sql` (lines 16‑105). Stores variant metadata keyed by `lesson_day`, `phase`, `age_bucket`, `language`, and optional `template` (persona). Includes `public_url`, duration, and quality metrics, plus a status column to gate RLS (anon can only read rows with `status = 'completed'`).
* **Lifecycle:** Generation scripts such as `scripts/heygen-day1-full-production.ts`, `scripts/generate-day-videos-heygen.ts`, and `scripts/upload-golden-lesson-videos.ts` upload MP4s to the `kelly-videos` storage bucket and then insert rows into `kelly_video_assets` with the resulting `public_url`. This table is also used for infographic images (`asset_type = 'image'`).
* **Playback:** `getVideoUrl` in `public/learn.html` (lines 11530‑11545) filters by day, phase, `template` (persona ID), `age_bucket`, and language to find a `public_url`. If a matching row exists the player uses that video.

### `kelly_motion_library`
* **Schema:** Created in `supabase/migrations/20251214_create_kelly_motion_library.sql` (lines 14‑53). Stores one clip per `persona_ageBucket` (`avatar_key`) and phase, plus HeyGen metadata and completion status.
* **API:** `/api/motion-clip` (lines 16‑76) uses the Supabase service role to pull a clip for a given `persona`, `age`, and `phase`, returning `{ videoUrl, duration }` or `{ fallback: true }`.
* **Playback:** `getMotionClipUrl` (lines 11564‑11641) calls that API. If no clip exists it recursively falls back to the Explorer-adult clip defined by `MVP_MOTION_ARCHETYPE` and `MVP_MOTION_AGE_BUCKET` (lines 11556‑11537). Any failure leaves `motionClipCache` entry as `null`, forcing the next step in the fallback chain.

### Fallback order when no video exists
1. **Atom video** – `lessonAtoms[n].videoUrl` (from `lesson_atoms.hd_video_url`).
2. **Variant registry** – `kelly_video_assets` lookup via `getVideoUrl()`.
3. **Motion library** – `/api/motion-clip` → `kelly_motion_library` via `getMotionClipUrl()`.
4. **Static fallback** – `kellyAudio.speak()` drives captions/avatars. `public/js/kelly-fallback-engine.js` adds a final safety net by showing Kelly’s headshot + audio when even the `<video>` element can’t play (see `getBestMedia()` and `FallbackPlayer` around lines 24‑205 in that file).

### Where rendered files live
* HeyGen renders are uploaded to the Supabase storage bucket `kelly-videos` (see `scripts/heygen-day1-full-production.ts` lines 140‑147) and also mirrored locally under `generated-videos/heygen-production` for inspection.
* The player accesses them via CDN URLs stored either in `lesson_atoms.hd_video_url` or `kelly_video_assets.public_url` and streams them directly into the `#kelly-video` element.

## Part 3 – Persona / Archetype System

### Manifest loading
* `CONFIG.MANIFEST_URL` is set to `/assets/kelly/kelly-personas-manifest.json` in `public/config.js` (lines 12‑66). `loadManifest()` (`public/learn.html` lines 10976‑11004) fetches this file to build the `KELLYS` array used by the carousel and “Teaching style” selector.
* The manifest (`public/assets/kelly/kelly-personas-manifest.json`) enumerates all 12 personas, their icons, colors, accessories, and per-age image paths.
* If the fetch 404s, the code logs `Manifest load failed` and uses the hard-coded `FALLBACK_KELLYS` array (lines 9653‑9665).

### Connection to the 12 archetypes
* `PERSONA_TO_ARCHETYPE` (`public/learn.html` lines 10877‑10885) maps each persona ID (`scientist`, `explorer`, etc.) to the Supabase archetype names (`"The Scientist"`, `"The Explorer"`, ...). This is how a UI selection is supposed to influence the loader.
* `getPhaseArchetype` (`public/learn.html` lines 9702‑9715) also wires each phase to a specific persona for contextual visuals (e.g., Hook always uses the Explorer even if the user never picked one).

### Why the dropdown is empty / ineffective today
* The settings panel simply shows the current `state.kellyId` (lines 15724‑15733) and a `Change Kelly` button that routes to a non-existent `character` scene (`showScene('character')`). There is no code outside of the screenshot helper (`prepareForScreenshot` lines 18543‑18591) that ever sets `state.kellyId`, so it remains `null` → `KELLYS[0]` (Scientist) for everyone.
* Even if the dropdown were populated, the content loader hard-codes `const archetype = 'The Scientist'` when calling `KellyLessonLoader.loadLesson` (line 11163), so lesson data would still come from the Scientist rows in `lesson_atoms`.
* The persona manifest therefore only affects cosmetics (color, avatar) today, not the Supabase queries. To make it functional the app would need to set `state.kellyId` from user input and pass `PERSONA_TO_ARCHETYPE[state.kellyId]` into `KellyLessonLoader.loadLesson` and `getAudioUrl`.

## Part 4 – Actual coverage vs. MVP target

| Asset / Data set | Actual count | Source / method |
| --- | --- | --- |
| Local JS lesson packs (`public/data/day-*-complete.js`) | 365 files | `Get-ChildItem public/data 'day-*-complete.js'` |
| Seed JSON (`public/lessons/day-*.json`) | 366 files | `Get-ChildItem public/lessons 'day-*.json'` |
| Supabase `lesson_atoms` rows | **20,433** | Queried with `@supabase/supabase-js` anon key in `tmp/supabase-counts.js` |
| Supabase `kelly_video_assets` rows where `asset_type = 'video'` | **312** | Same script as above |
| Supabase `kelly_motion_library` rows | **336** | Same script |
| Explorer/adult completed videos | **5** rows, all with `public_url` | `tmp/supabase-counts.js` filtered on `template='explorer', age_bucket='adult'` |

**Implication for the MVP scenario (Explorer × adult × 365 days × 7 phases × 2 options = 5,110 slots):** only 5 Explorer/adult records exist in `kelly_video_assets`, all pointing to Day 1 phases. The remaining 5,105 slots never reach the `getVideoUrl()` stage, so the player immediately falls through to `getMotionClipUrl()` and finally to the static Kelly + TTS path. There are zero Explorer/adult rows with missing URLs because no other rows exist.

Because `kelly_motion_library` currently has 336 rows (roughly the number of persona/age combinations times five phases), most lessons rely on these generic motion clips rather than day-specific videos. When clips are missing, the logs emit `"No motion clip for ..."` (`public/learn.html` line 11602) and `kellyAudio.speak()` keeps the lesson running silently.

### Takeaways
* Lesson metadata is sourced hierarchically: local packs → `/lessons/*.json` → Supabase (`core_lessons` + `lesson_atoms` + `lesson_shards`) → Cloudflare D1 → `/api/lessons` → emergency fallback.
* Phase scripts and options always come from the `lesson_atoms` payload no matter which layer supplied the data.
* Video playback depends on three registries in order (`lesson_atoms.hd_video_url`, `kelly_video_assets`, `kelly_motion_library`) before falling back to TTS and static imagery.
* The “Teaching style” selector is tied to `/assets/kelly/kelly-personas-manifest.json`, but `loadLessonRuntime` still requests `"The Scientist"`, so persona changes do not influence Supabase queries today.
* Against the 5,110-slot MVP target, only 5 Explorer/adult videos exist, 312 total video rows are registered, and 336 generic motion clips backfill the rest.
