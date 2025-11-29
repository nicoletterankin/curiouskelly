# Lesson Player Data Structure & Interface Mapping

## PHASE 1: DATA STRUCTURE AUDIT

### Supabase Tables

#### `core_lessons`
The master table containing 365 daily lessons.

**Key Columns:**
- `id` (UUID): Primary key
- `day_number` (Int, Unique): Day of year (1-365)
- `topic` (String): Main topic (e.g., "Rights", "The Sun")
- `universal_truth` (String): Core concept taught
- `difficulty_level` (String): Beginner/Intermediate/Advanced
- `ideal_age_range` (String): Target age group
- `learning_objectives` (JSON): Array of learning goals
- `recommended_videos` (JSON): Related video links
- `recommended_books` (JSON): Book recommendations
- Plus many other metadata fields

**Query Pattern:**
```javascript
const { data } = await supabase
  .from('core_lessons')
  .select('*')
  .eq('day_number', dayNumber)
  .single();
```

#### `lesson_atoms`
Contains the actual lesson content pieces organized by archetype and phase.

**Key Columns:**
- `id` (UUID): Primary key
- `core_lesson_id` (UUID, FK): Links to `core_lessons.id`
- `archetype` (String): One of 12 archetypes (e.g., "The Scientist", "The Explorer")
- `phase` (String): Phase name (e.g., "welcome", "teaching", "practice", "reflection", "wisdom")
- `content` (JSON): The actual content payload
- `created_at` (Timestamp): Creation date

**Unique Constraint:** `(core_lesson_id, archetype, phase)` - ensures one atom per lesson/archetype/phase combination

**Query Pattern:**
```javascript
const { data } = await supabase
  .from('lesson_atoms')
  .select('*')
  .eq('core_lesson_id', lessonId)
  .eq('archetype', 'The Scientist');
```

**Content Structure:**
The `content` JSON field can contain:
- `text`: Direct text string
- `script`: Script content
- `message`: Message content
- Or nested objects with various structures

#### `looms_shards` (Optional)
High-granularity content variations. May not exist in all deployments.

### Data Relationships

```
core_lessons (1) ──< (many) lesson_atoms
  - One lesson has many atoms (one per archetype/phase combination)
  - Typical: 5 phases × 12 archetypes = 60 atoms per lesson
```

## PHASE 2: INTERFACE MAPPING

### Before (Broken)
The original `player.html` was:
- ❌ Generating fake phases with hardcoded text
- ❌ Not loading from `lesson_atoms` table
- ❌ Fixed 5 phases regardless of actual data
- ❌ No archetype support

### After (Fixed)
The updated `player.html` now:
- ✅ Loads `core_lessons` by `day_number`
- ✅ Loads `lesson_atoms` filtered by `core_lesson_id` and `archetype`
- ✅ Dynamically creates phase dots based on actual atom count
- ✅ Extracts text from `content` JSON intelligently
- ✅ Maps phases to progress indicators correctly
- ✅ Supports archetype selection (default: "The Scientist")
- ✅ Comprehensive console logging for debugging

### UI Elements Mapping

| UI Element | Data Source | Notes |
|------------|-------------|-------|
| Day Badge | `core_lessons.day_number` | "Day X of 365" |
| Category | `core_lessons.difficulty_level` or `ideal_age_range` | Fallback to "General Knowledge" |
| Title | `core_lessons.topic` | Main topic |
| Description | `core_lessons.universal_truth` | Core concept |
| Phase Dots | `lesson_atoms.length` | Dynamically created based on atom count |
| Transcript Text | `lesson_atoms[phaseIndex].content` | Extracted via `extractTextFromContent()` |

### Phase Progression Logic

1. **Load Lesson:**
   - Query `core_lessons` for day number
   - Query `lesson_atoms` for that lesson + archetype
   - Sort atoms by phase order: `['welcome', 'teaching', 'practice', 'reflection', 'wisdom']`
   - Create phase dots matching atom count

2. **Play Phase:**
   - Get atom at `currentPhaseIndex`
   - Extract text from `atom.content` JSON
   - Display in transcript
   - Generate voice via ElevenLabs API
   - Update phase indicator

3. **Continue:**
   - Increment `currentPhaseIndex`
   - Play next phase
   - Mark previous phases as completed

4. **Completion:**
   - When `currentPhaseIndex >= lessonAtoms.length`
   - Show completion message
   - Disable continue button

## PHASE 3: DEBUGGING & LOGGING

### Console Logging Added

All major operations now log to console with prefixes:

- `🔍 [AUDIT]` - Data loading and structure inspection
- `🎬 [PLAYBACK]` - Phase playback operations
- `🖱️  [INTERACTION]` - User interactions (clicks)
- `🔄 [INTERACTION]` - Replay actions
- `✅` - Success operations
- `❌` - Errors
- `⚠️` - Warnings

### Example Log Output

```
🔍 [AUDIT] Loading Day 330 lesson...
✅ [AUDIT] core_lessons loaded: { id: "...", day_number: 330, topic: "Rights" }
📋 [AUDIT] Full core_lessons data: { ... }
🔍 [AUDIT] Loading lesson_atoms for archetype: The Scientist...
✅ [AUDIT] Found 5 lesson_atoms
📋 [AUDIT] lesson_atoms structure:
--- Atom 1 ---
Phase: welcome
Archetype: The Scientist
Content keys: ["text", "script"]
💬 [PLAYBACK] Extracted text (245 chars): Welcome! Today is Day 330...
```

### Testing

Add `?day=330` to URL to test specific day:
```
https://curiouskelly.com/player.html?day=330
```

## Data Model Summary

### Complete Data Flow

```
User visits player.html
  ↓
Calculate day number (or use ?day= param)
  ↓
Query core_lessons WHERE day_number = X
  ↓
Query lesson_atoms WHERE core_lesson_id = lesson.id AND archetype = 'The Scientist'
  ↓
Sort atoms by phase order
  ↓
Create UI phase dots (one per atom)
  ↓
User clicks "Start Lesson"
  ↓
Play phase 0: Extract text from atom.content → Generate voice → Play
  ↓
User clicks "Continue"
  ↓
Play phase 1: Extract text from atom.content → Generate voice → Play
  ↓
... (repeat for all phases)
  ↓
Lesson complete
```

### Key Functions

1. **`loadLesson(dayNumber)`** - Loads core_lessons and lesson_atoms, logs complete structure
2. **`extractTextFromContent(content)`** - Intelligently extracts text from JSON content
3. **`updatePhaseDots(count)`** - Dynamically creates phase progress indicators
4. **`playPhase(phaseIndex)`** - Plays a specific phase with voice synthesis
5. **`playText(text, phaseIndex)`** - Handles audio playback logic

## Known Issues & Fallbacks

1. **No lesson_atoms found:**
   - Falls back to basic lesson info
   - Shows generic welcome message
   - Still allows lesson to "load" but with limited content

2. **Archetype not found:**
   - Automatically uses first available archetype
   - Logs warning with available archetypes

3. **Content extraction fails:**
   - Falls back to JSON.stringify of content
   - Logs warning

4. **ElevenLabs API fails:**
   - Falls back to browser SpeechSynthesis API
   - Logs warning

## Next Steps

1. Add archetype selector UI
2. Add user progress tracking
3. Add lesson completion persistence
4. Add phase-specific expressions/animations
5. Add interactive elements (questions, choices)



