# Supabase Database Schema - VERIFIED NOV 27, 2024

## Production Tables

### `core_lessons` - 365 Daily Lessons

| Column | Type | Example |
|--------|------|---------|
| `id` | UUID | `9f8af9c5-66d6-40a0-a10c-b95a7940d25c` |
| `day_number` | INTEGER | 1, 2, 3... 365 |
| `topic` | TEXT | "Water", "Clouds", "Light" |
| `universal_truth` | TEXT | "Water transforms between solid, liquid, and gas..." |
| `marketing_headline` | TEXT | "Water Wonders: Discover the Magic of H2O!" |
| `marketing_tagline` | TEXT | "Water: It's not just wet!" |
| `marketing_pitch` | TEXT | "Dive into the fascinating world of water!" |
| `quick_quiz_questions` | JSONB | Quiz data |
| `reflection_prompts` | JSONB | Prompts array |
| `mastery_criteria` | TEXT | Success criteria |

### `lesson_atoms` - 21,915 Content Pieces

| Column | Type | Example |
|--------|------|---------|
| `id` | UUID | Primary key |
| `core_lesson_id` | UUID | FK to core_lessons.id |
| `archetype` | TEXT | "The Survivor" |
| `phase` | TEXT | "Fact1", "Fact2", "Fact3" |
| `content` | JSONB | `{script, options, responses}` |
| `created_at` | TIMESTAMP | |
| `visual_url` | TEXT | NULL |

### `lesson_shards` - 38,700 Demographic Variants

| Column | Type | Example |
|--------|------|---------|
| `id` | UUID | Primary key |
| `core_lesson_id` | UUID | FK to core_lessons.id |
| `age` | INTEGER | 5 |
| `region` | TEXT | "en" |
| `tone` | TEXT | "playful", "curious", "serious" |
| `birth_year` | INTEGER | 2020 |
| `script_content` | JSONB | Personalized script |

### `lessons` - 5 rows (TEST DATA ONLY)
Different schema - has `title` instead of `topic`. DO NOT USE.

## Frontend Query Examples

```javascript
// Load all lessons (for sidebar)
const { data } = await supabase
  .from('core_lessons')
  .select('id, day_number, topic, universal_truth')
  .order('day_number');

// Load lesson with content atoms
const { data } = await supabase
  .from('core_lessons')
  .select('id, day_number, topic, lesson_atoms(content, archetype, phase)')
  .eq('day_number', dayNumber)
  .single();
```

## User Tables

### `users` (`public.users`)
Stores public user profile information. This is separate from `auth.users` but linked by ID.
- **`id`** (UUID): Matches `auth.users.id`.
- **`email`** (String): User email.
- **`subscription_tier`** (String): 'free', 'scholar', etc.
- **`current_day`** (Int): The user's current progress day (1-365).
- **`streak_days`** (Int): Current streak count.

### `user_progress`
Tracks granular progress through specific lessons.
- **`user_id`**, **`lesson_id`**: Link user to lesson.
- **`completed`** (Boolean): Completion status.
- **`progress_percent`** (Int): 0-100.

## Authentication (`auth` schema)
Managed by Supabase Auth.
- **`auth.users`**: System user table.
- **`auth.identities`**: Linked social logins (Google, Apple, etc.).

## Marketing & Growth

### `affiliates`
- **`referral_code`**, **`commission_rate`**, **`earnings`**.

### `enterprise_inquiries`
- Captures leads from the enterprise landing page.

## Integration Notes

- **Frontend Loading**: The Lesson Player (`app.js`) queries `core_lessons` by `day_number` and joins `lesson_atoms` to fetch the content payload.
- **Fallback**: If no database record is found, the frontend falls back to static JSON files (`lessons/`).

























