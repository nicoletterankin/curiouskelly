# Supabase Database Schema

This document outlines the Supabase database schema used by the Curious Kelly platform.

## Production Tables (AS OF NOV 2024)

### `core_lessons` - 365 Daily Lessons
**Rows: 365** - The master curriculum.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `day_number` | INTEGER | Day 1-365 (unique) |
| `topic` | TEXT | Lesson title (e.g., "The Sun") |
| `universal_truth` | TEXT | Core concept |
| `learning_objectives` | JSONB | Goals array |
| `difficulty_level` | TEXT | beginner/intermediate/advanced |

### `lesson_atoms` - 21,918 Content Pieces
Content variants for different archetypes and phases.

| Column | Type | Description |
|--------|------|-------------|
| `core_lesson_id` | UUID | FK to core_lessons |
| `archetype` | TEXT | Persona (e.g., "The Scientist") |
| `phase` | TEXT | welcome/teaching/practice/wisdom |
| `content` | JSONB | The actual script/content |

### `lesson_shards` - 38,314 Demographic Variants  
Fine-grained content by age/region/tone.

### `lessons` - 5 rows (MINIMAL, use core_lessons instead)

## Frontend Query Example
```javascript
// Load all 365 lessons
const { data } = await supabase
  .from('core_lessons')
  .select('*')
  .order('day_number');

// Load lesson with atoms
const { data } = await supabase
  .from('core_lessons')
  .select('*, lesson_atoms(content)')
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






