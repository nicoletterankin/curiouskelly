# Supabase Database Schema

This document outlines the Supabase database schema used by the Curious Kelly platform.

## ⚠️ IMPORTANT: THE CORRECT TABLE IS `lessons`

**DO NOT USE:** `core_lessons`, `lesson_atoms`, `lesson_shards` - these are DEPRECATED/NEVER EXISTED.

## Production Tables

### `lessons` ← THE ONLY LESSON TABLE
The production table for the 365-day curriculum.

```sql
CREATE TABLE public.lessons (
  id UUID PRIMARY KEY,
  day_number INTEGER UNIQUE NOT NULL,  -- Day 1-365
  title TEXT NOT NULL,                  -- "The Sun - Our Star"
  subtitle TEXT,                        -- Brief description
  content JSONB NOT NULL,               -- PhaseDNA structure
  audio_url TEXT,
  duration_seconds INTEGER,
  difficulty TEXT,                      -- 'beginner', 'intermediate', 'advanced'
  tags TEXT[],
  is_published BOOLEAN DEFAULT false,   -- Only published lessons are shown
  created_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ
);
```

**Frontend Query Example:**
```javascript
const { data } = await supabase
  .from('lessons')
  .select('day_number, title, subtitle, content')
  .eq('is_published', true)
  .order('day_number');
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






