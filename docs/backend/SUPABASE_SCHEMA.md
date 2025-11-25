# Supabase Database Schema

This document outlines the Supabase database schema used by the Curious Kelly platform. The schema is designed to support the "Unified Aquarium" architecture, handling daily lessons, user progress, and authentication.

## Core Tables

### `core_lessons`
The master table for the 365-day curriculum.
- **`day_number`** (Int, Unique): The day of the year (1-365).
- **`topic`** (String): The main topic of the lesson (e.g., "The Sun", "Gravity").
- **`universal_truth`** (String): The core concept taught.
- **`learning_objectives`** (Json): Array of learning goals.
- **Metadata**: `hero_image_url`, `difficulty_level`, `ideal_age_range`, etc.

### `lesson_atoms`
Contains the actual content pieces (atoms) used to construct a lesson for different archetypes and phases.
- **`core_lesson_id`** (FK): Links to `core_lessons`.
- **`archetype`** (String): The persona/archetype the content is tailored for.
- **`phase`** (String): The phase of the lesson (e.g., "welcome", "teaching", "practice", "wisdom").
- **`content`** (Json): The actual content payload (script, choices, metadata).

### `lesson_shards`
High-granularity content variations based on specific user demographics.
- **`core_lesson_id`** (FK): Links to `core_lessons`.
- **`age`**, **`region`**, **`tone`**, **`birth_year`**: targeting parameters.
- **`script_content`** (Json): The specific script for this shard.

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






