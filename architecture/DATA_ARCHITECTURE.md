# 📊 Kelly Experience — Data Architecture

## Complete Schema Definitions for All Components

**Version:** 1.0  
**Date:** November 28, 2025

---

## Table of Contents

1. [Variant Matrix](#1-variant-matrix)
2. [Supabase Schema](#2-supabase-schema)
3. [Component Data Schemas](#3-component-data-schemas)
4. [API Contracts](#4-api-contracts)
5. [State Management](#5-state-management)

---

## 1. Variant Matrix

### The 5 Variant Axes

Kelly lessons adapt along **5 independent axes**. Any combination is valid.

```
┌─────────────────────────────────────────────────────────────────┐
│                     VARIANT MATRIX                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   AXIS 1: AGE              AXIS 2: LANGUAGE                    │
│   ├── 2-5   (tiny)         ├── en (English)                    │
│   ├── 6-12  (young)        ├── es (Español)                    │
│   ├── 13-17 (teen)         └── fr (Français)                   │
│   ├── 18-35 (adult)                                            │
│   ├── 36-60 (grown)        AXIS 3: TONE                        │
│   └── 61+   (wise)         ├── curious                         │
│                            ├── playful                          │
│   AXIS 4: DIFFICULTY       └── serious                         │
│   ├── 2 choices (standard)                                     │
│   └── 3 choices (challenge) AXIS 5: MODE                       │
│                            ├── 2D (image)                       │
│                            └── 3D (Unity)                       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TOTAL UNIQUE VARIANTS PER LESSON:                            │
│   6 ages × 3 languages × 3 tones × 2 difficulties = 108        │
│                                                                 │
│   TOTAL FOR 365 LESSONS:                                        │
│   365 × 108 = 39,420 unique lesson variants                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### How Variants Interact

| Axis       | Affects Content?        | Affects Audio?       | Affects Choices?   | Runtime Switchable? |
| ---------- | ----------------------- | -------------------- | ------------------ | ------------------- |
| Age        | Yes - vocabulary, depth | Yes - different TTS  | Yes - complexity   | Yes                 |
| Language   | Yes - translation       | Yes - language TTS   | Yes - translated   | Yes                 |
| Tone       | Yes - personality       | Yes - delivery style | Slightly           | Yes                 |
| Difficulty | No                      | No                   | Yes - # of options | Yes                 |
| Mode       | No                      | No                   | No                 | Yes                 |

---

## 2. Supabase Schema

### Tables Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUPABASE TABLES                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   core_lessons          (365 rows - one per day)               │
│        │                                                        │
│        ├──▶ lesson_phases       (5 per lesson = 1825 rows)     │
│        │         │                                              │
│        │         └──▶ phase_choices (2-3 per Q phase)          │
│        │                                                        │
│        └──▶ lesson_variants     (variants by age/lang/tone)    │
│                  │                                              │
│                  └──▶ variant_audio    (audio files per variant)│
│                                                                 │
│   users                 (user accounts)                         │
│        │                                                        │
│        ├──▶ user_progress       (lesson completion tracking)   │
│        │                                                        │
│        └──▶ user_preferences    (variant preferences)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Table: `core_lessons`

```sql
CREATE TABLE core_lessons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    day_number INTEGER UNIQUE NOT NULL CHECK (day_number >= 1 AND day_number <= 365),
    calendar_date VARCHAR(20) NOT NULL,  -- "January 1", "November 28", etc.

    -- Core content (universal, not variant-specific)
    topic VARCHAR(255) NOT NULL,
    topic_emoji VARCHAR(10),
    category VARCHAR(50),  -- science, history, arts, etc.
    tags TEXT[],

    -- Learning objectives (base version)
    universal_truth TEXT,
    core_principle TEXT,
    learning_essence TEXT,

    -- Metadata
    duration_min INTEGER DEFAULT 5,
    duration_max INTEGER DEFAULT 15,
    difficulty_base VARCHAR(20) DEFAULT 'beginner',  -- beginner, intermediate, advanced

    -- Feature flags
    has_audio BOOLEAN DEFAULT false,
    has_3d_animations BOOLEAN DEFAULT false,
    is_premium BOOLEAN DEFAULT false,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for date lookups
CREATE INDEX idx_core_lessons_day ON core_lessons(day_number);
```

### Table: `lesson_phases`

```sql
CREATE TABLE lesson_phases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    core_lesson_id UUID REFERENCES core_lessons(id) ON DELETE CASCADE,

    phase_order INTEGER NOT NULL CHECK (phase_order >= 1 AND phase_order <= 5),
    phase_type VARCHAR(20) NOT NULL,  -- 'welcome', 'question', 'wisdom', 'completion'
    phase_name VARCHAR(50) NOT NULL,  -- 'Welcome', 'Q1', 'Q2', 'Q3', 'Wisdom'

    -- Base content (will be overridden by variants)
    base_text TEXT NOT NULL,
    base_question TEXT,  -- For question phases
    base_hint TEXT,      -- Optional hint
    base_reflection TEXT, -- For wisdom phase

    -- Kelly expression for this phase
    kelly_expression VARCHAR(30) DEFAULT 'explaining',  -- curious, explaining, listening, wisdom, celebrating

    -- Timing
    estimated_seconds INTEGER DEFAULT 60,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(core_lesson_id, phase_order)
);

-- Index for lesson lookups
CREATE INDEX idx_lesson_phases_lesson ON lesson_phases(core_lesson_id);
```

### Table: `phase_choices` (NEW - Supports 2 or 3 choices)

```sql
CREATE TABLE phase_choices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phase_id UUID REFERENCES lesson_phases(id) ON DELETE CASCADE,

    choice_letter VARCHAR(1) NOT NULL,  -- 'A', 'B', 'C'
    choice_order INTEGER NOT NULL,       -- 1, 2, 3

    -- Base choice text (will be overridden by variants)
    base_text TEXT NOT NULL,

    -- Is this the "best" answer? (for analytics, not shown to user)
    is_preferred BOOLEAN DEFAULT false,

    -- Choice metadata
    choice_type VARCHAR(20) DEFAULT 'standard',  -- 'standard', 'nuanced', 'challenge'

    -- When difficulty=2, only choices with choice_order <= 2 are shown
    -- When difficulty=3, all choices are shown

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(phase_id, choice_letter)
);

-- Index for phase lookups
CREATE INDEX idx_phase_choices_phase ON phase_choices(phase_id);
```

### Table: `lesson_variants`

```sql
CREATE TABLE lesson_variants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    core_lesson_id UUID REFERENCES core_lessons(id) ON DELETE CASCADE,

    -- Variant dimensions
    age_group VARCHAR(10) NOT NULL,    -- '2-5', '6-12', '13-17', '18-35', '36-60', '61+'
    language VARCHAR(5) NOT NULL,       -- 'en', 'es', 'fr'
    tone VARCHAR(20) NOT NULL,          -- 'curious', 'playful', 'serious'

    -- Variant-specific content overrides
    topic_translated VARCHAR(255),      -- Topic in target language
    learning_objective_variant TEXT,    -- Adapted for age/tone

    -- Status
    generation_status VARCHAR(20) DEFAULT 'pending',  -- pending, generating, complete, error

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(core_lesson_id, age_group, language, tone)
);

-- Composite index for variant lookups
CREATE INDEX idx_lesson_variants_lookup
ON lesson_variants(core_lesson_id, age_group, language, tone);
```

### Table: `phase_variant_content`

```sql
CREATE TABLE phase_variant_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phase_id UUID REFERENCES lesson_phases(id) ON DELETE CASCADE,
    variant_id UUID REFERENCES lesson_variants(id) ON DELETE CASCADE,

    -- Variant-specific content
    text_content TEXT NOT NULL,         -- The actual speech for this variant
    question_content TEXT,              -- Question for this variant
    hint_content TEXT,
    reflection_content TEXT,

    -- Audio reference
    audio_url TEXT,
    audio_duration_ms INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(phase_id, variant_id)
);
```

### Table: `choice_variant_content`

```sql
CREATE TABLE choice_variant_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    choice_id UUID REFERENCES phase_choices(id) ON DELETE CASCADE,
    variant_id UUID REFERENCES lesson_variants(id) ON DELETE CASCADE,

    -- Variant-specific choice text
    text_content TEXT NOT NULL,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(choice_id, variant_id)
);
```

### Table: `users`

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY REFERENCES auth.users(id),
    email VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),

    -- Birthday for birthday lesson feature
    birthday_month INTEGER CHECK (birthday_month >= 1 AND birthday_month <= 12),
    birthday_day INTEGER CHECK (birthday_day >= 1 AND birthday_day <= 31),
    birthday_lesson_day INTEGER,  -- Calculated: which of 365 days is their birthday

    -- Progress
    current_day INTEGER DEFAULT 1,
    streak_days INTEGER DEFAULT 0,
    longest_streak INTEGER DEFAULT 0,
    total_lessons_completed INTEGER DEFAULT 0,

    -- Preferences (stored as current defaults)
    preferred_age_group VARCHAR(10) DEFAULT '18-35',
    preferred_language VARCHAR(5) DEFAULT 'en',
    preferred_tone VARCHAR(20) DEFAULT 'curious',
    preferred_difficulty INTEGER DEFAULT 2,  -- 2 or 3 choices
    preferred_mode VARCHAR(5) DEFAULT '2D',  -- '2D' or '3D'

    -- Account
    subscription_status VARCHAR(20) DEFAULT 'free',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Table: `user_progress`

```sql
CREATE TABLE user_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    core_lesson_id UUID REFERENCES core_lessons(id),
    day_number INTEGER NOT NULL,

    -- Completion status
    status VARCHAR(20) DEFAULT 'not_started',  -- not_started, in_progress, completed, skipped
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Which variant was used
    age_group_used VARCHAR(10),
    language_used VARCHAR(5),
    tone_used VARCHAR(20),
    difficulty_used INTEGER,
    mode_used VARCHAR(5),

    -- Phase progress
    current_phase INTEGER DEFAULT 1,
    phases_completed INTEGER DEFAULT 0,

    -- Choices made (for analytics)
    choices_made JSONB,  -- {"Q1": "A", "Q2": "B", "Q3": "A"}

    -- Time spent
    time_spent_seconds INTEGER DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(user_id, day_number)
);

-- Index for user progress lookups
CREATE INDEX idx_user_progress_user ON user_progress(user_id);
CREATE INDEX idx_user_progress_day ON user_progress(day_number);
```

---

## 3. Component Data Schemas

### Schema: Kelly Frame State

```typescript
interface KellyFrameState {
  // Current lesson
  lesson: {
    id: string;
    dayNumber: number;
    topic: string;
    topicEmoji: string;
  };

  // Current phase
  phase: {
    order: number; // 1-5
    type: 'welcome' | 'question' | 'wisdom' | 'completion';
    name: string; // 'Welcome', 'Q1', 'Q2', 'Q3', 'Wisdom'
    content: string; // Current speech text
    question?: string; // For question phases
    hint?: string;
    choices?: Choice[]; // 2 or 3 choices
    audioUrl?: string;
  };

  // Current variants
  variants: {
    age: '2-5' | '6-12' | '13-17' | '18-35' | '36-60' | '61+';
    language: 'en' | 'es' | 'fr';
    tone: 'curious' | 'playful' | 'serious';
    difficulty: 2 | 3; // Number of choices
    mode: '2D' | '3D';
  };

  // Kelly state
  kelly: {
    expression: 'curious' | 'explaining' | 'listening' | 'wisdom' | 'celebrating';
    isSpeaking: boolean;
    isWaitingForChoice: boolean;
  };

  // Progress
  progress: {
    phasesCompleted: number[]; // [1, 2] means phases 1 and 2 done
    choicesMade: Record<string, string>; // { "Q1": "A", "Q2": "B" }
    startedAt: Date;
    completedAt?: Date;
  };
}

interface Choice {
  letter: 'A' | 'B' | 'C';
  text: string;
  isSelected: boolean;
}
```

### Schema: Hub State

```typescript
interface HubState {
  // Today's lesson
  today: {
    dayNumber: number;
    date: string;
    topic: string;
    topicEmoji: string;
    objective: string;
    isCompleted: boolean;
  };

  // User stats
  stats: {
    streak: number;
    longestStreak: number;
    lessonsCompleted: number;
    percentComplete: number; // 0-100
  };

  // Calendar
  calendar: {
    currentMonth: number; // 0-11
    currentYear: number;
    days: CalendarDay[];
  };

  // Birthday
  birthday?: {
    month: number;
    day: number;
    lessonDay: number; // Which of 365 days
    topic: string;
  };
}

interface CalendarDay {
  dayNumber: number; // 1-365
  calendarDay: number; // 1-31
  month: number; // 0-11
  topic: string;
  status: 'completed' | 'today' | 'future' | 'missed';
  isBirthday: boolean;
}
```

### Schema: Lesson DNA (Enhanced)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LessonDNA",
  "description": "Complete lesson definition with all variants",
  "type": "object",
  "required": ["lesson_id", "day_number", "topic", "phases"],
  "properties": {
    "lesson_id": {
      "type": "string",
      "description": "Unique identifier"
    },
    "day_number": {
      "type": "integer",
      "minimum": 1,
      "maximum": 365
    },
    "calendar_date": {
      "type": "string",
      "description": "Human-readable date like 'November 28'"
    },
    "topic": {
      "type": "string"
    },
    "topic_emoji": {
      "type": "string"
    },
    "category": {
      "type": "string",
      "enum": ["science", "history", "arts", "math", "language", "social", "health", "technology"]
    },
    "phases": {
      "type": "array",
      "minItems": 5,
      "maxItems": 5,
      "items": {
        "$ref": "#/definitions/Phase"
      }
    },
    "variants": {
      "type": "object",
      "description": "Content variations by age/language/tone",
      "additionalProperties": {
        "$ref": "#/definitions/Variant"
      }
    }
  },
  "definitions": {
    "Phase": {
      "type": "object",
      "required": ["phase_order", "phase_type", "phase_name", "base_text"],
      "properties": {
        "phase_order": {
          "type": "integer",
          "minimum": 1,
          "maximum": 5
        },
        "phase_type": {
          "type": "string",
          "enum": ["welcome", "question", "wisdom", "completion"]
        },
        "phase_name": {
          "type": "string",
          "enum": ["Welcome", "Q1", "Q2", "Q3", "Wisdom"]
        },
        "base_text": {
          "type": "string"
        },
        "base_question": {
          "type": "string"
        },
        "base_hint": {
          "type": "string"
        },
        "kelly_expression": {
          "type": "string",
          "enum": ["curious", "explaining", "listening", "wisdom", "celebrating"]
        },
        "choices": {
          "type": "array",
          "minItems": 2,
          "maxItems": 3,
          "items": {
            "$ref": "#/definitions/Choice"
          },
          "description": "2 choices for standard, 3 for challenge mode"
        }
      }
    },
    "Choice": {
      "type": "object",
      "required": ["letter", "text"],
      "properties": {
        "letter": {
          "type": "string",
          "enum": ["A", "B", "C"]
        },
        "text": {
          "type": "string"
        },
        "is_preferred": {
          "type": "boolean",
          "default": false
        },
        "choice_type": {
          "type": "string",
          "enum": ["standard", "nuanced", "challenge"],
          "default": "standard"
        }
      }
    },
    "Variant": {
      "type": "object",
      "properties": {
        "topic_translated": {
          "type": "string"
        },
        "phases": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/PhaseVariant"
          }
        }
      }
    },
    "PhaseVariant": {
      "type": "object",
      "properties": {
        "phase_order": {
          "type": "integer"
        },
        "text": {
          "type": "string"
        },
        "question": {
          "type": "string"
        },
        "hint": {
          "type": "string"
        },
        "audio_url": {
          "type": "string"
        },
        "choices": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "letter": { "type": "string" },
              "text": { "type": "string" }
            }
          }
        }
      }
    }
  }
}
```

---

## 4. API Contracts

### Endpoint: Get Today's Lesson

```typescript
// GET /api/lessons/today
// Headers: Authorization: Bearer <token>

interface GetTodayResponse {
  success: boolean;
  data: {
    lesson: {
      id: string;
      dayNumber: number;
      date: string;
      topic: string;
      topicEmoji: string;
      category: string;
      objective: string;
      durationMin: number;
      durationMax: number;
    };
    userProgress: {
      status: 'not_started' | 'in_progress' | 'completed';
      currentPhase: number;
      phasesCompleted: number[];
    } | null;
    streak: number;
  };
}
```

### Endpoint: Get Lesson Content (with Variants)

```typescript
// GET /api/lessons/:dayNumber/content
// Query: ?age=13-17&language=en&tone=curious&difficulty=2

interface GetLessonContentRequest {
  dayNumber: number;
  age: '2-5' | '6-12' | '13-17' | '18-35' | '36-60' | '61+';
  language: 'en' | 'es' | 'fr';
  tone: 'curious' | 'playful' | 'serious';
  difficulty: 2 | 3;
}

interface GetLessonContentResponse {
  success: boolean;
  data: {
    lessonId: string;
    dayNumber: number;
    topic: string;
    phases: PhaseContent[];
    variantKey: string; // "13-17_en_curious" for caching
  };
}

interface PhaseContent {
  order: number;
  type: 'welcome' | 'question' | 'wisdom' | 'completion';
  name: string;
  text: string;
  question?: string;
  hint?: string;
  reflection?: string;
  kellyExpression: string;
  audioUrl?: string;
  choices?: ChoiceContent[]; // Only for question phases
}

interface ChoiceContent {
  letter: 'A' | 'B' | 'C';
  text: string;
}
```

### Endpoint: Update User Preferences

```typescript
// PUT /api/users/preferences
// Headers: Authorization: Bearer <token>

interface UpdatePreferencesRequest {
  preferredAgeGroup?: '2-5' | '6-12' | '13-17' | '18-35' | '36-60' | '61+';
  preferredLanguage?: 'en' | 'es' | 'fr';
  preferredTone?: 'curious' | 'playful' | 'serious';
  preferredDifficulty?: 2 | 3;
  preferredMode?: '2D' | '3D';
  birthdayMonth?: number;
  birthdayDay?: number;
}

interface UpdatePreferencesResponse {
  success: boolean;
  data: {
    preferences: {
      preferredAgeGroup: string;
      preferredLanguage: string;
      preferredTone: string;
      preferredDifficulty: number;
      preferredMode: string;
    };
    birthdayLessonDay?: number; // If birthday was set
  };
}
```

### Endpoint: Save Progress

```typescript
// POST /api/progress
// Headers: Authorization: Bearer <token>

interface SaveProgressRequest {
  dayNumber: number;
  currentPhase: number;
  choicesMade: Record<string, string>; // { "Q1": "A", "Q2": "B" }
  variantsUsed: {
    age: string;
    language: string;
    tone: string;
    difficulty: number;
    mode: string;
  };
  timeSpentSeconds: number;
  completed: boolean;
}

interface SaveProgressResponse {
  success: boolean;
  data: {
    progressId: string;
    newStreak: number;
    streakIncreased: boolean;
    totalCompleted: number;
  };
}
```

---

## 5. State Management

### Local Storage Keys

```typescript
const STORAGE_KEYS = {
  // User preferences (synced to server when logged in)
  PREF_AGE: 'kelly_pref_age', // '13-17'
  PREF_LANG: 'kelly_pref_language', // 'en'
  PREF_TONE: 'kelly_pref_tone', // 'curious'
  PREF_DIFF: 'kelly_pref_difficulty', // '2' or '3'
  PREF_MODE: 'kelly_pref_mode', // '2D' or '3D'

  // Current session
  CURRENT_LESSON: 'kelly_current_lesson', // Full lesson data
  CURRENT_PHASE: 'kelly_current_phase', // Phase number
  CHOICES_MADE: 'kelly_choices_made', // JSON of choices

  // Cache
  LESSON_CACHE: 'kelly_lesson_cache', // Cached lesson content
  LAST_SYNC: 'kelly_last_sync', // Last server sync time

  // Guest mode
  GUEST_MODE: 'kelly_guest_mode', // 'true' if guest
  GUEST_PROGRESS: 'kelly_guest_progress' // Local progress for guests
};
```

### State Transitions

```typescript
// Kelly Frame State Machine
type FrameState =
  | 'loading' // Loading lesson content
  | 'idle' // Kelly visible, waiting
  | 'speaking' // Kelly speaking phase content
  | 'waiting_choice' // Waiting for user to select A/B/C
  | 'transitioning' // Transitioning between phases
  | 'variant_loading' // Loading new variant (age/lang/tone change)
  | 'completed' // Lesson complete
  | 'error'; // Error state

// Valid transitions
const VALID_TRANSITIONS: Record<FrameState, FrameState[]> = {
  loading: ['idle', 'error'],
  idle: ['speaking'],
  speaking: ['waiting_choice', 'transitioning'], // Q phases wait, Welcome transitions
  waiting_choice: ['transitioning'],
  transitioning: ['speaking', 'completed'],
  variant_loading: ['speaking'],
  completed: ['idle'], // Start new lesson
  error: ['loading'] // Retry
};
```

---

## Sample Data: Citizenship Lesson (Day 333 / Nov 28)

```json
{
  "lesson_id": "citizenship-nov-28",
  "day_number": 333,
  "calendar_date": "November 28",
  "topic": "Citizenship",
  "topic_emoji": "🏛️",
  "category": "social",
  "universal_truth": "Active participation in community creates stronger societies",
  "core_principle": "Citizenship is both rights and responsibilities",

  "phases": [
    {
      "phase_order": 1,
      "phase_type": "welcome",
      "phase_name": "Welcome",
      "base_text": "Welcome! Today we're exploring citizenship – what it means to be part of a community.",
      "kelly_expression": "curious"
    },
    {
      "phase_order": 2,
      "phase_type": "question",
      "phase_name": "Q1",
      "base_text": "What does it mean to be a good citizen?",
      "base_question": "Is citizenship about following rules, or something deeper?",
      "base_hint": "Think about your community...",
      "kelly_expression": "explaining",
      "choices": [
        {
          "letter": "A",
          "text": "Following laws and paying taxes",
          "is_preferred": false,
          "choice_type": "standard"
        },
        {
          "letter": "B",
          "text": "Actively contributing to your community",
          "is_preferred": true,
          "choice_type": "standard"
        },
        {
          "letter": "C",
          "text": "Both following rules AND contributing actively",
          "is_preferred": true,
          "choice_type": "challenge"
        }
      ]
    },
    {
      "phase_order": 3,
      "phase_type": "question",
      "phase_name": "Q2",
      "base_text": "Citizenship exists at many levels.",
      "base_question": "Which matters more: local community or global citizenship?",
      "kelly_expression": "explaining",
      "choices": [
        {
          "letter": "A",
          "text": "Local – change starts in your neighborhood",
          "choice_type": "standard"
        },
        {
          "letter": "B",
          "text": "Global – we're all connected across borders",
          "choice_type": "standard"
        },
        {
          "letter": "C",
          "text": "Both are equally important and connected",
          "choice_type": "challenge"
        }
      ]
    },
    {
      "phase_order": 4,
      "phase_type": "question",
      "phase_name": "Q3",
      "base_text": "You have power to make a difference.",
      "base_question": "How can young people be good citizens?",
      "kelly_expression": "explaining",
      "choices": [
        {
          "letter": "A",
          "text": "Volunteer and help others",
          "choice_type": "standard"
        },
        {
          "letter": "B",
          "text": "Learn and share knowledge",
          "choice_type": "standard"
        },
        {
          "letter": "C",
          "text": "Vote, volunteer, and speak up for what's right",
          "choice_type": "challenge"
        }
      ]
    },
    {
      "phase_order": 5,
      "phase_type": "wisdom",
      "phase_name": "Wisdom",
      "base_text": "Here's something to think about: Every act of kindness, every time you stand up for someone, every moment you contribute – you're exercising citizenship. You don't need to wait until you can vote. You're already a citizen of your family, your school, your neighborhood, and our world.",
      "base_reflection": "What's one small act of citizenship you can do today?",
      "kelly_expression": "wisdom"
    }
  ],

  "variants": {
    "2-5_en_curious": {
      "topic_translated": "Being a Good Helper",
      "phases": [
        {
          "phase_order": 1,
          "text": "Hi friend! Today we're learning about being a good helper in your family and neighborhood!"
        },
        {
          "phase_order": 2,
          "text": "What makes someone a good helper?",
          "question": "Is a good helper someone who...",
          "choices": [
            { "letter": "A", "text": "Follows the rules at home" },
            { "letter": "B", "text": "Helps others and shares" },
            { "letter": "C", "text": "Does both!" }
          ]
        }
      ]
    },
    "13-17_en_curious": {
      "phases": [
        {
          "phase_order": 1,
          "text": "Hey! Today's topic hits different – we're exploring what citizenship actually means beyond just the textbook definition."
        }
      ]
    },
    "18-35_es_serious": {
      "topic_translated": "Ciudadanía",
      "phases": [
        {
          "phase_order": 1,
          "text": "Bienvenido. Hoy exploramos la ciudadanía – lo que significa ser parte activa de una comunidad."
        }
      ]
    }
  }
}
```

---

_This data architecture supports the complete Kelly Experience with all variant combinations, difficulty levels, and progress tracking._
