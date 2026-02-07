# V0.APP SUPABASE DATABASE HANDOFF

**Date:** February 4, 2026  
**Purpose:** Everything v0.app needs to connect to and display curriculum data

---

## 1. CONNECTION DETAILS

### Supabase Project URL
```
https://tvjalxxsyryjphkforjv.supabase.co
```

### Anon Key (safe to use client-side)
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI
```

### Environment Variables (add to .env.local)
```env
NEXT_PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI
```

---

## 2. DATABASE SCHEMA

### Table: `core_lessons` (730 rows total)

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `day_number` | INTEGER | 1-365 |
| `track` | TEXT | `'learn'` or `'grow'` |
| `topic` | TEXT | Lesson title (e.g., "Why Ice Floats") |
| `universal_truth` | TEXT | One-sentence truth (e.g., "Water expands when it freezes...") |
| `icon_emoji` | TEXT | Emoji for the lesson (e.g., "🧊") |
| `marketing_headline` | TEXT | Hook for marketing |
| `marketing_tagline` | TEXT | Short tagline |
| `marketing_pitch` | TEXT | Full marketing description |
| `learning_objectives` | JSONB | Array of objectives |
| `fun_facts` | JSONB | Array of fun facts |
| `quick_quiz_questions` | JSONB | Quiz data |
| `reflection_prompts` | JSONB | Prompts for reflection |

### Data Overview
- **365 LEARN lessons** (science, history, life skills)
- **365 GROW lessons** (AI fluency, meta-learning)
- Every day has BOTH a LEARN and GROW lesson

---

## 3. SUPABASE CLIENT SETUP

### Install Package
```bash
npm install @supabase/supabase-js
```

### Create Client (lib/supabase.ts)
```typescript
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
```

---

## 4. EXAMPLE QUERIES

### Get Today's Lessons (both tracks)
```typescript
const dayNumber = 35 // February 4th, 2026

const { data: lessons, error } = await supabase
  .from('core_lessons')
  .select('*')
  .eq('day_number', dayNumber)
  .order('track')

// Returns 2 rows: one GROW, one LEARN
```

### Get Single LEARN Lesson
```typescript
const { data: lesson, error } = await supabase
  .from('core_lessons')
  .select('*')
  .eq('day_number', 35)
  .eq('track', 'learn')
  .single()
```

### Get All Lessons for Sidebar/Calendar
```typescript
const { data: allLessons, error } = await supabase
  .from('core_lessons')
  .select('day_number, topic, icon_emoji, track')
  .order('day_number')
```

### Get LEARN Lessons Only
```typescript
const { data: learnLessons, error } = await supabase
  .from('core_lessons')
  .select('day_number, topic, icon_emoji, universal_truth')
  .eq('track', 'learn')
  .order('day_number')
```

### Get GROW Lessons Only
```typescript
const { data: growLessons, error } = await supabase
  .from('core_lessons')
  .select('day_number, topic, icon_emoji, universal_truth')
  .eq('track', 'grow')
  .order('day_number')
```

### Get Lessons for a Month (e.g., February = Days 32-59)
```typescript
const { data: februaryLessons, error } = await supabase
  .from('core_lessons')
  .select('*')
  .gte('day_number', 32)
  .lte('day_number', 59)
  .order('day_number')
  .order('track')
```

---

## 5. REACT HOOK EXAMPLE

```typescript
// hooks/useLesson.ts
import { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase'

interface Lesson {
  id: string
  day_number: number
  track: 'learn' | 'grow'
  topic: string
  universal_truth: string
  icon_emoji: string
  marketing_headline: string
}

export function useLesson(dayNumber: number, track: 'learn' | 'grow' = 'learn') {
  const [lesson, setLesson] = useState<Lesson | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchLesson() {
      setLoading(true)
      const { data, error } = await supabase
        .from('core_lessons')
        .select('*')
        .eq('day_number', dayNumber)
        .eq('track', track)
        .single()

      if (error) {
        setError(error.message)
      } else {
        setLesson(data)
      }
      setLoading(false)
    }

    fetchLesson()
  }, [dayNumber, track])

  return { lesson, loading, error }
}
```

---

## 6. SAMPLE COMPONENT

```tsx
// components/LessonCard.tsx
'use client'

import { useLesson } from '@/hooks/useLesson'

export function LessonCard({ dayNumber }: { dayNumber: number }) {
  const { lesson, loading, error } = useLesson(dayNumber, 'learn')

  if (loading) return <div className="animate-pulse h-32 bg-gray-200 rounded-lg" />
  if (error) return <div className="text-red-500">Error: {error}</div>
  if (!lesson) return null

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-4">
        <span className="text-4xl">{lesson.icon_emoji}</span>
        <div>
          <p className="text-sm text-gray-500">Day {lesson.day_number}</p>
          <h2 className="text-xl font-bold">{lesson.topic}</h2>
        </div>
      </div>
      <p className="text-gray-700">{lesson.universal_truth}</p>
    </div>
  )
}
```

---

## 7. DAY NUMBER MAPPING

| Month | Start Day | End Day |
|-------|-----------|---------|
| January | 1 | 31 |
| February | 32 | 59 |
| March | 60 | 90 |
| April | 91 | 120 |
| May | 121 | 151 |
| June | 152 | 181 |
| July | 182 | 212 |
| August | 213 | 243 |
| September | 244 | 273 |
| October | 274 | 304 |
| November | 305 | 334 |
| December | 335 | 365 |

### Calculate Day Number from Date
```typescript
function getDayNumber(date: Date = new Date()): number {
  const start = new Date(date.getFullYear(), 0, 0)
  const diff = date.getTime() - start.getTime()
  const oneDay = 1000 * 60 * 60 * 24
  return Math.floor(diff / oneDay)
}

// Usage: getDayNumber() returns today's day number (1-365)
```

---

## 8. SAMPLE DATA (February 4th = Day 35)

### LEARN Lesson (Day 35)
```json
{
  "day_number": 35,
  "track": "learn",
  "topic": "How Electricity Flows",
  "icon_emoji": "💡",
  "universal_truth": "Electric current is the flow of charged particles through a conductor, like water through a pipe.",
  "marketing_headline": "The invisible river powering your entire world"
}
```

### GROW Lesson (Day 35)
```json
{
  "day_number": 35,
  "track": "grow",
  "topic": "Asking Experts",
  "icon_emoji": "🎓",
  "universal_truth": "How to learn from people who know more"
}
```

---

## 9. QUICK TEST

Paste this in browser console on any page with Supabase loaded:

```javascript
// Quick test - fetch Day 1 lessons
const { createClient } = supabase
const client = createClient(
  'https://tvjalxxsyryjphkforjv.supabase.co',
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI'
)
client.from('core_lessons').select('*').eq('day_number', 1).then(console.log)
```

---

## 10. IMPORTANT NOTES

1. **Row Level Security (RLS)** is enabled - the anon key can only READ data
2. **No writes allowed** from client-side with anon key
3. **All 730 lessons are live** and ready to use
4. **Both tracks exist for every day** - always filter by `track` if you want one or the other
5. **Icons were just fixed** on Feb 4, 2026 - all icons now match their topics correctly

---

## CONTACT

If v0.app needs additional columns, different data structure, or has questions:
- The Supabase project is `tvjalxxsyryjphkforjv`
- Schema changes require the service role key (not provided here for security)

---

**This document contains everything needed to connect and display lessons from Supabase.**
