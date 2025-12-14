# 🎨 Lotd v0 Templates

Production-ready React/Next.js components for the Curious Kelly / Lesson of the Day platform.

These templates are designed to be used with [v0.dev](https://v0.app) and connect to your Supabase backend.

**Status:** ✅ TypeScript strict mode passing  
**Last verified:** December 11, 2025

---

## 📦 Installation

```bash
# Templates are part of the main repo
# Just import from the templates/v0 directory

# If using as standalone, install peer dependencies:
npm install react react-dom @supabase/supabase-js
```

---

## 📦 Template Library

### Factory & Admin

| Template | Description | Data Source |
|----------|-------------|-------------|
| `FactoryDayView` | 12×5 grid showing all archetypes × phases for a single day | `core_lessons`, `lesson_atoms`, `kelly_video_assets` |

### UI Components

| Template | Description | Data Source |
|----------|-------------|-------------|
| `ArchetypeCard` | Adaptive card component for any archetype | Static personas |
| `ArchetypeGrid` | 12-archetype selector grid | Static personas |
| `ArchetypeBadge` | Compact pill/tag for archetype display | Static personas |
| `LessonPreviewCard` | Marketing card with 4 variants (default, compact, hero, social) | `core_lessons` (optional) |

### Shared Libraries

| Module | Description |
|--------|-------------|
| `lib/personas` | All 12 archetypes with full metadata, types, and helpers |
| `lib/supabase` | Typed hooks and queries for Lotd database schema |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# .env.local
NEXT_PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here
```

### 2. Import Everything

```tsx
// Import from the barrel file
import { 
  // Components
  FactoryDayView,
  ArchetypeCard,
  ArchetypeGrid,
  ArchetypeBadge,
  LessonPreviewCard,
  
  // Personas data
  PERSONAS,
  PERSONA_IDS,
  PRIMARY_THREE,
  getPersonaImageUrl,
  getArchetypeColor,
  
  // Supabase hooks
  useLessonWithAtoms,
  useCoreLesson,
  useVideoAssets,
  useDayStats,
  useAllLessons,
  
  // Types
  type PersonaId,
  type Phase,
  type CoreLesson,
  type LessonAtom,
} from '@/templates/v0';
```

---

## 🎭 The 12 Archetypes

| ID | Name | Icon | Color | Tagline | Best For |
|----|------|------|-------|---------|----------|
| `scientist` | The Scientist | 🔬 | `#3b82f6` | Data-driven precision | Skeptics who need proof |
| `explorer` | The Explorer | 🧭 | `#eab308` | Wonder and discovery | Curious learners |
| `rebel` | The Rebel | ⚡ | `#ef4444` | Bold challenging spirit | Disengaged teens |
| `architect` | The Architect | 🏛️ | `#6b7280` | Methodical structure | Systematic thinkers |
| `diplomat` | The Diplomat | 🤝 | `#22c55e` | Inclusive harmony | Collaborative learners |
| `empath` | The Empath | 💗 | `#ec4899` | Nurturing warmth | Heart-centered learners |
| `macgyver` | The MacGyver | 🔧 | `#f97316` | Hands-on problem solver | Practical minds |
| `mystic` | The Mystic | ✨ | `#a855f7` | Profound serenity | Meaning-makers |
| `provider` | The Provider | 🛡️ | `#14b8a6` | Reassuring strength | Parents, caregivers |
| `storyteller` | The Storyteller | 📖 | `#f472b6` | Theatrical captivation | Visual/story learners |
| `strategist` | The Strategist | 🎯 | `#6366f1` | Sharp tactical mind | Competitive planners |
| `survivor` | The Survivor | 🏕️ | `#84cc16` | Grounded resilience | Pragmatic minds |

---

## 📐 Component Examples

### FactoryDayView

The 12×5 production grid for monitoring content generation:

```tsx
import { FactoryDayView } from '@/templates/v0';

// Basic usage
<FactoryDayView dayNumber={17} />

// With URL parameter
import { useSearchParams } from 'next/navigation';
const day = Number(useSearchParams().get('day')) || 1;
<FactoryDayView dayNumber={day} />
```

### ArchetypeCard

Adaptive card that styles itself based on archetype:

```tsx
import { ArchetypeCard, ArchetypeGrid, ArchetypeBadge } from '@/templates/v0';

// Full card with lesson info
<ArchetypeCard 
  archetypeId="scientist"
  topic="The Three Lives of Water"
  dayNumber={17}
  subtitle="Explore how water transforms between states"
  size="lg"
  onClick={() => navigate('/lesson/17')}
/>

// Selection grid (12 archetypes)
<ArchetypeGrid 
  selectedId="explorer"
  onSelect={(id) => setArchetype(id)}
/>

// Inline badge
<ArchetypeBadge archetypeId="rebel" />
```

### LessonPreviewCard

Marketing-ready card with 4 visual variants:

```tsx
import { LessonPreviewCard } from '@/templates/v0';

// Default card
<LessonPreviewCard 
  dayNumber={17}
  archetypeId="mystic"
  showTodayBadge
/>

// Compact (for sidebars/lists)
<LessonPreviewCard 
  dayNumber={17}
  archetypeId="explorer"
  variant="compact"
/>

// Hero (for landing pages)
<LessonPreviewCard 
  topic="The Three Lives of Water"
  universalTruth="Water is the only substance that naturally exists in all three states..."
  dayNumber={17}
  archetypeId="scientist"
  variant="hero"
  showTodayBadge
  onClick={() => startLesson(17)}
/>

// Social (1.91:1 aspect ratio for Twitter/OG)
<LessonPreviewCard 
  topic="The Three Lives of Water"
  dayNumber={17}
  archetypeId="storyteller"
  variant="social"
/>
```

---

## 🔌 Supabase Hooks

### useLessonWithAtoms

Fetch a lesson with all its content atoms:

```tsx
import { useLessonWithAtoms } from '@/templates/v0';

function LessonPlayer({ day }: { day: number }) {
  const { data, loading, error, refetch } = useLessonWithAtoms(day, 'explorer');
  
  if (loading) return <Spinner />;
  if (error) return <Error message={error} onRetry={refetch} />;
  
  return (
    <div>
      <h1>{data.topic}</h1>
      {data.atoms.map(atom => (
        <Phase key={atom.id} {...atom} />
      ))}
    </div>
  );
}
```

### useDayStats

Get content/video completion stats for a day:

```tsx
import { useDayStats } from '@/templates/v0';

const { data } = useDayStats(17);
// data = {
//   totalAtoms: 60,
//   atomsWithContent: 58,
//   atomsWithVideo: 45,
//   atomsGenerating: 3,
//   atomsFailed: 0,
// }
```

### useVideoAssets

Fetch all video assets for a lesson day:

```tsx
import { useVideoAssets } from '@/templates/v0';

const { data: videos } = useVideoAssets(17);
// data = [{ id, lesson_day, phase, status, video_public_url, ... }]
```

---

## 🔗 Data Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPABASE                                 │
├─────────────────────────────────────────────────────────────┤
│  core_lessons (365 rows)                                    │
│    • id (UUID), day_number (1-365), topic, universal_truth  │
│                                                             │
│  lesson_atoms (21,855 rows)                                 │
│    • id, core_lesson_id, archetype, phase, content (JSONB)  │
│    • 12 archetypes × 5 phases × 365 days                    │
│                                                             │
│  kelly_video_assets                                         │
│    • lesson_day, phase, archetype, status, video_public_url │
│                                                             │
│  CDN: tvjalxxsyryjphkforjv.supabase.co/storage/v1/...       │
│    • Kelly archetype images (head, clean, prop variants)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Design Tokens

All templates use the archetype's brand color for styling:

```tsx
import { PERSONAS, getArchetypeColor } from '@/templates/v0';

// Get full persona data
const scientist = PERSONAS['scientist'];
// { id, name, icon, tagline, color, images, ... }

// Get color with opacity
const bgColor = getArchetypeColor('scientist', 0.2);
// "rgba(59, 130, 246, 0.2)"

// Use in styles
<div style={{ 
  backgroundColor: `${scientist.color}20`,  // 20% opacity
  borderColor: scientist.color,             // Solid border
  boxShadow: `0 0 20px ${scientist.color}40`, // Glow
}}>
```

---

## 📁 File Structure

```
templates/v0/
├── package.json              # Dependencies & scripts
├── tsconfig.json             # TypeScript config (strict mode)
├── README.md                 # This file
├── index.ts                  # Barrel export
│
├── FactoryDayView.tsx        # 12×5 production grid
├── ArchetypeCard.tsx         # Adaptive card + grid + badge
├── LessonPreviewCard.tsx     # Marketing card (4 variants)
│
└── lib/
    ├── personas.ts           # 12 archetypes, types, helpers
    └── supabase.ts           # Hooks & queries
```

---

## 🔌 v0 Prompts

Use these prompts in v0.app to generate new components that use your templates:

### Lesson Player
> "Create a lesson player component that fetches from Supabase core_lessons and lesson_atoms tables, displays a 5-phase journey (Hook → Fact1 → Fact2 → Fact3 → Wisdom), uses Kelly archetype images from the CDN, and adapts colors based on the selected archetype's brand color."

### Progress Dashboard
> "Create a learner progress dashboard showing completed lessons, current streak, next lesson preview, and archetype preference. Use dark mode with archetype-colored accents."

### Archetype Quiz
> "Create a 5-question personality quiz that recommends one of 12 Kelly archetypes based on learning style preferences. Show results with archetype card and sample lesson hook."

---

## 📝 Creating New Templates

1. **Start with data**: What Supabase tables/views does this need?
2. **Import from lib/**: Use `PERSONAS`, hooks, and types from shared libs
3. **Use archetype tokens**: Colors, icons, images adapt automatically
4. **Dark mode first**: All templates assume `bg-gray-950` base
5. **Responsive**: Mobile-first, then expand for larger screens
6. **Export from index**: Add to `index.ts` barrel for clean imports

---

## 🔗 Related Resources

- `/public/assets/kelly/kelly-personas-manifest.json` - Full archetype metadata
- `/docs/architecture/CANONICAL_IDS_AND_TERMS.md` - Naming conventions
- `/docs/GOLDEN_THREE_ARCHETYPES.md` - Archetype deep dive
- `/docs/backend/SUPABASE_SCHEMA.md` - Database schema
- `/docs/V0_TEMPLATE_SYSTEM_PROPOSAL.md` - Business rationale

---

*TypeScript verified: December 11, 2025*  
*Maintainer: Engineering Team*



