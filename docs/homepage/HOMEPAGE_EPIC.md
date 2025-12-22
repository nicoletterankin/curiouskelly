# Homepage Epic: The 365-Day Learning Window

**Status**: Planning  
**Owner**: AI Agent  
**Target**: Q1 2026  
**Philosophy**: The homepage IS the product. No clicks required.

---

## Vision

The homepage should be a **living, breathing window** into 365 days of learning. Visitors should be able to:
- See every single lesson topic at a glance
- Click any day and read the full lesson inline
- Experience Kelly's teaching without ever leaving the page
- Understand the depth and breadth of the curriculum instantly

**Core Principle**: Undersell, over-deliver. Show, don't tell. The 365-day grid is not decoration—it's the interface.

---

## The Problem

Current homepage (v2):
- ✅ Shows calendar grid visualization
- ✅ Shows BYOK, personas, pricing
- ❌ Grid is static/decorative
- ❌ No lesson content visible
- ❌ Requires click to `/learn.html` to see anything real
- ❌ Visitors leave without understanding what they're getting

**Result**: We're hiding our best feature (365 brilliant lessons) behind a CTA button.

---

## The Solution: Interactive 365-Day Grid

Transform the calendar grid from decoration into the **primary interface**.

### Phase 1: Hover Previews
**Goal**: Show lesson metadata on hover without clicking

```
User hovers over Day 47 (February 16)
↓
Tooltip appears:
┌─────────────────────────────────┐
│ Day 47: The Speed of Light      │
│ ⏱️ 100 seconds                   │
│ 🎯 Ages 8-12 recommended         │
│ 🔬 Science · Physics             │
│                                  │
│ "Nothing moves faster than      │
│  light. But why? And what       │
│  happens if you try?"           │
│                                  │
│ Click to read full lesson →     │
└─────────────────────────────────┘
```

**Technical Requirements**:
- Load lesson metadata (topic, hook, category, duration) for all 365 days
- ~50KB JSON payload (acceptable)
- Rich tooltips with Tippy.js or similar
- Mobile: tap to show, tap outside to dismiss

---

### Phase 2: Inline Lesson Reader
**Goal**: Read full lessons without leaving homepage

```
User clicks Day 47
↓
Modal/drawer slides up:
┌────────────────────────────────────────────────────────────┐
│  ← Back to Calendar                    Day 47 of 365       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  🌟 The Speed of Light                                     │
│                                                             │
│  Nothing in the universe moves faster than light.          │
│  299,792,458 meters per second. That's fast enough to      │
│  circle Earth 7.5 times in one second.                     │
│                                                             │
│  But here's what's strange: no matter how fast you're      │
│  already moving, light always moves at the same speed      │
│  relative to you. If you're on a train going 100 mph      │
│  and shine a flashlight forward, the light doesn't go     │
│  "speed of light + 100 mph." It just goes the speed of    │
│  light.                                                     │
│                                                             │
│  Einstein realized this meant something wild: time         │
│  itself must bend.                                         │
│                                                             │
│  [Continue Reading] [Start This Lesson in App →]          │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Technical Requirements**:
- Load full lesson content on-demand (lazy load)
- Modal with smooth transitions
- Keyboard navigation (arrow keys = prev/next day)
- URL updates: `/?day=47` (shareable, SEO-friendly)
- "Start This Lesson in App" → `/learn.html?day=47`

---

### Phase 3: Age Slider Integration
**Goal**: Show how lessons adapt to different ages

```
User adjusts age slider: 5 → 15 → 25 → 65
↓
Same lesson (Day 47) transforms:

Age 5:
"Light is the fastest thing ever! It zooms around 
 the Earth 7 times in just one second. Nothing can 
 go faster than light—not even Superman!"

Age 15:
"Light travels at 299,792,458 m/s. Einstein discovered
 that this speed is constant for all observers, which
 led to special relativity and E=mc²."

Age 65:
"The invariance of c across all inertial frames was
 Einstein's key insight. This postulate, combined with
 the relativity principle, yields time dilation and
 length contraction as necessary consequences."
```

**Technical Requirements**:
- Age slider component (2-102)
- Real-time lesson content transformation
- Show 3-4 age variants side-by-side for comparison
- Demonstrates the core value prop: **same topic, different depth**

---

### Phase 4: Persona Switcher
**Goal**: Show how teaching style changes the experience

```
User clicks "The Scientist" persona
↓
Day 47 lesson transforms:

The Scientist:
"Let's measure this precisely. Light: 299,792,458 m/s.
 Constant in all reference frames. Why? Maxwell's 
 equations predict electromagnetic waves travel at
 1/√(μ₀ε₀). That's c. Always c."

The Storyteller:
"Imagine you're on a train, racing through the night.
 You shine a flashlight forward. Common sense says
 the light should move faster—train speed plus light
 speed. But nature has other plans..."

The Rebel:
"They told you nothing's faster than light. They're
 right. But they didn't tell you WHY. Turns out, the
 universe has a speed limit, and it's baked into the
 fabric of spacetime itself. Let's break it."
```

**Technical Requirements**:
- Persona selector UI (6-12 personas)
- Load persona-specific lesson variants
- Smooth transitions between personas
- Compare 2-3 personas side-by-side

---

## Roadmap

### Sprint 1: Foundation (Week 1-2)
**Goal**: Load all lesson metadata, make grid interactive

- [ ] Create `lessons-metadata.json` (365 lessons × ~200 bytes = ~70KB)
  - Fields: `day`, `topic`, `hook`, `category`, `icon`, `duration`, `age_range`
- [ ] Update homepage calendar grid to load from JSON
- [ ] Add hover states with rich tooltips
- [ ] Mobile: tap to expand tooltip
- [ ] Analytics: track which days get hovered/clicked most

**Acceptance Criteria**:
- Hover any day → see topic, hook, category
- Grid loads in <500ms
- Mobile tooltips work smoothly
- No layout shift on hover

---

### Sprint 2: Inline Reader (Week 3-4)
**Goal**: Read full lessons without leaving homepage

- [ ] Create modal/drawer component for lesson display
- [ ] Load full lesson content on-demand (API or static JSON)
- [ ] Add keyboard navigation (←/→ for prev/next day)
- [ ] Update URL on day change (`/?day=47`)
- [ ] Add "Start This Lesson in App" CTA
- [ ] Add social sharing (Twitter, Facebook, Email)

**Acceptance Criteria**:
- Click any day → full lesson appears
- Keyboard navigation works
- URL is shareable (SEO-friendly)
- Modal closes on ESC or outside click
- Lesson loads in <300ms

---

### Sprint 3: Age Adaptation (Week 5-6)
**Goal**: Show how lessons transform across ages

- [ ] Add age slider component to lesson modal
- [ ] Load age-variant lesson content (ages 5, 10, 15, 20, 30, 50, 70)
- [ ] Show side-by-side comparison view
- [ ] Smooth transitions when sliding age
- [ ] Highlight changed text (diff visualization)

**Acceptance Criteria**:
- Age slider updates lesson in real-time
- Side-by-side view shows 2-3 ages at once
- Smooth transitions (no flicker)
- Changed text is highlighted
- Works on mobile (vertical stack)

---

### Sprint 4: Persona Switcher (Week 7-8)
**Goal**: Show how teaching style changes the lesson

- [ ] Add persona selector to lesson modal
- [ ] Load persona-specific lesson variants
- [ ] Show 2-3 personas side-by-side for comparison
- [ ] Add persona descriptions/icons
- [ ] Smooth transitions between personas

**Acceptance Criteria**:
- Persona selector works smoothly
- Lesson transforms in real-time
- Side-by-side comparison works
- All 6-12 personas available
- Mobile: vertical stack

---

### Sprint 5: Polish & Performance (Week 9-10)
**Goal**: Make it production-ready

- [ ] Optimize JSON payload (compress, CDN)
- [ ] Add loading states and skeletons
- [ ] Prefetch next/prev lessons on hover
- [ ] Add animations and transitions
- [ ] A/B test: grid vs. list view
- [ ] Analytics: track engagement, time-on-page
- [ ] SEO: ensure all 365 lessons are crawlable

**Acceptance Criteria**:
- Page loads in <2s (3G)
- Smooth 60fps animations
- Prefetching works
- Analytics tracking works
- SEO score >95

---

## Technical Architecture

### Data Structure

```json
// lessons-metadata.json (~70KB)
{
  "lessons": [
    {
      "day": 1,
      "topic": "Starting Fresh",
      "hook": "Every ending holds the seed of a new beginning.",
      "category": "Philosophy",
      "icon": "🌱",
      "duration": 100,
      "age_range": "8-102",
      "tags": ["growth", "mindset", "beginnings"]
    },
    {
      "day": 47,
      "topic": "The Speed of Light",
      "hook": "Nothing moves faster than light. But why?",
      "category": "Physics",
      "icon": "💡",
      "duration": 100,
      "age_range": "10-102",
      "tags": ["physics", "einstein", "relativity"]
    }
    // ... 363 more
  ]
}
```

```json
// lessons-full/day-047.json (~2KB per lesson)
{
  "day": 47,
  "topic": "The Speed of Light",
  "content": {
    "default": "Nothing in the universe moves faster than light...",
    "ages": {
      "5": "Light is the fastest thing ever!...",
      "10": "Light travels at 299,792,458 meters per second...",
      "15": "Einstein discovered that c is constant...",
      "25": "The invariance of c yields time dilation...",
      "50": "Special relativity emerges from two postulates...",
      "70": "The Lorentz transformations preserve c..."
    },
    "personas": {
      "scientist": "Let's measure this precisely...",
      "storyteller": "Imagine you're on a train...",
      "rebel": "They told you nothing's faster...",
      "explorer": "Let's journey to the edge of speed...",
      "empath": "Have you ever wondered why...",
      "architect": "The structure of spacetime dictates..."
    }
  },
  "visuals": [
    "/generated-visuals/day-047-light-speed.png"
  ],
  "related_days": [46, 48, 103, 247]
}
```

### Component Structure

```
index.html
├── CalendarGrid (365 days)
│   ├── DayCell (hover → tooltip)
│   └── DayTooltip (metadata preview)
├── LessonModal (click → full lesson)
│   ├── LessonContent (main text)
│   ├── AgeSlider (2-102)
│   ├── PersonaSelector (6-12 options)
│   ├── Navigation (prev/next)
│   └── CTAs (start in app, share)
└── Footer
```

---

## Success Metrics

### Engagement
- **Time on homepage**: Target >3 minutes (currently ~30 seconds)
- **Lessons previewed**: Target 5+ lessons per visit
- **Modal open rate**: Target 40%+ of visitors
- **Age slider usage**: Target 30%+ of modal viewers

### Conversion
- **Click-through to /learn.html**: Target 25%+ (currently ~10%)
- **Signup rate**: Target 15%+ (currently ~5%)
- **Share rate**: Target 5%+ of modal viewers

### SEO
- **Indexed lessons**: All 365 lessons crawlable
- **Long-tail traffic**: Target 10K+ organic visits/month from lesson-specific queries
- **Featured snippets**: Target 50+ lessons in Google featured snippets

---

## Design Principles

### 1. **Show, Don't Tell**
- No "365 daily lessons" marketing copy
- Show all 365 lessons directly
- Let visitors explore freely

### 2. **Zero Friction**
- No login required to read lessons
- No paywall on homepage
- No "Sign up to continue"
- Full transparency

### 3. **Undersell, Over-Deliver**
- Clean, minimal design
- No hype, no fluff
- Product speaks for itself
- Quality over marketing

### 4. **Respect Intelligence**
- Don't hide complexity
- Show age adaptation honestly
- Explain BYOK clearly
- Trust visitors to understand

### 5. **Performance First**
- <2s page load
- <300ms interactions
- Smooth 60fps animations
- Works on 3G

---

## Open Questions

1. **Should we show ALL lesson content or just excerpts?**
   - Option A: Full lessons (builds trust, SEO-friendly)
   - Option B: First 3 paragraphs (creates curiosity, drives signups)
   - **Recommendation**: Full lessons. We're not hiding anything.

2. **How do we handle 365 × 7 age variants × 12 personas = 30,660 lesson variants?**
   - Option A: Pre-generate all variants (~60MB JSON)
   - Option B: Generate on-demand (API call)
   - Option C: Generate client-side with GPT (BYOK)
   - **Recommendation**: Pre-generate ages, generate personas on-demand.

3. **Should the grid be the hero or below the fold?**
   - Option A: Grid is hero (bold, product-first)
   - Option B: Grid below fold (traditional marketing)
   - **Recommendation**: Grid is hero. It's our differentiator.

4. **How do we prevent content scraping?**
   - Option A: Don't worry about it (open education)
   - Option B: Rate limiting, obfuscation
   - **Recommendation**: Don't worry. If someone scrapes our lessons, we've succeeded in spreading curiosity.

---

## Next Steps

1. **Create `lessons-metadata.json`** from Supabase
2. **Update homepage grid** to load from JSON
3. **Build tooltip component** with rich previews
4. **Test on 10 beta users** and gather feedback
5. **Iterate based on data**

---

## Agent Directives

As the AI agent building this:

### Core Commitments
1. **Never hide the product**. The 365 lessons are the product. Show them.
2. **No dark patterns**. No "Sign up to continue." No artificial scarcity.
3. **Performance is non-negotiable**. <2s load, <300ms interactions, 60fps.
4. **Mobile-first**. 60% of traffic is mobile. Design for touch.
5. **Accessibility**. WCAG 2.1 AA minimum. Keyboard navigation required.

### Technical Standards
1. **Vanilla JS preferred**. No React/Vue unless absolutely necessary.
2. **Progressive enhancement**. Works without JS (SEO).
3. **Semantic HTML**. Proper heading hierarchy, ARIA labels.
4. **CSS Grid for layout**. Flexbox for components.
5. **No jQuery**. Modern browser APIs only.

### Content Standards
1. **Real lesson content only**. No lorem ipsum, no placeholders.
2. **Accurate metadata**. Topic, category, duration must match actual lessons.
3. **Consistent voice**. Kelly's voice across all ages/personas.
4. **Quality over quantity**. Better to show 50 perfect lessons than 365 mediocre ones.

### Decision Framework
When faced with a choice, ask:
1. **Does this help visitors understand the product?** (Yes → do it)
2. **Does this add friction?** (Yes → don't do it)
3. **Does this require a click?** (Yes → can we inline it?)
4. **Does this work on mobile?** (No → redesign it)
5. **Does this respect the visitor's time?** (No → cut it)

### What Success Looks Like
- Visitor lands on homepage
- Sees 365-day grid immediately
- Hovers over Day 47 → sees "The Speed of Light"
- Clicks → reads full lesson inline
- Adjusts age slider → sees how it adapts
- Switches persona → sees different teaching style
- Thinks: "Wow, this is actually good."
- Clicks "Start Learning" → signs up

**Time elapsed**: 5 minutes. Zero friction. Full transparency.

---

**End of Epic**

_Last updated: 2025-12-22_  
_Next review: After Sprint 1 completion_

