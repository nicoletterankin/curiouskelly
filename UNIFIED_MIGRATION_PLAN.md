# Unified Kelly Experience - Migration Plan

## Vision
One scrollable homepage that contains everything. Kelly is always present, adaptable, and the star. Zero friction, infinite depth.

## Current State Audit

### Pages to Consolidate
1. **index.html** - Auth/Login (KEEP as entry, but add content below)
2. **pricing.html** - 3 pricing cards + gift section + features + FAQ
3. **careers.html** - Affiliate program with calculator
4. **perspectives.html** - Time machine slider + generational hooks
5. **curriculum.html** - 366 topics with age selector
6. **about.html** - Institute positioning + syllabus

### Elements to Extract

#### From `index.html` (Current Homepage)
- ✅ Clean auth flow (Google, Apple, Email)
- ✅ Kelly hero image (right panel)
- ✅ Footer structure (4 columns)
- ❌ Two-panel split (too rigid for unified scroll)

#### From `pricing.html`
- ✅ 3 pricing tiers (Monthly $9.99, Annual $99, Lifetime $299)
- ✅ Gift cards (3/6/12 month, lifetime)
- ✅ Features grid (6 features)
- ✅ FAQ section (6 questions)
- ✅ "7-day free trial" badge
- ❌ Separate navigation (consolidate)

#### From `careers.html`
- ✅ Affiliate calculator (interactive slider)
- ✅ 3 commission tiers (Scholar 20%, Fellow 25%, Ambassador 30%)
- ✅ Success stories (3 cards)
- ✅ Application form
- ✅ Founding 100 offer (30% forever)
- ❌ Separate header (consolidate)

#### From `perspectives.html`
- ✅ Time machine slider (1945-2020)
- ✅ Generation quick picks (6 buttons)
- ✅ 3 perspective cards (older/you/younger)
- ✅ Topic selector (10 sample topics)
- ✅ Age-specific hooks from Supabase
- ❌ Standalone page (integrate into curriculum section)

#### From `curriculum.html`
- ✅ Age bucket selector (6 buckets)
- ✅ 366 lesson cards with age-specific hooks
- ✅ Day number + topic + hook display
- ❌ Separate page (make it a deep-dive section)

#### From `about.html`
- ✅ "Institute" positioning
- ✅ Curriculum tracks (4 cards: Wonder, Foundations, Synthesis, Mastery)
- ✅ Syllabus preview (first 30 days)
- ❌ Separate Unity section (not needed on unified)

## Unified Structure (Single Scroll)

```
┌─────────────────────────────────────────┐
│ FIXED TOP NAV                           │
│ [✨ Kelly] [Today's Lesson] [Curriculum]│
│ [Pricing] [Careers] [Login] [🎛️ Kelly] │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ HERO SECTION                            │
│ Kelly avatar (always visible, animated) │
│ "Your Favorite Teacher Ever"            │
│ [Start Learning] [See Curriculum]       │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ TODAY'S LESSON                          │
│ Day 334: How Money Works                │
│ [Join Live Class] [Watch 2D] [Listen]   │
│ Kelly avatar adapts to selected mode    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ CURRICULUM EXPLORER                     │
│ Age Selector: [2-5] [6-12] [13-17]...  │
│ 366 topics with age-specific hooks      │
│ [Expand to see all variations] →        │
│   ↓ Opens Perspectives Time Machine     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ PERSPECTIVES (Collapsed by default)     │
│ Time Machine Slider (1945-2020)         │
│ See same topic through different eyes   │
│ 3 comparison cards (older/you/younger)  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ PRICING                                 │
│ 3 cards: Monthly, Annual (featured),    │
│ Lifetime. Inline checkout modal.        │
│ Gift section below.                     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ CAREERS / AFFILIATES                    │
│ "Build Your Career Teaching the World"  │
│ Interactive earnings calculator          │
│ 3 commission tiers                      │
│ Application form (inline)               │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ FOOTER                                  │
│ Explore | About | Social | Download     │
│ Final approved links only               │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ KELLY CONTROL PANEL (Floating)          │
│ [2D] [3D] [Audio Only] [Full Screen]    │
│ [Solo] [Social] [Settings]              │
│ Always accessible, bottom-right         │
└─────────────────────────────────────────┘
```

## Kelly Avatar Controller

### Modes
1. **2D** - Flat image, expressions change
2. **3D** - Unity avatar (when available)
3. **Audio Only** - Voice only, no visual
4. **Image Only** - Silent, visual only
5. **Full Screen** - Immersive mode

### Social Modes
1. **Solo** - Just you and Kelly
2. **Social** - See other learners (future)

### Visual States
- **Idle** - Gentle animation, waiting
- **Teaching** - Animated gestures
- **Listening** - Attentive pose
- **Celebrating** - Success animation

## Navigation Strategy

### Top Nav (Fixed)
- Logo (✨ Curious Kelly)
- Today's Lesson (jumps to section)
- Curriculum (jumps to section)
- Pricing (jumps to section)
- Careers (jumps to section)
- Login (opens modal)
- Kelly Controls (opens floating panel)

### No Page Redirects
- Login → Modal overlay
- Checkout → Modal overlay (Stripe embedded)
- Settings → Slide-in panel
- All navigation is smooth scroll or modal

## Brand Consistency

### Colors (LOCKED)
- Primary: Kelly Blue (#2563eb)
- Hover: #1d4ed8
- Background: #0a0a0b
- Text Primary: #fafafa
- Text Secondary: #a1a1aa
- Success: #22c55e
- **NO ORANGE ANYWHERE**

### Typography
- Headlines: Fraunces (serif, elegant)
- Body: Inter (sans-serif, clean)
- Monospace: For data/code

### Kelly Images (Approved Only)
- Hero: `/assets/kelly/hero/kelly-hero.jpeg`
- Expressions: `/public/images/expressions/` (9 expressions)
- Production: `/assets/kelly/production/` (53 images)

## Implementation Order

1. ✅ Audit complete (this document)
2. Build Kelly avatar controller component
3. Create unified navigation
4. Migrate sections one by one:
   - Hero
   - Today's Lesson
   - Curriculum + Perspectives (integrated)
   - Pricing
   - Careers
   - Footer
5. Add login/checkout modals
6. Polish animations and transitions
7. Test all interactions
8. Deploy as new index.html

## Success Criteria

- ✅ Single scroll, no page reloads
- ✅ Kelly always visible and adaptable
- ✅ Deep curriculum exploration without leaving page
- ✅ Login/checkout as modals
- ✅ All orange replaced with Kelly Blue
- ✅ Top and bottom navigation functional
- ✅ Mobile responsive
- ✅ Sub-second interactions
- ✅ "High-end creative agency" polish

## Next Steps

Execute the implementation in order. Start with the Kelly avatar controller, then build the unified page section by section.



