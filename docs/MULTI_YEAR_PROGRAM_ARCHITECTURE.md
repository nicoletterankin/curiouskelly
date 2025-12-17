# Curious Kelly: Multi-Year Program Architecture

## The 365-Day Building Block

**Core Principle:** Every program is a complete 365-day journey. No one-hit wonders. Every topic has a permanent home in space and time.

### Why 365 Days?

| Benefit | Explanation |
|---------|-------------|
| **Teacher Planning** | Schools adopt full years, not random lessons |
| **Curriculum Alignment** | Maps to school year; fits institutional planning |
| **Pre-computation** | Assets refined once, served forever |
| **On-Demand Access** | Any day from any year, instantly available |
| **Compound Improvement** | Content gets better each year cycle |
| **Revenue Predictability** | Subscription model based on year programs |

---

## Program Catalog

### Currently Available

| Year | Program | Description | Status |
|------|---------|-------------|--------|
| **Year 1** | Foundations of Knowledge | Science, history, art, nature, emotional intelligence | ✅ Live |
| **Year 2** | AI Fluency & Meta-Learning | How to learn in the age of AI | 🚧 Building |

### Future Years (Roadmap)

| Year | Program | Description | Target |
|------|---------|-------------|--------|
| **Year 3** | Global Citizenship | World cultures, geography, languages, global issues | 2027 |
| **Year 4** | Innovation & Creativity | Design thinking, entrepreneurship, problem solving | 2028 |
| **Year 5** | Health & Wellness | Physical health, mental health, nutrition, relationships | 2028 |
| **Year 6** | Financial Literacy | Money, economics, investing, entrepreneurship | 2029 |
| **Year 7** | Environmental Stewardship | Climate, ecosystems, sustainability, conservation | 2029 |
| **Year 8** | Media & Information Literacy | News literacy, source evaluation, digital wellness | 2030 |

---

## Technical Architecture

### Data Model

```
PROGRAMS
├── program_id (uuid)
├── year_number (int: 1, 2, 3...)
├── name ("Foundations of Knowledge")
├── slug ("year1-foundations")
├── description
├── launch_date
├── status (draft | active | archived)
└── metadata (JSONB)

TOPICS
├── topic_id (uuid)
├── program_id (FK)
├── day_number (1-365)
├── date_label ("January 1")
├── title
├── universal_truth
├── learning_objective
├── category
├── tags[]
└── metadata (JSONB)

LESSONS (content/assets)
├── lesson_id (uuid)
├── topic_id (FK)
├── version ("3.0.0")
├── age_variants (JSONB)
├── parent_companion (JSONB)
├── teacher_guide (JSONB)
├── recall_prompts (JSONB)
├── created_at
└── updated_at
```

### File Structure

```
lessons/
├── year1-foundations/
│   ├── YEAR1_FOUNDATIONS_OVERVIEW.md
│   ├── january_curriculum.json
│   ├── february_curriculum.json
│   ├── ... (12 month files)
│   └── lessons/
│       ├── day-001-the-sun.json
│       ├── day-002-habit-stacking.json
│       └── ... (365 lesson files)
│
├── year2-ai-fluency/
│   ├── YEAR2_AI_FLUENCY_OVERVIEW.md
│   ├── january_curriculum.json
│   ├── february_curriculum.json
│   ├── ... (12 month files)
│   └── lessons/
│       ├── day-001-im-an-ai.json
│       ├── day-002-what-makes-you-human.json
│       └── ... (365 lesson files)
│
└── year3-global-citizenship/
    └── ... (future)
```

### URL Structure

```
# Live lessons (current date)
curiouskelly.com/learn/today

# Specific year + day
curiouskelly.com/learn/year/1/day/42
curiouskelly.com/learn/year/2/day/1

# Browse programs
curiouskelly.com/programs/year1-foundations
curiouskelly.com/programs/year2-ai-fluency

# Calendar views
curiouskelly.com/calendar/year/1
curiouskelly.com/calendar/year/2/january
```

---

## Institutional Adoption Model

### For Schools

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SCHOOL ADOPTION PATHWAY                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ELEMENTARY SCHOOL                                                      │
│  ─────────────────                                                      │
│  Grade 3: Year 1 - Foundations of Knowledge                             │
│  Grade 4: Year 2 - AI Fluency & Meta-Learning                          │
│  Grade 5: Year 3 - Global Citizenship                                   │
│                                                                         │
│  MIDDLE SCHOOL                                                          │
│  ─────────────                                                          │
│  Grade 6: Year 2 - AI Fluency (if not done in elementary)              │
│  Grade 7: Year 4 - Innovation & Creativity                              │
│  Grade 8: Year 5 - Health & Wellness                                    │
│                                                                         │
│  HIGH SCHOOL                                                            │
│  ───────────                                                            │
│  Grade 9-10: Year 6 - Financial Literacy                                │
│  Grade 11-12: Year 7 - Environmental Stewardship                        │
│                                                                         │
│  ADULT EDUCATION                                                        │
│  ────────────────                                                       │
│  Any Year - Self-selected based on needs                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pricing Tiers

| Tier | Access | Price | Ideal For |
|------|--------|-------|-----------|
| **Individual** | 1 year program | $4.99/month | Individual learners |
| **Family** | All years, 5 accounts | $9.99/month | Families |
| **Classroom** | 1 year, 35 students | $99/year | Single teacher |
| **School** | All years, unlimited | $999/year | K-12 schools |
| **District** | All years, all schools | Custom | Districts |

### Teacher Earnings Integration

| Action | Teacher Earns | Notes |
|--------|---------------|-------|
| Student completes lesson | 1¢ | Passive income from teaching |
| Referral → subscription | 10-25% | Commission tier based on usage |
| Content contribution | Revenue share | If teacher creates lessons |
| School adoption | Bonus | Referral bonus for institutional sale |

---

## Planning & Discovery

### Annual Calendar View

Schools can see the full year at a glance:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  YEAR 2: AI FLUENCY - 2026-2027 SCHOOL YEAR                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  AUGUST 2026                                                            │
│  Pre-planning: Review Year 2 overview, download teacher guides          │
│                                                                         │
│  SEPTEMBER 2026 - Foundations (Days 1-30)                               │
│  ┌────┬────┬────┬────┬────┬────┬────┐                                   │
│  │ M  │ T  │ W  │ T  │ F  │ S  │ S  │                                   │
│  ├────┼────┼────┼────┼────┼────┼────┤                                   │
│  │  1 │  2 │  3 │  4 │  5 │  6 │  7 │ ← Week 1: What is AI?            │
│  │  8 │  9 │ 10 │ 11 │ 12 │ 13 │ 14 │ ← Week 2: AI Capabilities        │
│  │ 15 │ 16 │ 17 │ 18 │ 19 │ 20 │ 21 │ ← Week 3: Learning Foundations   │
│  │ 22 │ 23 │ 24 │ 25 │ 26 │ 27 │ 28 │ ← Week 4: Your Learning Style    │
│  │ 29 │ 30 │    │    │    │    │    │                                   │
│  └────┴────┴────┴────┴────┴────┴────┘                                   │
│                                                                         │
│  OCTOBER 2026 - Questioning (Days 31-61)                                │
│  ... (continues through May/June)                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Flexibility Options

| Model | Description | Days/Week |
|-------|-------------|-----------|
| **Daily** | One lesson per school day | 5 |
| **Intensive** | One lesson per day including weekends | 7 |
| **Weekly** | One deep lesson per week | 1 |
| **Self-Paced** | Student-driven, no schedule | Variable |

For a 180-day school year at 5 days/week:
- **Option A:** Compress Year 1 to school year (skip weekends)
- **Option B:** Spread Year 1 across 1.5 school years
- **Option C:** Do 3 lessons/week, complete in ~2 years

---

## Content Versioning

### Living Content Model

```
YEAR 1: FOUNDATIONS
├── 2024 Edition (v1.0) ← Original release
├── 2025 Edition (v1.1) ← Updated facts, improved scripts
├── 2026 Edition (v1.2) ← Community feedback integrated
└── 2027 Edition (v2.0) ← Major refresh based on usage data

Content improves each cycle:
- Usage analytics identify weak lessons
- Teacher feedback improves pedagogy
- Fact-checking updates outdated claims
- A/B testing optimizes engagement
```

### Backward Compatibility

- Old URLs still work (redirect to current version)
- Schools can lock to specific edition if needed
- All changes documented in changelog
- Major changes require opt-in for institutions

---

## Discovery & Browse

### For Learners

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BROWSE PROGRAMS                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  🌟 YEAR 1: Foundations of Knowledge                                    │
│  ─────────────────────────────────────                                  │
│  The building blocks: science, history, art, and emotional              │
│  intelligence. Perfect starting point for all ages.                     │
│  [365 lessons] [Ages 2-102] [8 min/day]                                 │
│  [START YEAR 1]                                                         │
│                                                                         │
│  🤖 YEAR 2: AI Fluency & Meta-Learning                                  │
│  ────────────────────────────────────                                   │
│  How to learn in the age of AI. Critical thinking, verification,       │
│  and the skills that matter when AI is everywhere.                      │
│  [365 lessons] [Ages 6-102] [8 min/day]                                 │
│  [START YEAR 2]                                                         │
│                                                                         │
│  🌍 YEAR 3: Global Citizenship (Coming 2027)                            │
│  ────────────────────────────────────────────                           │
│  World cultures, geography, languages, and global issues.               │
│  [NOTIFY ME]                                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### For Educators

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EDUCATOR RESOURCES                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  📅 CURRICULUM CALENDARS                                                │
│  Download full-year planning documents                                  │
│  • Year 1 Calendar (PDF)                                                │
│  • Year 2 Calendar (PDF)                                                │
│  • Cross-Curricular Alignment Matrix                                    │
│                                                                         │
│  📋 TEACHER GUIDES                                                      │
│  5-minute prep guides for every lesson                                  │
│  • Download All Guides (ZIP)                                            │
│  • Browse by Month                                                      │
│  • Search by Topic                                                      │
│                                                                         │
│  📊 ASSESSMENT RESOURCES                                                │
│  • Rubrics for each month's theme                                       │
│  • Portfolio templates                                                   │
│  • Progress tracking tools                                              │
│                                                                         │
│  💰 EARN-TO-TEACH PROGRAM                                               │
│  Earn income while teaching with Kelly                                  │
│  • Learn More                                                           │
│  • View Your Earnings                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Timeline

### Phase 1: Complete Year 2 (Q1 2026)

- [ ] Complete all 365 topics (12 month files) ✅ January done
- [ ] Generate lesson files for February-December
- [ ] Create teacher guides for each month
- [ ] Build Year 2 calendar view
- [ ] Test with pilot teachers

### Phase 2: Multi-Year Infrastructure (Q2 2026)

- [ ] Database schema for multi-program
- [ ] URL routing for year/day
- [ ] Program selection UI
- [ ] Educator portal MVP
- [ ] Institutional pricing integration

### Phase 3: Launch (Q3 2026)

- [ ] Year 2 public launch
- [ ] School outreach campaign
- [ ] Teacher earnings program expansion
- [ ] Year 3 planning begins

---

## Connection to Existing Systems

### Calendar System Integration

The existing `365_day_calendar.json` and `generate_unified_calendar.py` work with Year 1. For Year 2:

1. Create parallel `year2_365_day_calendar.json`
2. Update generator to accept year parameter
3. Calendar UI shows year selector
4. Each year has independent but parallel structure

### Lesson Player Integration

The `KellyOS` class in `app.js` needs:

```javascript
// Add program selection
this.state = {
  ...existingState,
  program: 'year1-foundations',  // or 'year2-ai-fluency'
  programYear: 1
};

// Fetch lesson based on program
async fetchDailyLesson(dayNumber = 1, programYear = 1) {
  const program = programYear === 1 ? 'core_lessons' : 'year2_lessons';
  // ... existing fetch logic with table selection
}
```

### Supabase Schema Addition

```sql
-- Programs table
CREATE TABLE programs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  year_number INTEGER NOT NULL UNIQUE,
  name VARCHAR(100) NOT NULL,
  slug VARCHAR(50) NOT NULL UNIQUE,
  description TEXT,
  status VARCHAR(20) DEFAULT 'draft',
  created_at TIMESTAMP DEFAULT NOW()
);

-- Link lessons to programs
ALTER TABLE core_lessons ADD COLUMN program_id UUID REFERENCES programs(id);

-- Insert programs
INSERT INTO programs (year_number, name, slug, status) VALUES
(1, 'Foundations of Knowledge', 'year1-foundations', 'active'),
(2, 'AI Fluency & Meta-Learning', 'year2-ai-fluency', 'building');
```

---

## Success Metrics

### Per-Year Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Lesson completion rate | 85%+ | Analytics |
| Teacher satisfaction | 4.5/5 | Survey |
| Student engagement | 90%+ | Time on task |
| Recall accuracy (7-day) | 50%+ | Recall prompts |
| Institutional adoption | 100 schools Year 1 | Sales |

### Cross-Year Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Year 1 → Year 2 progression | 60%+ | User journey |
| Multi-year retention | 70%+ Year 2 | Cohort analysis |
| Educator advocacy (NPS) | 70+ | Survey |
| Revenue per learner | $50+/year | Finance |

---

## Conclusion

The multi-year architecture transforms Curious Kelly from a product into a **platform**—a complete educational infrastructure that:

1. **Teachers** can adopt for whole school years with zero content creation
2. **Schools** can plan curriculum across grade levels
3. **Governments** can mandate as baseline digital/AI literacy
4. **Learners** can progress through over years of growth
5. **Assets** compound in quality year over year

Each 365-day year is a complete, polished, pre-computed journey. Multiple years create pathways. The platform grows more valuable as it grows larger.

---

*Document created: December 16, 2025*
*Version: 1.0*
*Contact: hello@curiouskelly.com*
