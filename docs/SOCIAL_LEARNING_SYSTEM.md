# Social Learning System - Complete Guide

> *"The social experience is simulated. The learning is real."*

---

## Philosophy

**Why we simulate social learning:**

Humans evolved to learn together. We watch others, ask questions, feel belonging. This isn't optional—it's how our brains work.

Social media exploited this need with addictive, unpredictable feedback loops. Kelly offers something different: **predictable, safe, growth-oriented social mirrors**.

**What makes Kelly different:**

| Social Media | Curious Kelly |
|--------------|---------------|
| Hidden algorithms | Transparent simulation |
| Variable rewards (addictive) | Predictable content |
| Engagement optimization | Learning optimization |
| No user control | Full user control |
| Status competition | Growth mindset focus |
| Deceptive (bots as real) | Every comment marked ✨ |

---

## Comment Design Principles

### ✅ DO: Growth Mindset

**Normalize struggle:**
- "I had to replay that part"
- "Still processing this"
- "Confused but curious"
- "I don't fully get it yet"

**Show questions as good:**
- "Wait, can someone explain that again?"
- "Is that always true?"
- "But what about when...?"

**Celebrate small wins:**
- "That makes sense now"
- "I see the connection"
- "Good takeaway"

### ❌ DON'T: Hyperbole & Manipulation

**Never use:**
- "Mind = BLOWN 🤯"
- "Best teacher EVER!"
- "I'm literally addicted"
- "Game changer!"
- "This is EVERYTHING"
- "Kelly > my professors"

**Never manipulate:**
- No FOMO ("Only 5% understand this!")
- No guilt ("Don't give up like others!")
- No pressure ("Everyone's doing it!")

---

## Phase-Specific Comments

### Welcome Phase
*The lesson is starting. People are settling in.*

Good examples:
- "Morning everyone 👋"
- "Day 47, here we go"
- "Don't know much about {topic} yet"
- "Watching with my kids today"

### Hook Phase
*Kelly introduces the topic and key idea.*

Good examples:
- "I never thought of it that way"
- "Interesting framing"
- "This connects to something at work"
- "Not sure I follow yet" (normalize confusion!)

### Question Phases (Q1, Q2, Q3)
*Learners are thinking and choosing answers.*

Good examples:
- "Hmm, not sure about this one"
- "Both seem reasonable"
- "Going with my gut"
- "Changed my mind twice"
- "Anyone else unsure here?"

### Wisdom Phase
*Kelly shares the key insight/takeaway.*

Good examples:
- "That's helpful"
- "I'll remember that"
- "Good takeaway"
- "This applies to a lot of things"
- "Going to think about this more"

### Complete Phase
*The lesson is ending.*

Good examples:
- "Good lesson today"
- "See everyone tomorrow"
- "Day X done ✓"
- "Thanks, learned something"

---

## Persona System

### 60 Diverse Personas

We maintain 60 carefully crafted personas representing global diversity:

**Demographics:**
- Ages: 7-72 (children, teens, young adults, adults, seniors)
- Countries: 30+ nations across all continents
- Backgrounds: Students, teachers, professionals, retirees

**Each persona has:**
- Unique ID (e.g., `emma-us`, `yuki-jp`)
- Name appropriate to their culture
- Age and age group
- Country and flag
- Avatar image
- Bio/description
- Learning style

### Avatar Images

Avatars are stored at `/images/learners/[persona-id].jpg`

**Generation options:**
1. AI-generated (FLUX via Replicate)
2. Licensed stock photos
3. SVG placeholders (fallback)

**Requirements:**
- Diverse, authentic representation
- Friendly, approachable expressions
- Professional quality
- Square format (128x128 minimum)

---

## Technical Implementation

### Files

```
public/js/
├── learner-personas.js    # Persona data and helpers
├── social-comments.js     # Comment banks and generator
└── chat-overlay.js        # UI overlay (existing)

scripts/
├── generate_social_comments.py  # Claude-powered comment generation
└── generate_learner_avatars.py  # Avatar image generation

sql/
├── lesson_comments.sql    # Comments table
└── learner_personas.sql   # Personas table + enhanced schema
```

### Database Schema

```sql
-- Personas
CREATE TABLE learner_personas (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  age INT NOT NULL,
  country_code TEXT NOT NULL,
  country_flag TEXT NOT NULL,
  age_group TEXT NOT NULL,
  bio TEXT,
  learning_style TEXT,
  avatar_url TEXT
);

-- Comments (enhanced)
CREATE TABLE lesson_comments (
  id UUID PRIMARY KEY,
  lesson_day INT NOT NULL,
  phase TEXT NOT NULL,
  option_context TEXT,
  persona_id TEXT REFERENCES learner_personas(id),
  persona_name TEXT NOT NULL,
  persona_country TEXT NOT NULL,
  persona_flag TEXT NOT NULL,
  comment_text TEXT NOT NULL,
  comment_type TEXT NOT NULL,
  mood TEXT
);
```

### Comment Generation

```bash
# Generate for one lesson
python scripts/generate_social_comments.py --day 1

# Generate for a range
python scripts/generate_social_comments.py --range 1-30

# Generate for all 365 (takes time + API credits)
python scripts/generate_social_comments.py --all
```

### Avatar Generation

```bash
# Create placeholders for all personas
python scripts/generate_learner_avatars.py --placeholders

# Generate with AI (requires REPLICATE_API_TOKEN)
python scripts/generate_learner_avatars.py --all
```

---

## Trust & Safety Compliance

### Required Disclosures

1. **✨ Indicator**: Every simulated comment marked with sparkle
2. **Tooltip**: Tap badge to learn more
3. **Settings**: Master toggle to disable

### User Controls

```javascript
// Preferences stored in localStorage
kellySimulatedContentPrefs = {
  enabled: true,      // Master toggle
  showIndicators: true // Show ✨ marks
}
```

### Content Review Checklist

Before deploying new comments:

- [ ] Purpose Check: Does this serve learning, not engagement?
- [ ] Manipulation Check: Could this create pressure, FOMO, or anxiety?
- [ ] Hyperbole Check: No "mind blown", "best ever", etc.?
- [ ] Growth Mindset Check: Does it normalize struggle?
- [ ] Diversity Check: Does it represent diverse perspectives?
- [ ] Disclosure Check: Is it clearly marked?

---

## Metrics

### Health Metrics (Good)
- User satisfaction with social features
- Learning outcomes with/without simulated content
- % understanding content is simulated

### Warning Metrics (Bad)
- Users thinking content is real
- Complaints about simulated content
- Users feeling deceived

---

## Quick Start

1. **Run SQL migrations:**
   ```sql
   -- In Supabase SQL editor
   \i sql/learner_personas.sql
   ```

2. **Generate comments:**
   ```bash
   python scripts/generate_social_comments.py --range 1-30
   ```

3. **Create avatars:**
   ```bash
   python scripts/generate_learner_avatars.py --placeholders
   ```

4. **Test locally:**
   - Navigate to `/learn.html`
   - Comments should appear with ✨ indicator
   - Click badge to see disclosure

---

## Philosophy Summary

> **"Humble, clear, growth-mindset."**

Every comment should feel like it came from a thoughtful peer in a real classroom—someone who asks questions, admits confusion, celebrates understanding, and treats learning as a journey, not a performance.

No hyperbole. No manipulation. Just authentic social learning.

---

*Last updated: December 2025*
*Document owner: Trust & Safety*
*Contact: hello@curiouskelly.com*

