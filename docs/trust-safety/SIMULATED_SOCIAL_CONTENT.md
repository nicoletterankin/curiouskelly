# SIMULATED SOCIAL CONTENT

> *"The social experience is simulated. The learning is real."*

---

## What We Simulate

### 1. Peer Learner Comments
**What**: Comments that appear to be from other learners during lessons  
**Example**: "I never thought about it that way!" — Emma, 34  
**Purpose**: Creates sense of shared learning experience  
**Marking**: ✨ icon + "Simulated learner" on hover/tap

### 2. Age-Perspective Responses
**What**: How different age groups might respond to the same content  
**Example**: "My grandkids explained this to me last week!" — Simulated, 70s  
**Purpose**: Shows that learning spans generations, normalizes asking questions  
**Marking**: ✨ icon + "Simulated perspective"

### 3. Questions Other Learners "Asked"
**What**: "Other learners asked: Why does the sky look red at sunset?"  
**Purpose**: Normalizes curiosity, surfaces common questions  
**Marking**: ✨ icon + "Common question (simulated)"

### 4. Learning Journey Milestones
**What**: "Learners like you typically feel confused here—that's normal!"  
**Purpose**: Normalizes struggle, provides emotional support  
**Marking**: ✨ icon + "Based on learning patterns"

### 5. Discussion Prompts
**What**: Simulated discussion threads showing diverse viewpoints  
**Purpose**: Models productive disagreement, shows multiple perspectives  
**Marking**: ✨ icon + "Simulated discussion"

---

## What We DON'T Simulate

### Real Data Only
- Actual number of learners (when we have real data)
- Actual completion rates (when measured)
- Actual user testimonials (when given with permission)
- Actual community posts (when we have real community)

### Never Fake
- Reviews or ratings
- Press coverage
- Award wins
- User testimonials presented as real

---

## Visual Marking System

### Primary Indicator: ✨ Sparkle Icon

The sparkle (✨) appears next to all simulated social content. It was chosen because:
- It matches our brand (Curious Kelly uses ✨)
- It's gentle, not alarming
- It suggests "magic" not "fake"
- It's accessible and recognizable

### Secondary Indicator: Tooltip/Label

On hover (desktop) or tap (mobile):
```
┌─────────────────────────────────────┐
│ ✨ Simulated Learner                │
│                                     │
│ This comment was created to show    │
│ diverse learning perspectives.      │
│ Kelly simulates social interaction  │
│ to support your learning journey.   │
│                                     │
│ [Turn off simulated content]        │
└─────────────────────────────────────┘
```

### Tertiary Indicator: Settings Page

Clear explanation in Settings:
```
SIMULATED SOCIAL CONTENT
━━━━━━━━━━━━━━━━━━━━━━━━

Kelly shows comments and reactions from 
simulated learners to create a supportive 
social learning environment.

This content is AI-generated and marked 
with ✨. No real user data is shown.

[Toggle: ON/OFF]

Why we do this →
```

---

## Technical Implementation

### Content Schema

```json
{
  "type": "social_content",
  "is_simulated": true,
  "simulation_type": "peer_comment",
  "display_name": "Emma",
  "display_age": 34,
  "display_location": null,
  "content": "I never thought about it that way!",
  "educational_purpose": "normalize_insight_moments",
  "disclosure": {
    "icon": "✨",
    "label": "Simulated learner",
    "tooltip": "This comment was created to show diverse learning perspectives."
  }
}
```

### CSS Classes

```css
/* Simulated content wrapper */
.simulated-content {
  position: relative;
}

/* Sparkle indicator */
.simulated-indicator {
  position: absolute;
  top: 4px;
  right: 4px;
  font-size: 0.8rem;
  opacity: 0.8;
}

/* Hover state */
.simulated-content:hover .simulated-tooltip {
  display: block;
}

/* Reduced opacity when user prefers */
.simulated-reduced .simulated-content {
  opacity: 0.6;
}

/* Hidden when user disables */
.simulated-hidden .simulated-content {
  display: none;
}
```

### User Preference Storage

```javascript
// Simulated content preferences
const simulatedContentPrefs = {
  enabled: true,           // Master toggle
  showIndicators: true,    // Show ✨ icons
  showTooltips: true,      // Show explanatory tooltips
  types: {
    peerComments: true,    // Individual type controls
    ageResponses: true,
    questions: true,
    milestones: true,
    discussions: true
  }
};
```

---

## Content Guidelines

### Simulated Comments Must:

1. **Be educational** — Every comment teaches something or normalizes learning
2. **Show diversity** — Ages, backgrounds, learning styles
3. **Model growth mindset** — Show struggle, questions, confusion as normal
4. **Be kind** — No negativity, criticism, or discouragement
5. **Be realistic** — Sound like real humans, not AI-perfect
6. **Include mistakes** — Typos, informal language, personality

### Simulated Comments Must NOT:

1. **Manipulate emotions** — No guilt, fear, or FOMO
2. **Create competition** — No "fastest learner" or rankings
3. **Pressure purchases** — No "I upgraded and it's amazing!"
4. **Be parasocial** — No "I feel like Kelly is my friend"
5. **Reference real people** — No celebrities, public figures
6. **Be political/religious** — Neutral content only

### Example Good vs. Bad

✅ **Good**: "Wait, I'm confused about the gravity part. Can you explain again?" — Marcus, 16  
❌ **Bad**: "This is SO easy, I can't believe anyone would struggle with this!"

✅ **Good**: "My grandson helped me understand this! Never too old to learn." — Simulated, 70s  
❌ **Bad**: "Everyone in my family uses Kelly now. You should tell your family too!"

✅ **Good**: "I've watched this three times and finally got it. Don't give up!"  
❌ **Bad**: "Only 5% of learners understand this. Are you in the top 5%?"

---

## Age-Specific Considerations

### For Young Learners (2-12)

- Simulated peers are clearly marked as "Kelly's learning friends"
- Names are obviously fictional (Sunny, Max, Luna)
- No age display for simulated comments
- Extra-clear disclosure for parents in Settings

### For Teens (13-17)

- Realistic names but clearly marked
- Shows diverse teen perspectives
- Models healthy social learning (vs. social media comparison)
- Extra prominent controls in Settings

### For Adults (18+)

- Full adult personas with age ranges
- Professional and life-stage diversity
- Shows learning is lifelong
- Standard disclosure

### For Seniors (55+)

- Intergenerational comments prominent
- Shows tech learning is normal
- Emphasizes "never too late"
- Larger disclosure text for accessibility

---

## Ethical Review Process

### Before Deploying New Simulated Content:

1. **Purpose Check**: Does this serve learning, not engagement?
2. **Manipulation Check**: Could this create pressure, FOMO, or anxiety?
3. **Disclosure Check**: Is it clearly marked?
4. **Diversity Check**: Does it represent diverse perspectives?
5. **Growth Mindset Check**: Does it normalize struggle?
6. **Exit Check**: Can users easily turn it off?

### Quarterly Review

- Review all simulated content types
- Check user feedback about simulated content
- Verify disclosure systems are working
- Update guidelines based on learnings

---

## Metrics We Track

### Health Metrics (Good)
- % of users who understand content is simulated (survey)
- User satisfaction with social features
- Learning outcomes with/without simulated content

### Warning Metrics (Bad)
- Users reporting they thought content was real
- Users feeling deceived
- Complaints about simulated content

---

## FAQ

**Q: Why simulate at all? Why not wait for real users?**  
A: Social learning is psychologically necessary. Learners need to feel they're not alone. We simulate responsibly while building real community.

**Q: Isn't this just lying?**  
A: No. Lying is deception. We fully disclose that content is simulated. Every piece is marked. Users can turn it off. That's transparency, not deception.

**Q: What about when we have real users?**  
A: Real user content will be clearly marked as real. We'll never mix real and simulated without clear labels.

**Q: Could simulated content be addictive?**  
A: We specifically design against addiction: no variable rewards, no notifications, no social comparison, no engagement optimization.

---

*Last updated: December 2025*  
*Document owner: Trust & Safety*  
*Contact: hello@curiouskelly.com*





