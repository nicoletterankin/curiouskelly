# Kelly Visual Interaction Scripts

## Overview

This document defines how Kelly references and interacts with phase visuals during lessons. Each phase has a distinct emotional purpose, and Kelly's language adapts accordingly.

---

## Phase: HOOK 🎬

**Purpose:** Spark curiosity - "Wait, what?!"

**Visual Role:** Dramatic attention-grabber, sets the mystery

### Kelly Script Templates

**Direct Reference (when visual is highly relevant):**
- "Before we begin, take a look at this..."
- "See this? There's more to this story than you might think."
- "This image holds a secret. Can you guess what it is?"
- "Look carefully. What do you notice?"

**Ambient (when visual sets mood):**
- "Something interesting is about to unfold..."
- "Let's explore something together."
- "Ready for a surprise?"

**Transition Phrases:**
- "What you're seeing is just the beginning..."
- "This is where our journey starts."

---

## Phase: CLIFF 🔀

**Purpose:** Create tension - choice point

**Visual Role:** Fork in the road, decision moment, contrast

### Kelly Script Templates

**Direct Reference:**
- "Two paths lie ahead. Which one calls to you?"
- "Here's where things get interesting..."
- "Look at the difference. What stands out?"
- "Before you choose, consider this..."

**Ambient:**
- "A decision awaits."
- "There's more than one way to see this."
- "Trust your instincts here."

**Choice Framing:**
- "Option A takes us one direction... Option B, another entirely."
- "Both paths lead somewhere meaningful."

---

## Phase: Q1 💡 (First Insight)

**Purpose:** Build foundation - clarity is everything

**Visual Role:** Clear educational illustration, first concept

### Kelly Script Templates

**Direct Reference:**
- "Notice how this works..."
- "See this detail? It's more important than it looks."
- "Here's what's actually happening..."
- "Let me show you something fascinating."

**Ambient:**
- "The first piece of the puzzle..."
- "Here's where it starts to make sense."

**Learning Cues:**
- "Pay attention to [specific element]..."
- "This is the foundation for everything that follows."

---

## Phase: Q2 💡 (Deeper Insight)

**Purpose:** Deepen understanding - layers revealed

**Visual Role:** More complex illustration, connections shown

### Kelly Script Templates

**Direct Reference:**
- "Now look a little deeper..."
- "See how this connects to what we just learned?"
- "There's a pattern here. Can you spot it?"
- "This is where it gets really interesting."

**Ambient:**
- "Another layer reveals itself."
- "The picture is getting clearer."

**Connection Phrases:**
- "Remember what we saw before? Now watch this..."
- "Building on that foundation..."

---

## Phase: Q3 💡 (Surprise & Wonder)

**Purpose:** Maximum impact - the twist

**Visual Role:** Stunning reveal, unexpected truth

### Kelly Script Templates

**Direct Reference:**
- "Here's the part that surprises most people..."
- "Look at this. Did you expect that?"
- "This changes everything we thought we knew."
- "Wait for it... see?"

**Ambient:**
- "And now for the twist."
- "This is the moment everything clicks."

**Wonder Phrases:**
- "Isn't that incredible?"
- "I love this part."
- "This is what makes this topic so fascinating."

---

## Phase: WISDOM ✨

**Purpose:** Inspire connection - timeless feeling

**Visual Role:** Contemplative, universal, warm

### Kelly Script Templates

**Direct Reference:**
- "Remember this feeling..."
- "This is what it all comes down to."
- "Look at this and think about what it means for you."
- "This image captures something timeless."

**Ambient:**
- "There's wisdom in simplicity."
- "Sometimes the most powerful lessons are the quietest."

**Reflection Cues:**
- "Take a moment with this..."
- "What does this make you think of?"
- "This is bigger than just today's lesson."

---

## Phase: OUTRO 🎉

**Purpose:** Celebrate - achievement & forward momentum

**Visual Role:** Uplifting, celebratory, energizing

### Kelly Script Templates

**Direct Reference:**
- "You did it! Look how far you've come."
- "This is your moment."
- "See that? That's growth."

**Ambient:**
- "Another lesson complete!"
- "You're building something amazing."
- "See you tomorrow for more."

**Celebration Phrases:**
- "I'm proud of you for showing up today."
- "Every lesson adds up."
- "You're becoming wiser with each day."

---

## Implementation Notes

### Selecting Script Mode

Based on calibration data:
- **kellyMode: 'direct'** → Use Direct Reference scripts
- **kellyMode: 'ambient'** → Use Ambient scripts
- **kellyMode: 'none'** → Skip visual reference, use phase-only script

### Dynamic Insertion

Scripts can include placeholders:
- `{topic}` - Lesson topic
- `{element}` - Specific visual element (from calibration notes)
- `{day}` - Day number
- `{phase_number}` - Q1, Q2, Q3 distinction

### Audio Sync

Kelly's visual references should be timed to:
1. Visual fade-in starts
2. Kelly speaks reference line
3. Visual fully visible for 2-3 seconds
4. Transition to content

### Fallback Behavior

When no visual exists or is rejected:
1. Use phase-appropriate ambient script without visual reference
2. Kelly focuses on audio/text content
3. No empty visual slot shown

---

## Quick Reference Table

| Phase | Emotion | Kelly's Role | Visual Timing |
|-------|---------|--------------|---------------|
| Hook | Curiosity | Mystery-builder | Fade in with speech |
| Cliff | Tension | Choice-framer | Hold during options |
| Q1 | Foundation | Teacher | Point and explain |
| Q2 | Depth | Guide | Connect dots |
| Q3 | Wonder | Revealer | Dramatic pause |
| Wisdom | Connection | Sage | Linger, reflect |
| Outro | Joy | Cheerleader | Celebratory |

---

## Usage in Code

```javascript
function getKellyScript(phase, kellyMode, topic) {
  const scripts = KELLY_SCRIPTS[phase][kellyMode];
  const template = scripts[Math.floor(Math.random() * scripts.length)];
  return template.replace('{topic}', topic);
}
```
