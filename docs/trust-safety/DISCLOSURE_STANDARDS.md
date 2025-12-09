# DISCLOSURE STANDARDS

> *"If it's simulated, it's marked. Always. Everywhere. No exceptions."*

---

## The Standard

**Every piece of AI-generated social content must be visually marked with:**

1. A primary indicator (✨ sparkle)
2. An accessible tooltip/label on interaction
3. A path to learn more
4. A path to settings/controls

---

## Visual Hierarchy

### Level 1: Primary Indicator

The ✨ sparkle appears on ALL simulated content, ALL the time.

**Placement:**
- After the author name
- In the top-right corner of the content block
- Consistent across all platforms

**Styling:**
```css
.simulated-indicator {
  font-size: 0.9em;
  opacity: 0.85;
  cursor: help;
  user-select: none;
}
```

**Example:**
```
"I love how this connects to yesterday's lesson!" — Maya ✨
```

---

### Level 2: Tooltip (On Interaction)

When user hovers (desktop) or taps (mobile) the ✨:

```
┌─────────────────────────────────────────┐
│  ✨ Simulated Learner                   │
│                                         │
│  This comment was created by Kelly to   │
│  show diverse learning perspectives.    │
│                                         │
│  • All simulated content is marked      │
│  • You can turn this off in Settings    │
│                                         │
│  [Learn more]        [Go to Settings]   │
└─────────────────────────────────────────┘
```

**Requirements:**
- Appears within 100ms of interaction
- Disappears on click outside or Escape key
- Contains explanation, not just label
- Links to more info AND settings

---

### Level 3: Learn More Page

Linked from tooltip, accessible from Settings:

```
✨ ABOUT SIMULATED CONTENT

Why Kelly Shows Simulated Learners
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Learning is social. Humans evolved to learn by watching 
others, asking questions together, and feeling part of 
a community.

Kelly shows AI-generated comments from simulated learners 
to give you that sense of shared learning—without the 
harmful effects of social media.

What Makes This Different From Social Media
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Social Media:          Kelly:
• Hidden algorithms    • Everything marked
• Optimized for        • Optimized for 
  addiction              learning
• No user control      • Full control in
                         Settings
• Comparison &         • Growth mindset
  competition
• Variable rewards     • Predictable, safe

Your Controls
━━━━━━━━━━━━━

You're in charge:

• Turn off all simulated content [Settings →]
• Choose which types you see
• Choose how prominently it's marked

Questions?
━━━━━━━━━━

Contact us: hello@curiouskelly.com
Read our Trust & Safety policy: [Link]
```

---

## Context-Specific Disclosures

### In Lessons

```
┌──────────────────────────────────────────────────────┐
│  💬 WHAT OTHER LEARNERS SAID                         │
│  ─────────────────────────────────────────────────── │
│                                                      │
│  "Wait, I'm confused. How does gravity work          │
│   in space if there's no air?"                       │
│                                    — Jordan, 14 ✨   │
│                                                      │
│  "My teacher explained this differently. Both        │
│   ways make sense to me now!"                        │
│                                    — Priya, 28 ✨    │
│                                                      │
│  ─────────────────────────────────────────────────── │
│  ✨ These are simulated learners. [Learn more]       │
└──────────────────────────────────────────────────────┘
```

### In Discussions

```
┌──────────────────────────────────────────────────────┐
│  ✨ SIMULATED DISCUSSION                             │
│  This conversation was created to show different     │
│  perspectives on today's topic.                      │
│  ─────────────────────────────────────────────────── │
│                                                      │
│  [Discussion content...]                             │
│                                                      │
│  ─────────────────────────────────────────────────── │
│  Want to see real discussions? Join our community    │
│  when it launches. [Get notified]                    │
└──────────────────────────────────────────────────────┘
```

### For Children (Ages 2-12)

```
┌──────────────────────────────────────────────────────┐
│  🌟 KELLY'S LEARNING FRIENDS                         │
│                                                      │
│  "I like asking questions too!"                      │
│                                    — Sunny 🌟        │
│                                                      │
│  ─────────────────────────────────────────────────── │
│  Sunny is one of Kelly's pretend learning friends!   │
│  Kelly made them up to help you feel less alone.     │
└──────────────────────────────────────────────────────┘
```

For parents in Settings:
```
KELLY'S LEARNING FRIENDS
━━━━━━━━━━━━━━━━━━━━━━━━

Your child sees AI-generated "learning friends" 
with names like Sunny, Max, and Luna.

These characters:
✓ Model asking questions
✓ Show that confusion is normal
✓ Are clearly marked as pretend
✓ Can be turned off below

[Toggle: ON / OFF]
```

---

## Technical Implementation

### HTML Structure

```html
<div class="simulated-content" 
     data-simulated="true"
     data-type="peer-comment">
  
  <p class="content-text">
    "I love how this connects to yesterday's lesson!"
  </p>
  
  <footer class="content-meta">
    <span class="author-name">Maya</span>
    <button class="simulated-indicator" 
            aria-label="This is simulated content. Click to learn more."
            aria-expanded="false"
            aria-controls="disclosure-tooltip-123">
      ✨
    </button>
  </footer>
  
  <div id="disclosure-tooltip-123" 
       class="disclosure-tooltip" 
       role="tooltip"
       hidden>
    <h4>✨ Simulated Learner</h4>
    <p>This comment was created by Kelly to show diverse learning perspectives.</p>
    <div class="tooltip-actions">
      <a href="/about/simulated-content">Learn more</a>
      <a href="/settings/social">Settings</a>
    </div>
  </div>
</div>
```

### CSS

```css
/* Simulated content wrapper */
.simulated-content {
  position: relative;
}

/* Sparkle indicator */
.simulated-indicator {
  background: none;
  border: none;
  cursor: help;
  font-size: 0.9em;
  opacity: 0.85;
  padding: 2px 4px;
  margin-left: 4px;
  border-radius: 4px;
  transition: opacity 0.2s, background 0.2s;
}

.simulated-indicator:hover,
.simulated-indicator:focus {
  opacity: 1;
  background: rgba(255, 255, 255, 0.1);
}

/* Tooltip */
.disclosure-tooltip {
  position: absolute;
  bottom: 100%;
  right: 0;
  background: var(--bg-elevated);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 16px;
  width: 280px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  z-index: 100;
}

.disclosure-tooltip[hidden] {
  display: none;
}

/* Enhanced disclosure mode */
.disclosure-enhanced .simulated-content::before {
  content: "✨ Simulated";
  font-size: 0.75rem;
  color: var(--text-muted);
  display: block;
  margin-bottom: 4px;
}

/* Maximum disclosure mode */
.disclosure-maximum .simulated-content {
  border: 1px dashed var(--kelly-gold);
  padding: 12px;
  border-radius: 8px;
  background: rgba(245, 158, 11, 0.05);
}

.disclosure-maximum .simulated-content::before {
  content: "✨ SIMULATED CONTENT - This is AI-generated";
  font-size: 0.8rem;
  color: var(--kelly-gold);
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
}
```

### JavaScript

```javascript
class SimulatedContentDisclosure {
  constructor() {
    this.initTooltips();
    this.loadUserPreferences();
  }
  
  initTooltips() {
    document.querySelectorAll('.simulated-indicator').forEach(btn => {
      btn.addEventListener('click', (e) => this.toggleTooltip(e.target));
      btn.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          this.toggleTooltip(e.target);
        }
      });
    });
    
    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.simulated-content')) {
        this.closeAllTooltips();
      }
    });
    
    // Close on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.closeAllTooltips();
      }
    });
  }
  
  toggleTooltip(button) {
    const tooltipId = button.getAttribute('aria-controls');
    const tooltip = document.getElementById(tooltipId);
    const isExpanded = button.getAttribute('aria-expanded') === 'true';
    
    this.closeAllTooltips();
    
    if (!isExpanded) {
      tooltip.hidden = false;
      button.setAttribute('aria-expanded', 'true');
    }
  }
  
  closeAllTooltips() {
    document.querySelectorAll('.disclosure-tooltip').forEach(t => t.hidden = true);
    document.querySelectorAll('.simulated-indicator').forEach(b => {
      b.setAttribute('aria-expanded', 'false');
    });
  }
  
  loadUserPreferences() {
    const prefs = JSON.parse(localStorage.getItem('simulatedContentPrefs') || '{}');
    
    // Apply disclosure mode
    document.body.classList.remove('disclosure-standard', 'disclosure-enhanced', 'disclosure-maximum');
    document.body.classList.add(`disclosure-${prefs.disclosureMode || 'standard'}`);
    
    // Hide content if disabled
    if (prefs.enabled === false) {
      document.body.classList.add('simulated-hidden');
    }
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  new SimulatedContentDisclosure();
});
```

---

## Accessibility Requirements

### Screen Readers

- ✨ icon has `aria-label` explaining it's simulated
- Tooltip content is announced when opened
- Focus management returns to trigger after tooltip closes

### Keyboard Navigation

- Tab reaches ✨ indicator
- Enter/Space opens tooltip
- Tab navigates within tooltip
- Escape closes tooltip
- Focus trap within tooltip when open

### Visual Accessibility

- ✨ icon has sufficient contrast
- Tooltip text meets WCAG AA contrast
- Focus states are visible
- Works with high contrast mode

### Cognitive Accessibility

- Language is simple and clear
- Explanation is brief
- Actions are obvious
- Consistent placement everywhere

---

## Testing Requirements

### Before Launch

- [ ] ✨ appears on ALL simulated content
- [ ] Tooltip opens on hover/tap/keyboard
- [ ] Tooltip contains explanation
- [ ] Tooltip links to learn more AND settings
- [ ] Settings toggle works
- [ ] User preference persists
- [ ] Screen reader announces correctly
- [ ] Works on mobile
- [ ] Works in all themes/modes

### Ongoing

- [ ] Quarterly audit of all simulated content for marking
- [ ] User comprehension survey (do they understand?)
- [ ] Accessibility audit annually

---

## Edge Cases

### When Content is Both Real and Simulated

If we ever mix real user content with simulated content:

```
┌──────────────────────────────────────────────────────┐
│  💬 FROM THE COMMUNITY                               │
│  ─────────────────────────────────────────────────── │
│                                                      │
│  "Great lesson today!"                               │
│                      — @RealUser123 ✓ (Verified)     │
│                                                      │
│  "I had the same question!"                          │
│                      — Alex ✨ (Simulated)            │
│                                                      │
└──────────────────────────────────────────────────────┘
```

- Real users get ✓ verified badge
- Simulated users get ✨ and "(Simulated)" label
- NEVER ambiguous

### When Disclosure Breaks

If disclosure system fails to load:

1. Simulated content should NOT display
2. Error logged for immediate fix
3. Fallback to server-rendered disclosure if JS fails

```javascript
// Fail-safe: if disclosure can't init, hide simulated content
try {
  new SimulatedContentDisclosure();
} catch (e) {
  console.error('Disclosure system failed:', e);
  document.body.classList.add('simulated-hidden');
  // Alert T&S team
  reportDisclosureFailure(e);
}
```

---

*"Disclosure isn't a feature. It's a promise."*

---

*Last updated: December 2025*  
*Document owner: Trust & Safety*  
*Contact: hello@curiouskelly.com*




