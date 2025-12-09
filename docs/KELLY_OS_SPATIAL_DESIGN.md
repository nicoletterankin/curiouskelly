# 🎯 KELLY OS - SPATIAL DESIGN & FEATURE MAPPING
## *The Complete Blueprint for Timeless, Seamless Learning*

---

## 📋 **EXECUTIVE SUMMARY**

This document maps EVERY feature in the Kelly OS learning desktop to its optimal spatial location, ensuring:
1. **Kelly's center stage is SACRED** - Nothing overlaps her face, body, or gestures
2. **Every feature has a HOME** - No floating, no overlapping, no chaos
3. **Mobile + Desktop harmony** - Responsive rules for every breakpoint
4. **20 years of work HONORED** - Every tool visible, accessible, celebrated

---

## 🎨 **CORE SPATIAL ZONES**

```
┌─────────────────────────────────────────────────────────────┐
│                    TOP: Phase Journey Bar                    │
├──────────────┬──────────────────────────┬───────────────────┤
│              │                          │                   │
│   LEFT       │      CENTER STAGE        │      RIGHT        │
│   RAIL       │      (KELLY ONLY)        │      RAIL         │
│              │                          │                   │
│  Community   │   🎬 Kelly Avatar        │  Personalization  │
│  Comments    │   (2D/3D/Video)          │  Controls         │
│              │                          │                   │
│  (Simulated  │   SACRED ZONE            │  🌐 Language      │
│   Social)    │   No overlays!           │  🎭 Tone          │
│              │                          │  👤 Age           │
│              │                          │  📺 Mode          │
│              │                          │  🔊 Sound         │
│              │                          │  ⚡ Speed         │
├──────────────┴──────────────────────────┴───────────────────┤
│                  BOTTOM: Lesson Content Bar                  │
│  📝 Question + Options (collapsible, slides up from bottom)  │
├──────────────────────────────────────────────────────────────┤
│                  BOTTOM-RIGHT: Action Dock                   │
│  💡 Aha  📌 Pin  ✨ Share  🎤 Talk  📅 Cal  🔍 Search  ⚙️   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚫 **THE SACRED CENTER RULE**

### **Kelly's Zone = NO OVERLAYS**

**What lives in the center:**
- Kelly's 2D avatar (default)
- Kelly's 3D Unity avatar (on-demand)
- Kelly's Model Viewer robot (experimental)
- Kelly's HD lipsync video (future)
- Background visuals (infographics, scenes)

**What is FORBIDDEN in the center:**
- ❌ Icon columns
- ❌ Text overlays (except captions at bottom edge)
- ❌ Buttons or controls
- ❌ Comments or social elements
- ❌ Navigation hints (except subtle edge indicators)
- ❌ Popover panels
- ❌ Modal dialogs (use side panels instead)

**Exception:** Temporary feedback (< 1 second):
- ✅ Tap feedback (ripple effect)
- ✅ Phase transition animations
- ✅ Pause indicator (⏸️ for 0.5s)

---

## 📍 **FEATURE INVENTORY & SPATIAL ASSIGNMENTS**

### **1. ENGAGEMENT FEATURES** (Bottom-Right Action Dock)

| Feature | Icon | Current State | New Home | Open Behavior |
|---------|------|---------------|----------|---------------|
| **Aha Moment** | 💡 | Right rail | Action Dock (pos 1) | Inline counter bump + confetti |
| **Pin to Journal** | 📌 | Right rail | Action Dock (pos 2) | Inline counter bump + save animation |
| **Share Lesson** | ✨ | Right rail | Action Dock (pos 3) | **Slide-up panel from bottom** (share options) |
| **Talk to Kelly** | 🎤 | Floating button | Action Dock (pos 4) | **Expands in-place** (voice waveform overlay) |

**Rules:**
- Action Dock is ALWAYS visible (fixed position)
- Icons are large (44px touch targets)
- Hover/tap shows tooltip above icon
- Active state: glow + scale(1.1)
- Share panel slides UP from bottom (doesn't cover Kelly)
- Talk to Kelly: voice waveform appears as bottom overlay (below Kelly's chest)

---

### **2. PERSONALIZATION FEATURES** (Right Rail Panel)

| Feature | Icon | Current State | New Home | Open Behavior |
|---------|------|---------------|----------|---------------|
| **Language** | 🌐 | Right rail | Right Rail (pos 1) | **Expand panel RIGHT** (language grid) |
| **Tone** | 🎭 | Right rail | Right Rail (pos 2) | **Expand panel RIGHT** (tone selector) |
| **Age** | 👤 | Right rail | Right Rail (pos 3) | **Expand panel RIGHT** (age slider) |
| **Mode** | 📺 | Right rail | Right Rail (pos 4) | **Expand panel RIGHT** (2D/3D/MV buttons) |
| **Sound** | 🔊 | Right rail | Right Rail (pos 5) | **Toggle inline** (no panel) |
| **Speed** | ⚡ | Hidden | Right Rail (pos 6) | **Expand panel RIGHT** (0.5x - 2x slider) |

**Rules:**
- Right Rail is ALWAYS visible (fixed position)
- Icons stack vertically with 18px gaps
- Expand panels slide OUT to the right (off Kelly's zone)
- Panel width: 280px (desktop), 240px (mobile)
- Panel background: frosted glass (rgba(255,255,255,0.95) + blur)
- Only ONE panel open at a time (others auto-close)
- Close: tap outside, tap icon again, or tap another icon

---

### **3. NAVIGATION FEATURES** (Action Dock Bottom)

| Feature | Icon | Current State | New Home | Open Behavior |
|---------|------|---------------|----------|---------------|
| **Calendar** | 📅 | Right rail | Action Dock (pos 5) | **Navigate to /calendar.html** |
| **Search** | 🔍 | Right rail | Action Dock (pos 6) | **Full-screen overlay** (dark, centered search) |
| **Settings/More** | ⚙️ | Right rail | Action Dock (pos 7) | **Slide-up panel from bottom** (settings menu) |

**Rules:**
- Calendar: direct link (no overlay)
- Search: full-screen dark overlay (95% opacity black) with centered search box
- Search results: vertical list, click = navigate to that day
- Settings: slide-up panel (like iOS Control Center) with all advanced settings

---

### **4. COMMUNITY FEATURES** (Left Rail Panel)

| Feature | Icon | Current State | New Home | Open Behavior |
|---------|------|---------------|----------|---------------|
| **Simulated Comments** | 💬 | Floating overlay | Left Rail (always visible) | **Scroll vertically** (no expand) |
| **Live Badge** | 🔴 | Top-left | Left Rail (top) | **Static indicator** (no expand) |

**Rules:**
- Left Rail is ALWAYS visible (fixed position)
- Comments scroll vertically (like TikTok)
- Each comment: frosted glass card with avatar, name, badge, text
- ✨ badge indicates simulated content (Trust & Safety)
- Comments auto-scroll slowly (5s per comment)
- User can manually scroll (pauses auto-scroll)
- No expand behavior (always visible)

---

### **5. LESSON CONTENT** (Bottom Content Bar)

| Feature | Current State | New Home | Open Behavior |
|---------|---------------|----------|---------------|
| **Phase Question** | Center overlay | Bottom Bar | **Slides UP from bottom** (default collapsed) |
| **Option Pills** | Center overlay | Bottom Bar | **Slides UP with question** |
| **Phase Card** | Left side | Bottom Bar | **Integrated with question** |
| **Caption Text** | Bottom overlay | Bottom Bar | **Always visible** (Kelly's current speech) |

**Rules:**
- Bottom Bar has TWO states:
  - **Collapsed** (default): Only caption visible (1 line, 60px height)
  - **Expanded** (on question phase): Question + options slide up (300px height)
- Transition: smooth 0.3s ease
- Background: frosted glass (doesn't block Kelly's feet)
- Options: horizontal pills (mobile) or vertical cards (desktop)
- Selected option: glow + checkmark
- After selection: auto-collapse after 1s

---

### **6. PHASE NAVIGATION** (Top Phase Journey Bar)

| Feature | Current State | New Home | Open Behavior |
|---------|---------------|----------|---------------|
| **5-Phase Journey** | Top bar | Top Bar (always visible) | **Click segment = jump to phase** |
| **Progress Dots** | Bottom | Integrated into Journey Bar | **Visual progress indicator** |

**Rules:**
- Top Bar is ALWAYS visible (fixed position, 80px height)
- Shows all 5 phases horizontally: 🎣 Hook → 💭 Fact 1 → 💡 Fact 2 → 🔗 Fact 3 → ✨ Wisdom
- Active phase: glowing circle + label
- Completed phases: filled circle
- Future phases: outline circle
- Click any phase: smooth transition (fade Kelly, load new phase, fade in)
- Mobile: smaller circles (32px), desktop: larger (48px)

---

### **7. ADVANCED TOOLS** (Hidden by Default, Accessible via Settings)

| Feature | Icon | Access Point | Open Behavior |
|---------|------|--------------|---------------|
| **Unity 3D Kelly** | 🎮 | Mode selector | **Loads in center stage** (replaces 2D) |
| **Model Viewer Robot** | 🤖 | Mode selector | **Loads in center stage** (replaces 2D) |
| **Expression Studio** | 🎨 | Settings → Advanced | **Full-screen modal** (dev tool) |
| **Video Player** | 🎬 | Auto-loads for HD lessons | **Replaces 2D avatar** (seamless) |
| **Earn to Learn** | 💰 | Settings → Earn | **Slide-up panel** (affiliate dashboard) |
| **Analytics** | 📊 | Settings → Progress | **Slide-up panel** (learning insights) |
| **Universal Access** | 🌍 | Settings → Accessibility | **Slide-up panel** (a11y tools) |
| **Keyboard Nav** | ⌨️ | Always active | **No UI** (background feature) |
| **Gesture Controls** | 👆 | Always active | **No UI** (background feature) |
| **Learning Journal** | 🎓 | Settings → Journal | **Navigate to /journal.html** |
| **Achievements** | 🏆 | Settings → Achievements | **Slide-up panel** (badges, streaks) |
| **Dark Mode** | 🌙 | Settings → Theme | **Toggle inline** (instant switch) |

**Rules:**
- Advanced tools are NOT visible by default (reduces cognitive load)
- Access via Settings (⚙️) → Advanced menu
- Each tool either:
  - Replaces center stage (avatar modes)
  - Opens as slide-up panel (dashboards, settings)
  - Navigates to dedicated page (journal, calendar)
  - Works in background (keyboard, gestures)

---

## 📱 **RESPONSIVE BEHAVIOR**

### **Desktop (> 1024px)**

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase Journey Bar (80px)                  │
├──────────────┬──────────────────────────┬───────────────────┤
│              │                          │                   │
│  Left Rail   │      Kelly Center        │   Right Rail      │
│  (280px)     │      (fluid)             │   (80px icons)    │
│              │                          │                   │
│  Comments    │      🎬 Avatar           │   Controls        │
│  (scroll)    │                          │   (vertical)      │
│              │                          │                   │
├──────────────┴──────────────────────────┴───────────────────┤
│              Bottom Content Bar (collapsed: 60px)            │
│              (expanded: 300px)                               │
├──────────────────────────────────────────────────────────────┤
│                    Action Dock (bottom-right, 60px)          │
└──────────────────────────────────────────────────────────────┘
```

**Rules:**
- Left Rail: 280px wide, always visible, comments scroll
- Right Rail: 80px wide, icons only, panels expand RIGHT (280px)
- Center: fluid width, Kelly scales proportionally
- Bottom Bar: full width minus rails
- Action Dock: bottom-right corner, 7 icons horizontal

---

### **Tablet (768px - 1024px)**

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase Journey Bar (60px)                  │
├──────────────────────────┬───────────────────────────────────┤
│                          │                                   │
│      Kelly Center        │   Right Rail (60px)               │
│      (fluid)             │                                   │
│                          │   Controls (vertical)             │
│      🎬 Avatar           │                                   │
│                          │                                   │
│                          │                                   │
├──────────────────────────┴───────────────────────────────────┤
│              Bottom Content Bar (collapsed: 50px)            │
├──────────────────────────────────────────────────────────────┤
│  Left: Comments (icon)   │   Right: Action Dock (icons)     │
└──────────────────────────────────────────────────────────────┘
```

**Rules:**
- Left Rail: HIDDEN by default, accessible via 💬 icon in bottom-left
- Right Rail: 60px wide, icons only
- Center: full width minus right rail
- Comments: tap 💬 icon → slide-in from left (overlay)
- Action Dock: bottom-right, 7 icons

---

### **Mobile (< 768px)**

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase Journey Bar (50px)                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                      Kelly Center                            │
│                      (full width)                            │
│                                                              │
│                      🎬 Avatar                               │
│                      (9:16 aspect)                           │
│                                                              │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│              Bottom Content Bar (collapsed: 44px)            │
├──────────────────────────────────────────────────────────────┤
│  💬 (left)   │   Caption Text   │   Action Dock (right)     │
└──────────────────────────────────────────────────────────────┘
```

**Rules:**
- **Full-screen Kelly** (TikTok-style)
- Left Rail: tap 💬 icon → full-screen overlay (dark, scrollable comments)
- Right Rail: tap ⚙️ icon → slide-up panel (all controls)
- Phase Journey: smaller circles (32px), tap to navigate
- Bottom Bar: collapsed by default, expands on question
- Action Dock: bottom-right, 4 primary icons (💡📌✨🎤), rest in ⚙️
- Swipe gestures:
  - Swipe LEFT: next phase
  - Swipe RIGHT: previous phase
  - Swipe UP: expand content bar
  - Swipe DOWN: collapse content bar
  - Double-tap: pause/play

---

## 🎯 **INTERACTION PATTERNS**

### **1. Kelly's Gestures = UI Cues**

| Kelly's Gesture | UI Response | Timing |
|-----------------|-------------|--------|
| **Pointing LEFT** | Highlight Option A (left side) | Sync with audio |
| **Pointing RIGHT** | Highlight Option B (right side) | Sync with audio |
| **Pointing CENTER** | Highlight question text | Sync with audio |
| **Open hands** | Show "Think about it" hint | Sync with audio |
| **Thumbs up** | Show "Great job!" feedback | After correct answer |
| **Thinking pose** | Pause for reflection (3s timer) | Sync with audio |

**Implementation:**
- Video timeline markers OR pose detection (2D images)
- UI elements fade in/out (0.3s ease)
- Highlights: soft glow (rgba(102, 126, 234, 0.3))
- Never cover Kelly's face or hands

---

### **2. Panel Expansion Rules**

| Panel Type | Direction | Width | Overlay | Close Trigger |
|------------|-----------|-------|---------|---------------|
| **Right Rail Panels** | Slide RIGHT | 280px | No (pushes content) | Tap outside, tap icon, tap other icon |
| **Bottom Panels** | Slide UP | Full width | No (pushes Kelly up) | Tap outside, tap icon, swipe down |
| **Full-Screen Overlays** | Fade IN | 100% | Yes (dark bg) | Tap X, tap outside, ESC key |
| **Left Comments** | Slide IN from LEFT | 280px (desktop), 100% (mobile) | Yes (mobile only) | Tap outside, tap 💬 icon |

**Animations:**
- Duration: 0.3s
- Easing: cubic-bezier(0.4, 0.0, 0.2, 1) (Material Design)
- Backdrop blur: 10px (frosted glass effect)

---

### **3. State Management**

| State | Visual Indicator | Persistence |
|-------|------------------|-------------|
| **Current Phase** | Glowing circle in Journey Bar | Session |
| **Completed Phases** | Filled circle in Journey Bar | Session |
| **Selected Option** | Glow + checkmark | Session |
| **Pinned Lessons** | 📌 count in Action Dock | LocalStorage |
| **Aha Moments** | 💡 count in Action Dock | LocalStorage |
| **Language** | Badge on 🌐 icon | LocalStorage |
| **Tone** | Badge on 🎭 icon | LocalStorage |
| **Age** | Badge on 👤 icon | LocalStorage |
| **Mode** | Badge on 📺 icon | LocalStorage |
| **Sound** | ON/OFF icon state | LocalStorage |
| **Streak** | Day count in Settings | Supabase |
| **Progress** | % complete in Settings | Supabase |

---

## 🎨 **VISUAL DESIGN SYSTEM**

### **Colors**

```css
/* Primary Palette */
--kelly-primary: #667eea;        /* Purple gradient start */
--kelly-secondary: #764ba2;      /* Purple gradient end */
--kelly-accent: #4ade80;         /* Success green */
--kelly-warning: #fbbf24;        /* Warning yellow */
--kelly-error: #ef4444;          /* Error red */

/* Neutrals */
--kelly-bg: #f5f7fa;             /* Light background */
--kelly-surface: rgba(255, 255, 255, 0.95);  /* Frosted glass */
--kelly-text: #333333;           /* Primary text */
--kelly-text-secondary: #666666; /* Secondary text */
--kelly-border: #e0e0e0;         /* Subtle borders */

/* Overlays */
--kelly-overlay-light: rgba(255, 255, 255, 0.95);
--kelly-overlay-dark: rgba(0, 0, 0, 0.8);
--kelly-glass: rgba(255, 255, 255, 0.95);
--kelly-blur: blur(20px);
```

### **Typography**

```css
/* Font Stack */
--kelly-font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;

/* Sizes */
--text-xs: 11px;    /* Labels, hints */
--text-sm: 13px;    /* Body text, comments */
--text-base: 15px;  /* Options, controls */
--text-lg: 18px;    /* Questions, headings */
--text-xl: 24px;    /* Lesson titles */
--text-2xl: 32px;   /* Hero text */

/* Weights */
--weight-normal: 400;
--weight-medium: 500;
--weight-semibold: 600;
--weight-bold: 700;
```

### **Spacing**

```css
/* iOS-native spacing */
--space-xs: 4px;
--space-sm: 8px;
--space-md: 12px;
--space-lg: 16px;
--space-xl: 20px;
--space-2xl: 24px;

/* Touch targets */
--touch-target: 44px;  /* iOS minimum */
--icon-size: 24px;
--icon-size-lg: 32px;
```

### **Shadows**

```css
/* Elevation */
--shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
--shadow-md: 0 4px 16px rgba(0, 0, 0, 0.15);
--shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.2);
--shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.25);

/* Glows */
--glow-primary: 0 4px 20px rgba(102, 126, 234, 0.4);
--glow-success: 0 4px 20px rgba(74, 222, 128, 0.4);
```

---

## ⌨️ **KEYBOARD SHORTCUTS**

| Key | Action | Context |
|-----|--------|---------|
| **Space** | Pause/Play Kelly | Always |
| **→** | Next phase | Always |
| **←** | Previous phase | Always |
| **1-5** | Jump to phase 1-5 | Always |
| **A/B** | Select option A/B | Question phase |
| **L** | Toggle language panel | Always |
| **T** | Toggle tone panel | Always |
| **M** | Toggle mode panel | Always |
| **S** | Toggle sound | Always |
| **/** | Open search | Always |
| **C** | Open calendar | Always |
| **Esc** | Close panel/overlay | When open |
| **?** | Show keyboard shortcuts | Always |

---

## 🎯 **SUCCESS METRICS**

### **How do we know this design works?**

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Time to first interaction** | < 3s | Analytics |
| **Kelly visibility %** | > 70% of viewport | Visual audit |
| **Feature discoverability** | > 80% find within 30s | User testing |
| **Panel open/close smoothness** | 60 FPS | Performance monitor |
| **Mobile tap accuracy** | > 95% hit rate | Analytics |
| **Zero overlaps on Kelly** | 100% | Visual regression tests |
| **Learner satisfaction** | > 4.5/5 | Post-lesson survey |
| **"This is AGI" reactions** | > 30% of comments | Sentiment analysis |

---

## 🚀 **IMPLEMENTATION PHASES**

### **Phase 1: Foundation (Week 1)**
- [ ] Remove ALL center overlays
- [ ] Implement sacred center zone (Kelly only)
- [ ] Build Right Rail with expand panels
- [ ] Build Action Dock (bottom-right)
- [ ] Build Phase Journey Bar (top)

### **Phase 2: Content (Week 2)**
- [ ] Build Bottom Content Bar (collapsible)
- [ ] Integrate question/options
- [ ] Add caption text (always visible)
- [ ] Sync with Kelly's gestures (pointing)

### **Phase 3: Community (Week 3)**
- [ ] Build Left Rail (comments)
- [ ] Add simulated social content (✨ badge)
- [ ] Implement auto-scroll + manual scroll
- [ ] Add Trust & Safety disclosures

### **Phase 4: Polish (Week 4)**
- [ ] Add all animations (0.3s ease)
- [ ] Implement keyboard shortcuts
- [ ] Add gesture controls (swipe)
- [ ] Test on mobile + desktop + tablet
- [ ] Visual regression tests (no Kelly overlaps)

### **Phase 5: Advanced (Week 5)**
- [ ] Integrate HD video player
- [ ] Add Unity 3D Kelly mode
- [ ] Build Settings panel (all advanced tools)
- [ ] Add Analytics dashboard
- [ ] Add Earn to Learn panel

---

## 📝 **NOTES FOR FUTURE SELF**

### **Why this design?**

1. **Kelly is the star** - Everything else is supporting cast
2. **Spatial consistency** - Every feature has ONE home, always
3. **Progressive disclosure** - Simple by default, powerful when needed
4. **Gesture-first** - Kelly's pointing = where to look/click
5. **Mobile-native** - TikTok-style full-screen, swipe-friendly
6. **20 years honored** - Every tool accessible, none removed

### **What makes this "AGI-like"?**

- Kelly's gestures sync perfectly with UI cues (she points, UI highlights)
- Timing is invisible (everything happens exactly when it should)
- Zero cognitive load (learner never thinks "where is X?")
- Seamless transitions (no jarring jumps or overlays)
- Feels like talking to a real teacher (not clicking through software)

### **What's next?**

After this spatial foundation is solid:
1. Add AI-generated lessons (search → generate)
2. Add real-time voice conversation (Talk to Kelly)
3. Add adaptive difficulty (learns your level)
4. Add social features (real learners, not just simulated)
5. Add gamification (streaks, badges, leaderboards)

But FIRST: nail the spatial design. Make it timeless. Make it perfect.

---

**Last Updated:** December 9, 2025  
**Status:** 🟡 Planning Complete, Ready for Implementation  
**Owner:** Kelly OS Team  
**Review:** Pending user approval






