# 🔒 THE UI GENERATION SPEC
## Canonical Layout & Interaction Patterns for Curious Kelly

**Last Updated:** December 19, 2025  
**Status:** LOCKED - Do not deviate without explicit approval

---

## ⚠️ CRITICAL RULES (NEVER VIOLATE)

1. **NEVER create new UI elements** - Modify existing elements, don't duplicate
2. **NEVER drop phase options** - Every phase has EXACTLY 2 options (A/B), 2 paths, 2 feedback responses
3. **NEVER put chat in bottom-right** - Kelly is NOT a chatbot. Chat lives in LEFT PANEL via logo click.
4. **NEVER mix icon styles** - One style per zone (emoji OR svg, not both)
5. **NEVER break the phase flow** - 7 phases, each with 2 options = 14 interaction points per lesson
6. **Kelly IS the logo** - Her face is the brand. Clicking it opens social/chat panel.

---

## 🧠 THE CORE MENTAL MODEL

### Kelly = Your Learning Calendar

```
┌─────────────────────────────────────────────────────────────────┐
│  "What day is it?"                                              │
│                                                                  │
│  December 19 = "Being Where You Are" day                        │
│  December 20 = [Tomorrow's Topic] day                           │
│  December 21 = [Next Topic] day                                 │
│  ...                                                            │
│  365 days = 365 topics = The LEARN track                        │
│  + 365 more = The GROW track (AI-powered)                       │
│                                                                  │
│  "Kelly is my calendar."                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Integration:** Google Calendar, Apple Calendar, Outlook, etc.
- Each lesson = a calendar event
- Topic becomes the event title
- Kelly syncs with your life

---

## 🧭 NAVIGATION RULES (SINGLE PAGE APP)

### ⚠️ CRITICAL: Never Navigate Away
```
✅ ALLOWED: showScene('lesson'), showScene('journey'), showScene('settings')
✅ ALLOWED: openPanel('left'), openPanel('right')
✅ ALLOWED: URL params like ?day=353&phase=2
❌ FORBIDDEN: window.location.href = '/other-page'
❌ FORBIDDEN: <a href="/other-page">
❌ FORBIDDEN: Any navigation that leaves learn.html
```

### Every Click Has ONE Behavior

| Element | Click Behavior | Never Does |
|---------|----------------|------------|
| **Kelly Logo** (top-left) | `openPanel('left')` → Shows comments + chat | ~~Navigate to home~~ |
| **Time/Date** (top-right) | `showScene('journey')` → Calendar tab | ~~Nothing~~ |
| **Learn/Grow Toggle** | Switch track, reload current day | ~~Navigate away~~ |
| **Topic Title** | Expand lesson detail panel | ~~Search~~ |
| **Phase Dot** | Jump to that phase in current lesson | ~~Open new lesson~~ |
| **Search Icon** | Expand search bar in header | ~~Open new page~~ |
| **📊 Infographic** | Show phase visual overlay | ~~Navigate~~ |
| **🔖 Bookmark** | Toggle bookmark for current phase | ~~Navigate~~ |
| **⚙️ Settings** | `showScene('settings')` | ~~New tab~~ |
| **Calendar Day Cell** | Load that day's lesson | ~~Navigate away~~ |

---

## 📅 CALENDAR VIEW (Journey Scene)

### Calendar Tabs (4 views, same data)
| Tab | View | Purpose |
|-----|------|---------|
| **📅 Calendar** | Month grid | Visual overview, tap any day |
| **📆 Week** | 7-day horizontal | Quick week navigation |
| **📚 Curriculum** | Category list + search | Find by topic/category |
| **🔖 Saved** | Bookmarked moments | Return to favorites |

### Day Cell States
```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  ✓ 15   │  │ ● 19    │  │   20    │  │ 🔒 25   │
│ Dreams  │  │ Present │  │ Tomorrow│  │ Future  │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
 Completed    TODAY        Available    Locked
 (green)      (blue glow)  (normal)     (dimmed)
```

### Calendar Click Behavior
```javascript
// When user clicks a calendar day cell:
function onCalendarDayClick(dayNumber) {
  // 1. Close journey scene
  closeScene('journey');
  
  // 2. Update URL (no page reload)
  history.pushState({}, '', `?day=${dayNumber}`);
  
  // 3. Load lesson data
  await loadLessonData(dayNumber);
  
  // 4. Show lesson scene
  showScene('lesson');
  
  // 5. Start at Hook phase
  goToPhase(0);
}
```

---

## 🔍 SEARCH SYSTEM

### What Search Finds
| Priority | Searches | Example Query | Result |
|----------|----------|---------------|--------|
| 1 | **Day number** | "day 45", "45" | Jump to Day 45 |
| 2 | **Topic title** | "dreams", "being present" | Lessons matching topic |
| 3 | **Category** | "mind", "science", "emotions" | Lessons in category |
| 4 | **Script content** | "neural pathways" | Lessons mentioning term |

### Search Backend Query
```sql
-- Full-text search on core_lessons
SELECT 
  cl.day_number,
  cl.topic,
  cl.category,
  ts_rank(
    to_tsvector('english', cl.topic || ' ' || COALESCE(cl.marketing_headline, '')),
    plainto_tsquery('english', $SEARCH_TERM)
  ) as relevance
FROM core_lessons cl
WHERE 
  -- Direct day match
  cl.day_number::text = $SEARCH_TERM
  OR
  -- Topic/headline match
  to_tsvector('english', cl.topic || ' ' || COALESCE(cl.marketing_headline, '')) 
    @@ plainto_tsquery('english', $SEARCH_TERM)
  OR
  -- Category match
  cl.category ILIKE '%' || $SEARCH_TERM || '%'
ORDER BY 
  CASE WHEN cl.day_number::text = $SEARCH_TERM THEN 0 ELSE 1 END,
  relevance DESC
LIMIT 20;
```

### Search UI Experience
```
┌─────────────────────────────────────────────────────┐
│ 🔍 [Search lessons...                          ] × │
├─────────────────────────────────────────────────────┤
│ 📅 Day 45 - Why We Dream                   [→]     │
│ 📅 Day 123 - Lucid Dreaming               [→]     │
│ 📅 Day 201 - Sleep and Memory             [→]     │
│                                                     │
│ 🏷️ Category: Mind & Brain (23 lessons)    [→]     │
└─────────────────────────────────────────────────────┘
```

### Search Result Click
```javascript
function onSearchResultClick(dayNumber) {
  // 1. Close search
  closeSearch();
  
  // 2. Update URL
  history.pushState({}, '', `?day=${dayNumber}`);
  
  // 3. Load and show lesson
  await loadLessonData(dayNumber);
  showScene('lesson');
}
```

---

## 🚀 5 WAYS TO JUMP BETWEEN DAYS

### Method 1: Calendar Grid (Journey Scene)
- Tap any day cell in month view
- **Backend:** No query needed, day_number is on cell

### Method 2: Search
- Type day number or topic
- **Backend:** Full-text search on core_lessons

### Method 3: URL Parameter
- Direct link: `?day=45`
- **Backend:** Simple SELECT by day_number

### Method 4: Week Navigation
- Left/right arrows in week view
- **Backend:** Preloaded week data

### Method 5: Bookmarks
- Tap saved bookmark → returns to that day + phase
- **Backend:** User's bookmark record includes day_number + phase

### Method 6 (Bonus): Tomorrow Preview
- Outro phase shows "Tomorrow: [Topic]" with tap to preview
- **Backend:** day_number + 1

---

## ⚙️ SETTINGS STRUCTURE (LOCKED)

### Settings Sections
```
SETTINGS
├── Preferences
│   ├── Auto-Play (toggle)
│   └── Show Captions (toggle)
│
├── Language
│   └── Display Language → [Language Picker Overlay]
│
├── Your Learning
│   ├── Learning Journey → [Stats subpanel]
│   └── Subscription → [Plan details]
│
├── Account
│   └── Profile → [Profile subpanel]
│
├── Support
│   ├── Help & FAQ → [FAQ subpanel]
│   └── Message Kelly → [Contact form]
│
└── Legal & Privacy
    ├── Privacy → [Privacy policy]
    ├── Terms → [Terms of service]
    └── Your Data → [Export/delete options]
```

### Settings Click Behavior
```javascript
// Settings items open subpanels, NOT new pages
function onSettingsItemClick(itemId) {
  switch(itemId) {
    case 'btn-language':
      openOverlay('overlay-picker'); // Language picker
      break;
    case 'btn-help':
      openSettingsSubpanel('help'); // Slide-in subpanel
      break;
    case 'btn-privacy':
      openSettingsSubpanel('privacy');
      break;
    // etc.
  }
}
```

---

## 🌍 LANGUAGE SWITCHER

### Current Languages (Precomputed per CLAUDE.md)
| Code | Language | Status |
|------|----------|--------|
| `en` | English | ✅ Complete |
| `es` | Spanish | ✅ Complete |
| `fr` | French | ✅ Complete |

### Language Affects
- Kelly's script (audio + text)
- UI labels (via i18n-core.js)
- Simulated comments
- Kelly personality greetings (via i18n-kelly.js)

### Language Does NOT Affect
- Phase structure (always 7 phases)
- Option count (always 2)
- Visual assets (language-neutral images)
- Day/topic mapping (same 365 days)

### Language Switch Backend
```sql
-- When language changes, load translated atoms
SELECT content, visual_url, hd_video_url
FROM lesson_atoms
WHERE core_lesson_id = $LESSON_ID
  AND archetype = $ARCHETYPE
  AND language = $NEW_LANGUAGE;  -- 'en', 'es', 'fr'

-- Also fetch translated audio
SELECT url, phase
FROM kelly_video_assets
WHERE day_number = $DAY_NUMBER
  AND language = $NEW_LANGUAGE
  AND asset_type = 'audio';
```

### Language Picker UI
```
┌─────────────────────────────────────────┐
│         🌍 Choose Language              │
├─────────────────────────────────────────┤
│  ✓ 🇺🇸 English                          │
│    🇪🇸 Español                          │
│    🇫🇷 Français                         │
├─────────────────────────────────────────┤
│  ⚠️ More languages coming soon!         │
│  Sponsor a language →                   │
└─────────────────────────────────────────┘
```

---

## 📐 THE 7-PHASE STRUCTURE (LOCKED)

### Every Phase Has 2 Options

| Phase | Options | Purpose |
|-------|---------|---------|
| **Hook** | Option A, Option B | How to approach today's curiosity |
| **Cliff** | Option A, Option B | Choose your learning path |
| **Fact 1 (Q1)** | Option A, Option B | First knowledge checkpoint |
| **Fact 2 (Q2)** | Option A, Option B | Second knowledge checkpoint |
| **Fact 3 (Q3)** | Option A, Option B | Third knowledge checkpoint |
| **Wisdom** | Option A, Option B | How this applies to your life |
| **Outro** | Option A, Option B | What to explore next |

### Total Interactions Per Lesson
```
7 phases × 2 options = 14 choice points
7 phases × 2 feedback responses = 14 Kelly responses
7 phases × 1 infographic = 7 visuals
7 phases × 2 option card images = 14 option visuals
──────────────────────────────────────────────────
TOTAL: 49 interactive elements per lesson
```

### Option Structure (Per Phase)
```json
{
  "phase": "hook",
  "script": "Kelly's opening line...",
  "options": [
    {
      "id": "A",
      "label": "Option A text",
      "imageUrl": "https://...",
      "responseScript": "Kelly's response to A...",
      "quality": "best"
    },
    {
      "id": "B", 
      "label": "Option B text",
      "imageUrl": "https://...",
      "responseScript": "Kelly's response to B...",
      "quality": "good"
    }
  ],
  "visualUrl": "https://...",
  "simulatedComments": [
    { "emoji": "🧠", "text": "This reminds me of...", "author": "curious_maya" },
    { "emoji": "💡", "text": "I never thought about it that way!", "author": "science_sam" }
  ]
}
```

---

## 🖼️ SCREEN LAYOUT (LOCKED)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER (TOP)                                                                 │
│  [🔍]  [←]        [ LEARN ║ GROW ]        "Topic Title"        4:27 AM      │
│                                                                  FRI DEC 19  │
├─────────────────────────────────────────────────────────────────────────────┤
│                           PHASE BAR (7 phases)                               │
│   ●━━━●━━━●━━━●━━━●━━━●━━━●                                                 │
│  Hook Cliff Q1  Q2  Q3 Wisdom Outro                                         │
├──────────────┬──────────────────────────────────────────────┬───────────────┤
│              │                                              │               │
│  LEFT PANEL  │           SACRED CENTER (KELLY)              │  RIGHT PANEL  │
│  (Comments)  │                                              │  (Settings/   │
│              │                                              │   Journey)    │
│ ┌──────────┐ │                                              │               │
│ │ 💬 Chat  │ │         [KELLY VIDEO/IMAGE]                  │ ┌───────────┐ │
│ │          │ │                                              │ │ 📊 Info   │ │
│ │ Comments │ │                                              │ │ 🔖 Save   │ │
│ │ for this │ │                                              │ │ ⚙️ Set    │ │
│ │ phase... │ │                                              │ └───────────┘ │
│ │          │ │                                              │               │
│ │ ───────  │ │                                              │               │
│ │ 💡 Got it│ │                                              │               │
│ │ 🤯 Wow   │ │                                              │               │
│ │ 💭 More  │ │                                              │               │
│ │ ───────  │ │                                              │               │
│ │ [Type..] │ │                                              │               │
│ └──────────┘ │                                              │               │
│              │                                              │               │
├──────────────┴──────────────────────────────────────────────┴───────────────┤
│ BOTTOM ZONE (Lesson Controls)                                                │
│                                                                              │
│  Caption: "Kelly's current script line..."                                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │   [OPTION A IMAGE]          [OPTION B IMAGE]            │                │
│  │   "Option A Label"          "Option B Label"            │                │
│  └─────────────────────────────────────────────────────────┘                │
│                                                                              │
│              ◀️ PREV     [ ▶️ PLAY ]     NEXT ▶️                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 ZONE ASSIGNMENTS (LOCKED)

### HEADER (Top)
| Element | Position | Purpose |
|---------|----------|---------|
| Search icon (🔍) | Left | Expand to search bar |
| Back arrow (←) | Left | Return to previous view |
| Learn/Grow toggle | **CENTER** | Primary mode switch |
| Topic title | Center-right | Current lesson topic |
| Time | **RIGHT** | Current time (e.g., 4:27 AM) |
| Date | **RIGHT** | Current date (e.g., FRI DEC 19) |

### LEFT PANEL (Kelly Logo Click)
| Element | Position | Purpose |
|---------|----------|---------|
| Panel header | Top | "💬 Talk to Kelly" |
| Phase comments | Scrollable | Simulated student comments for current phase |
| Reaction buttons | Below comments | `💡 Got it` `🤯 Wow` `💭 More` |
| Chat input | Bottom | "Type something to Kelly..." |

**Trigger:** Click Kelly's logo/avatar anywhere it appears

### SACRED CENTER (Kelly)
| Element | Position | Purpose |
|---------|----------|---------|
| Kelly video/image | Center | The lesson presentation |
| Gradient overlay | Bottom | Caption readability |
| Tap zones | Full area | Prev/Center/Next tap targets |

**NEVER put UI elements over Kelly's face.**

### RIGHT PANEL (Settings/Journey)
| Element | Position | Purpose |
|---------|----------|---------|
| 📊 Infographic | Top | Phase-specific visual |
| 🔖 Bookmark | Middle | Save this lesson |
| ⚙️ Settings | Bottom | Open settings |

**Trigger:** Swipe from right edge or tap side actions

### BOTTOM ZONE (Lesson Controls)
| Element | Position | Purpose |
|---------|----------|---------|
| Caption text | Top of zone | Kelly's current script |
| Option A card | Left | First choice (with image) |
| Option B card | Right | Second choice (with image) |
| Playback controls | Bottom | Prev / Play / Next |
| Auto-advance toggle | Right of controls | Enable/disable auto-play |

---

## 🚫 WHAT DOES NOT BELONG WHERE

### ❌ NEVER in Bottom-Right
- Chat input (Kelly is not a chatbot)
- Comment feed
- Reaction buttons

### ❌ NEVER in Sacred Center
- Settings buttons
- Navigation that blocks Kelly's face
- Floating icons over video

### ❌ NEVER in Header
- Playback controls (those go bottom)
- Comments
- Reactions

### ❌ NEVER Floating/Orphaned
- Icons without a zone assignment
- New buttons without spec approval
- Duplicate functionality

---

## 📝 SIMULATED COMMENTS SPEC

### Comments Are Phase-Linked
Every phase needs pre-written simulated comments:

```
lesson_atoms.content.simulatedComments: [
  {
    "emoji": "✨",           // Required: Trust & Safety indicator
    "text": "Comment text",
    "author": "curious_maya",
    "timestamp": "2m ago"    // Relative time (simulated)
  }
]
```

### Comment Requirements Per Phase
| Phase | Min Comments | Tone |
|-------|-------------|------|
| Hook | 2-3 | Curiosity, excitement |
| Cliff | 2-3 | Deliberation, preference sharing |
| Q1 | 2-3 | "I didn't know that!", learning |
| Q2 | 2-3 | Deeper engagement, connections |
| Q3 | 2-3 | Application, personal stories |
| Wisdom | 2-3 | Reflection, gratitude |
| Outro | 1-2 | Anticipation, farewell |

### Trust & Safety Rules (from CLAUDE.md)
- ALL simulated comments marked with ✨ indicator
- Master toggle in Settings to disable
- Never claim simulated users are real
- Never hide disclosure indicators

---

## 🔘 ICON STYLE GUIDE (LOCKED)

### Zone: Left Panel (Social)
**Style:** Emoji only
- 💬 Chat
- 💡 Got it
- 🤯 Wow
- 💭 More
- ✨ Simulated indicator

### Zone: Right Panel (Actions)
**Style:** Emoji only
- 📊 Infographic
- 🔖 Bookmark
- ⚙️ Settings

### Zone: Header
**Style:** Monochrome SVG
- Search icon
- Back arrow
- Learn/Grow toggle styling

### Zone: Bottom (Playback)
**Style:** Monochrome SVG
- ◀️ Previous (SVG)
- ▶️ Play/Pause (SVG)
- ▶️ Next (SVG)

**Rule:** Never mix emoji and SVG in the same zone.

---

## 🔄 PHASE FLOW INTEGRITY

### The Golden Rule
```
Every phase MUST have:
├── script (Kelly's words)
├── options[2] (exactly 2)
│   ├── option A
│   │   ├── label
│   │   ├── imageUrl (512×512)
│   │   ├── responseScript
│   │   └── quality ("best" | "good" | "redirect")
│   └── option B
│       ├── label
│       ├── imageUrl (512×512)
│       ├── responseScript
│       └── quality
├── visualUrl (infographic, 1920×1080)
└── simulatedComments[2-3]
```

### If Options Are Missing
```
❌ BROKEN: Phase without 2 options
❌ BROKEN: Phase without response scripts
❌ BROKEN: Phase without visual
❌ BROKEN: Phase without simulated comments

✅ REQUIRED: Generate missing content before lesson is "ready"
```

### Verification Query
```sql
SELECT 
  la.phase,
  jsonb_array_length(la.content->'options') as option_count,
  la.visual_url IS NOT NULL as has_visual,
  jsonb_array_length(la.content->'simulatedComments') as comment_count
FROM lesson_atoms la
JOIN core_lessons cl ON la.core_lesson_id = cl.id
WHERE cl.day_number = $DAY_NUMBER
  AND la.archetype = 'The Scientist';

-- EXPECTED: 7 rows, each with option_count=2, has_visual=true, comment_count>=2
```

---

## 📱 RESPONSIVE BEHAVIOR

### Desktop (>1024px)
- Left panel: Slide-in from left (280px width)
- Right panel: Slide-in from right (320px width)
- Both can be open simultaneously

### Tablet (768-1024px)
- Panels overlay content (semi-transparent backdrop)
- Only one panel open at a time

### Mobile (<768px)
- Panels are full-screen overlays
- Bottom sheet for quick actions
- Swipe gestures for panel open/close

---

## ✅ UI VERIFICATION CHECKLIST

Before deploying any UI changes:

### Layout Verification
- [ ] Header has: Search (left), Learn/Grow (center), Topic + Time/Date (right)
- [ ] Left panel opens on Kelly logo click
- [ ] Left panel contains: Comments, Reactions, Chat input
- [ ] Right panel contains: Infographic, Bookmark, Settings
- [ ] Bottom zone has: Caption, 2 Option cards, Playback controls
- [ ] No floating orphan icons

### Phase Verification
- [ ] All 7 phases have exactly 2 options
- [ ] All options have image cards (512×512)
- [ ] All options have response scripts
- [ ] All phases have infographic visual
- [ ] All phases have 2-3 simulated comments

### Style Verification
- [ ] Left panel icons are emoji
- [ ] Right panel icons are emoji
- [ ] Playback controls are SVG
- [ ] No mixed styles in same zone
- [ ] ✨ indicator on all simulated comments

---

## 🚨 COMMON MISTAKES & FIXES

### "I created a new floating button"
**Problem:** Adding UI without zone assignment
**Fix:** Delete it. Assign functionality to existing element.

### "Options only showing on Cliff phase"
**Problem:** Only Cliff has choices implemented
**Fix:** ALL 7 phases need 2 options. Generate missing atoms.

### "Chat is in bottom-right corner"
**Problem:** Treating Kelly like a chatbot widget
**Fix:** Move to left panel. Chat triggers via Kelly logo click.

### "Reactions are in bottom center"
**Problem:** Reactions mixed with playback controls
**Fix:** Move reactions to left panel (comments section)

### "I added a new icon style"
**Problem:** Inconsistent visual language
**Fix:** Use existing style for that zone (emoji or SVG)

---

## 📂 KEY FILES

### UI Implementation
| File | Purpose |
|------|---------|
| `public/learn.html` | Main lesson player (14k+ lines) |
| `public/js/kelly-lesson-loader.js` | Data loading |
| `docs/UI_PATTERN_AUDIT.md` | Duplicate pattern warnings |

### Spec References
| Document | Purpose |
|----------|---------|
| `LESSON_GENERATION_SPEC.md` | Content generation rules |
| `UI_GENERATION_SPEC.md` | This document - layout rules |
| `docs/trust-safety/SIMULATED_SOCIAL_CONTENT.md` | Comment safety rules |

---

## 📝 CHANGE LOG

| Date | Change | Author |
|------|--------|--------|
| 2025-12-19 | Initial spec created | Claude |
| 2025-12-19 | Locked: 2 options per phase, Kelly logo → left panel, reactions in comments | Claude |
| 2025-12-19 | Added: Navigation rules (SPA), 5 ways to jump days, Search system, Calendar view, Settings structure, Language switcher | Claude |

---

## 🎯 SUCCESS CRITERIA

The UI is **PRODUCTION READY** when:

### Layout
1. ✅ Kelly logo click opens left panel with comments + reactions + chat
2. ✅ Time/Date anchored in top-right
3. ✅ Learn/Grow toggle in header center
4. ✅ Search in top-left (icon expands to bar)
5. ✅ Reactions (Got it/Wow/More) are in left panel, not bottom
6. ✅ No floating orphan icons
7. ✅ Single icon style per zone

### Phase Flow
8. ✅ All 7 phases display 2 option cards
9. ✅ Simulated comments visible per phase with ✨ markers
10. ✅ Phase flow never breaks (always 2 options available)

### Navigation (Single Page App)
11. ✅ Calendar day click loads lesson without page reload
12. ✅ Search results load lesson without page reload
13. ✅ All navigation uses showScene() / openPanel()
14. ✅ URL updates via history.pushState (never href navigation)
15. ✅ Back button works correctly (popstate handler)

### Calendar & Journey
16. ✅ 4 journey tabs work (Calendar, Week, Curriculum, Bookmarks)
17. ✅ Calendar shows correct completion states
18. ✅ Week navigation works (prev/next)
19. ✅ Curriculum search filters lessons

### Language
20. ✅ Language picker shows EN/ES/FR
21. ✅ Language switch reloads atoms + audio (no page refresh)
22. ✅ Kelly's personality matches selected language

---

*This spec is the source of truth for UI. When in doubt, follow this document.*
