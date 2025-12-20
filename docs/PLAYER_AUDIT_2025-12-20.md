# Kelly Player Audit - December 20, 2025

## Current State Summary

### ✅ WHAT'S WORKING

#### Core Player
- **365 Learn Track Lessons** - All with topics, marketing headlines, reflection prompts
- **20,481 Lesson Atoms** - Content units for all phases
- **330 Visual Commons** - 269 active, all with public URLs
- **2,265 Video Assets** - 241 validated (10.6%)

#### UI/UX (Just Completed)
- ✅ Kelly Panel opens on logo click
- ✅ Learn 365 / Grow 365 toggle in Kelly Panel
- ✅ Phase navigation (1-7) in Kelly Panel  
- ✅ Got it / Wow / More reactions
- ✅ Simulated comments feed (phase-aware)
- ✅ "Ask Kelly anything..." chat input
- ✅ Journey panel with week view
- ✅ Calendar integration ready
- ✅ Settings with BYOK sections
- ✅ Polished choice cards (A/B badges hidden)
- ✅ Header simplified (logo + time only)
- ✅ Controls relocated to bottom-right

---

## 🔴 GAPS TO FIX (Priority Order)

### 1. GROW TRACK ✅ FIXED
**Status: Database ready, starter lessons created**
- ✅ Added `track` column to `core_lessons` table
- ✅ Created unique constraint on (day_number, track)
- ✅ Added 7 starter Grow lessons (AI fluency curriculum)
- ✅ Player now passes track to lesson loader
- 🟡 Need to generate remaining 358 Grow track lessons

**Current State:**
- Learn track: 365 lessons
- Grow track: 7 lessons (Days 1-7)

### 2. API KEY CONFUSION
**Impact: Students don't know which key does what**

Current state:
- **Google AI API Key** → Visuals (in Settings)
- **OpenAI/Anthropic API Key** → Live chat with Kelly (in BYOK section)

Problems:
- Two different places to enter keys
- Unclear messaging about what each does
- Settings has "Google AI API Key" but BYOK section has "OpenAI"

**Consolidation needed in Settings:**
- "🎨 Visual Generation" → Google AI key
- "💬 Live Chat with Kelly" → OpenAI/Anthropic key

### 3. SIMULATED COMMENTS ARE GENERIC
**Impact: Comments don't feel connected to the actual lesson**

Current comments are phase-generic:
```javascript
hook: ['Here we go!', 'Love this topic', 'Ready to learn']
cliff: ['Tough choice…', 'Going with A', 'B feels right']
```

**Should be lesson-specific:**
- Use `reflection_prompts` from `core_lessons`
- Generate topic-aware reactions
- Example for Day 355 "Why You Get Up": "Purpose is everything!", "Made me think about my morning routine"

### 4. KELLY PANEL PHASES NOT POPULATED
**Impact: Phase items show as numbers, not teaching moments**

Current: Just shows "1, 2, 3, 4, 5"
Missing:
- Phase names (Hook, Cliff, Q1, Q2, Wisdom)
- Phase descriptions ("A surprising question...")
- Teaching moment previews on hover
- Click to jump to phase

### 5. VISUAL LOADING NEVER COMPLETES
**Impact: Shows "🎨 Visual loading..." indefinitely**

The Kelly Panel visual placeholder isn't being populated with:
- Current phase visual from Visual Commons
- Choice visuals during cliff phases
- Lesson thumbnail as fallback

### 6. CHAT INPUT NOT WIRED
**Impact: Typing in "Ask Kelly anything..." does nothing without API key**

The `sendToKelly()` function exists but:
- Needs BYOK key to work
- No feedback when no key is set
- Should show "Set up your API key in Settings to chat with Kelly"

---

## 📊 CONTENT DEPTH AUDIT

### Reflection Prompts (Ready to Use)
All 365 lessons have reflection prompts:
- Day 1: "What surprised you most about leaves?"
- Day 17: "What did you learn about your own movement habits?"
- Day 355: "What are you grateful for in your life right now?"

**Not displayed anywhere in UI** - should be in Kelly Panel or wisdom phase.

### Video Coverage
- 2,265 total assets
- Only 241 validated (10.6%)
- Most lessons will play audio-only with Kelly image

### Visual Coverage
- 330 visuals in Visual Commons
- 269 active
- ~0.9 visuals per lesson (need 7 per lesson for full coverage)

---

## 🎯 STUDENT JOURNEY GAPS

### What Students See When They...

1. **Open the app** → Working ✅
2. **Click Kelly logo** → Kelly Panel opens ✅
3. **Click Learn/Grow toggle** → Toggle animates but Grow does nothing 🔴
4. **Look at phase navigation** → Just numbers, no context 🟡
5. **Try to chat with Kelly** → Nothing happens without API key 🟡
6. **Make a choice** → Works but visuals may not load 🟡
7. **Complete a lesson** → Progress saves ✅
8. **Check their journey** → Calendar works ✅

---

## 📋 ACTION ITEMS

### Immediate (Today)
1. [ ] Wire reflection prompts to Kelly Panel comments
2. [ ] Add "no API key" message to chat input
3. [ ] Populate phase names in Kelly Panel

### This Week
4. [ ] Create Grow track schema and starter lessons
5. [ ] Consolidate API key settings UI
6. [ ] Generate more phase-specific simulated comments

### Before Launch
7. [ ] Generate 7 visuals per lesson (2,555 total needed)
8. [ ] Validate remaining 2,024 videos
9. [ ] Full E2E testing of all 365 days

---

## 🔧 TECHNICAL DEBT

1. `state.journeyTrack` is set but never used to filter lessons
2. Phase hover hints exist in CSS but no JS populates them
3. Visual Commons loader runs but Kelly Panel doesn't receive results
4. Multiple BYOK entry points cause confusion

---

*Generated by Claude audit - December 20, 2025*
