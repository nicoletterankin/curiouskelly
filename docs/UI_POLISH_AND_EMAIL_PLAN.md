# UI Polish & Email Content Plan

**Created:** December 16, 2025  
**Status:** Active Sprint  
**Goal:** Tight, contained UI + full lesson delivery via email

---

## 🎯 NORTH STAR

> "A tight learning box where everything stays in bounds, and learners can choose their channel — app OR email — for the full experience."

---

## PART 1: UI POLISH — PANEL & OVERLAY SYSTEM

### Problem Statement

Current overlays (paywall modal, settings panel, search, commons) can stack awkwardly. When multiple panels are open, content gets cut off or overlaps. The experience should feel contained, like a single focused "box."

### Design Principles

1. **One Thing at a Time** — Only one major overlay open at once
2. **In The Box** — All content contained, no clipping or overflow
3. **Clear Hierarchy** — User always knows what's on top
4. **Smooth Transitions** — Panels animate cleanly, no jarring jumps

### Current Z-Index Layers

| Layer | Z-Index | Usage |
|-------|---------|-------|
| Background | 0 | Kelly video |
| Kelly | 1 | Avatar frame |
| Bottom overlay | 80-90 | Choices, speech |
| Controls | 100-150 | Action buttons, nav |
| Modals | 200-500 | Settings, paywall |
| Search | 700 | Full-screen search |
| Toasts | 300-1000 | Notifications |
| Commons | 10000 | Full-screen commons |

### Required Fixes

#### 1. Mutual Exclusion for Major Panels
```
When SETTINGS opens → close PAYWALL, SEARCH, COMMONS
When PAYWALL opens → close SETTINGS, SEARCH, COMMONS  
When SEARCH opens → close SETTINGS, PAYWALL, COMMONS
```

**Implementation:** Add `closePanelExclusively()` function that manages one-at-a-time behavior.

#### 2. Consistent Modal Container
All modals should:
- Use `max-height: calc(100vh - safe-top - safe-bottom - 40px)`
- Have `overflow-y: auto` for scrolling within
- Respect safe areas on all edges
- Never allow content to clip outside the modal

#### 3. Paywall Modal Containment
- Fix: Modal content should have internal scroll, not overflow
- All pricing tiers visible without cut-off
- Close button always accessible

#### 4. Settings Panel Polish
- Full-screen slide-up from bottom
- Proper backdrop that blocks interaction behind
- Close when backdrop tapped
- Sections: Profile, Preferences, Subscription, Help, Legal

---

## PART 2: EMAIL-FIRST LESSON DELIVERY

### Problem Statement

Some learners want to consume lessons entirely via email — read the content, see the pictures, engage at their pace. Currently emails are teasers that drive to the app.

### New Vision

**Daily email = Complete standalone lesson experience**

Learners can:
- Read the full lesson text
- See all images/illustrations
- Answer questions in their head
- Get the wisdom/takeaway
- OPTIONALLY click to app for video/voice experience

### Email Content Structure

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ CURIOUS KELLY — Day [N]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 TODAY'S LEARN LESSON
[Topic Title]

[Hero Image - 600px wide]

[Full lesson text - 2-3 paragraphs]

🤔 Quick question: [Question from the lesson]
(Think about it... answer below!)

[Second image if applicable]

[Answer reveal section]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 TODAY'S GROW LESSON  
[Skill Title]

[Skill illustration]

[Full skill content]

💡 Try this: [Activity prompt]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ DAILY WISDOM
"[Wisdom quote for the day]"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Button: Experience with Kelly →]
Watch today's lesson with voice & video

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stay curious! ✨
— Kelly

[Unsubscribe] | [Preferences] | [curiouskelly.com]
```

### Email Templates Needed

1. **Daily Lesson Email** (LEARN + GROW combined)
2. **LEARN-only Email** (for day-unlock purchasers)
3. **GROW-only Email** (for skill-focused track)
4. **Weekly Digest** (summary of 7 days)

### Technical Implementation

1. **Content Pipeline:** Extract full text from PhaseDNA files
2. **Image Pipeline:** Attach lesson images as hosted URLs
3. **Email Service:** Use existing Supabase + email provider
4. **Personalization:** Include learner name, streak count
5. **Preferences:** Let users choose email frequency & content type

### Email Settings (User Controls)

- [ ] Daily LEARN lesson
- [ ] Daily GROW lesson  
- [ ] Weekly digest only
- [ ] No emails (app only)

---

## PART 3: IMPLEMENTATION CHECKLIST

### Phase A: UI Containment (Today)
- [ ] Add mutual exclusion for overlays
- [ ] Fix paywall modal max-height
- [ ] Test settings + paywall interaction
- [ ] Verify no content clipping on small screens

### Phase B: Email Templates (This Week)
- [ ] Design HTML email template
- [ ] Test across email clients (Gmail, Apple Mail, Outlook)
- [ ] Create content extraction script from PhaseDNA
- [ ] Set up email preference management

### Phase C: Full Email Delivery (Post-Launch)
- [ ] Daily automated sends
- [ ] Streak tracking in emails
- [ ] Image hosting pipeline
- [ ] Analytics (opens, clicks)

---

## SUCCESS METRICS

### UI
- Zero overlapping panels in user testing
- All content readable without scrolling into hidden areas
- Settings/paywall/search work independently

### Email
- 40%+ open rate on daily emails
- Users report "I read the lesson" without opening app
- Email-only users maintain streaks

---

## ASSETS REFERENCED

- `/public/css/learn.css` — Main lesson styles
- `/public/css/kelly-os.css` — OS-level tokens
- `/public/learn.html` — Lesson player
- `/curious-kellly/content/` — PhaseDNA files for extraction

---

*Stay curious. Stay contained. ✨*
