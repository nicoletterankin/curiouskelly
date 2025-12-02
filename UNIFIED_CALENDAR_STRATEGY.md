# 🎯 Unified Calendar Strategy: "The Kelly Today Experience"

## One View to Rule Them All

**Strategic Recommendation for Curious Kelly**  
**Date:** November 28, 2025

---

## The Problem

You currently have **4 different calendar/lesson views** scattered across your product:

| Location           | What it shows            | State                               |
| ------------------ | ------------------------ | ----------------------------------- |
| `app.html` sidebar | Lesson list + Cal toggle | Cramped, covers content on mobile   |
| `curriculum.html`  | Marketing syllabus grid  | Beautiful but disconnected from app |
| `calendar.html`    | Side panel calendar      | Completely broken                   |
| `player.html`      | Simple lesson delivery   | No calendar context                 |

**Result:** Learners don't know where to go. The same 365 lessons exist in different forms, different layouts, different interaction patterns. There's no **anchor point** for daily habit formation.

---

## The Insight

Your product has a **superpower** that you're not exploiting:

> **Everyone in the world learns the SAME topic on the SAME day.**

- November 28 = Citizenship (for everyone)
- Your birthday = YOUR special lesson
- January 1 = The Sun (shared New Year ritual)

This creates **shared cultural moments**. It's why Wordle went viral — everyone plays the same puzzle. Kelly could be the "Wordle of learning."

---

## The Recommendation: ONE UNIFIED VIEW

### Eliminate calendar.html entirely.

### Deprecate curriculum.html as standalone (redirect to marketing hero).

### Create ONE calendar experience that serves EVERYWHERE.

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE KELLY TODAY HUB                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │     🎓 TODAY'S LESSON                                   │   │
│  │     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━      │   │
│  │                                                         │   │
│  │     November 28, 2025                                   │   │
│  │                                                         │   │
│  │     ╔═══════════════════════════════════════════════╗   │   │
│  │     ║                                               ║   │   │
│  │     ║            CITIZENSHIP                        ║   │   │
│  │     ║                                               ║   │   │
│  │     ║     "Participating in and contributing       ║   │   │
│  │     ║      to community"                           ║   │   │
│  │     ║                                               ║   │   │
│  │     ╚═══════════════════════════════════════════════╝   │   │
│  │                                                         │   │
│  │     [  🎬 START TODAY'S LESSON  ]   ← BIG, unmissable  │   │
│  │                                                         │   │
│  │     🔥 Your streak: 7 days                              │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  📅 YOUR YEAR AT A GLANCE                               │   │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │   │
│  │                                                         │   │
│  │  [JAN] [FEB] [MAR] [APR] [MAY] [JUN]  ← Month tabs     │   │
│  │  [JUL] [AUG] [SEP] [OCT] [NOV] [DEC]                   │   │
│  │                                                         │   │
│  │   S   M   T   W   T   F   S                             │   │
│  │  ┌───┬───┬───┬───┬───┬───┬───┐                         │   │
│  │  │   │   │   │   │ 1 │ 2 │ 3 │  ← Past: ✓ completed   │   │
│  │  ├───┼───┼───┼───┼───┼───┼───┤     or ○ missed        │   │
│  │  │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │                         │   │
│  │  ├───┼───┼───┼───┼───┼───┼───┤  ← TODAY: ★ pulsing    │   │
│  │  │11 │12 │13 │14 │15 │16 │17 │                         │   │
│  │  ├───┼───┼───┼───┼───┼───┼───┤  ← Future: preview     │   │
│  │  │18 │19 │20 │21 │22 │23 │24 │                         │   │
│  │  ├───┼───┼───┼───┼───┼───┼───┤  ← Birthday: 🎂        │   │
│  │  │25 │26 │27 │★28│29 │30 │   │                         │   │
│  │  └───┴───┴───┴───┴───┴───┴───┘                         │   │
│  │                                                         │   │
│  │  🎂 Your birthday (Mar 15): "Creative Writing"         │   │
│  │     [Preview your birthday lesson]                     │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  📊 YOUR PROGRESS                                       │   │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │   │
│  │                                                         │   │
│  │   332/365 completed     ████████████████░░░ 91%         │   │
│  │                                                         │   │
│  │   7-day streak 🔥       Best: 45 days                   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## How This ONE View Serves ALL Use Cases

### 1. **App Experience** (app.html)

- Replace the broken sidebar with **The Kelly Today Hub** as a slide-up bottom sheet
- Or: Make the Hub the **default landing state** before lesson starts
- Kelly is visible in background; Hub overlays
- One tap to start today's lesson

### 2. **Marketing (Replace curriculum.html)**

- **Same component**, different context
- Hero shows today's lesson as social proof: "Today, learners worldwide are exploring Citizenship"
- Calendar below lets prospects browse the year
- CTA changes to "Start Free Trial" instead of "Start Lesson"

### 3. **Birthday Discovery**

- User enters birthday during onboarding OR in settings
- Their birthday is **always highlighted** on the calendar with 🎂
- Special preview mode: "On March 15, you'll learn about Creative Writing!"
- Creates anticipation and emotional connection

### 4. **Mobile Experience**

- Hub is **mobile-first** by design
- Full-screen on phone
- Today's lesson is always visible above the fold
- Calendar scrolls below
- No sidebar covering content — it IS the content

---

## Technical Implementation Path

### Phase 1: Build the Component

Create ONE `<kelly-today-hub>` component (or HTML partial) with these features:

- Today's lesson hero card
- Interactive 365-day calendar
- Birthday highlight system
- Progress/streak display
- Responsive: works at any size

### Phase 2: Deploy to App

- Replace app.html sidebar with full-screen Hub as default view
- Lesson player overlays when started
- Hub is accessible via back/menu button

### Phase 3: Deploy to Marketing

- Embed Hub on homepage or replace curriculum.html
- Modify CTAs for non-authenticated users
- Same data, different action buttons

### Phase 4: Delete Legacy

- Delete `calendar.html`
- Redirect `/curriculum.html` → `/` with Hub
- Remove sidebar calendar code from app.html

---

## Why This Works

### 1. **Single Source of Truth**

One calendar component. One data source. One mental model.

### 2. **TODAY is King**

Every time you open the app, you see TODAY'S lesson. Not a list to scroll. Not options to choose. TODAY.

This is the Wordle secret: Remove choice paralysis.

### 3. **Shared Experience Creates Community**

"What was your Kelly lesson today?"  
"Citizenship! What a topic for Thanksgiving weekend..."

Marketing writes itself. Users share organically.

### 4. **Birthday = Emotional Hook**

People WILL find their birthday. When they see "Creative Writing" is their birthday lesson, they're emotionally invested. They'll wait for it. They'll share it.

### 5. **Mobile-First Solves Desktop**

By designing for full-screen phone first, you automatically:

- Simplify the layout
- Eliminate sidebar issues
- Create touch-friendly targets
- Force focus on what matters

---

## What NOT to Do

❌ Don't create "dozens of awesome calendar views"  
— That's the problem you already have. More views = more confusion.

❌ Don't make the calendar the primary element  
— TODAY is primary. Calendar is context.

❌ Don't let users pick any lesson any time  
— That breaks the shared experience magic. Today's lesson is today's lesson.  
— (Allow catch-up for past lessons, but don't promote it)

❌ Don't hide progress in a settings menu  
— Streak and completion % drive habit formation. Always visible.

---

## The North Star Metric

**Daily Active Learners (DAL)**

If THE KELLY TODAY HUB works, you'll see:

- Higher lesson start rate (one clear CTA)
- Higher completion rate (no paralysis)
- Higher return rate (streak motivation)
- More birthday-related shares (organic growth)

---

## Summary

| Before                           | After                                      |
| -------------------------------- | ------------------------------------------ |
| 4 scattered calendar views       | 1 unified Hub                              |
| Sidebar covers content on mobile | Hub IS the content                         |
| Users lost in lesson lists       | Today's lesson is unmissable               |
| No birthday feature              | Birthday highlighted, creates anticipation |
| Curriculum disconnected from app | Same component serves both                 |
| "What should I learn?"           | "Today we're learning Citizenship"         |

---

## Next Step

**Design the Kelly Today Hub in Figma first.**

Before writing any code, create a high-fidelity mockup showing:

1. Mobile view (iPhone 14 Pro size)
2. Tablet view (iPad)
3. Desktop view (1440px)

Show: Today's lesson, calendar grid, birthday highlight, progress bar, streak counter.

Once approved, implement as a single reusable component.

---

_"The best interface is no interface. The second-best is one that gives you exactly one thing to do."_

— Your students should never wonder where to click. They open Kelly, they see TODAY, they start learning.







