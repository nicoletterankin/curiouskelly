# 🚨 DO THIS NOW - December 7, 2025
## Your Immediate Next Actions

---

## ⏰ TODAY'S MISSION (Dec 7)

You have **10 days** until launch. Here's exactly what to do TODAY.

---

## 🎯 HOUR 1-2: Verify Database & Content

**Already done for you:**
- ✅ 365 core lessons in database
- ✅ 21,855 lesson atoms (content pieces)
- ✅ 2,196 age hooks (age-specific content)
- ✅ 72 Kelly dialog templates

**Run this SQL to verify:**
```sql
-- Execute in Supabase SQL Editor
SELECT 
  cl.day_number, 
  cl.topic, 
  COUNT(la.id) as atom_count
FROM core_lessons cl
LEFT JOIN lesson_atoms la ON la.core_lesson_id = cl.id
GROUP BY cl.day_number, cl.topic
ORDER BY cl.day_number
LIMIT 10;
```

---

## 🎯 HOUR 2-4: Wire Lesson Player to Supabase

The Lesson Player V2 is at `curious-kellly/lesson-player-v2/`.

**Current State:** Loads from local JSON files.  
**Need:** Load from Supabase database.

**Key file to modify:** `curious-kellly/lesson-player-v2/js/app.js`

**Replace this function:**
```javascript
async fetchDailyLesson() {
    // CURRENT: Loads from local file
    let res = await fetch(`lessons/the-sun-dna.json`);
}
```

**With this:**
```javascript
async fetchDailyLesson() {
    // NEW: Load Day 1 from Supabase
    const { data: lesson, error } = await this.supabase
        .from('core_lessons')
        .select('*')
        .eq('day_number', 1)
        .single();
    
    if (error) {
        console.error('Error fetching lesson:', error);
        return;
    }
    
    // Get atoms for this lesson
    const { data: atoms } = await this.supabase
        .from('lesson_atoms')
        .select('*')
        .eq('core_lesson_id', lesson.id);
    
    this.state.currentLesson = {
        ...lesson,
        atoms: atoms || []
    };
    
    this.renderLessonState();
}
```

---

## 🎯 HOUR 4-6: Test Landing Page

**File:** `legacy_marketing_site.html`

1. Open in browser
2. Check all sections load
3. Verify Unity WebGL container (can fail gracefully)
4. Test "Give as Gift" buttons (will fail - that's OK)
5. Check mobile responsive

---

## 🎯 HOUR 6-8: Setup Stripe (If keys available)

**If you have Stripe keys:**

1. Create products in Stripe Dashboard:
   - `Personal Annual` - $199/year
   - `Family Annual` - $299/year  
   - `Gift 12 Months` - $199 one-time

2. Get Price IDs (format: `price_xxx`)

3. Add to `.env`:
```bash
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_PRICE_ANNUAL=price_xxx
STRIPE_PRICE_GIFT_12MO=price_xxx
```

4. Checkout code already exists at:
   - `functions/handlers/stripe-checkout.ts` ✅
   - `_archive/curious-kellly/backend/src/api/checkout.js` ✅

---

## 📋 END OF DAY CHECKLIST

Before you stop today, verify:

- [ ] Database query returns Day 1 lesson
- [ ] Lesson Player loads (even with errors)
- [ ] Landing page displays correctly
- [ ] You know what Stripe products you need
- [ ] BURNDOWN_SPRINT_DEC7_17.md saved to reference tomorrow

---

## 📁 KEY FILES FOR TODAY

| File | What to do |
|------|------------|
| `BURNDOWN_SPRINT_DEC7_17.md` | Your 10-day execution plan |
| `SPRINT_ARTIFACTS_INDEX.md` | Index of all files |
| `curious-kellly/lesson-player-v2/js/app.js` | Wire to Supabase |
| `legacy_marketing_site.html` | Test landing page |
| `functions/handlers/stripe-checkout.ts` | Reference for Stripe |

---

## 🛑 BLOCKERS? HERE'S WHAT TO DO

### "I don't have Stripe keys"
→ **Skip Stripe today.** Focus on Lesson Player and Landing Page.

### "Supabase isn't connecting"
→ Check the anon key in app.js is correct:
```javascript
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIs...'; // This should work
```

### "Unity WebGL won't load"
→ **That's OK!** The landing page works without Unity. It's a "nice to have."

### "I'm overwhelmed"
→ **Pick ONE thing.** Just make the Lesson Player show Day 1's title from the database. That's enough for today.

---

## 🏁 TOMORROW (Dec 8) PREVIEW

- Finish Lesson Player wiring
- Create Stripe products
- Test checkout flow
- Start on email templates

---

## 💪 REMEMBER

> "A shipped product beats a perfect product."

You have:
- ✅ 365 lessons ready
- ✅ Working landing page
- ✅ Database with all content
- ✅ Checkout code written

You just need to connect the pieces. **You can do this.**

---

**START NOW. Open `curious-kellly/lesson-player-v2/js/app.js` and make Day 1 load from Supabase.**

🚀



