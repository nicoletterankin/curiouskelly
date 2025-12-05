# Technical Blueprint: Lifetime Learner Experience

> Implementation plan to wire up spiral learning across learn.html, emails, and commons.

---

## Phase 1: Database Schema (Supabase Migration)

### 1.1 Enhance Users Table

```sql
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS kelly_remembers BOOLEAN DEFAULT true;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS first_lesson_at TIMESTAMPTZ;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS longest_streak INTEGER DEFAULT 0;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS total_lessons_completed INTEGER DEFAULT 0;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS unique_lessons_completed INTEGER DEFAULT 0;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS years_completed INTEGER DEFAULT 0;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS timezone TEXT DEFAULT 'America/New_York';

COMMENT ON COLUMN public.users.kelly_remembers IS 'User consent for Kelly to remember their history';
COMMENT ON COLUMN public.users.years_completed IS 'How many times they have completed all 365 lessons';
```

### 1.2 Create Lesson History Table

```sql
CREATE TABLE public.lesson_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  lesson_day INTEGER NOT NULL CHECK (lesson_day >= 1 AND lesson_day <= 366),
  
  -- When
  completed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  year_completed INTEGER NOT NULL, -- e.g., 2025
  
  -- How many times
  view_number INTEGER NOT NULL DEFAULT 1, -- 1st, 2nd, 3rd time seeing this
  
  -- What they did
  answers JSONB, -- { "q1": "A", "q2": "B", "q3": "C" }
  notes TEXT, -- Personal annotations
  time_spent_seconds INTEGER DEFAULT 0,
  
  -- What layer they saw
  layer TEXT DEFAULT 'foundation' CHECK (layer IN ('foundation', 'exploration', 'mastery', 'teaching')),
  
  -- Context
  user_age_at_completion INTEGER, -- Age when they completed it
  
  UNIQUE(user_id, lesson_day, year_completed)
);

-- Indexes for fast queries
CREATE INDEX idx_lesson_history_user ON public.lesson_history(user_id);
CREATE INDEX idx_lesson_history_day ON public.lesson_history(lesson_day);
CREATE INDEX idx_lesson_history_year ON public.lesson_history(year_completed);
CREATE INDEX idx_lesson_history_user_day ON public.lesson_history(user_id, lesson_day);

-- Enable RLS
ALTER TABLE public.lesson_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own history" ON public.lesson_history
  FOR SELECT USING (auth.uid() = user_id);
  
CREATE POLICY "Users can insert own history" ON public.lesson_history
  FOR INSERT WITH CHECK (auth.uid() = user_id);
```

### 1.3 Create Milestones Table

```sql
CREATE TABLE public.milestones (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  milestone_type TEXT NOT NULL CHECK (milestone_type IN (
    'first_lesson',
    'streak_7', 'streak_30', 'streak_100', 'streak_365', 'streak_1000',
    'year_complete_1', 'year_complete_5', 'year_complete_10',
    'decade_together',
    'birthday_lesson',
    'lessons_50', 'lessons_100', 'lessons_200', 'lessons_365'
  )),
  
  achieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  celebration_shown BOOLEAN DEFAULT false,
  metadata JSONB, -- Extra context like { "streak_count": 365 }
  
  UNIQUE(user_id, milestone_type)
);

CREATE INDEX idx_milestones_user ON public.milestones(user_id);

ALTER TABLE public.milestones ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own milestones" ON public.milestones
  FOR SELECT USING (auth.uid() = user_id);
```

### 1.4 Create Commons Aggregate Table

```sql
CREATE TABLE public.commons_answers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  lesson_day INTEGER NOT NULL,
  question_id TEXT NOT NULL, -- 'q1', 'q2', 'q3'
  answer_value TEXT NOT NULL, -- 'A', 'B', 'C'
  year INTEGER NOT NULL,
  
  -- Aggregates (updated via trigger or cron)
  count INTEGER DEFAULT 0,
  percentage NUMERIC(5,2) DEFAULT 0,
  
  UNIQUE(lesson_day, question_id, answer_value, year)
);

CREATE INDEX idx_commons_day ON public.commons_answers(lesson_day);
CREATE INDEX idx_commons_year ON public.commons_answers(year);
```

---

## Phase 2: API Endpoints

### 2.1 `/api/lesson-history/[day]` - Get User's History for a Lesson

```typescript
// GET /api/lesson-history/[day]
// Returns: User's history for this lesson across all years

interface LessonHistoryResponse {
  hasSeenBefore: boolean;
  viewCount: number;
  history: {
    year: number;
    completedAt: string;
    answers: Record<string, string>;
    notes: string | null;
    ageAtCompletion: number;
    layer: string;
  }[];
  recommendedLayer: 'foundation' | 'exploration' | 'mastery' | 'teaching';
  isBirthdayLesson: boolean;
}
```

### 2.2 `/api/lesson-complete` - Record Completion

```typescript
// POST /api/lesson-complete
// Body: { lessonDay, answers, notes, timeSpent }
// Returns: { milestones: Milestone[], streakUpdate: {...} }

interface LessonCompleteRequest {
  lessonDay: number;
  answers: Record<string, string>;
  notes?: string;
  timeSpentSeconds: number;
}

interface LessonCompleteResponse {
  success: boolean;
  viewNumber: number; // "This is your 3rd time learning this"
  newMilestones: Milestone[];
  streak: {
    current: number;
    longest: number;
    isNewRecord: boolean;
  };
  yearProgress: {
    completed: number;
    remaining: number;
    percentComplete: number;
  };
}
```

### 2.3 `/api/reflection/[day]` - Get Reflection Data

```typescript
// GET /api/reflection/[day]
// Returns: Year-over-year comparison of answers

interface ReflectionResponse {
  canReflect: boolean; // Only if seen 2+ times
  timeline: {
    year: number;
    age: number;
    answers: Record<string, string>;
  }[];
  insights: string[]; // "Your answer to Q1 changed from A to B"
}
```

### 2.4 `/api/commons/[day]` - Get Aggregate Data

```typescript
// GET /api/commons/[day]
// Returns: How everyone answered, by year

interface CommonsResponse {
  currentYear: {
    q1: { A: 45, B: 32, C: 23 };
    q2: { A: 50, B: 30, C: 20 };
    q3: { A: 40, B: 35, C: 25 };
  };
  historical: {
    [year: number]: {
      q1: { A: number, B: number, C: number };
      // ...
    };
  };
  userVsCommons: {
    q1: { userAnswer: 'A', popularAnswer: 'B', userPercentile: 45 };
    // ...
  };
}
```

---

## Phase 3: Frontend Integration (learn.html)

### 3.1 On Lesson Load

```javascript
// In lesson player initialization
async function initLessonWithHistory(lessonDay) {
  // 1. Check if user has kelly_remembers enabled
  const user = await getUser();
  if (!user.kelly_remembers) {
    return loadFreshLesson(lessonDay);
  }
  
  // 2. Fetch history
  const history = await fetch(`/api/lesson-history/${lessonDay}`);
  
  // 3. Show returning learner experience
  if (history.hasSeenBefore) {
    showReturningLearnerBanner({
      viewCount: history.viewCount,
      lastSeen: history.history[0].completedAt,
      previousAnswers: history.history[0].answers
    });
  }
  
  // 4. Check birthday
  if (history.isBirthdayLesson && isUserBirthday()) {
    showBirthdayCelebration();
  }
  
  // 5. Load appropriate layer
  loadLessonLayer(lessonDay, history.recommendedLayer);
}
```

### 3.2 Returning Learner Banner

```html
<!-- Shows when user has seen lesson before -->
<div id="returning-learner-banner" class="hidden">
  <div class="banner-content">
    <span class="view-count">You've learned this <strong>3 times</strong></span>
    <span class="last-seen">Last time: December 2024</span>
    <button onclick="showReflection()">See how you've grown →</button>
  </div>
</div>
```

### 3.3 Reflection Modal

```html
<!-- Shows year-over-year answer comparison -->
<div id="reflection-modal" class="hidden">
  <h2>Your Journey with This Lesson</h2>
  
  <div class="timeline">
    <div class="year-entry">
      <span class="year">2024</span>
      <span class="age">Age 28</span>
      <span class="answer">You chose: "Money is freedom"</span>
    </div>
    <div class="year-entry">
      <span class="year">2025</span>
      <span class="age">Age 29</span>
      <span class="answer">You chose: "Money is a tool"</span>
    </div>
  </div>
  
  <p class="insight">Your perspective has evolved. That's growth.</p>
</div>
```

### 3.4 Birthday Lesson Experience

```javascript
function showBirthdayCelebration() {
  const yearsWithLesson = calculateYearsWithBirthdayLesson();
  
  const modal = document.createElement('div');
  modal.innerHTML = `
    <div class="birthday-celebration">
      <h1>🎂 Happy Birthday!</h1>
      <p>This lesson is yours. It has been for ${yearsWithLesson} years.</p>
      <p>Let's learn it together, one more time.</p>
      <button onclick="closeBirthdayModal()">Begin →</button>
    </div>
  `;
  document.body.appendChild(modal);
}
```

### 3.5 On Lesson Complete

```javascript
async function onLessonComplete(lessonDay, answers, notes, timeSpent) {
  const response = await fetch('/api/lesson-complete', {
    method: 'POST',
    body: JSON.stringify({ lessonDay, answers, notes, timeSpentSeconds: timeSpent })
  });
  
  const result = await response.json();
  
  // Show view number
  if (result.viewNumber > 1) {
    showToast(`That's ${result.viewNumber} times you've learned this. ✨`);
  }
  
  // Show new milestones
  for (const milestone of result.newMilestones) {
    showMilestoneCelebration(milestone);
  }
  
  // Update streak display
  updateStreakDisplay(result.streak);
  
  // Show year progress
  showYearProgress(result.yearProgress);
}
```

---

## Phase 4: Email Integration (Resend)

### 4.1 Email Types & Triggers

| Email | Trigger | Template |
|-------|---------|----------|
| Birthday | Cron: Daily at midnight user timezone | `birthday-lesson` |
| Streak 7 | On milestone achieved | `streak-celebration` |
| Streak 30 | On milestone achieved | `streak-celebration` |
| Streak 365 | On milestone achieved | `streak-legendary` |
| Year Complete | On 365th unique lesson | `year-complete` |
| Anniversary | Cron: Annual on first_lesson_at | `anniversary` |
| Miss You | Cron: 7 days inactive | `gentle-return` |

### 4.2 Birthday Email Template

```html
<!-- Subject: Happy birthday, {name} -->
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

{name} —<br><br>

Today is yours.<br><br>

Your birthday lesson is waiting. You've learned it {viewCount} times now. 
Each year, it means something a little different.<br><br>

<a href="{birthdayLessonUrl}" style="color: #1e3a5f; text-decoration: underline;">
  Learn it again today.
</a><br><br>

I hope your year is filled with wonder.<br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

### 4.3 Year Complete Email Template

```html
<!-- Subject: You did it. 365 lessons. -->
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

{name} —<br><br>

365 lessons. 365 times you chose curiosity.<br><br>

This is your {yearNumber} complete year learning with me. 
That's {totalLessonsCompleted} moments of wonder.<br><br>

Most people don't finish things. You did. That's rare. That's beautiful.<br><br>

Here's to another year of learning together.<br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

### 4.4 Cron Jobs (Vercel)

```typescript
// api/cron/birthday-emails.ts
// Runs daily at 00:00 UTC

export default async function handler(req, res) {
  // Get users whose birthday is today (accounting for timezone)
  const birthdayUsers = await getBirthdayUsers();
  
  for (const user of birthdayUsers) {
    const birthdayLesson = getBirthdayLesson(user.birthday);
    const viewCount = await getViewCount(user.id, birthdayLesson);
    
    await sendEmail({
      to: user.email,
      template: 'birthday-lesson',
      data: {
        name: user.name,
        viewCount,
        birthdayLessonUrl: `https://curiouskelly.com/day/${birthdayLesson}`
      }
    });
  }
}
```

---

## Phase 5: Commons Integration

### 5.1 Historical View Component

```html
<div id="commons-historical">
  <h3>How learners answered over time</h3>
  
  <div class="chart">
    <!-- Bar chart showing answer distribution by year -->
    <div class="year-bar" data-year="2025">
      <div class="answer-a" style="width: 45%">A: 45%</div>
      <div class="answer-b" style="width: 32%">B: 32%</div>
      <div class="answer-c" style="width: 23%">C: 23%</div>
    </div>
    <div class="year-bar" data-year="2024">
      <div class="answer-a" style="width: 40%">A: 40%</div>
      <div class="answer-b" style="width: 35%">B: 35%</div>
      <div class="answer-c" style="width: 25%">C: 25%</div>
    </div>
  </div>
  
  <p class="insight">
    In 2024, most learners chose B. This year, A is winning.
    <span class="user-context">You chose A. You're with the 45%.</span>
  </p>
</div>
```

### 5.2 Aggregate Update (After Each Completion)

```typescript
// Called after lesson complete
async function updateCommonsAggregate(lessonDay, answers) {
  const year = new Date().getFullYear();
  
  for (const [questionId, answerValue] of Object.entries(answers)) {
    await supabase.rpc('increment_commons_answer', {
      p_lesson_day: lessonDay,
      p_question_id: questionId,
      p_answer_value: answerValue,
      p_year: year
    });
  }
}

// Supabase function
CREATE OR REPLACE FUNCTION increment_commons_answer(
  p_lesson_day INTEGER,
  p_question_id TEXT,
  p_answer_value TEXT,
  p_year INTEGER
) RETURNS void AS $$
BEGIN
  INSERT INTO commons_answers (lesson_day, question_id, answer_value, year, count)
  VALUES (p_lesson_day, p_question_id, p_answer_value, p_year, 1)
  ON CONFLICT (lesson_day, question_id, answer_value, year)
  DO UPDATE SET count = commons_answers.count + 1;
END;
$$ LANGUAGE plpgsql;
```

---

## Phase 6: Layer System

### 6.1 Determine Recommended Layer

```typescript
function getRecommendedLayer(
  viewCount: number,
  userAge: number,
  hasCompletedYear: boolean
): 'foundation' | 'exploration' | 'mastery' | 'teaching' {
  
  // Teaching layer: 10+ views OR is a parent/educator
  if (viewCount >= 10) return 'teaching';
  
  // Mastery layer: 5+ views OR 18+ and 3+ views
  if (viewCount >= 5 || (userAge >= 18 && viewCount >= 3)) return 'mastery';
  
  // Exploration layer: 2+ views OR 13+
  if (viewCount >= 2 || userAge >= 13) return 'exploration';
  
  // Foundation: default
  return 'foundation';
}
```

### 6.2 Content Variants (Already in lesson_shards)

The `lesson_shards` table already has age-based content. We map layers to age buckets:

| Layer | Age Bucket | Description |
|-------|------------|-------------|
| Foundation | 5-8 | Simple, concrete, visual |
| Exploration | 9-14 | Nuance, edge cases |
| Mastery | 15-22 | Philosophy, application |
| Teaching | 23+ | How to share with others |

---

## Implementation Order

### Sprint 1: Database & Core APIs (Day 1-2)
1. ✅ Run schema migrations
2. ✅ Create `/api/lesson-history/[day]`
3. ✅ Create `/api/lesson-complete`
4. ✅ Create `/api/reflection/[day]`

### Sprint 2: Frontend Integration (Day 3-4)
1. ✅ Add history check on lesson load
2. ✅ Add returning learner banner
3. ✅ Add reflection modal
4. ✅ Add birthday detection
5. ✅ Update completion flow

### Sprint 3: Email System (Day 5)
1. ✅ Create birthday email cron
2. ✅ Create year-complete email trigger
3. ✅ Create anniversary email cron
4. ✅ Create gentle-return email cron

### Sprint 4: Commons (Day 6)
1. ✅ Add historical aggregate view
2. ✅ Add year-over-year comparison
3. ✅ Add "your place in history" context

### Sprint 5: Polish (Day 7)
1. ✅ Test full flow
2. ✅ Add celebration animations
3. ✅ Privacy controls UI
4. ✅ Documentation

---

## Files to Create/Modify

### New Files
- `api/lesson-history/[day].ts`
- `api/lesson-complete.ts`
- `api/reflection/[day].ts`
- `api/commons/[day].ts`
- `api/cron/birthday-emails.ts`
- `api/cron/anniversary-emails.ts`
- `api/cron/gentle-return.ts`
- `public/js/lesson-history.js`
- `public/js/reflection.js`
- `public/js/milestones.js`

### Modified Files
- `public/learn.html` - Add history UI components
- `public/js/lesson-player.js` - Integrate history checks
- `public/commons.html` - Add historical view
- `vercel.json` - Add new cron jobs

---

## Privacy & Consent

### Kelly Remembers Toggle

```html
<div class="setting-row">
  <label>
    <input type="checkbox" id="kelly-remembers" checked>
    Kelly remembers my learning history
  </label>
  <p class="setting-description">
    When enabled, Kelly tracks your progress and shows how your 
    understanding evolves over time. You can export or delete 
    this data anytime.
  </p>
</div>
```

### Data Export

```typescript
// GET /api/export-my-data
// Returns all user data as JSON download
```

### Data Deletion

```typescript
// DELETE /api/delete-my-history
// Removes all lesson_history and milestones for user
```

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Return rate on seen lessons | +20% engagement | Compare completion rate 1st vs 2nd+ view |
| Birthday lesson completion | 80%+ | Track birthday lesson completions |
| Year completion rate | 5% of active users | Count users with years_completed >= 1 |
| Reflection engagement | 30% click rate | Track reflection modal opens |
| Kelly Remembers opt-in | 90%+ | Count users with kelly_remembers = true |

---

*This blueprint is ready for implementation. Estimated time: 7 days.*

