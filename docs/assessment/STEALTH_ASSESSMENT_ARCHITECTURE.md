# 🔮 Stealth Assessment Architecture

**Created:** December 16, 2025  
**Status:** DESIGN SPECIFICATION  
**Philosophy:** Assessment that feels like play, not testing

---

## The Invisible Test Philosophy

> "The best test is one you never know you're taking."

Traditional assessments create anxiety and often measure test-taking ability rather than knowledge. Kelly's approach is **embedded assessment** — we observe natural learning behaviors and build a rich understanding of each learner without ever asking them to "take a test."

### Core Principles

1. **Never Feel Like a Test** — No "quiz," "test," or "assessment" language
2. **Continuous Calibration** — Every lesson refines our understanding
3. **Behavioral Signals > Answers** — How they learn matters more than what they get right
4. **Growth-Oriented Display** — Show "Your Learning Journey," not "Your Score"
5. **Zero Anxiety** — Learners see insights about themselves, never grades

---

## What We Observe (The Invisible Signals)

### 🎯 Response Quality Metrics

| Signal | What It Tells Us | How We Capture It |
|--------|------------------|-------------------|
| `first_try_accuracy` | Conceptual understanding | Track if they select "best" option first |
| `option_quality_sequence` | Learning pattern | Array: ['best', 'good', 'redirect', 'best'] |
| `hint_usage` | Need for scaffolding | Count of "no choice" hints shown |
| `redirect_recovery` | Resilience after mistakes | Does redirect → good choice next? |

### ⏱️ Timing Metrics

| Signal | What It Tells Us | How We Capture It |
|--------|------------------|-------------------|
| `time_to_first_choice` | Confidence/deliberation | ms from options shown to selection |
| `phase_durations` | Engagement depth | Time spent per phase (welcome→wisdom) |
| `audio_replays` | Comprehension needs | Count of audio segment replays |
| `video_replay_count` | Visual learning preference | Times video was rewatched |
| `rushing_detected` | Possible disengagement | Consistent <2s choices |
| `exploring_detected` | Deep engagement | Consistent 5-25s thoughtful choices |

### 🔄 Engagement Patterns

| Signal | What It Tells Us | How We Capture It |
|--------|------------------|-------------------|
| `completion_rate` | Overall engagement | Completed phases / Total phases |
| `session_abandonment_phase` | Where we lose them | Last phase before exit (if not complete) |
| `return_rate` | Habit formation | Days between lesson completions |
| `wisdom_engagement` | Values reflection content | Time spent on wisdom phase |
| `choice_revision_count` | Deliberative thinking | If we allow choice changes, track them |

### 📊 Subject Proficiency Signals

| Signal | What It Tells Us | How We Capture It |
|--------|------------------|-------------------|
| `topic_performance` | Subject strengths | Accuracy grouped by lesson tags |
| `difficulty_comfort` | Level appropriateness | Success rate by lesson difficulty |
| `archetype_resonance` | Learning personality | Which archetype gets best engagement |
| `age_bracket_fit` | Content level match | Performance vs. selected age setting |

---

## Stealth Placement: The First 3 Lessons

Instead of a separate "placement test," the first 3 lessons Kelly delivers are **calibration lessons** that:
- Feel completely normal
- Cover deliberately varied difficulty
- Include strategically designed questions
- Build our initial learner profile

### Lesson 1 (Day 1): Baseline Establishment

```
Purpose: Establish baseline timing and behavior patterns
What we learn:
- Natural response speed (are they deliberate or quick?)
- Comfort with the format (do they need hints?)
- Engagement style (do they explore or rush?)
- First-try accuracy baseline

Kelly's View: "Getting to know you"
```

### Lesson 2 (Day 2): Difficulty Calibration

```
Purpose: Test response to varied difficulty
What we learn:
- How they handle easy vs. hard questions
- Frustration tolerance (do they persist or abandon?)
- Self-awareness (do they adjust when struggling?)

Kelly's View: "Seeing how you learn"
```

### Lesson 3 (Day 3): Pattern Confirmation

```
Purpose: Confirm initial patterns, adjust if needed
What we learn:
- Consistency of behavior
- Learning velocity (improving over sessions?)
- Preferred engagement depth

Kelly's View: "Your learning personality is emerging"
```

### After First Week: Initial Profile Complete

Kelly can now populate the "Your Learning Journey" section in Settings with:
- Engagement style (Explorer, Deliberator, Speedrunner)
- Subject comfort zones
- Optimal session length
- Streak likelihood prediction

---

## Data Model

### New Table: `learner_observations`

Captures all invisible signals per lesson session.

```sql
CREATE TABLE IF NOT EXISTS public.learner_observations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
  lesson_id UUID REFERENCES public.lessons(id) ON DELETE SET NULL,
  day_number INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  
  -- Response quality
  first_try_correct BOOLEAN,
  option_quality_sequence TEXT[], -- ['best', 'good', 'redirect', 'best', 'good']
  hints_used INTEGER DEFAULT 0,
  redirects_count INTEGER DEFAULT 0,
  redirect_recoveries INTEGER DEFAULT 0, -- good choice after redirect
  
  -- Timing (all in milliseconds)
  phase_durations JSONB, -- {"welcome": 15000, "q1": 32000, ...}
  choice_timings INTEGER[], -- [3200, 4100, 2800, 5500] per phase
  avg_choice_time INTEGER,
  rushed_choices_count INTEGER DEFAULT 0, -- choices < 2000ms
  deliberate_choices_count INTEGER DEFAULT 0, -- choices 5000-25000ms
  
  -- Engagement
  audio_replays INTEGER DEFAULT 0,
  video_replays INTEGER DEFAULT 0,
  pauses_count INTEGER DEFAULT 0,
  total_session_duration INTEGER, -- ms
  completed BOOLEAN DEFAULT false,
  abandoned_at_phase TEXT, -- NULL if completed
  
  -- Context
  archetype TEXT,
  age_setting TEXT,
  language TEXT DEFAULT 'en',
  device_type TEXT, -- 'mobile', 'tablet', 'desktop'
  
  -- Timestamps
  started_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  
  UNIQUE(user_id, session_id)
);

-- Enable RLS
ALTER TABLE public.learner_observations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own observations" ON public.learner_observations
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own observations" ON public.learner_observations
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Indexes for analytics
CREATE INDEX idx_observations_user_id ON public.learner_observations(user_id);
CREATE INDEX idx_observations_day ON public.learner_observations(day_number);
CREATE INDEX idx_observations_completed ON public.learner_observations(completed);
```

### New Table: `learner_insights`

Aggregated, computed insights shown to user in Settings.

```sql
CREATE TABLE IF NOT EXISTS public.learner_insights (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
  
  -- Overall Profile
  engagement_style TEXT CHECK (engagement_style IN (
    'explorer', 'deliberator', 'speedrunner', 'reflector', 'undetermined'
  )) DEFAULT 'undetermined',
  learning_velocity TEXT CHECK (learning_velocity IN (
    'accelerating', 'steady', 'warming_up', 'undetermined'
  )) DEFAULT 'undetermined',
  
  -- Proficiency (0-100 scale, computed from observations)
  overall_mastery INTEGER DEFAULT 0,
  subject_proficiencies JSONB DEFAULT '{}', -- {"science": 72, "life-skills": 85}
  
  -- Engagement Metrics
  avg_session_duration INTEGER, -- ms
  optimal_session_length INTEGER, -- recommended based on patterns
  best_time_of_day TEXT, -- 'morning', 'afternoon', 'evening'
  streak_reliability DECIMAL(3,2), -- 0.00 to 1.00
  
  -- Strengths & Growth Areas (positive framing)
  strengths TEXT[], -- ['quick thinker', 'persistent', 'curious']
  growth_areas TEXT[], -- ['taking time to reflect', 'exploring options']
  
  -- Archetype Affinity
  preferred_archetype TEXT,
  archetype_scores JSONB DEFAULT '{}', -- {"explorer": 0.8, "scientist": 0.6}
  
  -- Progress Tracking
  lessons_analyzed INTEGER DEFAULT 0,
  confidence_level DECIMAL(3,2) DEFAULT 0.00, -- How confident we are in profile
  last_analyzed_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.learner_insights ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own insights" ON public.learner_insights
  FOR SELECT USING (auth.uid() = user_id);

-- Trigger for updated_at
CREATE TRIGGER update_learner_insights_updated_at BEFORE UPDATE ON public.learner_insights
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### Extended `user_progress` Table

Add observation aggregates to existing progress tracking:

```sql
ALTER TABLE public.user_progress ADD COLUMN IF NOT EXISTS 
  observation_summary JSONB DEFAULT '{}';
-- Contains: {first_try_correct, hints_used, session_duration, choice_quality}
```

---

## Collection Implementation

### JavaScript Observation Tracker

```javascript
// STEALTH OBSERVATION SYSTEM
// Collects behavioral signals invisibly during normal lesson flow

class LearnerObserver {
  constructor() {
    this.sessionId = crypto.randomUUID();
    this.observations = {
      phaseTimings: {},
      choiceTimings: [],
      optionQualities: [],
      hintsUsed: 0,
      audioReplays: 0,
      videoReplays: 0,
      rushingDetected: false,
      exploringDetected: false,
      pausesCount: 0
    };
    this.phaseStartTime = null;
    this.optionsShownTime = null;
  }

  // Called when a phase renders
  onPhaseStart(phaseName) {
    this.phaseStartTime = Date.now();
    this.observations.phaseTimings[phaseName] = { startedAt: this.phaseStartTime };
  }

  // Called when options appear
  onOptionsShown() {
    this.optionsShownTime = Date.now();
  }

  // Called when user makes a choice
  onChoice(optionQuality) {
    const choiceTime = Date.now() - this.optionsShownTime;
    
    this.observations.choiceTimings.push(choiceTime);
    this.observations.optionQualities.push(optionQuality);
    
    // Detect patterns
    if (choiceTime < 2000) {
      this.observations.rushingDetected = 
        this.observations.choiceTimings.filter(t => t < 2000).length >= 3;
    }
    if (choiceTime >= 5000 && choiceTime <= 25000) {
      this.observations.exploringDetected = 
        this.observations.choiceTimings.filter(t => t >= 5000 && t <= 25000).length >= 2;
    }
  }

  // Called when hint is shown
  onHintShown() {
    this.observations.hintsUsed++;
  }

  // Called when audio is replayed
  onAudioReplay() {
    this.observations.audioReplays++;
  }

  // Called when phase ends
  onPhaseEnd(phaseName) {
    if (this.observations.phaseTimings[phaseName]) {
      this.observations.phaseTimings[phaseName].duration = 
        Date.now() - this.observations.phaseTimings[phaseName].startedAt;
    }
  }

  // Calculate first-try accuracy
  getFirstTryAccuracy() {
    const bestChoices = this.observations.optionQualities.filter(q => q === 'best').length;
    return bestChoices / Math.max(this.observations.optionQualities.length, 1);
  }

  // Calculate redirect recovery rate
  getRedirectRecoveryRate() {
    let redirects = 0;
    let recoveries = 0;
    
    for (let i = 0; i < this.observations.optionQualities.length - 1; i++) {
      if (this.observations.optionQualities[i] === 'redirect') {
        redirects++;
        if (['best', 'good'].includes(this.observations.optionQualities[i + 1])) {
          recoveries++;
        }
      }
    }
    
    return redirects > 0 ? recoveries / redirects : 1;
  }

  // Get summary for saving
  getSummary() {
    return {
      sessionId: this.sessionId,
      firstTryAccuracy: this.getFirstTryAccuracy(),
      optionQualitySequence: this.observations.optionQualities,
      hintsUsed: this.observations.hintsUsed,
      redirectsCount: this.observations.optionQualities.filter(q => q === 'redirect').length,
      redirectRecoveryRate: this.getRedirectRecoveryRate(),
      phaseDurations: this.observations.phaseTimings,
      choiceTimings: this.observations.choiceTimings,
      avgChoiceTime: this.observations.choiceTimings.reduce((a, b) => a + b, 0) / 
                     Math.max(this.observations.choiceTimings.length, 1),
      rushedChoicesCount: this.observations.choiceTimings.filter(t => t < 2000).length,
      deliberateChoicesCount: this.observations.choiceTimings.filter(t => t >= 5000 && t <= 25000).length,
      audioReplays: this.observations.audioReplays,
      videoReplays: this.observations.videoReplays,
      rushingDetected: this.observations.rushingDetected,
      exploringDetected: this.observations.exploringDetected,
      totalDuration: Date.now() - Object.values(this.observations.phaseTimings)[0]?.startedAt
    };
  }
}
```

### Integration Points in `learn.html`

```javascript
// Initialize observer when lesson loads
let observer = null;

function initLessonObserver() {
  observer = new LearnerObserver();
}

// Hook into existing functions:

// In renderPhase():
observer?.onPhaseStart(phaseName);
// When options appear:
observer?.onOptionsShown();

// In selectChoice():
observer?.onChoice(selectedOption.quality || 'good');
observer?.onPhaseEnd(currentPhase);

// In showStuckHint():
observer?.onHintShown();

// When audio replays:
observer?.onAudioReplay();

// In completeLesson():
const summary = observer?.getSummary();
await saveObservation(summary);
```

### Save Observation to Supabase

```javascript
async function saveObservation(summary) {
  if (!summary || !isLoggedIn()) return;
  
  const observation = {
    user_id: getUserId(),
    lesson_id: currentLesson?.id,
    day_number: state.currentDay,
    session_id: summary.sessionId,
    
    first_try_correct: summary.firstTryAccuracy >= 0.8,
    option_quality_sequence: summary.optionQualitySequence,
    hints_used: summary.hintsUsed,
    redirects_count: summary.redirectsCount,
    redirect_recoveries: Math.round(summary.redirectRecoveryRate * summary.redirectsCount),
    
    phase_durations: summary.phaseDurations,
    choice_timings: summary.choiceTimings,
    avg_choice_time: Math.round(summary.avgChoiceTime),
    rushed_choices_count: summary.rushedChoicesCount,
    deliberate_choices_count: summary.deliberateChoicesCount,
    
    audio_replays: summary.audioReplays,
    video_replays: summary.videoReplays,
    total_session_duration: summary.totalDuration,
    completed: true,
    
    archetype: state.kellyId,
    age_setting: state.ageBucket,
    device_type: detectDeviceType()
  };

  await supabase.from('learner_observations').insert(observation);
}
```

---

## Insight Computation (Background Job)

Run periodically to update `learner_insights` from raw `learner_observations`.

```sql
-- Function to recompute insights for a user
CREATE OR REPLACE FUNCTION compute_learner_insights(target_user_id UUID)
RETURNS VOID AS $$
DECLARE
  obs_count INTEGER;
  avg_accuracy DECIMAL;
  avg_session INTEGER;
  rush_ratio DECIMAL;
  deliberate_ratio DECIMAL;
  engagement_type TEXT;
  velocity_type TEXT;
BEGIN
  -- Count observations
  SELECT COUNT(*) INTO obs_count
  FROM learner_observations
  WHERE user_id = target_user_id AND completed = true;
  
  IF obs_count < 3 THEN
    -- Not enough data yet
    RETURN;
  END IF;
  
  -- Calculate averages
  SELECT 
    AVG(CASE WHEN first_try_correct THEN 1 ELSE 0 END),
    AVG(total_session_duration),
    AVG(rushed_choices_count::DECIMAL / GREATEST(array_length(choice_timings, 1), 1)),
    AVG(deliberate_choices_count::DECIMAL / GREATEST(array_length(choice_timings, 1), 1))
  INTO avg_accuracy, avg_session, rush_ratio, deliberate_ratio
  FROM learner_observations
  WHERE user_id = target_user_id AND completed = true;
  
  -- Determine engagement style
  IF rush_ratio > 0.5 THEN
    engagement_type := 'speedrunner';
  ELSIF deliberate_ratio > 0.5 THEN
    engagement_type := 'deliberator';
  ELSIF avg_accuracy > 0.7 AND deliberate_ratio > 0.3 THEN
    engagement_type := 'explorer';
  ELSE
    engagement_type := 'reflector';
  END IF;
  
  -- Determine learning velocity (compare recent 5 vs. first 5)
  -- (Simplified - full implementation would be more sophisticated)
  velocity_type := 'steady';
  
  -- Upsert insights
  INSERT INTO learner_insights (
    user_id,
    engagement_style,
    learning_velocity,
    overall_mastery,
    avg_session_duration,
    lessons_analyzed,
    confidence_level,
    last_analyzed_at
  ) VALUES (
    target_user_id,
    engagement_type,
    velocity_type,
    LEAST(ROUND(avg_accuracy * 100), 100),
    ROUND(avg_session),
    obs_count,
    LEAST(obs_count / 20.0, 1.0), -- Confidence builds over 20 lessons
    NOW()
  )
  ON CONFLICT (user_id) DO UPDATE SET
    engagement_style = EXCLUDED.engagement_style,
    learning_velocity = EXCLUDED.learning_velocity,
    overall_mastery = EXCLUDED.overall_mastery,
    avg_session_duration = EXCLUDED.avg_session_duration,
    lessons_analyzed = EXCLUDED.lessons_analyzed,
    confidence_level = EXCLUDED.confidence_level,
    last_analyzed_at = EXCLUDED.last_analyzed_at;
    
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

---

## User-Facing Display: "Your Learning Journey"

### Settings Panel Section

```html
<div class="settings-section learning-journey">
  <h3>🌟 Your Learning Journey</h3>
  
  <div class="journey-overview">
    <div class="insight-card engagement-style">
      <div class="insight-emoji">🧭</div>
      <div class="insight-title">Your Style</div>
      <div class="insight-value">Explorer</div>
      <div class="insight-description">
        You love to understand deeply before moving on
      </div>
    </div>
    
    <div class="insight-card mastery">
      <div class="insight-emoji">📈</div>
      <div class="insight-title">Growth</div>
      <div class="insight-progress">
        <div class="progress-bar" style="width: 72%"></div>
      </div>
      <div class="insight-description">
        You've explored 72% of the learning landscape
      </div>
    </div>
    
    <div class="insight-card velocity">
      <div class="insight-emoji">🚀</div>
      <div class="insight-title">Momentum</div>
      <div class="insight-value">Accelerating!</div>
      <div class="insight-description">
        Your recent lessons show faster understanding
      </div>
    </div>
  </div>
  
  <div class="strengths-section">
    <h4>✨ Your Superpowers</h4>
    <div class="strength-badges">
      <span class="badge">🎯 First-try thinker</span>
      <span class="badge">🔁 Great at bouncing back</span>
      <span class="badge">🌱 Curious explorer</span>
    </div>
  </div>
  
  <div class="growth-section">
    <h4>🌱 Areas to Nurture</h4>
    <div class="growth-badges">
      <span class="badge gentle">Take your time with hard ones</span>
      <span class="badge gentle">The wisdom phase loves attention</span>
    </div>
  </div>
  
  <div class="subject-mastery">
    <h4>📚 Subject Exploration</h4>
    <div class="subject-bars">
      <div class="subject">
        <span class="name">Science</span>
        <div class="bar"><div class="fill" style="width: 85%"></div></div>
        <span class="label">Expert Explorer</span>
      </div>
      <div class="subject">
        <span class="name">Life Skills</span>
        <div class="bar"><div class="fill" style="width: 68%"></div></div>
        <span class="label">Growing Strong</span>
      </div>
      <div class="subject">
        <span class="name">History</span>
        <div class="bar"><div class="fill" style="width: 45%"></div></div>
        <span class="label">Just Starting</span>
      </div>
    </div>
  </div>
  
  <p class="privacy-note">
    Kelly learns how you learn to be a better teacher for you.
    This data is private and only visible to you.
  </p>
</div>
```

### Styling (CSS)

```css
.learning-journey {
  background: linear-gradient(135deg, #f8f4ff 0%, #e8f4ff 100%);
  border-radius: 16px;
  padding: 24px;
  margin: 16px 0;
}

.journey-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
  margin: 20px 0;
}

.insight-card {
  background: white;
  border-radius: 12px;
  padding: 16px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.insight-emoji {
  font-size: 32px;
  margin-bottom: 8px;
}

.insight-title {
  font-size: 12px;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.insight-value {
  font-size: 20px;
  font-weight: 600;
  color: #2d3748;
  margin: 4px 0;
}

.insight-description {
  font-size: 11px;
  color: #718096;
  line-height: 1.4;
}

.strength-badges, .growth-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0;
}

.badge {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 13px;
}

.badge.gentle {
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.subject-bars {
  margin: 16px 0;
}

.subject {
  display: grid;
  grid-template-columns: 100px 1fr 100px;
  align-items: center;
  gap: 12px;
  margin: 8px 0;
}

.bar {
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
}

.bar .fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border-radius: 4px;
  transition: width 0.6s ease;
}

.privacy-note {
  font-size: 11px;
  color: #a0aec0;
  text-align: center;
  margin-top: 20px;
  font-style: italic;
}
```

---

## Privacy & Trust Considerations

### What We DON'T Do

❌ Share observation data with third parties  
❌ Use data for advertising  
❌ Compare learners to each other  
❌ Show "grades" or "scores" that feel judgmental  
❌ Use data to limit access to content  
❌ Store data indefinitely without user control  

### What We DO

✅ Keep all data private to the individual user  
✅ Allow users to export their data  
✅ Allow users to delete their learning history  
✅ Use data only to improve Kelly's teaching for that user  
✅ Present insights in growth-oriented, encouraging language  
✅ Explain clearly what data we collect and why  

### Settings Controls

```html
<div class="privacy-controls">
  <h4>🔒 Your Data, Your Control</h4>
  
  <div class="toggle-row">
    <label>Allow Kelly to learn how you learn</label>
    <input type="checkbox" id="observation-enabled" checked>
    <span class="helper">Helps Kelly teach you better</span>
  </div>
  
  <div class="action-buttons">
    <button class="btn-secondary" onclick="exportMyData()">
      📤 Export My Learning Data
    </button>
    <button class="btn-danger" onclick="deleteMyHistory()">
      🗑️ Delete Learning History
    </button>
  </div>
</div>
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create `learner_observations` table
- [ ] Create `learner_insights` table
- [ ] Implement JavaScript `LearnerObserver` class
- [ ] Hook into existing lesson player events

### Phase 2: Collection (Week 2)
- [ ] Wire up all observation points in `learn.html`
- [ ] Test data collection with sample sessions
- [ ] Verify RLS policies work correctly
- [ ] Add observation summary to `user_progress`

### Phase 3: Insights (Week 3)
- [ ] Implement `compute_learner_insights` function
- [ ] Set up periodic computation (via Supabase Edge Function or cron)
- [ ] Test insight accuracy with sample data

### Phase 4: Display (Week 4)
- [ ] Build "Your Learning Journey" UI component
- [ ] Integrate into Settings panel
- [ ] Add privacy controls
- [ ] User testing for emotional response (should feel positive!)

### Phase 5: Refinement (Ongoing)
- [ ] Tune engagement style detection
- [ ] Add subject proficiency tracking
- [ ] Improve strength/growth area identification
- [ ] A/B test different display styles

---

## Success Metrics

### Technical Success
- 100% of lesson completions have observations saved
- Insights update within 24 hours of new data
- No user-reported privacy concerns

### Experience Success
- Users report "Your Learning Journey" feels encouraging, not judgmental
- Users engage with the insights (view rate > 50%)
- Users don't realize they're being "tested"

### Learning Success
- Kelly's adaptive hints improve over time (fewer redirects needed)
- Learners who view insights maintain higher streaks
- Subject proficiency predictions match self-reported confidence

---

## Appendix: Engagement Style Definitions

| Style | Characteristics | Kelly's Approach |
|-------|-----------------|------------------|
| **Explorer** | Takes time, tries different options, asks "why" | Give deeper context, more wisdom |
| **Deliberator** | Reads carefully, considers before choosing | Allow more thinking time, fewer nudges |
| **Speedrunner** | Quick choices, efficient completion | Celebrate efficiency, offer depth optionally |
| **Reflector** | Pauses often, may replay content | Validate thinking, patience in pacing |

---

**End of Stealth Assessment Architecture**

*"The best learning happens when you forget you're learning."*
