# Kelly OS App Content Map

**Purpose:** Maps each Kelly OS app to its content sources and data requirements

---

## App Content Mapping

### 📚 Knowledge Library
**Purpose:** Browse and explore all lesson content

**Content Sources:**
- `core_lessons.*` - All lesson metadata
- `core_lessons.recommended_videos` - External video resources
- `core_lessons.recommended_books` - Book recommendations
- `core_lessons.extended_explanation` - Deep dive content
- `core_lessons.fun_facts` - Interesting facts
- `core_lessons.common_misconceptions` - What people get wrong
- `core_lessons.real_world_applications` - Practical uses
- `lesson_atoms.content` - Phase-specific scripts

**Display:**
- Lesson cards with thumbnails
- Topic search/filter
- Resource links (videos, books)
- Extended explanations
- Related lessons

**Files:**
- `public/lesson-detail.html`
- `public/js/golden-v5-data-loader.js`

---

### 🔬 Research Lab
**Purpose:** Community-contributed expert knowledge and citations

**Content Sources:**
- `commons_lesson_notes` - Community notes
  - `type: 'expert_context'` - Deep domain knowledge
  - `type: 'source_citation'` - Academic sources
  - `type: 'historical_note'` - Historical connections
  - `type: 'teaching_tip'` - Pedagogical advice
  - `type: 'common_misconception'` - What learners misunderstand
  - `type: 'real_world_example'` - Practical applications
- `commons_user_contributions` - Contributor stats

**Display:**
- Expert notes per lesson
- Source citations
- Upvoted contributions
- Verified content badges

**Files:**
- `api/commons/notes.ts`
- `docs/features/LEARNER_COMMONS.md`

---

### 💻 Code Lab
**Purpose:** Interactive coding challenges tied to lessons

**Content Sources:**
- ❌ `lesson_code_challenges` - **TO CREATE**
  - `language` - 'javascript', 'python'
  - `starter_code` - Template code
  - `solution_code` - Answer
  - `test_cases` - Automated tests
  - `hints` - Progressive hints

**Display:**
- Code editor (Monaco/CodeMirror)
- Test runner
- Solution reveal
- Progress tracking

**Files:**
- None (new app)

---

### 📝 Learning Journal
**Purpose:** Personal notes and reflections on lessons

**Content Sources:**
- `lesson_history.notes` - User's personal notes
- `core_lessons.reflection_prompts` - Prompts for journaling
- `lesson_history.answers` - User's lesson answers
- `lesson_history.completed_at` - When completed

**Display:**
- Journal entries per lesson
- Reflection prompts
- Answer history
- Search notes
- Export journal

**Files:**
- `api/lesson-complete.ts`
- `api/reflection.ts`

---

### 📅 365 Calendar
**Purpose:** Visual calendar of all 365 lessons

**Content Sources:**
- `core_lessons.day_number` - Day 1-365
- `core_lessons.topic` - Lesson topic
- `core_lessons.thumbnail_url` - Calendar thumbnail
- `user_progress.completed` - Completion status
- `user_progress.completed_at` - When completed

**Display:**
- Calendar grid (365 days)
- Color-coded completion status
- Click to view lesson
- Current day highlight
- Streak visualization

**Files:**
- `public/calendar.html`

---

### 🎨 Visual Atlas
**Purpose:** Browse all visual assets (infographics, videos, images)

**Content Sources:**
- `lesson_visuals` - Visual asset registry
  - `infographic_url` - Educational diagrams
  - `infographic_urls` - Multiple infographics
  - `thumbnail_url` - Lesson thumbnails
  - `illustration_url` - Illustrations
- `lesson_atoms.visual_url` - Phase-specific visuals
- `lesson_atoms.hd_video_url` - HD video URLs
- `core_lessons.hero_image_url` - Hero images

**Display:**
- Gallery grid
- Filter by type (infographic, video, image)
- Filter by lesson/day
- Full-screen viewer
- Download options

**Files:**
- `api/visual/check.ts`
- `api/visual/generate.ts`
- `api/visual/stats.ts`

---

### 🔊 Audio Player
**Purpose:** Audio-only lesson playback

**Content Sources:**
- TTS system (`/api/tts`) - Generate audio on-demand
- Pre-generated audio (if available)
- `lesson_atoms.content.script` - Text to convert to speech

**Display:**
- Audio player controls
- Playlist (all lessons)
- Speed controls (0.5x - 2x)
- Background playback
- Download audio

**Files:**
- `public/js/kelly-audio.js`
- `api/tts.ts`
- `infrastructure/cloudflare/tts-worker/src/index.js`

---

### 💬 Ask Kelly
**Purpose:** Conversational AI chat with Kelly

**Content Sources:**
- ElevenLabs Conversational AI
- `lesson_atoms.content.script` - Lesson context
- `core_lessons.topic` - Current topic
- System prompts (persona-specific)
- ❌ `conversation_history` - **TO CREATE** for persistence

**Display:**
- Chat interface
- Voice input/output
- Conversation history
- Lesson context awareness
- Expression indicators

**Files:**
- `public/js/kelly-conversation.js`
- `api/elevenlabs-signed-url.ts`

---

### 🎯 Quiz Arena
**Purpose:** Standalone quiz interface

**Content Sources:**
- `core_lessons.quick_quiz_questions` - Quiz questions
  - `question` - Question text
  - `options` - Answer choices
  - `correct` - Correct answer
  - `explanation` - Why answer is correct

**Display:**
- Quiz interface
- Multiple choice questions
- Score tracking
- Explanation after answer
- Review mode

**Files:**
- `public/learn.html` (quiz section)

---

### 🗺️ World Map
**Purpose:** Geographic visualization of lesson locations

**Content Sources:**
- `api/geo-context` - User's location
- ❌ `core_lessons.related_locations` - **TO ADD**
  - `name` - Location name
  - `lat` - Latitude
  - `lng` - Longitude
  - `country` - Country code
  - `description` - Why it's relevant

**Display:**
- Interactive world map
- Location markers per lesson
- Click marker → view lesson
- Filter by region/country
- User's location highlight

**Files:**
- `api/geo-context.ts`

---

### ⏰ Timeline
**Purpose:** Historical timeline of lesson events

**Content Sources:**
- `core_lessons.historical_context` - Unstructured text
- ❌ `lesson_timeline_events` - **TO CREATE**
  - `event_name` - Event name
  - `year` - Year (or approximate)
  - `era` - Historical era
  - `description` - Event description
  - `related_lessons` - Other lessons mentioning this

**Display:**
- Timeline visualization
- Filter by era
- Click event → view lessons
- Year navigation
- Related events

**Files:**
- None (new app)

---

### 🃏 Flashcards
**Purpose:** Study mode with flashcards

**Content Sources:**
- ❌ `lesson_flashcards` - **TO CREATE** (auto-generate from quiz)
  - `front` - Question/term
  - `back` - Answer/definition
  - `hint` - Progressive hint
  - `source` - 'quiz', 'atom', 'reflection'
- Can derive from `quick_quiz_questions`

**Display:**
- Flashcard interface
- Flip animation
- Study mode (spaced repetition)
- Progress tracking
- Custom flashcard creation

**Files:**
- None (new app)

---

### 🛠️ Workshop
**Purpose:** Hands-on activities and creative projects

**Content Sources:**
- `core_lessons.hands_on_activities` - Activity instructions
- `core_lessons.creative_prompts` - Creative project prompts
- `core_lessons.challenge_questions` - Extended challenges
- `core_lessons.real_world_applications` - Practical projects

**Display:**
- Activity cards
- Step-by-step instructions
- Project templates
- Submission gallery (future)
- Share projects

**Files:**
- None (new app)

---

### 🏆 Trophy Room
**Purpose:** Achievements, badges, and progress

**Content Sources:**
- `users.streak_days` - Current streak
- `users.longest_streak` - Best streak
- `user_progress.completed` - Completed lessons count
- ❌ `user_badges` - **TO CREATE** (badge unlocks)
- ❌ `badge_definitions` - **TO CREATE** (badge metadata)

**Badge Types:**
- Streak badges (7 days, 30 days, 100 days)
- Lesson count badges (10, 50, 100, 365)
- Time-based badges (Early Bird, Night Owl)
- Interaction badges (Ask Kelly 5 questions)
- Language badges (Polyglot)
- Sharing badges (Teacher)

**Display:**
- Badge grid
- Unlocked/locked states
- Progress indicators
- Streak visualization
- Leaderboard (future)

**Files:**
- `public/me.html`
- `public/learn.html` (badge definitions)

---

### 👨‍👩‍👧‍👦 Family Hub
**Purpose:** Group learning with family/friends

**Content Sources:**
- `learning_groups` - Group definitions
- `group_members` - Group membership
- `user_progress` - Individual progress (aggregated)
- `daily_lesson_stats` - Group stats

**Display:**
- Group dashboard
- Member progress
- Group streak
- Invite members
- Group chat (future)

**Files:**
- `api/family/link.ts`
- `api/family/members.ts`

---

### 👤 My Profile
**Purpose:** User profile and settings

**Content Sources:**
- `users.*` - Profile data
- `users.streak_days` - Current streak
- `users.current_day` - Progress day
- `user_progress` - Lesson completion stats
- `users.subscription_tier` - Subscription status

**Display:**
- Profile info
- Progress stats
- Streak counter
- Settings
- Subscription management

**Files:**
- `public/me.html`

---

## Content Flow Diagram

```
core_lessons (365 lessons)
├── lesson_atoms (21,915 content pieces)
│   ├── visual_url → Visual Atlas
│   ├── hd_video_url → Visual Atlas, Audio Player
│   └── content.script → Audio Player, Ask Kelly
│
├── quick_quiz_questions → Quiz Arena, Flashcards
├── reflection_prompts → Learning Journal
├── recommended_videos → Knowledge Library
├── recommended_books → Knowledge Library
├── hands_on_activities → Workshop
├── creative_prompts → Workshop
├── historical_context → Timeline (needs extraction)
└── related_locations → World Map (needs addition)

user_progress → 365 Calendar, My Profile, Trophy Room
lesson_history.notes → Learning Journal
commons_lesson_notes → Research Lab
learning_groups → Family Hub
```

---

## Content Generation Priority

### Phase 1: Extract Existing (Week 1)
- ✅ All content types accessible
- ✅ Visual assets ready
- ✅ Quiz questions ready
- ✅ Notes system ready

### Phase 2: Create Missing Tables (Week 2)
- ⚠️ `lesson_flashcards` (auto-generate from quiz)
- ⚠️ `conversation_history` (persist chat)
- ⚠️ `user_badges` + `badge_definitions` (migrate badges)

### Phase 3: Add New Content (Weeks 3-4)
- ⚠️ `related_locations` in `core_lessons` (for World Map)
- ⚠️ `lesson_timeline_events` (extract from historical_context)
- ❌ `lesson_code_challenges` (for Code Lab - longer term)

---

**End of Content Map**


