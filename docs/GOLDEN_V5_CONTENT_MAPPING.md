# 🗺️ GOLDEN V5: CONTENT MAPPING & INTEGRATION

**Date:** December 9, 2025  
**Status:** 🚧 IN PROGRESS  
**Goal:** Map ALL content, icons, and features to Golden V5's spatial intelligence system

---

## 📊 **DATABASE SCHEMA OVERVIEW**

### **Core Tables:**
1. **`core_lessons`** (365 rows) - Master lesson data
   - `day_number`, `topic`, `universal_truth`, `icon_emoji`
   - `estimated_duration`, `ideal_age_range`, `difficulty_level`
   - Marketing fields, learning objectives, resources

2. **`lesson_atoms`** (20,341 rows) - Archetype-specific content
   - `archetype` (The Explorer, The Scientist, The Rebel, etc.)
   - `phase` (Hook, Fact1, Fact2, Fact3, Wisdom)
   - `content` (JSONB: script, options, kellyPose, kellyEmotion)

3. **`kelly_video_assets`** (2,008 rows) - HD video files
   - `day_number`, `phase`, `template`, `asset_type`
   - `storage_path`, `public_url`, `duration_seconds`
   - `quality_tier`, `resolution`, `status`

4. **`users`** - User profiles and progress
   - `current_day`, `streak_days`, `preferences`
   - `subscription_tier`, `subscription_status`

5. **`lesson_history`** - Completion tracking
   - `lesson_day`, `completed_at`, `answers`, `time_spent_seconds`

---

## 🎬 **DAY 1: "STARTING FRESH"** (REAL DATA)

### **Core Lesson Data:**
```json
{
  "day_number": 1,
  "topic": "Starting Fresh",
  "universal_truth": "Fresh starts provide psychological permission to change—the calendar creates natural reset points.",
  "icon_emoji": "🌅",
  "estimated_duration": 8,
  "ideal_age_range": "All ages (content adapts to learner)",
  "difficulty_level": "Beginner"
}
```

### **Phase Structure (The Explorer Archetype):**

#### **HOOK Phase:**
```json
{
  "script": "Have you ever felt the thrill of starting fresh? Imagine standing at the edge of a new adventure, where every choice can lead to exciting changes. Today, we're going to explore how a fresh start can empower us to make the changes we crave in our lives!",
  "options": [
    {
      "letter": "A",
      "text": "What are some ways we can create a fresh start in our lives?",
      "quality": "good",
      "response": "Great question! A fresh start can be as simple as setting new goals or changing your environment. What specific area of your life are you thinking about refreshing?"
    },
    {
      "letter": "B",
      "text": "How can I take advantage of fresh starts to improve my habits?",
      "quality": "best",
      "response": "That's an excellent thought! Fresh starts allow us to reset our habits. Consider breaking down your goals into manageable steps and celebrate small victories along the way."
    },
    {
      "letter": "C",
      "text": "Why do you think fresh starts are so psychologically impactful?",
      "quality": "redirect",
      "response": "That's a fascinating reflection! Fresh starts symbolize new possibilities and can help us let go of past limitations, making room for growth and exploration."
    }
  ],
  "kellyPose": "explaining",
  "kellyEmotion": "curious"
}
```

#### **FACT1 Phase:**
```json
{
  "script": "Did you know that studies show people are more likely to pursue their goals after a significant date, like New Year's Day or a birthday? This phenomenon, known as the 'fresh start effect,' highlights how these points in time can act as psychological cues, motivating us to make meaningful changes in our lives.",
  "options": [
    {
      "letter": "A",
      "text": "What are some examples of fresh starts?",
      "quality": "good",
      "response": "Fresh starts can come in many forms, like starting a new job, moving to a new city, or even just beginning a new week. Each of these moments gives us the chance to redefine our goals and behaviors."
    },
    {
      "letter": "B",
      "text": "How can I create my own fresh start?",
      "quality": "best",
      "response": "To create your own fresh start, consider setting a specific date to begin a new habit or project. Write down your goals and visualize what success looks like to you, embracing the excitement of this new journey."
    },
    {
      "letter": "C",
      "text": "Why do fresh starts feel so motivating?",
      "quality": "redirect",
      "response": "Fresh starts feel motivating because they allow us to leave behind past failures and start anew. This psychological reset can inspire hope and determination, making us feel empowered to chase our dreams."
    }
  ],
  "kellyPose": "explaining",
  "kellyEmotion": "curious"
}
```

#### **FACT2 Phase:**
```json
{
  "script": "Starting fresh can be an exhilarating experience, as it opens the door to new possibilities and perspectives. This opportunity for renewal allows us to shed old habits and embrace change, fostering growth in our personal and professional lives.",
  "options": [
    {
      "letter": "A",
      "text": "What are some ways to embrace a fresh start?",
      "quality": "good",
      "response": "Embracing a fresh start can involve setting new goals, decluttering your space, or even picking up a new hobby that excites you."
    },
    {
      "letter": "B",
      "text": "How can I maintain my motivation during a fresh start?",
      "quality": "best",
      "response": "Maintaining motivation can be achieved by setting small, achievable milestones and celebrating each success along the way to keep the momentum going."
    },
    {
      "letter": "C",
      "text": "What does it mean to truly start fresh?",
      "quality": "redirect",
      "response": "To truly start fresh means to let go of past limitations and approach life with a renewed mindset, ready to explore and discover new paths."
    }
  ],
  "kellyPose": "explaining",
  "kellyEmotion": "curious"
}
```

#### **FACT3 Phase:**
```json
{
  "script": "Did you know that starting fresh can actually rewire your brain? When we embrace new beginnings, our minds release dopamine, the feel-good hormone, which encourages us to seek out new experiences and opportunities. This powerful reset allows us to let go of past limitations and explore uncharted territories in our lives!",
  "options": [
    {
      "letter": "A",
      "text": "Curious question about Starting Fresh",
      "quality": "good",
      "response": "What are some new beginnings you've been thinking about lately?"
    },
    {
      "letter": "B",
      "text": "Practical question about Starting Fresh",
      "quality": "best",
      "response": "How can setting specific goals help you make the most of a fresh start?"
    },
    {
      "letter": "C",
      "text": "Thoughtful reflection on Starting Fresh",
      "quality": "redirect",
      "response": "In what ways do you think a fresh start could change your perspective on past challenges?"
    }
  ],
  "kellyPose": "explaining",
  "kellyEmotion": "curious"
}
```

#### **WISDOM Phase:**
```json
{
  "script": "Starting fresh allows us to embrace change and explore new possibilities. It acts as a psychological reset, encouraging us to let go of past limitations and embark on a journey of self-discovery and growth.",
  "options": [
    {
      "letter": "A",
      "text": "What are some ways we can create a fresh start in our lives?",
      "quality": "good",
      "response": "One way to create a fresh start is to set new goals or intentions. It's like turning the page to a new chapter! What changes are you curious about exploring?"
    },
    {
      "letter": "B",
      "text": "How can I practically implement a fresh start in my daily routine?",
      "quality": "best",
      "response": "You can start by setting specific, achievable goals for each day. This could mean organizing your space or dedicating time to a new hobby. Small changes can lead to a significant fresh start!"
    },
    {
      "letter": "C",
      "text": "What does it mean to truly embrace the idea of starting fresh?",
      "quality": "redirect",
      "response": "Embracing a fresh start means acknowledging your past but choosing to focus on the possibilities ahead. It's a commitment to growth and exploration, allowing yourself to be open to new experiences and perspectives."
    }
  ],
  "kellyPose": "explaining",
  "kellyEmotion": "curious"
}
```

---

## 🎨 **KELLY POSES & EMOTIONS**

### **Poses (from `kellyPose`):**
- `explaining` - Kelly gesturing, teaching
- `hello` - Kelly waving, welcoming
- `listening` - Kelly attentive, nodding
- `thinking` - Kelly hand on chin
- `excited` - Kelly animated, enthusiastic
- `clasp` - Kelly hands together, warm
- `pointing_left` - Kelly gesturing left
- `pointing_right` - Kelly gesturing right

### **Emotions (from `kellyEmotion`):**
- `curious` - Inquisitive, engaged
- `encouraging` - Supportive, warm
- `excited` - Enthusiastic, energetic
- `thoughtful` - Reflective, contemplative
- `warm` - Kind, gentle
- `confident` - Assured, steady

---

## 🎯 **MAPPING TO GOLDEN V5 UI**

### **1. Lesson Info Panel (Top-Left Safe Zone):**
```javascript
{
  lessonDay: "Day 1 of 365",
  lessonTitle: core_lessons.topic, // "Starting Fresh"
  lessonMeta: `${icon_emoji} • ${estimated_duration} min`
}
```

### **2. Phase Journey Bar (Top Center):**
```javascript
phases = [
  { id: 'Hook', emoji: '🎣', label: 'Hook', status: 'active' },
  { id: 'Fact1', emoji: '💭', label: 'Fact 1', status: 'pending' },
  { id: 'Fact2', emoji: '💡', label: 'Fact 2', status: 'pending' },
  { id: 'Fact3', emoji: '🔗', label: 'Fact 3', status: 'pending' },
  { id: 'Wisdom', emoji: '✨', label: 'Wisdom', status: 'pending' }
]
```

### **3. Lesson Content Panel (Bottom - Collapsible):**
```javascript
{
  phaseQuestion: lesson_atoms[currentPhase].content.script,
  options: lesson_atoms[currentPhase].content.options.map(opt => ({
    letter: opt.letter,
    text: opt.text,
    quality: opt.quality,
    response: opt.response
  }))
}
```

### **4. Action Dock (Bottom-Right Safe Zone):**
```javascript
actions = [
  { id: 'aha', emoji: '💡', label: 'Aha Moment', handler: recordAhaMoment },
  { id: 'pin', emoji: '📌', label: 'Pin to Journal', handler: pinToJournal },
  { id: 'share', emoji: '✨', label: 'Share Lesson', handler: shareLesson },
  { id: 'talk', emoji: '🎤', label: 'Talk to Kelly', handler: startVoiceChat }
]
```

### **5. Kelly Video (Full-Screen Wallpaper):**
```javascript
{
  videoId: `${day_number}-${phase}`, // "1-Hook"
  videoUrl: kelly_video_assets.public_url,
  manifestUrl: `/videos/${day_number}-${phase}-safe-zones.json`,
  kellyPose: lesson_atoms[currentPhase].content.kellyPose,
  kellyEmotion: lesson_atoms[currentPhase].content.kellyEmotion
}
```

---

## 📱 **RESPONSIVE DESIGN REQUIREMENTS**

### **Mobile (< 768px):**
- **Lesson Info Panel:** Smaller font, compact padding
- **Phase Journey Bar:** Horizontal scroll if needed, smaller circles
- **Content Panel:** Full-width, slides up from bottom
- **Action Dock:** Stacked vertically or 2x2 grid
- **Kelly Video:** Portrait mode (9:16 aspect ratio)

### **Tablet (768px - 1024px):**
- **Lesson Info Panel:** Medium size, balanced spacing
- **Phase Journey Bar:** Full width, visible all phases
- **Content Panel:** 80% width, centered
- **Action Dock:** Horizontal row, 4 buttons
- **Kelly Video:** Adaptive (16:9 or 9:16)

### **Desktop (> 1024px):**
- **Lesson Info Panel:** Full size, generous spacing
- **Phase Journey Bar:** Full width, large circles
- **Content Panel:** 60% width, centered
- **Action Dock:** Horizontal row, larger buttons
- **Kelly Video:** Landscape mode (16:9 aspect ratio)

---

## 🎬 **VIDEO REQUIREMENTS (UPDATED)**

### **Resolution & Quality:**
- **Minimum:** 1280x720 (720p)
- **Target:** 1920x1080 (1080p)
- **Aspirational:** 3840x2160 (4K)
- **Bitrate:** 8+ Mbps for 1080p
- **Format:** MP4 (H.264 codec)

### **Duration:**
- **Hook:** 15-30 seconds
- **Fact1-3:** 20-40 seconds each
- **Wisdom:** 30-60 seconds
- **Total per day:** 2-4 minutes

### **Kelly Appearance:**
- **Consistent face:** Use Kelly LoRA for all frames
- **Consistent outfit:** Blue sweater (canonical)
- **Consistent lighting:** Soft, warm, professional
- **Consistent background:** Phase-specific (see `PHASE_BACKGROUNDS`)

### **Spatial Intelligence Requirements:**
- **Face position:** Center-top (42% x, 12% y)
- **Face size:** 16% width, 20% height
- **Hand positions:** Vary by pose (see manifest)
- **Safe zones:** Pre-computed for each video segment
- **Manifest FPS:** 10 FPS (sufficient for UI sync)

### **Lip-Sync Requirements:**
- **Accuracy:** 95%+ (Sync Labs lipsync-2)
- **Audio source:** ElevenLabs (Kelly voice)
- **Phoneme alignment:** Montreal Forced Aligner
- **Blendshape timeline:** 30 FPS for smooth animation

---

## 🔧 **IMPLEMENTATION CHECKLIST**

### **Phase 1: Wire Real Data** ✅ NEXT
- [ ] Create `GoldenV5DataLoader` class
- [ ] Fetch Day 1 from `core_lessons` table
- [ ] Fetch all phases from `lesson_atoms` (archetype: "The Explorer")
- [ ] Map content to UI components
- [ ] Test with real data

### **Phase 2: Responsive Design**
- [ ] Test mobile view (iPhone 14 Pro, Pixel 7)
- [ ] Test tablet view (iPad Pro, Galaxy Tab)
- [ ] Test desktop view (1920x1080, 2560x1440)
- [ ] Adjust safe zones for portrait/landscape
- [ ] Test orientation changes

### **Phase 3: Video Integration**
- [ ] Query `kelly_video_assets` for Day 1 videos
- [ ] Download/upload to local test environment
- [ ] Generate safe zone manifests for each video
- [ ] Test video switching between phases
- [ ] Verify lip-sync accuracy

### **Phase 4: Action Handlers**
- [ ] Implement "Aha Moment" (save to `lesson_history.answers`)
- [ ] Implement "Pin to Journal" (future feature)
- [ ] Implement "Share Lesson" (social sharing)
- [ ] Implement "Talk to Kelly" (voice chat)

### **Phase 5: User Progress**
- [ ] Track phase completion
- [ ] Update `user_progress` table
- [ ] Show completion checkmarks
- [ ] Calculate time spent
- [ ] Award streak on completion

---

## 🎯 **VIDEO GENERATION REQUIREMENTS (REFINED)**

Based on Golden V5's spatial intelligence system, here are the REFINED requirements for video generation:

### **1. Pose Consistency:**
- Each phase should have a **specific Kelly pose** that matches the content
- Hook: `excited` (hands gesturing, leaning forward)
- Fact1-3: `explaining` (one hand raised, other relaxed)
- Wisdom: `warm` (hand over heart, sincere)

### **2. Safe Zone Planning:**
- **Pre-compute safe zones during video generation**
- Use MediaPipe Pose to detect Kelly's face/hands in each frame
- Generate manifest JSON with 10 FPS sampling
- Include gesture metadata (pointing_left, pointing_right, etc.)

### **3. Background Consistency:**
- Hook: Warm modern classroom (bright, inviting)
- Fact1: Cozy study room (books, warm lighting)
- Fact2: Professional studio (clean, focused)
- Fact3: Minimalist space (simple, elegant)
- Wisdom: Warm intimate setting (soft, reflective)

### **4. Motion & Gestures:**
- **Natural motion:** Use MiniMax Video-01 for realistic gestures
- **Gesture sync:** Match Kelly's hand movements to script emphasis
- **Eye contact:** Kelly should look at camera (engaging learner)
- **Facial expressions:** Match `kellyEmotion` (curious, encouraging, etc.)

### **5. Quality Assurance:**
- **Face audit:** Verify Kelly's face is consistent across all frames
- **Sweater color check:** Ensure blue sweater is visible and correct
- **Lighting check:** Consistent soft lighting, no harsh shadows
- **Background check:** No distracting elements, clean composition

---

## 🚀 **NEXT STEPS**

1. ✅ **Map content structure** (DONE)
2. 🚧 **Wire real data to Golden V5** (IN PROGRESS)
3. ⏳ **Test responsive design**
4. ⏳ **Generate Day 1 videos with refined requirements**
5. ⏳ **Create safe zone manifests**
6. ⏳ **Test full day flow**

---

**Built with precision by the Curious Kelly team.**  
**December 9, 2025**  
**✨ Every detail matters. ✨**







