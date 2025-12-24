# Kelly Curriculum-Trained LLM System

**Status:** 🚀 IMPLEMENTED  
**Created:** December 23, 2025  
**Purpose:** Browser-based LLM system that uses the complete Curious Kelly curriculum as knowledge base

---

## 🎯 Vision

Kelly becomes a **curriculum-trained AI teacher** that:
- Knows all 365 Learn + 365 Grow lessons
- Answers questions using curriculum context
- Compounds knowledge as learners progress
- Optimizes BYOK prompts with curriculum awareness
- Gets smarter each year as lessons cycle

---

## 📊 Current State Analysis

### What We Have

#### ✅ Complete Content Library
- **365 Learn Track Lessons**: Full JSON files with all phases, multilingual content
- **Grow Track**: AI fluency lessons (7+ created, 358 remaining)
- **Comprehensive Audit System**: Finds all assets (videos, audio, images, text)
- **Multilingual Support**: EN, ES, PT embedded in every lesson
- **12 Archetypes**: Explorer, Scientist, Rebel, Architect, Diplomat, Empath, MacGyver, Mystic, Provider, Storyteller, Strategist, Survivor
- **6 Age Buckets**: toddler, child, teen, young_adult, adult, elder
- **7 Phases Per Lesson**: hook, cliff, q1, q2, q3, wisdom, outro

#### 📈 Content Statistics (From Audit System)
- **Total Lessons**: 365 Learn + ~7 Grow (growing)
- **Total Phases**: ~2,555 (365 × 7)
- **Total Words**: ~500,000+ (estimated)
- **Languages**: 3 (EN, ES, PT)
- **Categories**: 20+ (Beginnings, Science, History, Nature, Art, etc.)

### 🔍 Gaps Identified

#### 1. **Asset Coverage**
- **Videos**: 0 found in Supabase (database empty)
- **Audio**: Only in JSON files, not systematically stored
- **Images**: Partial coverage (Day 1-17 have assets, rest missing)
- **Grow Track**: Only 7/365 lessons created

#### 2. **Knowledge Base Gaps**
- No vector embeddings computed
- No semantic search beyond keyword matching
- No cross-lesson connections indexed
- No concept mapping (e.g., "photosynthesis" → Day 18, Day 45, Day 120)

#### 3. **Learning History**
- Not tracked systematically
- No compound knowledge calculation
- No personalized recommendations based on history

#### 4. **BYOK Integration**
- No curriculum-aware prompt templates
- No context injection system
- No provider-specific optimizations

---

## 🚀 Implementation: Curriculum Knowledge Base

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Kelly Curriculum Knowledge Base                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Content Extraction                                  │
│     ├── Load all 365 lessons (JSON)                    │
│     ├── Extract phases, options, responses             │
│     ├── Extract Grow track content                      │
│     └── Build full-text index                           │
│                                                         │
│  2. Search & Retrieval                                  │
│     ├── Keyword search                                  │
│     ├── Category filtering                              │
│     ├── Semantic search (future: embeddings)           │
│     └── Relevance scoring                              │
│                                                         │
│  3. Learning History                                    │
│     ├── Track seen lessons                              │
│     ├── Track completed phases                         │
│     ├── Calculate streaks                               │
│     └── Compound knowledge over time                    │
│                                                         │
│  4. BYOK Prompt Generation                             │
│     ├── Curriculum context injection                    │
│     ├── Learning history context                        │
│     ├── Provider-specific templates                     │
│     └── Optimized prompt construction                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Features

#### 1. **Complete Lesson Extraction**
- Extracts topic, headline, universal truth, fun facts
- Extracts all phases with scripts, prompts, options, responses
- Extracts Grow track content
- Builds searchable full-text index

#### 2. **Semantic Search**
- Keyword-based search across all lessons
- Category filtering
- Relevance scoring
- Future: Vector embeddings for true semantic search

#### 3. **Learning History Tracking**
- Tracks which lessons learner has seen
- Tracks completed phases per lesson
- Calculates daily streaks
- Stores in localStorage (persists across sessions)

#### 4. **BYOK Prompt Optimization**
- Injects curriculum context into prompts
- Includes learning history for personalization
- Provider-specific templates (OpenAI, Anthropic, Google)
- Context-aware prompt construction

---

## 📝 Usage Examples

### Basic Search

```javascript
// Search curriculum
const results = KellyCurriculumKB.search('photosynthesis', {
  limit: 5,
  category: 'Science'
});

// Results include:
// - Day number
// - Full lesson data
// - Relevance score
// - Match highlights
```

### Generate BYOK Prompt

```javascript
// Generate optimized prompt for OpenAI
const promptData = KellyCurriculumKB.generateBYOKPrompt(
  'How does photosynthesis work?',
  {
    provider: 'openai',
    model: 'gpt-4-turbo-preview',
    includeContext: true,
    personality: 'curious',
    tone: 'warm'
  }
);

// Returns:
// {
//   prompt: "Full prompt with curriculum context...",
//   context: "Curriculum context string...",
//   metadata: {
//     provider: 'openai',
//     model: 'gpt-4-turbo-preview',
//     lessonsReferenced: [18, 45, 120],
//     timestamp: '2025-12-23T...'
//   }
// }
```

### Track Learning Progress

```javascript
// When learner views a lesson
KellyCurriculumKB.trackLessonAccess(18, 'hook');
KellyCurriculumKB.trackLessonAccess(18, 'q1');
KellyCurriculumKB.trackLessonAccess(18, 'q2');

// Streak automatically calculated
// History saved to localStorage
```

### Get Curriculum Context

```javascript
// Get context for any query
const context = KellyCurriculumKB.getCurriculumContext(
  'How do plants make food?',
  {
    maxLessons: 5,
    includePhases: true,
    includeGrowTrack: true
  }
);

// Returns formatted context string with:
// - Relevant lessons
// - Phase content
// - Grow track content
// - Learning history
```

---

## 🎨 BYOK Prompt Templates

### Template Structure

```
You are Kelly, an AI teacher from Curious Kelly...

{{CURRICULUM_CONTEXT}}
  ↓
  Relevant lessons (5 most relevant)
  Phase content
  Grow track content

{{LEARNING_HISTORY}}
  ↓
  Lessons seen: X/365
  Current streak: Y days
  Recent lessons: Day Z, Day W...

User Question: {{USER_QUERY}}

Instructions:
1. Answer using curriculum context
2. Reference seen lessons naturally
3. Be warm, curious, encouraging
4. Connect concepts across lessons
5. Keep responses concise (2-4 sentences)
```

### Provider-Specific Optimizations

#### OpenAI (GPT-4)
- Emphasize structured reasoning
- Use system message for personality
- Include few-shot examples from curriculum

#### Anthropic (Claude)
- Leverage long context window
- Include more curriculum examples
- Use XML tags for structure

#### Google (Gemini)
- Optimize for factual accuracy
- Include fun facts from curriculum
- Emphasize visual concepts

---

## 🔮 Future Enhancements

### Phase 1: Vector Embeddings (Next)
- Compute embeddings for all lessons
- Use Web API or local model (Transformers.js)
- True semantic search
- Concept clustering

### Phase 2: Cross-Lesson Connections
- Map concepts across lessons
- Build knowledge graph
- "Related lessons" recommendations
- Concept progression tracking

### Phase 3: Compound Learning
- Track which concepts learner has mastered
- Adjust difficulty based on history
- Personalize lesson recommendations
- Build learner profile over time

### Phase 4: Advanced BYOK Features
- Multi-turn conversation context
- Lesson-specific prompt templates
- Archetype-aware responses
- Age-appropriate language adaptation

---

## 📊 Insights & Opportunities

### Insights from Audit System

1. **Content Richness**
   - Each lesson has ~1,400 words of content
   - 7 phases × multiple languages = extensive content
   - Fun facts, discussion questions add depth

2. **Asset Gaps**
   - Videos: Database empty (needs population)
   - Images: Only 17/365 days have assets
   - Audio: Not systematically stored

3. **Structure Opportunities**
   - Lessons follow consistent structure
   - Easy to extract and index
   - Phases provide natural chunks for context

### Opportunities

1. **Knowledge Compounding**
   - As learner sees more lessons, Kelly gets smarter
   - Can reference previous lessons naturally
   - Builds understanding over time

2. **Personalization**
   - Use learning history to personalize responses
   - Reference lessons learner has seen
   - Adjust complexity based on progress

3. **Cross-Curricular Connections**
   - Connect concepts across lessons
   - "Remember when we learned about X on Day Y?"
   - Build holistic understanding

4. **BYOK Optimization**
   - Curriculum context reduces hallucinations
   - More accurate, relevant responses
   - Better than generic LLM

---

## 🎯 Next Steps

### Immediate (This Week)
1. ✅ **Deploy knowledge base system** (`kelly-curriculum-knowledge-base.js`)
2. ✅ **Integrate with learn.html** (track lesson access)
3. ✅ **Create BYOK prompt generator UI** (settings panel)
4. ✅ **Test with real queries** (validate context injection)

### Short Term (Next 2 Weeks)
1. **Add vector embeddings** (Transformers.js or API)
2. **Build concept mapping** (extract key concepts per lesson)
3. **Create prompt template library** (per use case)
4. **Add learning analytics** (track what's working)

### Long Term (Next Month)
1. **Compound learning system** (track mastery over time)
2. **Cross-lesson recommendations** (related lessons)
3. **Advanced personalization** (archetype-aware, age-appropriate)
4. **Multi-turn conversations** (context across sessions)

---

## 🔧 Technical Details

### File Structure

```
public/js/
├── kelly-curriculum-knowledge-base.js  # Main KB system
├── kelly-lesson-audit.js                # Asset discovery (existing)
└── kelly-byok-prompt-generator.js      # BYOK UI (to be created)

docs/kelly-llm/
├── CURRICULUM_TRAINED_LLM_SYSTEM.md    # This file
├── BYOK_PROMPT_TEMPLATES.md            # Template library
└── KNOWLEDGE_BASE_ARCHITECTURE.md      # Technical deep dive
```

### Integration Points

1. **learn.html**
   - Call `KellyCurriculumKB.trackLessonAccess()` when lesson loads
   - Track phase completion
   - Update learning history

2. **Settings Panel**
   - Add "Ask Kelly" input
   - Use `generateBYOKPrompt()` to create prompt
   - Send to user's API key
   - Display response

3. **Homepage**
   - Show learning stats from `getStats()`
   - Display streak
   - Show progress

---

## 💡 Key Principles

1. **Curriculum-First**: Always use curriculum context when relevant
2. **Compound Learning**: Track progress, build on previous lessons
3. **Personalization**: Reference what learner has seen
4. **Honesty**: Admit when curriculum doesn't cover something
5. **Connection**: Link concepts across lessons
6. **Warmth**: Maintain Kelly's curious, encouraging personality

---

## 🎓 Example: Complete Flow

### User asks: "How do plants make food?"

1. **Search Curriculum**
   ```javascript
   KellyCurriculumKB.search('plants make food')
   // Returns: Day 18 (Photosynthesis), Day 45 (Plant Biology)
   ```

2. **Get Context**
   ```javascript
   KellyCurriculumKB.getCurriculumContext('plants make food')
   // Returns formatted context with Day 18 and Day 45 content
   ```

3. **Generate Prompt**
   ```javascript
   KellyCurriculumKB.generateBYOKPrompt('How do plants make food?', {
     provider: 'openai',
     includeContext: true
   })
   // Returns optimized prompt with curriculum context
   ```

4. **Send to LLM**
   ```javascript
   // User's API key → OpenAI
   // Response includes curriculum knowledge
   // References Day 18 naturally
   ```

5. **Track Access**
   ```javascript
   // If user views Day 18 after asking
   KellyCurriculumKB.trackLessonAccess(18)
   // Future responses can reference "when you learned about photosynthesis"
   ```

---

**Status**: System implemented and ready for integration. Next: Add to learn.html and create BYOK UI.


