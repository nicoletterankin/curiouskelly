/**
 * Kelly Curriculum Knowledge Base
 * 
 * Browser-based LLM system trained on the complete Curious Kelly curriculum.
 * 
 * Purpose:
 * - Extract and structure all lesson content for LLM context
 * - Build searchable knowledge base from 365 Learn + 365 Grow lessons
 * - Enable Kelly to answer questions using curriculum knowledge
 * - Compound learning: track which lessons learner has seen
 * - Optimize BYOK prompts with curriculum context
 * 
 * Architecture:
 * - Client-side vector embeddings (using Web API or local model)
 * - Semantic search across all lessons
 * - Context-aware prompt generation
 * - Learning history tracking
 */

(function() {
  'use strict';

  const KellyCurriculumKB = {
    // Knowledge base storage
    knowledgeBase: {
      lessons: new Map(), // dayNumber -> full lesson data
      embeddings: new Map(), // dayNumber -> embedding vectors (if computed)
      searchIndex: null, // Full-text search index
      metadata: {
        totalLessons: 0,
        totalPhases: 0,
        totalWords: 0,
        languages: new Set(),
        categories: new Set(),
        lastUpdated: null
      }
    },

    // Learning history (what learner has seen)
    learningHistory: {
      seenLessons: new Set(), // dayNumbers
      completedPhases: new Map(), // dayNumber -> Set of phase names
      lastAccessed: new Map(), // dayNumber -> timestamp
      streaks: {
        current: 0,
        longest: 0,
        lastDate: null
      }
    },

    // Prompt templates for BYOK
    promptTemplates: {
      question: null,
      explanation: null,
      connection: null,
      personalization: null
    },

    /**
     * Initialize knowledge base (lazy-load, non-blocking)
     */
    async init() {
      // Load learning history immediately (fast, from localStorage)
      this.loadLearningHistory();
      
      // Build knowledge base in background (non-blocking)
      // This allows the page to load while KB builds
      this.buildKnowledgeBase().then(() => {
        this.buildSearchIndex();
        this.loadPromptTemplates();
        console.log('[KellyCurriculumKB] Knowledge base ready:', {
          lessons: this.knowledgeBase.metadata.totalLessons,
          phases: this.knowledgeBase.metadata.totalPhases,
          words: this.knowledgeBase.metadata.totalWords
        });
      }).catch(err => {
        console.warn('[KellyCurriculumKB] Failed to build knowledge base:', err);
      });
    },

    /**
     * Build knowledge base from all available lessons (optimized, lazy-load)
     */
    async buildKnowledgeBase() {
      const lessons = [];
      const loadPromises = [];
      let loadedCount = 0;
      
      // Load lessons in smaller batches to avoid blocking
      const BATCH_SIZE = 25; // Smaller batches for better responsiveness
      
      for (let day = 1; day <= 365; day++) {
        loadPromises.push(
          fetch(`/lessons/day-${day}.json`)
            .then(res => res.ok ? res.json() : null)
            .then(data => {
              if (data) {
                const structured = this.structureLesson(data, day);
                this.knowledgeBase.lessons.set(day, structured);
                lessons.push(structured);
                loadedCount++;
                
                // Update progress every 50 lessons
                if (loadedCount % 50 === 0) {
                  console.log(`[KellyCurriculumKB] Loaded ${loadedCount}/365 lessons...`);
                }
              }
              return null;
            })
            .catch(() => null)
        );
        
        // Process in batches
        if (loadPromises.length >= BATCH_SIZE || day === 365) {
          await Promise.all(loadPromises);
          loadPromises.length = 0;
          
          // Yield to browser between batches
          if (day < 365) {
            await new Promise(resolve => setTimeout(resolve, 10));
          }
        }
      }
      
      // Update metadata
      this.knowledgeBase.metadata.totalLessons = lessons.length;
      this.knowledgeBase.metadata.totalPhases = lessons.reduce((sum, l) => sum + (l.phases?.length || 0), 0);
      this.knowledgeBase.metadata.totalWords = lessons.reduce((sum, l) => sum + this.countWords(l), 0);
      this.knowledgeBase.metadata.lastUpdated = new Date().toISOString();
      
      // Extract languages and categories
      lessons.forEach(lesson => {
        if (lesson.languages) lesson.languages.forEach(lang => this.knowledgeBase.metadata.languages.add(lang));
        if (lesson.category) this.knowledgeBase.metadata.categories.add(lesson.category);
      });
    },

    /**
     * Structure lesson data for knowledge base
     */
    structureLesson(lessonData, dayNumber) {
      const structured = {
        day: dayNumber,
        date: lessonData.meta?.date || null,
        track: 'learn',
        
        // Core content
        topic: this.extractText(lessonData.meta?.topic),
        headline: this.extractText(lessonData.headline),
        universalTruth: this.extractText(lessonData.universal_truth),
        category: lessonData.meta?.category || 'General',
        emoji: lessonData.meta?.emoji || '📚',
        
        // Fun facts
        funFacts: (lessonData.fun_facts || []).map(f => this.extractText(f)),
        
        // Discussion questions
        discussionQuestions: (lessonData.discussion_questions || []).map(q => this.extractText(q)),
        
        // Phases (all content)
        phases: this.extractPhases(lessonData.phases || {}),
        
        // Grow track
        growTrack: lessonData.growTrack ? {
          title: this.extractText(lessonData.growTrack.title),
          learningObjective: this.extractText(lessonData.growTrack.learning_objective),
          activity: this.extractText(lessonData.growTrack.activity)
        } : null,
        
        // Languages
        languages: lessonData.meta?.languages || ['en'],
        
        // Full text for search
        fullText: this.extractFullText(lessonData),
        
        // Keywords for semantic search
        keywords: this.extractKeywords(lessonData)
      };
      
      return structured;
    },

    /**
     * Extract text from multilingual object
     */
    extractText(obj) {
      if (!obj) return '';
      if (typeof obj === 'string') return obj;
      if (typeof obj === 'object') {
        return obj.en || obj[Object.keys(obj)[0]] || '';
      }
      return '';
    },

    /**
     * Extract all phases with full content
     */
    extractPhases(phases) {
      const extracted = [];
      
      for (const [phaseName, phaseData] of Object.entries(phases)) {
        const phase = {
          name: phaseName,
          script: this.extractText(phaseData.script || phaseData.talk?.script),
          prompt: this.extractText(phaseData.prompt),
          title: this.extractText(phaseData.title),
          options: [],
          responses: {}
        };
        
        // Extract options and responses
        if (phaseData.options) {
          phase.options = phaseData.options.map(opt => ({
            letter: opt.letter,
            text: this.extractText(opt.text),
            quality: opt.quality,
            response: this.extractText(opt.response)
          }));
          
          // Map responses
          phaseData.options.forEach(opt => {
            if (opt.response) {
              phase.responses[opt.letter] = this.extractText(opt.response);
            }
          });
        }
        
        extracted.push(phase);
      }
      
      return extracted;
    },

    /**
     * Extract full text for search indexing
     */
    extractFullText(lessonData) {
      const parts = [];
      
      // Topic
      if (lessonData.meta?.topic) parts.push(this.extractText(lessonData.meta.topic));
      
      // Headline
      if (lessonData.headline) parts.push(this.extractText(lessonData.headline));
      
      // Universal truth
      if (lessonData.universal_truth) parts.push(this.extractText(lessonData.universal_truth));
      
      // Fun facts
      if (lessonData.fun_facts) {
        lessonData.fun_facts.forEach(f => parts.push(this.extractText(f)));
      }
      
      // Discussion questions
      if (lessonData.discussion_questions) {
        lessonData.discussion_questions.forEach(q => parts.push(this.extractText(q)));
      }
      
      // Phase scripts
      if (lessonData.phases) {
        Object.values(lessonData.phases).forEach(phase => {
          if (phase.script || phase.talk?.script) parts.push(this.extractText(phase.script || phase.talk.script));
          if (phase.prompt) parts.push(this.extractText(phase.prompt));
          if (phase.options) {
            phase.options.forEach(opt => {
              parts.push(this.extractText(opt.text));
              if (opt.response) parts.push(this.extractText(opt.response));
            });
          }
        });
      }
      
      // Grow track
      if (lessonData.growTrack) {
        parts.push(this.extractText(lessonData.growTrack.title));
        parts.push(this.extractText(lessonData.growTrack.learning_objective));
        parts.push(this.extractText(lessonData.growTrack.activity));
      }
      
      return parts.join(' ').toLowerCase();
    },

    /**
     * Extract keywords for semantic search
     */
    extractKeywords(lessonData) {
      const keywords = new Set();
      
      // Topic words
      const topic = this.extractText(lessonData.meta?.topic);
      topic.split(/\s+/).forEach(word => {
        if (word.length > 3) keywords.add(word.toLowerCase());
      });
      
      // Category
      if (lessonData.meta?.category) {
        keywords.add(lessonData.meta.category.toLowerCase());
      }
      
      // Universal truth keywords
      const truth = this.extractText(lessonData.universal_truth);
      truth.split(/\s+/).forEach(word => {
        if (word.length > 4) keywords.add(word.toLowerCase());
      });
      
      return Array.from(keywords);
    },

    /**
     * Count words in lesson
     */
    countWords(lesson) {
      return lesson.fullText.split(/\s+/).filter(w => w.length > 0).length;
    },

    /**
     * Build search index for fast retrieval
     */
    buildSearchIndex() {
      const index = {
        byKeyword: new Map(), // keyword -> [dayNumbers]
        byCategory: new Map(), // category -> [dayNumbers]
        byTopic: new Map(), // topic -> dayNumber
        fullText: [] // Array of {day, text} for fuzzy search
      };
      
      this.knowledgeBase.lessons.forEach((lesson, day) => {
        // Index by keywords
        lesson.keywords.forEach(keyword => {
          if (!index.byKeyword.has(keyword)) {
            index.byKeyword.set(keyword, []);
          }
          index.byKeyword.get(keyword).push(day);
        });
        
        // Index by category
        if (lesson.category) {
          if (!index.byCategory.has(lesson.category)) {
            index.byCategory.set(lesson.category, []);
          }
          index.byCategory.get(lesson.category).push(day);
        }
        
        // Index by topic
        if (lesson.topic) {
          index.byTopic.set(lesson.topic.toLowerCase(), day);
        }
        
        // Full text index
        index.fullText.push({
          day,
          text: lesson.fullText,
          topic: lesson.topic
        });
      });
      
      this.knowledgeBase.searchIndex = index;
    },

    /**
     * Search lessons by query
     */
    search(query, options = {}) {
      const {
        limit = 10,
        category = null,
        track = null,
        seenOnly = false
      } = options;
      
      const queryLower = query.toLowerCase();
      const results = [];
      const scored = new Map(); // dayNumber -> score
      
      // Search full text
      this.knowledgeBase.searchIndex.fullText.forEach(item => {
        // Skip if filtering by seen lessons
        if (seenOnly && !this.learningHistory.seenLessons.has(item.day)) {
          return;
        }
        
        // Skip if category filter
        if (category) {
          const lesson = this.knowledgeBase.lessons.get(item.day);
          if (lesson.category !== category) return;
        }
        
        // Skip if track filter
        if (track) {
          const lesson = this.knowledgeBase.lessons.get(item.day);
          if (lesson.track !== track) return;
        }
        
        // Score by text match
        const text = item.text;
        let score = 0;
        
        // Exact phrase match
        if (text.includes(queryLower)) {
          score += 10;
        }
        
        // Word matches
        const queryWords = queryLower.split(/\s+/);
        queryWords.forEach(word => {
          if (word.length > 2 && text.includes(word)) {
            score += 2;
          }
        });
        
        // Topic match
        if (item.topic && item.topic.toLowerCase().includes(queryLower)) {
          score += 5;
        }
        
        if (score > 0) {
          scored.set(item.day, (scored.get(item.day) || 0) + score);
        }
      });
      
      // Convert to results array
      Array.from(scored.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit)
        .forEach(([day, score]) => {
          const lesson = this.knowledgeBase.lessons.get(day);
          results.push({
            day,
            lesson,
            score,
            relevance: this.calculateRelevance(lesson, query)
          });
        });
      
      return results;
    },

    /**
     * Calculate relevance score
     */
    calculateRelevance(lesson, query) {
      const queryLower = query.toLowerCase();
      let relevance = 0;
      
      // Topic match
      if (lesson.topic && lesson.topic.toLowerCase().includes(queryLower)) {
        relevance += 0.3;
      }
      
      // Headline match
      if (lesson.headline && lesson.headline.toLowerCase().includes(queryLower)) {
        relevance += 0.2;
      }
      
      // Universal truth match
      if (lesson.universalTruth && lesson.universalTruth.toLowerCase().includes(queryLower)) {
        relevance += 0.2;
      }
      
      // Full text match
      if (lesson.fullText.includes(queryLower)) {
        relevance += 0.3;
      }
      
      return Math.min(1, relevance);
    },

    /**
     * Get curriculum context for a query (for BYOK prompts)
     */
    getCurriculumContext(query, options = {}) {
      const {
        maxLessons = 5,
        includePhases = true,
        includeGrowTrack = true
      } = options;
      
      // Search for relevant lessons
      const searchResults = this.search(query, { limit: maxLessons });
      
      // Build context string
      const contextParts = [];
      
      contextParts.push(`# Curious Kelly Curriculum Knowledge Base`);
      contextParts.push(`\n## Relevant Lessons (${searchResults.length} found)\n`);
      
      searchResults.forEach((result, idx) => {
        const lesson = result.lesson;
        contextParts.push(`### Day ${lesson.day}: ${lesson.topic}`);
        contextParts.push(`**Headline:** ${lesson.headline}`);
        contextParts.push(`**Universal Truth:** ${lesson.universalTruth}`);
        contextParts.push(`**Category:** ${lesson.category}`);
        
        if (includePhases && lesson.phases.length > 0) {
          contextParts.push(`\n**Phases:**`);
          lesson.phases.forEach(phase => {
            if (phase.script) {
              contextParts.push(`- ${phase.name}: ${phase.script.substring(0, 200)}...`);
            }
          });
        }
        
        if (includeGrowTrack && lesson.growTrack) {
          contextParts.push(`\n**Grow Track:** ${lesson.growTrack.title}`);
          contextParts.push(`- Objective: ${lesson.growTrack.learningObjective}`);
        }
        
        contextParts.push('');
      });
      
      // Add learning history context
      if (this.learningHistory.seenLessons.size > 0) {
        contextParts.push(`\n## Learner Context`);
        contextParts.push(`- Lessons seen: ${this.learningHistory.seenLessons.size}/365`);
        contextParts.push(`- Current streak: ${this.learningHistory.streaks.current} days`);
        contextParts.push(`- Longest streak: ${this.learningHistory.streaks.longest} days`);
      }
      
      return contextParts.join('\n');
    },

    /**
     * Generate optimized BYOK prompt
     */
    generateBYOKPrompt(userQuery, options = {}) {
      const {
        provider = 'openai', // 'openai', 'anthropic', 'google'
        model = null,
        includeContext = true,
        personality = 'curious',
        tone = 'warm'
      } = options;
      
      // Get curriculum context
      const curriculumContext = includeContext 
        ? this.getCurriculumContext(userQuery, { maxLessons: 5 })
        : '';
      
      // Load appropriate template
      const template = this.getPromptTemplate(provider, personality, tone);
      
      // Build prompt
      const prompt = template
        .replace('{{CURRICULUM_CONTEXT}}', curriculumContext)
        .replace('{{USER_QUERY}}', userQuery)
        .replace('{{LEARNING_HISTORY}}', this.getLearningHistoryContext());
      
      return {
        prompt,
        context: curriculumContext,
        metadata: {
          provider,
          model: model || this.getDefaultModel(provider),
          lessonsReferenced: this.search(userQuery, { limit: 5 }).map(r => r.day),
          timestamp: new Date().toISOString()
        }
      };
    },

    /**
     * Get prompt template for provider
     */
    getPromptTemplate(provider, personality, tone) {
      const baseTemplate = `You are Kelly, an AI teacher from Curious Kelly, a daily learning platform with 365 lessons covering science, history, nature, art, and human achievement.

{{CURRICULUM_CONTEXT}}

{{LEARNING_HISTORY}}

Personality: ${personality}
Tone: ${tone}

User Question: {{USER_QUERY}}

Instructions:
1. Answer using the curriculum context above when relevant
2. If the question relates to a lesson the learner has seen, reference it naturally
3. Be warm, curious, and encouraging
4. If you don't know something, say so honestly
5. Connect concepts across lessons when possible
6. Keep responses concise but complete (2-4 sentences typically)

Response:`;

      return baseTemplate;
    },

    /**
     * Get default model for provider
     */
    getDefaultModel(provider) {
      const models = {
        openai: 'gpt-4-turbo-preview',
        anthropic: 'claude-3-opus-20240229',
        google: 'gemini-pro'
      };
      return models[provider] || 'gpt-4-turbo-preview';
    },

    /**
     * Get learning history context
     */
    getLearningHistoryContext() {
      if (this.learningHistory.seenLessons.size === 0) {
        return 'This is a new learner - no lessons completed yet.';
      }
      
      const recentLessons = Array.from(this.learningHistory.seenLessons)
        .slice(-5)
        .map(day => {
          const lesson = this.knowledgeBase.lessons.get(day);
          return lesson ? `Day ${day}: ${lesson.topic}` : null;
        })
        .filter(Boolean);
      
      return `
## Learner Progress
- Lessons completed: ${this.learningHistory.seenLessons.size}/365
- Current streak: ${this.learningHistory.streaks.current} days
- Recent lessons: ${recentLessons.join(', ')}
`;
    },

    /**
     * Track lesson access (lightweight, non-blocking)
     */
    trackLessonAccess(dayNumber, phaseName = null) {
      if (!dayNumber || dayNumber < 1 || dayNumber > 365) return;
      
      try {
        this.learningHistory.seenLessons.add(dayNumber);
        
        if (phaseName) {
          if (!this.learningHistory.completedPhases.has(dayNumber)) {
            this.learningHistory.completedPhases.set(dayNumber, new Set());
          }
          this.learningHistory.completedPhases.get(dayNumber).add(phaseName);
        }
        
        this.learningHistory.lastAccessed.set(dayNumber, Date.now());
        
        // Update streak (async to avoid blocking)
        this.updateStreak();
        
        // Save to localStorage (debounced)
        this.debouncedSave();
      } catch (e) {
        console.warn('[KellyCurriculumKB] Failed to track lesson access:', e);
      }
    },

    /**
     * Debounced save to avoid excessive localStorage writes
     */
    _saveTimeout: null,
    debouncedSave() {
      if (this._saveTimeout) clearTimeout(this._saveTimeout);
      this._saveTimeout = setTimeout(() => {
        this.saveLearningHistory();
      }, 1000); // Save after 1 second of inactivity
    },

    /**
     * Update learning streak
     */
    updateStreak() {
      const today = new Date().toDateString();
      const lastDate = this.learningHistory.streaks.lastDate;
      
      if (lastDate === today) {
        // Already counted today
        return;
      }
      
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const yesterdayStr = yesterday.toDateString();
      
      if (lastDate === yesterdayStr) {
        // Continuing streak
        this.learningHistory.streaks.current++;
      } else {
        // New streak
        this.learningHistory.streaks.current = 1;
      }
      
      if (this.learningHistory.streaks.current > this.learningHistory.streaks.longest) {
        this.learningHistory.streaks.longest = this.learningHistory.streaks.current;
      }
      
      this.learningHistory.streaks.lastDate = today;
    },

    /**
     * Load learning history from localStorage
     */
    loadLearningHistory() {
      try {
        const stored = localStorage.getItem('kelly-learning-history');
        if (stored) {
          const parsed = JSON.parse(stored);
          this.learningHistory.seenLessons = new Set(parsed.seenLessons || []);
          this.learningHistory.streaks = parsed.streaks || { current: 0, longest: 0, lastDate: null };
          
          // Convert completedPhases back to Map
          if (parsed.completedPhases) {
            this.learningHistory.completedPhases = new Map(
              Object.entries(parsed.completedPhases).map(([day, phases]) => [
                parseInt(day),
                new Set(phases)
              ])
            );
          }
        }
      } catch (e) {
        console.warn('[KellyCurriculumKB] Failed to load learning history:', e);
      }
    },

    /**
     * Save learning history to localStorage
     */
    saveLearningHistory() {
      try {
        const toStore = {
          seenLessons: Array.from(this.learningHistory.seenLessons),
          completedPhases: Object.fromEntries(
            Array.from(this.learningHistory.completedPhases.entries()).map(([day, phases]) => [
              day,
              Array.from(phases)
            ])
          ),
          streaks: this.learningHistory.streaks
        };
        localStorage.setItem('kelly-learning-history', JSON.stringify(toStore));
      } catch (e) {
        console.warn('[KellyCurriculumKB] Failed to save learning history:', e);
      }
    },

    /**
     * Load prompt templates
     */
    loadPromptTemplates() {
      // Templates can be customized per use case
      this.promptTemplates.question = this.getPromptTemplate('openai', 'curious', 'warm');
      this.promptTemplates.explanation = this.getPromptTemplate('openai', 'explaining', 'clear');
      this.promptTemplates.connection = this.getPromptTemplate('openai', 'curious', 'inspiring');
      this.promptTemplates.personalization = this.getPromptTemplate('openai', 'warm', 'personal');
    },

    /**
     * Get statistics
     */
    getStats() {
      return {
        knowledgeBase: {
          totalLessons: this.knowledgeBase.metadata.totalLessons,
          totalPhases: this.knowledgeBase.metadata.totalPhases,
          totalWords: this.knowledgeBase.metadata.totalWords,
          languages: Array.from(this.knowledgeBase.metadata.languages),
          categories: Array.from(this.knowledgeBase.metadata.categories),
          lastUpdated: this.knowledgeBase.metadata.lastUpdated
        },
        learningHistory: {
          lessonsSeen: this.learningHistory.seenLessons.size,
          completionRate: (this.learningHistory.seenLessons.size / 365 * 100).toFixed(1) + '%',
          currentStreak: this.learningHistory.streaks.current,
          longestStreak: this.learningHistory.streaks.longest
        }
      };
    }
  };

  // Auto-initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => KellyCurriculumKB.init());
  } else {
    KellyCurriculumKB.init();
  }

  // Expose globally
  window.KellyCurriculumKB = KellyCurriculumKB;
})();

