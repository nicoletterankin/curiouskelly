/**
 * Social Comments System v4 - Humble, Growth-Mindset Focused
 * ===========================================================
 * 
 * PHILOSOPHY:
 * Comments should feel like sitting in a classroom with thoughtful peers.
 * They normalize struggle, celebrate small wins, and show learning is human.
 * 
 * NO: "Mind = BLOWN 🤯" "Best teacher EVER!" "I'm addicted!"
 * YES: "I had to replay that part twice" "This connects to yesterday" "Good question"
 * 
 * TRUST & SAFETY:
 * - Every comment marked with ✨
 * - Models growth mindset (struggle is normal)
 * - No manipulation, no FOMO, no pressure
 * - User can disable anytime
 */

// ═══════════════════════════════════════════════════════════════════
// COMMENT BANKS - Humble, Clear, Growth-Mindset
// ═══════════════════════════════════════════════════════════════════

const SOCIAL_COMMENTS = {
  
  // ═══════════════════════════════════════════════════════════════
  // WELCOME PHASE - Settling in, ready to learn
  // ═══════════════════════════════════════════════════════════════
  welcome: [
    // Greetings (simple, warm)
    { text: "Morning everyone 👋", mood: "friendly" },
    { text: "Here for today's lesson", mood: "ready" },
    { text: "Coffee ready, let's learn", mood: "casual" },
    { text: "Good evening from my timezone", mood: "friendly" },
    { text: "Made it! Almost forgot today", mood: "honest" },
    { text: "Back again", mood: "steady" },
    { text: "Day {dayNumber} 📅", mood: "tracking" },
    { text: "Watching with my kids today", mood: "family" },
    { text: "First time here, hi everyone", mood: "new" },
    { text: "Let's see what today's about", mood: "curious" },
    
    // Topic anticipation
    { text: "Interesting topic today", mood: "curious" },
    { text: "I've wondered about {topic} before", mood: "curious" },
    { text: "Don't know much about {topic}, excited to learn", mood: "open" },
    { text: "{topic}! This should be good", mood: "interested" },
    { text: "Never thought about {topic} much", mood: "honest" },
  ],

  // ═══════════════════════════════════════════════════════════════
  // HOOK PHASE - First insight, opening the topic
  // ═══════════════════════════════════════════════════════════════
  hook: [
    // Genuine interest
    { text: "Okay, that's a good way to start", mood: "engaged" },
    { text: "I never thought of it that way", mood: "reflective" },
    { text: "That's an interesting framing", mood: "thoughtful" },
    { text: "Huh, I didn't know that", mood: "learning" },
    { text: "Makes sense so far", mood: "following" },
    { text: "This is connecting some dots for me", mood: "connecting" },
    
    // Questions (growth mindset - asking is good)
    { text: "Wait, can someone explain that again?", mood: "confused" },
    { text: "I'm not sure I follow yet", mood: "honest" },
    { text: "Interesting... but why though?", mood: "questioning" },
    { text: "Is that always true?", mood: "critical" },
    
    // Connection to life
    { text: "I've seen this in my own life", mood: "connecting" },
    { text: "My grandmother used to talk about this", mood: "personal" },
    { text: "This explains something from work", mood: "practical" },
  ],

  // ═══════════════════════════════════════════════════════════════
  // QUESTION PHASES (Q1, Q2, Q3) - Thinking, choosing, uncertain
  // ═══════════════════════════════════════════════════════════════
  question: [
    // Honest uncertainty (normalizing not knowing)
    { text: "Hmm, not sure about this one", mood: "uncertain" },
    { text: "Let me think...", mood: "thinking" },
    { text: "This is harder than I thought", mood: "struggling" },
    { text: "I want to say A but...", mood: "uncertain" },
    { text: "Both options seem reasonable", mood: "torn" },
    { text: "I'm guessing here honestly", mood: "honest" },
    { text: "Going with my gut", mood: "deciding" },
    { text: "Changed my mind twice already", mood: "processing" },
    
    // Thinking out loud
    { text: "If what Kelly said is true, then...", mood: "reasoning" },
    { text: "Based on the intro, I think...", mood: "reasoning" },
    { text: "This connects to the first point", mood: "connecting" },
    { text: "Wait, let me reread the question", mood: "careful" },
    
    // Social (comparing without competing)
    { text: "Anyone else stuck on this?", mood: "communal" },
    { text: "Curious what others picked", mood: "social" },
    { text: "I picked {choice}, we'll see", mood: "sharing" },
  ],

  // ═══════════════════════════════════════════════════════════════
  // ANSWER REVEAL - After seeing if they got it right
  // ═══════════════════════════════════════════════════════════════
  answer_correct: [
    { text: "Got it! That makes sense", mood: "satisfied" },
    { text: "Okay good, my reasoning was right", mood: "confirmed" },
    { text: "Yes! The clue was in the intro", mood: "connected" },
    { text: "Phew, wasn't sure about that one", mood: "relieved" },
    { text: "I almost second-guessed myself", mood: "honest" },
  ],
  
  answer_wrong: [
    // Growth mindset - wrong is okay, learning is the point
    { text: "Ah, I see where I went wrong", mood: "learning" },
    { text: "That's actually a better answer", mood: "accepting" },
    { text: "I missed that connection", mood: "reflecting" },
    { text: "Good to know for next time", mood: "forward" },
    { text: "The explanation helps", mood: "grateful" },
    { text: "I was overthinking it", mood: "honest" },
    { text: "Makes sense now", mood: "understanding" },
  ],

  // ═══════════════════════════════════════════════════════════════
  // WISDOM PHASE - The main insight/conclusion
  // ═══════════════════════════════════════════════════════════════
  wisdom: [
    // Genuine appreciation (not hyperbolic)
    { text: "That's a helpful way to think about it", mood: "grateful" },
    { text: "I'll remember that", mood: "storing" },
    { text: "Good takeaway", mood: "satisfied" },
    { text: "Simple but true", mood: "appreciative" },
    { text: "This applies to a lot of things", mood: "connecting" },
    
    // Reflection
    { text: "I need to sit with this one", mood: "thoughtful" },
    { text: "Going to think about this more", mood: "processing" },
    { text: "This changes how I see {topic}", mood: "shifted" },
    { text: "I want to share this with someone", mood: "sharing" },
    
    // Practical application
    { text: "I can use this at work", mood: "practical" },
    { text: "This explains something I noticed", mood: "connecting" },
    { text: "Going to try this approach", mood: "applying" },
  ],

  // ═══════════════════════════════════════════════════════════════
  // COMPLETE PHASE - Lesson finished
  // ═══════════════════════════════════════════════════════════════
  complete: [
    // Simple wrap-up
    { text: "Good lesson today", mood: "satisfied" },
    { text: "Thanks, learned something new", mood: "grateful" },
    { text: "See everyone tomorrow", mood: "closing" },
    { text: "Day {dayNumber} done ✓", mood: "tracking" },
    { text: "Short but good", mood: "satisfied" },
    { text: "That went fast", mood: "surprised" },
    
    // Looking forward
    { text: "Curious about tomorrow's topic", mood: "anticipating" },
    { text: "On to the next one", mood: "steady" },
    { text: "Building the streak 📊", mood: "tracking" },
    
    // Social
    { text: "Good learning with you all", mood: "communal" },
    { text: "My favorite 5 minutes of the day", mood: "appreciative" },
  ],

  // ═══════════════════════════════════════════════════════════════
  // STRUGGLE COMMENTS - Normalize confusion (IMPORTANT)
  // ═══════════════════════════════════════════════════════════════
  struggle: [
    { text: "I had to replay that part", mood: "honest" },
    { text: "Still processing this", mood: "thinking" },
    { text: "This is new to me", mood: "learning" },
    { text: "I don't fully get it yet", mood: "honest" },
    { text: "Can someone explain differently?", mood: "asking" },
    { text: "Third time watching this one", mood: "persisting" },
    { text: "The more I learn, the more questions I have", mood: "growth" },
    { text: "Confused but curious", mood: "growth" },
  ],

  // ═══════════════════════════════════════════════════════════════
  // REACTIONS - Quick, authentic responses
  // ═══════════════════════════════════════════════════════════════
  reactions: [
    { text: "👍", mood: "positive" },
    { text: "Interesting", mood: "engaged" },
    { text: "True", mood: "agreeing" },
    { text: "Hmm", mood: "thinking" },
    { text: "📝", mood: "noting" },
    { text: "Good point", mood: "appreciative" },
    { text: "💡", mood: "insight" },
    { text: "Never knew", mood: "learning" },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// TOPIC-SPECIFIC TEMPLATES
// These get filled in with the actual lesson topic
// ═══════════════════════════════════════════════════════════════════

const TOPIC_TEMPLATES = {
  welcome: [
    "Don't know much about {topic} yet",
    "My {relation} always talked about {topic}",
    "{topic} — this should be interesting",
  ],
  hook: [
    "So {topic} is about {insight}?",
    "I never connected {topic} to that",
    "This is a new way to think about {topic}",
  ],
  wisdom: [
    "Now I understand {topic} better",
    "The {topic} lesson was worth it",
    "Will think about {topic} differently now",
  ],
};

// ═══════════════════════════════════════════════════════════════════
// AGE-SPECIFIC COMMENT VARIATIONS
// Same sentiment, different voice
// ═══════════════════════════════════════════════════════════════════

const AGE_VARIATIONS = {
  child: {
    confused: "I don't get it yet",
    excited: "Cool!",
    learning: "Ohhh I see",
  },
  teen: {
    confused: "Wait what",
    excited: "This is actually interesting",
    learning: "That makes sense now",
  },
  adult: {
    confused: "I'm not sure I follow",
    excited: "This is fascinating",
    learning: "That clarifies things",
  },
  senior: {
    confused: "Could you explain that again?",
    excited: "Wonderful insight",
    learning: "I've learned something valuable today",
  },
};

// ═══════════════════════════════════════════════════════════════════
// COMMENT GENERATOR
// ═══════════════════════════════════════════════════════════════════

class CommentGenerator {
  constructor(options = {}) {
    this.lessonDay = options.lessonDay || 1;
    this.topic = options.topic || 'today\'s topic';
    this.usedComments = new Set();
  }
  
  setLesson(dayNumber, topic) {
    this.lessonDay = dayNumber;
    this.topic = topic;
    this.usedComments.clear();
  }
  
  /**
   * Get a comment for a specific phase
   */
  getComment(phase, persona, options = {}) {
    const bank = SOCIAL_COMMENTS[phase] || SOCIAL_COMMENTS.reactions;
    
    // Filter to avoid repeats
    const available = bank.filter(c => !this.usedComments.has(c.text));
    if (available.length === 0) {
      this.usedComments.clear(); // Reset if exhausted
    }
    
    const comment = available[Math.floor(Math.random() * available.length)] || bank[0];
    this.usedComments.add(comment.text);
    
    // Fill in templates
    let text = comment.text
      .replace('{topic}', this.topic)
      .replace('{dayNumber}', this.lessonDay)
      .replace('{choice}', options.choice || 'A');
    
    // Age-appropriate variation
    if (persona && AGE_VARIATIONS[persona.ageGroup] && comment.mood) {
      const variation = AGE_VARIATIONS[persona.ageGroup][comment.mood];
      if (variation && Math.random() > 0.7) {
        text = variation;
      }
    }
    
    return {
      text,
      mood: comment.mood,
      persona: persona ? {
        name: persona.name,
        flag: persona.flag,
        avatar: persona.avatar,
        id: persona.id,
      } : null,
    };
  }
  
  /**
   * Get a batch of comments for a phase
   */
  getCommentBatch(phase, count = 5) {
    const personas = window.getUniquePersonas ? 
      window.getUniquePersonas(count) : 
      [];
    
    return Array(count).fill(null).map((_, i) => {
      return this.getComment(phase, personas[i] || null);
    });
  }
  
  /**
   * Get struggle comment (important for growth mindset)
   */
  getStruggleComment(persona) {
    return this.getComment('struggle', persona);
  }
  
  /**
   * Get answer reaction based on correctness
   */
  getAnswerReaction(isCorrect, persona) {
    const phase = isCorrect ? 'answer_correct' : 'answer_wrong';
    return this.getComment(phase, persona);
  }
}

// ═══════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════

window.SOCIAL_COMMENTS = SOCIAL_COMMENTS;
window.CommentGenerator = CommentGenerator;
window.AGE_VARIATIONS = AGE_VARIATIONS;

