// Emergency fallback lessons - used when ALL data sources fail
// These are hardcoded and always available

const EMERGENCY_LESSONS = {
  1: {
    day_number: 1,
    title: "The Power of Curiosity",
    subtitle: "Why asking questions changes everything",
    marketing_hook: "What makes curious people more successful?",
    greeting: "Welcome to Curious Kelly! I'm so excited to start this journey with you. Today, we explore the most powerful tool you already have - your curiosity.",
    content: `Every great discovery, invention, and breakthrough started with a simple question.

**Why does this matter?**

Curious people learn faster, solve problems better, and live more fulfilling lives. Studies show that curiosity activates the brain's reward system - learning literally feels good when you're genuinely curious.

**Today's Challenge:**

Think of something you've always wondered about. It could be anything - how planes fly, why the sky is blue, or how your favorite food is made. Write it down. That question is the first step on your learning journey.

**Remember:** There are no stupid questions. Only unexplored curiosities.`,
    category: "mindset",
    difficulty: "beginner",
    duration_estimate: 5,
    topic: "The Power of Curiosity",
    marketing_headline: "What makes curious people more successful?",
    marketing_tagline: "Why asking questions changes everything",
    universal_truth: "Every great discovery started with a simple question.",
    hero_image_url: "/images/fallback-lesson.png"
  },
  
  2: {
    day_number: 2,
    title: "The 5-Minute Rule",
    subtitle: "How small starts lead to big changes",
    marketing_hook: "The secret successful people use to beat procrastination",
    greeting: "Great to see you back! Today we learn a simple trick that can change how you approach any challenge.",
    content: `The hardest part of any task is starting. That's why the 5-minute rule exists.

**The Rule:**

Commit to doing something for just 5 minutes. That's it. After 5 minutes, you can stop guilt-free.

**Why it works:**

Once you start, you usually keep going. The brain's resistance is highest before beginning. Once you're in motion, continuing is easier than stopping.

**Try it today:**

Pick something you've been putting off. Set a timer for 5 minutes. Start. See what happens.`,
    category: "productivity",
    difficulty: "beginner",
    duration_estimate: 5,
    topic: "The 5-Minute Rule",
    marketing_headline: "The secret successful people use to beat procrastination",
    marketing_tagline: "How small starts lead to big changes",
    universal_truth: "The hardest part of any task is starting.",
    hero_image_url: "/images/fallback-lesson.png"
  },
  
  3: {
    day_number: 3,
    title: "Your Brain on Learning",
    subtitle: "Understanding how you actually learn",
    marketing_hook: "The science behind why some things stick and others don't",
    greeting: "Hello, curious mind! Today we peek inside your incredible brain.",
    content: `Your brain is constantly rewiring itself. Every time you learn something new, neurons form new connections.

**Key insight:**

Repetition strengthens neural pathways. That's why practice works. The first time you do something, it's hard. The hundredth time, it's automatic.

**The learning cycle:**

1. Encounter something new (confusion)
2. Practice it (struggle)
3. Rest and sleep (consolidation)
4. Return to it (mastery)

**Today's takeaway:**

Struggle is not a sign you're bad at learning. It's a sign learning is happening.`,
    category: "science",
    difficulty: "beginner",
    duration_estimate: 5,
    topic: "Your Brain on Learning",
    marketing_headline: "The science behind why some things stick and others don't",
    marketing_tagline: "Understanding how you actually learn",
    universal_truth: "Your brain physically changes when you learn.",
    hero_image_url: "/images/fallback-lesson.png"
  },
  
  4: {
    day_number: 4,
    title: "The Art of Asking",
    subtitle: "Better questions lead to better answers",
    marketing_hook: "How to ask questions that actually get you somewhere",
    greeting: "Curious friend! Today we level up your question-asking skills.",
    content: `Not all questions are created equal.

**Closed questions** get yes/no answers:
"Did you like the movie?"

**Open questions** invite exploration:
"What did the movie make you feel?"

**The best questions:**
- Start with "How" or "Why"
- Can't be answered with one word
- Make you think before answering

**Practice:**

Take any topic you're curious about. Instead of asking "What is X?", ask "Why does X work that way?" or "How might X be different?"`,
    category: "communication",
    difficulty: "beginner",
    duration_estimate: 5,
    topic: "The Art of Asking",
    marketing_headline: "How to ask questions that actually get you somewhere",
    marketing_tagline: "Better questions lead to better answers",
    universal_truth: "Better questions lead to better answers.",
    hero_image_url: "/images/fallback-lesson.png"
  },
  
  5: {
    day_number: 5,
    title: "Failure is Data",
    subtitle: "Reframing mistakes as information",
    marketing_hook: "Why the most successful people fail the most",
    greeting: "Welcome back! Today's lesson might change how you see setbacks forever.",
    content: `Every failure contains information. The question is: are you reading it?

**The reframe:**

Instead of "I failed" → "I learned what doesn't work"
Instead of "I'm bad at this" → "I'm still learning this"

**Famous 'failures':**
- WD-40 is named after 39 failed formulas
- James Dyson made 5,126 prototypes before his vacuum worked
- J.K. Rowling was rejected by 12 publishers

**Today's practice:**

Think of a recent mistake. What information was hidden in that failure? What will you do differently next time?`,
    category: "mindset",
    difficulty: "beginner",
    duration_estimate: 5,
    topic: "Failure is Data",
    marketing_headline: "Why the most successful people fail the most",
    marketing_tagline: "Reframing mistakes as information",
    universal_truth: "Every failure contains information if you're willing to read it.",
    hero_image_url: "/images/fallback-lesson.png"
  },
  
  6: {
    day_number: 6,
    title: "The Power of Yet",
    subtitle: "One word that transforms your mindset",
    marketing_hook: "The tiny word that separates growth from stagnation",
    greeting: "Hello again! Today we add one powerful word to your vocabulary.",
    content: `There's a huge difference between these two statements:

"I can't do this."
"I can't do this YET."

**The growth mindset:**

Adding "yet" transforms a limitation into a timeline. It acknowledges where you are while believing in where you're going.

**Try it:**
- "I don't understand calculus" → "I don't understand calculus yet"
- "I can't speak Spanish" → "I can't speak Spanish yet"
- "I'm not good at public speaking" → "I'm not good at public speaking yet"

**Feel the difference?**

The first is a closed door. The second is an open road.`,
    category: "mindset",
    difficulty: "beginner",
    duration_estimate: 5,
    topic: "The Power of Yet",
    marketing_headline: "The tiny word that separates growth from stagnation",
    marketing_tagline: "One word that transforms your mindset",
    universal_truth: "Adding 'yet' transforms limitations into timelines.",
    hero_image_url: "/images/fallback-lesson.png"
  },
  
  7: {
    day_number: 7,
    title: "Week 1 Reflection",
    subtitle: "Looking back to move forward",
    marketing_hook: "The practice that 10x's your learning",
    greeting: "Congratulations on completing your first week! Let's reflect on how far you've come.",
    content: `You've shown up for 7 days. That's not nothing - that's everything.

**This week you learned:**
- Curiosity is your superpower
- 5 minutes is enough to start
- Your brain physically changes when you learn
- Better questions get better answers
- Failure is just data
- "Yet" opens doors

**Reflection questions:**
1. Which lesson resonated most with you?
2. What's one thing you did differently this week?
3. What are you curious to learn next week?

**The truth:**

Most people who start things don't make it to day 7. You did. That says something about you.`,
    category: "reflection",
    difficulty: "beginner",
    duration_estimate: 5,
    topic: "Week 1 Reflection",
    marketing_headline: "The practice that 10x's your learning",
    marketing_tagline: "Looking back to move forward",
    universal_truth: "Reflection turns experience into wisdom.",
    hero_image_url: "/images/fallback-lesson.png"
  }
};

// Generic fallback for any day not in the emergency list
const GENERIC_FALLBACK = {
  title: "A Moment to Reflect",
  subtitle: "Sometimes the best lesson is patience",
  greeting: "Hey there! We're experiencing some technical difficulties, but that's okay - it happens to everyone. Let's use this moment to practice patience and curiosity.",
  content: `While we work on getting your full lesson loaded, here's a quick thought:

**Today's Mini-Lesson: Resilience**

Did you know that the most successful people aren't those who never face problems, but those who learn to adapt when things don't go as planned?

Right now, our systems are experiencing some issues. But you showed up anyway. That's what matters. You made the choice to learn something today.

**Your Challenge:**

Think of a time when something didn't work out as planned. What did you learn from that experience?

We'll be back with your full lesson soon. Thank you for your patience! 💛`,
  category: "life-skills",
  difficulty: "beginner",
  duration_estimate: 3,
  topic: "A Moment to Reflect",
  marketing_headline: "Sometimes the best lesson is patience",
  marketing_tagline: "Learning to adapt when things don't go as planned",
  universal_truth: "Resilience is the ability to adapt when things don't go as planned.",
  hero_image_url: "/images/fallback-lesson.png"
};

// Get emergency lesson by day number
function getEmergencyLesson(dayNumber) {
  if (EMERGENCY_LESSONS[dayNumber]) {
    return { ...EMERGENCY_LESSONS[dayNumber], source: 'emergency' };
  }
  return { 
    ...GENERIC_FALLBACK, 
    day_number: dayNumber,
    source: 'generic-fallback' 
  };
}

// Export for use
window.EMERGENCY_LESSONS = EMERGENCY_LESSONS;
window.GENERIC_FALLBACK = GENERIC_FALLBACK;
window.getEmergencyLesson = getEmergencyLesson;

console.log('🆘 Emergency lessons loaded (7 lessons + generic fallback)');


