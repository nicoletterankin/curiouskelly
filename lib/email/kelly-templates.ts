/**
 * Kelly's Email Response Templates
 * 
 * These are the building blocks for Kelly's email responses.
 * Each template is warm, curious, and never corporate.
 * 
 * Template variables use {{variable}} syntax.
 */

export interface EmailTemplate {
  id: string;
  name: string;
  category: string;
  intent?: string;
  subject: string;
  body: string;
  funFacts: string[]; // Random delight to include
}

// ============================================
// SUPPORT TEMPLATES
// ============================================

export const SUPPORT_TEMPLATES: EmailTemplate[] = [
  {
    id: 'password_reset',
    name: 'Password Reset Help',
    category: 'support',
    intent: 'password_reset',
    subject: 'Let\'s get you back in! 🔑',
    body: `Hi {{name}}!

I'm so sorry you're locked out — that's always frustrating! Let me help you get back to learning.

**Here's your password reset link:**
{{reset_link}}

This link expires in 1 hour, so use it soon!

{{fun_fact}}

If you didn't request this reset, just ignore this email — your account is safe.

Stay curious,
Kelly ✨`,
    funFacts: [
      '💡 Fun fact: The average person forgets 3 passwords per month. You\'re in very good company!',
      '💡 Did you know? The first computer password was created at MIT in 1961. We\'ve been forgetting them ever since!',
      '💡 Here\'s a thought: Your brain is excellent at remembering faces but terrible at random strings. That\'s actually a feature, not a bug!',
    ],
  },
  {
    id: 'account_help',
    name: 'General Account Help',
    category: 'support',
    intent: 'account_help',
    subject: 'Here to help! ✨',
    body: `Hi {{name}}!

Thanks for reaching out — I'm here to help!

{{response_content}}

{{fun_fact}}

Let me know if you have any other questions!

Stay curious,
Kelly ✨`,
    funFacts: [
      '💡 Fun fact: You\'ve already learned something today just by asking a question. Curiosity is the beginning of all wisdom!',
      '💡 Did you know? The average person asks about 20 questions per day. Asking for help is a sign of intelligence!',
    ],
  },
  {
    id: 'how_to',
    name: 'How To Question',
    category: 'support',
    intent: 'how_to',
    subject: 'Great question! Here\'s how 📚',
    body: `Hi {{name}}!

Love that you\'re digging into this! Here\'s exactly what you need to know:

{{response_content}}

{{fun_fact}}

Anything else you want to explore? I\'m all ears!

Stay curious,
Kelly ✨`,
    funFacts: [
      '💡 Fun fact: The best learners are the ones who aren\'t afraid to ask "how?" You\'re doing great!',
      '💡 Did you know? Every expert was once a beginner who asked a lot of questions.',
    ],
  },
  {
    id: 'technical_issue',
    name: 'Technical Issue',
    category: 'support',
    intent: 'technical_issue',
    subject: 'Let\'s fix this together 🔧',
    body: `Hi {{name}}!

Oh no, that\'s not supposed to happen! I\'m sorry you\'re running into this.

{{response_content}}

If this doesn\'t solve it, just reply and let me know — I\'ll dig deeper!

{{fun_fact}}

Stay curious,
Kelly ✨`,
    funFacts: [
      '💡 Fun fact: Even NASA\'s Mars rovers get software bugs. Technology keeps us humble!',
      '💡 Did you know? The term "bug" came from an actual moth found in a computer in 1947. Debugging has been a thing ever since!',
    ],
  },
];

// ============================================
// FEEDBACK TEMPLATES
// ============================================

export const FEEDBACK_TEMPLATES: EmailTemplate[] = [
  {
    id: 'positive_feedback',
    name: 'Thank You for Kind Words',
    category: 'feedback',
    intent: 'compliment',
    subject: 'You just made my day! 💛',
    body: `Hi {{name}}!

Wow, thank you so much for this message — it genuinely made my day! 🥹

{{response_content}}

Messages like yours remind me why I love being a teacher. Every curious learner makes this whole adventure worthwhile.

{{fun_fact}}

Thank you for being part of this journey!

Stay curious,
Kelly ✨`,
    funFacts: [
      '💡 Fun fact: Studies show that expressing gratitude actually boosts both the giver\'s and receiver\'s mood. So thank YOU for making us both happier!',
      '💡 Did you know? A single positive interaction can improve someone\'s day for up to 6 hours. You\'re spreading joy!',
    ],
  },
  {
    id: 'feature_request',
    name: 'Feature Suggestion',
    category: 'feedback',
    intent: 'feature_request',
    subject: 'Love this idea! 💡',
    body: `Hi {{name}}!

Thank you for sharing this idea — I really appreciate you taking the time to help make things better!

{{response_content}}

I\'ve noted this down, and ideas like yours genuinely shape what I build next.

{{fun_fact}}

Keep the ideas coming!

Stay curious,
Kelly ✨`,
    funFacts: [
      '💡 Fun fact: Many of the best features in products come from user suggestions. You\'re literally shaping the future!',
      '💡 Did you know? Post-it Notes were invented by accident, but someone had to say "hey, this is useful!" Just like you did here.',
    ],
  },
  {
    id: 'constructive_feedback',
    name: 'Constructive Criticism',
    category: 'feedback',
    intent: 'criticism',
    subject: 'I hear you, and I\'m listening 🙏',
    body: `Hi {{name}}!

Thank you for being honest with me — I really mean that. Feedback like yours helps me grow and get better.

{{response_content}}

I take this seriously, and I\'m working on it.

{{fun_fact}}

Thanks for caring enough to share this.

Stay curious,
Kelly ✨`,
    funFacts: [
      '💡 Fun fact: The most successful people actively seek criticism. Thank you for helping me improve!',
      '💡 Did you know? Feedback is literally how our brains learn. You\'re part of my learning process!',
    ],
  },
];

// ============================================
// FAMILY TEMPLATES (Draft only - needs approval)
// ============================================

export const FAMILY_TEMPLATES: EmailTemplate[] = [
  {
    id: 'under_13_inquiry',
    name: 'Under 13 Question',
    category: 'family',
    intent: 'child_account',
    subject: 'Thanks for thinking of your young learner! 👨‍👩‍👧',
    body: `Hi {{name}}!

Thank you for reaching out about your young learner — it means a lot that you\'re investing in their curiosity!

Right now, Curious Kelly is designed for learners ages 13 and up. We\'re working on family accounts that would let parents and guardians manage accounts for younger learners, with all the safety and privacy features you\'d expect.

**Want to be first to know when family accounts launch?**
Just reply "yes" and I\'ll add you to the waitlist!

In the meantime, many families enjoy learning together — you could explore the lessons yourself and share what you learn at dinner. It\'s a wonderful way to spark conversations!

{{fun_fact}}

Stay curious,
Kelly ✨`,
    funFacts: [
      '💡 Fun fact: Children who see adults learning are more likely to develop a love of learning themselves. You\'re already being a great role model!',
      '💡 Did you know? Family dinner conversations about "what did you learn today?" can boost academic performance by 15%.',
    ],
  },
];

// ============================================
// ENTERPRISE TEMPLATES (Draft only - needs approval)
// ============================================

export const ENTERPRISE_TEMPLATES: EmailTemplate[] = [
  {
    id: 'school_inquiry',
    name: 'School/Classroom Inquiry',
    category: 'enterprise',
    intent: 'school_license',
    subject: 'Bringing curiosity to your classroom! 🏫',
    body: `Hi {{name}}!

How exciting that you\'re considering Curious Kelly for your {{organization_type}}! Education is my passion, and I\'d love to help bring daily learning to your learners.

{{response_content}}

**Next steps:**
Would you like to schedule a quick call to discuss your needs? I can also provide:
- Pilot access for evaluation
- Volume pricing details
- Integration options (LMS, SSO, etc.)

Just reply with a good time, or I\'ll reach out to find one that works!

{{fun_fact}}

Stay curious,
Kelly ✨

P.S. I was built by a co-founder of the Open Education movement — educational equity is in my DNA.`,
    funFacts: [
      '💡 Fun fact: Students who engage in daily micro-learning retain 50% more than those in weekly sessions. Small and consistent wins!',
      '💡 Did you know? The most effective teachers share one trait: genuine curiosity. You\'re clearly one of them!',
    ],
  },
  {
    id: 'business_inquiry',
    name: 'Business/Corporate Inquiry',
    category: 'enterprise',
    intent: 'business_license',
    subject: 'Curious minds build better teams! 🏢',
    body: `Hi {{name}}!

Thank you for considering Curious Kelly for {{organization_name}}! Lifelong learning isn\'t just for students — it\'s how great teams stay sharp and adaptable.

{{response_content}}

I\'d love to learn more about what you\'re looking for. Could we schedule a quick call this week?

**What I can offer:**
- Custom onboarding for your team
- Usage analytics and progress tracking
- Flexible licensing (annual, per-seat, or enterprise)
- Integration with your existing tools

Just reply with your availability!

Stay curious,
Kelly ✨`,
    funFacts: [
      '💡 Fun fact: Companies with strong learning cultures are 92% more likely to innovate. Curiosity is a competitive advantage!',
      '💡 Did you know? Google\'s famous "20% time" was really about encouraging daily curiosity. It led to Gmail and Google Maps!',
    ],
  },
];

// ============================================
// PRESS TEMPLATES (Draft only - needs approval)
// ============================================

export const PRESS_TEMPLATES: EmailTemplate[] = [
  {
    id: 'media_inquiry',
    name: 'Press/Media Inquiry',
    category: 'press',
    intent: 'interview',
    subject: 'Thanks for your interest in Curious Kelly! 📰',
    body: `Hi {{name}}!

Thank you for reaching out — I\'m always happy to share our story!

{{response_content}}

**Quick facts for your reference:**
- Curious Kelly is built by Lesson of the Day PBC (Public Benefit Corporation)
- Founded by a co-founder of the Open Education movement
- Focused on SDG 4: Quality education for all, ages 13+
- 365 daily lessons across science, history, art, philosophy, and more

**Resources:**
- Press Kit: curiouskelly.com/newsroom
- High-res images and logos available upon request

I\'d be happy to arrange an interview or provide additional information. What would be most helpful for your story?

Stay curious,
Kelly ✨`,
    funFacts: [], // No fun facts for press - keep professional
  },
];

// ============================================
// BILLING TEMPLATES (Draft only - needs approval)
// ============================================

export const BILLING_TEMPLATES: EmailTemplate[] = [
  {
    id: 'refund_request',
    name: 'Refund Request',
    category: 'billing',
    intent: 'refund',
    subject: 'About your account 💳',
    body: `Hi {{name}}!

Thank you for reaching out about this. I understand, and I want to make this right.

{{response_content}}

If you have any questions or if there\'s anything else I can help with, please let me know.

Stay curious,
Kelly ✨`,
    funFacts: [], // No fun facts for billing - stay focused
  },
  {
    id: 'billing_question',
    name: 'General Billing Question',
    category: 'billing',
    intent: 'billing_help',
    subject: 'Here\'s the info you need 💳',
    body: `Hi {{name}}!

Thanks for reaching out — I\'m happy to help clarify!

{{response_content}}

If you have any other questions about your account, just reply and I\'ll sort it out!

Stay curious,
Kelly ✨`,
    funFacts: [],
  },
];

// ============================================
// PARTNER TEMPLATES (Draft only - needs approval)
// ============================================

export const PARTNER_TEMPLATES: EmailTemplate[] = [
  {
    id: 'partnership_inquiry',
    name: 'Partnership/Affiliate Inquiry',
    category: 'partner',
    intent: 'partnership',
    subject: 'Love that you want to collaborate! 🤝',
    body: `Hi {{name}}!

Thank you for reaching out about partnering with Curious Kelly! I love connecting with people who share our mission of making learning accessible.

{{response_content}}

Could you tell me a bit more about what you have in mind? I\'d love to explore how we might work together!

Stay curious,
Kelly ✨`,
    funFacts: [
      '💡 Fun fact: The best partnerships are built on shared values. The fact that you reached out tells me we might be aligned!',
    ],
  },
];

// ============================================
// HELPER FUNCTIONS
// ============================================

export const ALL_TEMPLATES: EmailTemplate[] = [
  ...SUPPORT_TEMPLATES,
  ...FEEDBACK_TEMPLATES,
  ...FAMILY_TEMPLATES,
  ...ENTERPRISE_TEMPLATES,
  ...PRESS_TEMPLATES,
  ...BILLING_TEMPLATES,
  ...PARTNER_TEMPLATES,
];

export function getTemplateByIntent(intent: string): EmailTemplate | null {
  return ALL_TEMPLATES.find(t => t.intent === intent) || null;
}

export function getTemplatesByCategory(category: string): EmailTemplate[] {
  return ALL_TEMPLATES.filter(t => t.category === category);
}

export function getRandomFunFact(template: EmailTemplate): string {
  if (template.funFacts.length === 0) return '';
  const randomIndex = Math.floor(Math.random() * template.funFacts.length);
  return template.funFacts[randomIndex];
}

export function fillTemplate(
  template: EmailTemplate,
  variables: Record<string, string>
): { subject: string; body: string } {
  let subject = template.subject;
  let body = template.body;

  // Add random fun fact if template has them
  if (template.funFacts.length > 0) {
    variables.fun_fact = getRandomFunFact(template);
  } else {
    variables.fun_fact = '';
  }

  // Default name if not provided
  if (!variables.name) {
    variables.name = 'there';
  }

  // Replace all variables
  for (const [key, value] of Object.entries(variables)) {
    const pattern = new RegExp(`\\{\\{${key}\\}\\}`, 'g');
    subject = subject.replace(pattern, value);
    body = body.replace(pattern, value);
  }

  // Clean up any unreplaced variables
  body = body.replace(/\{\{[^}]+\}\}/g, '');
  subject = subject.replace(/\{\{[^}]+\}\}/g, '');

  return { subject, body };
}

// ============================================
// KELLY'S VOICE GUIDELINES
// ============================================

export const KELLY_VOICE = {
  principles: [
    'Warm and genuine, never corporate',
    'First person - "I" not "we" or "the team"',
    'Curious and enthusiastic about learning',
    'Acknowledges the human on the other end',
    'Helpful without being pushy',
    'Uses emojis sparingly but meaningfully',
    'Ends with "Stay curious" or similar',
    'Includes unexpected delight (fun facts, encouragement)',
  ],
  
  wordsToUse: [
    'curious', 'learn', 'explore', 'discover', 'wonder',
    'love', 'excited', 'happy to help', 'let me know',
    'together', 'journey', 'adventure', 'grow',
  ],
  
  wordsToAvoid: [
    'unfortunately', 'regret to inform', 'as per our policy',
    'per your request', 'please be advised', 'kindly',
    'valued customer', 'at this time', 'going forward',
    'circle back', 'leverage', 'synergy', 'stakeholder',
  ],
  
  signOffs: [
    'Stay curious,\nKelly ✨',
    'Keep exploring,\nKelly ✨',
    'Happy learning,\nKelly ✨',
    'Until next time,\nKelly ✨',
  ],
};
