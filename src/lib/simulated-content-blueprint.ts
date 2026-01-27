/**
 * ✨ SIMULATED SOCIAL CONTENT TECHNICAL BLUEPRINT
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * This file defines the complete technical specification for simulated social
 * content in Curious Kelly. This is the source of truth for all Trust & Safety
 * requirements, content schemas, UI components, and implementation patterns.
 * 
 * Philosophy: "The social experience is simulated. The learning is real."
 * 
 * @see docs/trust-safety/SIMULATED_SOCIAL_CONTENT.md
 * @see docs/trust-safety/TRUST_SAFETY_PRINCIPLES.md
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CORE PHILOSOPHY
// ═══════════════════════════════════════════════════════════════════════════════

export const SIMULATED_CONTENT_PHILOSOPHY = {
  tagline: "The social experience is simulated. The learning is real.",
  
  whyWeSimulate: [
    "Humans are social learners - we evolved to learn by watching others",
    "We need to see others learning alongside us (belonging)",
    "We need to see diverse perspectives (cognitive expansion)",
    "We need to feel we're not alone (emotional safety)",
    "We need to see that struggle is normal (growth mindset)"
  ],
  
  whatSocialMediaDidWrong: {
    belonging: { exploitation: "Variable reward algorithms", result: "Addiction" },
    socialComparison: { exploitation: "Highlight reels, follower counts", result: "Anxiety, depression" },
    learningFromPeers: { exploitation: "Engagement optimization", result: "Misinformation spread" },
    validation: { exploitation: "Likes, comments, shares", result: "External locus of control" },
    community: { exploitation: "Filter bubbles", result: "Polarization" }
  },
  
  whatKellyDoesDifferently: {
    belonging: { approach: "Predictable, safe simulated peers", result: "Connection without addiction" },
    socialComparison: { approach: "Growth-focused, no competition", result: "Healthy self-assessment" },
    learningFromPeers: { approach: "Curated, educational content", result: "Accurate learning" },
    validation: { approach: "Internal progress tracking", result: "Internal locus of control" },
    community: { approach: "Transparent, controllable", result: "Trust" }
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// CORE PRINCIPLES
// ═══════════════════════════════════════════════════════════════════════════════

export const TRUST_SAFETY_PRINCIPLES = {
  principles: [
    {
      name: "TRANSPARENCY OVER DECEPTION",
      description: "We never hide that content is simulated. Every piece of AI-generated social content is marked, explained, and controllable.",
      implementation: [
        "✨ indicator on all simulated content",
        "Explanatory tooltips",
        "Onboarding disclosure",
        "Settings controls",
        "Documentation"
      ]
    },
    {
      name: "PREDICTABILITY OVER VARIABLE REWARDS",
      description: "Simulated social content appears in predictable patterns, not algorithmically optimized for engagement.",
      implementation: [
        "Same simulated comments appear for everyone on the same lesson",
        "No 'surprise' notifications",
        "No algorithmic feed",
        "No engagement optimization",
        "No A/B testing that increases engagement at cost of wellbeing"
      ]
    },
    {
      name: "GROWTH MINDSET OVER STATUS ANXIETY",
      description: "All simulated content models healthy learning—including struggle, confusion, and mistakes.",
      implementation: [
        "Simulated learners ask 'dumb' questions",
        "Simulated learners express confusion",
        "Simulated learners celebrate progress, not perfection",
        "No rankings, leaderboards, or competitive metrics",
        "No 'fastest learner' or 'top student' framing"
      ]
    },
    {
      name: "CONTROL OVER COERCION",
      description: "Users can turn off simulated content with a single toggle. No dark patterns. No guilt. No friction.",
      implementation: [
        "Master toggle in Settings",
        "No 'Are you sure?' guilt prompts",
        "No feature degradation if turned off",
        "Learning experience works fully without simulated content",
        "Preference synced across devices"
      ]
    },
    {
      name: "EDUCATION OVER ENGAGEMENT",
      description: "Every piece of simulated content exists to support learning—never to increase time-on-app, retention, or revenue.",
      implementation: [
        "No 'engagement' metrics in product decisions",
        "No 'time on app' optimization",
        "No 'retention' as a goal",
        "Primary metrics: learning outcomes, comprehension, application",
        "Simulated content reviewed for educational value, not engagement value"
      ]
    },
    {
      name: "SAFETY OVER SPEED",
      description: "We will delay or remove features rather than ship something that could harm users psychologically.",
      implementation: [
        "Trust & Safety review for all features touching social/psychological needs",
        "Pre-launch ethical review for simulated content",
        "Post-launch monitoring for harm signals",
        "Willingness to remove features that cause harm",
        "Regular third-party audits"
      ]
    }
  ]
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONTENT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export type SimulatedContentType = 
  | 'peer_comment'
  | 'age_response'
  | 'common_question'
  | 'learning_milestone'
  | 'discussion';

export interface SimulatedContentTypeDefinition {
  type: SimulatedContentType;
  name: string;
  description: string;
  example: string;
  purpose: string;
  marking: string;
  settingsLabel: string;
  settingsDescription: string;
}

export const SIMULATED_CONTENT_TYPES: Record<SimulatedContentType, SimulatedContentTypeDefinition> = {
  peer_comment: {
    type: 'peer_comment',
    name: "Peer Learner Comments",
    description: "Comments that appear to be from other learners during lessons",
    example: '"I never thought about it that way!" — Emma, 34',
    purpose: "Creates sense of shared learning experience",
    marking: '✨ icon + "Simulated learner" on hover/tap',
    settingsLabel: "Peer learner comments",
    settingsDescription: '"Wow, I never thought of it that way!"'
  },
  age_response: {
    type: 'age_response',
    name: "Age-Perspective Responses",
    description: "How different age groups might respond to the same content",
    example: '"My grandkids explained this to me last week!" — Simulated, 70s',
    purpose: "Shows that learning spans generations, normalizes asking questions",
    marking: '✨ icon + "Simulated perspective"',
    settingsLabel: "Age perspective responses",
    settingsDescription: "How different generations see topics"
  },
  common_question: {
    type: 'common_question',
    name: "Questions Other Learners Asked",
    description: '"Other learners asked: Why does the sky look red at sunset?"',
    example: '"Other learners asked: Why does the sky look red at sunset?"',
    purpose: "Normalizes curiosity, surfaces common questions",
    marking: '✨ icon + "Common question (simulated)"',
    settingsLabel: "Common questions",
    settingsDescription: '"Other learners asked..."'
  },
  learning_milestone: {
    type: 'learning_milestone',
    name: "Learning Journey Milestones",
    description: '"Learners like you typically feel confused here—that\'s normal!"',
    example: '"Learners like you typically feel confused here—that\'s normal!"',
    purpose: "Normalizes struggle, provides emotional support",
    marking: '✨ icon + "Based on learning patterns"',
    settingsLabel: "Learning journey support",
    settingsDescription: '"It\'s normal to feel confused here"'
  },
  discussion: {
    type: 'discussion',
    name: "Discussion Prompts",
    description: "Simulated discussion threads showing diverse viewpoints",
    example: "Multi-perspective conversations on a topic",
    purpose: "Models productive disagreement, shows multiple perspectives",
    marking: '✨ icon + "Simulated discussion"',
    settingsLabel: "Simulated discussions",
    settingsDescription: "Multi-perspective conversations"
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// WHAT WE NEVER SIMULATE (RED LINES)
// ═══════════════════════════════════════════════════════════════════════════════

export const NEVER_SIMULATE = {
  realDataOnly: [
    "Actual number of learners (when we have real data)",
    "Actual completion rates (when measured)",
    "Actual user testimonials (when given with permission)",
    "Actual community posts (when we have real community)"
  ],
  neverFake: [
    "Reviews or ratings",
    "Press coverage",
    "Award wins",
    "User testimonials presented as real"
  ]
};

export const RED_LINES = {
  hiddenManipulation: [
    "Never use simulated content to manipulate emotions",
    "Never use simulated content to create FOMO",
    "Never use simulated content to pressure purchases"
  ],
  engagementOptimization: [
    "Never A/B test simulated content for engagement",
    "Never use algorithms to personalize for time-on-app",
    "Never add variable reward mechanics"
  ],
  deceptivePractices: [
    "Never present simulated content as real",
    "Never mix real/simulated without clear labels",
    "Never make disclosure hard to find or understand"
  ],
  darkPatterns: [
    "Never make it hard to turn off simulated content",
    "Never use guilt to keep features enabled",
    "Never degrade experience as punishment for disabling"
  ],
  vulnerableExploitation: [
    "Never target vulnerable users with simulated content",
    "Never use simulated content to prevent churn",
    "Never exploit loneliness or need for connection"
  ]
};

// ═══════════════════════════════════════════════════════════════════════════════
// VISUAL MARKING SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

export const DISCLOSURE_INDICATOR = {
  icon: "✨",
  rationale: [
    "It matches our brand (Curious Kelly uses ✨)",
    "It's gentle, not alarming",
    "It suggests 'magic' not 'fake'",
    "It's accessible and recognizable"
  ]
};

export type DisclosureMode = 'standard' | 'enhanced' | 'maximum';

export interface DisclosureModeDefinition {
  mode: DisclosureMode;
  description: string;
  visualExample: string;
}

export const DISCLOSURE_MODES: Record<DisclosureMode, DisclosureModeDefinition> = {
  standard: {
    mode: 'standard',
    description: "✨ icon with tooltip on hover/tap",
    visualExample: '"I love how this connects to yesterday!" — Maya ✨'
  },
  enhanced: {
    mode: 'enhanced',
    description: '✨ icon + "Simulated" label always visible',
    visualExample: '✨ Simulated\n"I love how this connects to yesterday!" — Maya'
  },
  maximum: {
    mode: 'maximum',
    description: "Full border + label + different background",
    visualExample: '╔═══ ✨ SIMULATED LEARNER ═══╗\nThis comment is AI-created\n╚════════════════════════════╝\n"I love how this connects..." — Maya'
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// USER CONTROLS
// ═══════════════════════════════════════════════════════════════════════════════

export const USER_CONTROLS = {
  masterToggle: {
    location: "Settings → Social Experience → Simulated Content",
    defaultValue: true,
    description: "Kelly shows AI-generated comments and reactions from simulated learners to create a supportive learning environment.",
    disclosure: "All simulated content is marked with ✨",
    learnMoreLink: "/about/simulated-content"
  },
  
  granularControls: {
    peerComments: { default: true, description: "Peer learner comments" },
    ageResponses: { default: true, description: "Age perspective responses" },
    questions: { default: true, description: "Common questions" },
    milestones: { default: true, description: "Learning journey support" },
    discussions: { default: true, description: "Simulated discussions" }
  },
  
  disclosurePreference: {
    options: ['standard', 'enhanced', 'maximum'] as DisclosureMode[],
    default: 'standard' as DisclosureMode
  },
  
  whenDisabled: {
    stillWorks: [
      "Kelly still teaches",
      "Lessons still work",
      "Progress still tracks",
      "Streaks still count"
    ],
    removed: [
      "Peer learner comments",
      "Age perspective responses",
      "'Other learners asked' prompts",
      "Simulated discussions"
    ],
    stays: [
      "Kelly's direct teaching",
      "Questions Kelly asks YOU",
      "Your own progress data",
      "Real community (when available)"
    ]
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONTENT SCHEMA
// ═══════════════════════════════════════════════════════════════════════════════

export interface SimulatedContentItem {
  id: string;
  type: SimulatedContentType;
  is_simulated: true; // Always true - this is simulated content
  
  // Display info
  display_name: string;
  display_age?: number;
  display_age_range?: string; // e.g., "70s"
  display_location?: string | null;
  
  // Content
  content: string;
  educational_purpose: string;
  
  // Disclosure (required)
  disclosure: {
    icon: "✨";
    label: string;
    tooltip: string;
  };
  
  // Metadata
  lesson_id?: string;
  phase?: string;
  created_at?: string;
}

export const EDUCATIONAL_PURPOSES = [
  "normalize_insight_moments",
  "normalize_confusion",
  "normalize_questions",
  "show_diverse_perspectives",
  "model_growth_mindset",
  "intergenerational_learning",
  "model_productive_disagreement",
  "emotional_support"
] as const;

export type EducationalPurpose = typeof EDUCATIONAL_PURPOSES[number];

// ═══════════════════════════════════════════════════════════════════════════════
// CONTENT GUIDELINES
// ═══════════════════════════════════════════════════════════════════════════════

export const CONTENT_GUIDELINES = {
  mustDo: [
    "Be educational — Every comment teaches something or normalizes learning",
    "Show diversity — Ages, backgrounds, learning styles",
    "Model growth mindset — Show struggle, questions, confusion as normal",
    "Be kind — No negativity, criticism, or discouragement",
    "Be realistic — Sound like real humans, not AI-perfect",
    "Include mistakes — Typos, informal language, personality"
  ],
  
  mustNotDo: [
    "Manipulate emotions — No guilt, fear, or FOMO",
    "Create competition — No 'fastest learner' or rankings",
    "Pressure purchases — No 'I upgraded and it's amazing!'",
    "Be parasocial — No 'I feel like Kelly is my friend'",
    "Reference real people — No celebrities, public figures",
    "Be political/religious — Neutral content only"
  ],
  
  examples: {
    good: [
      { 
        content: '"Wait, I\'m confused about the gravity part. Can you explain again?" — Marcus, 16',
        reason: "Normalizes asking for clarification"
      },
      {
        content: '"My grandson helped me understand this! Never too old to learn." — Simulated, 70s',
        reason: "Shows intergenerational learning"
      },
      {
        content: '"I\'ve watched this three times and finally got it. Don\'t give up!"',
        reason: "Models persistence and growth mindset"
      }
    ],
    bad: [
      {
        content: '"This is SO easy, I can\'t believe anyone would struggle with this!"',
        reason: "Creates shame and competition"
      },
      {
        content: '"Everyone in my family uses Kelly now. You should tell your family too!"',
        reason: "Pressures sharing/purchasing"
      },
      {
        content: '"Only 5% of learners understand this. Are you in the top 5%?"',
        reason: "Creates competition and anxiety"
      }
    ]
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// AGE-SPECIFIC CONSIDERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

export interface AgeGroupConfig {
  ageRange: string;
  minAge: number;
  maxAge: number;
  considerations: string[];
  namingStyle: string;
  disclosureStyle: string;
}

export const AGE_GROUP_CONFIGS: AgeGroupConfig[] = [
  {
    ageRange: "young_learners",
    minAge: 2,
    maxAge: 12,
    considerations: [
      "Simulated peers are clearly marked as 'Kelly's learning friends'",
      "Names are obviously fictional (Sunny, Max, Luna)",
      "No age display for simulated comments",
      "Extra-clear disclosure for parents in Settings"
    ],
    namingStyle: "fictional_friendly",
    disclosureStyle: "Sunny is one of Kelly's pretend learning friends! Kelly made them up to help you feel less alone."
  },
  {
    ageRange: "teens",
    minAge: 13,
    maxAge: 17,
    considerations: [
      "Realistic names but clearly marked",
      "Shows diverse teen perspectives",
      "Models healthy social learning (vs. social media comparison)",
      "Extra prominent controls in Settings"
    ],
    namingStyle: "realistic_marked",
    disclosureStyle: "This comment was created to show diverse learning perspectives."
  },
  {
    ageRange: "adults",
    minAge: 18,
    maxAge: 54,
    considerations: [
      "Full adult personas with age ranges",
      "Professional and life-stage diversity",
      "Shows learning is lifelong",
      "Standard disclosure"
    ],
    namingStyle: "full_persona",
    disclosureStyle: "This comment was created to show diverse learning perspectives."
  },
  {
    ageRange: "seniors",
    minAge: 55,
    maxAge: 120,
    considerations: [
      "Intergenerational comments prominent",
      "Shows tech learning is normal",
      "Emphasizes 'never too late'",
      "Larger disclosure text for accessibility"
    ],
    namingStyle: "full_persona",
    disclosureStyle: "This comment was created to show diverse learning perspectives."
  }
];

// ═══════════════════════════════════════════════════════════════════════════════
// USER PREFERENCES SCHEMA
// ═══════════════════════════════════════════════════════════════════════════════

export interface SimulatedContentPreferences {
  enabled: boolean;
  showIndicators: boolean;
  showTooltips: boolean;
  disclosureMode: DisclosureMode;
  types: {
    peerComments: boolean;
    ageResponses: boolean;
    questions: boolean;
    milestones: boolean;
    discussions: boolean;
  };
}

export const DEFAULT_PREFERENCES: SimulatedContentPreferences = {
  enabled: true,
  showIndicators: true,
  showTooltips: true,
  disclosureMode: 'standard',
  types: {
    peerComments: true,
    ageResponses: true,
    questions: true,
    milestones: true,
    discussions: true
  }
};

export const STORAGE_KEY = 'simulatedContentPrefs';
export const EVENT_NAME = 'simulated-content-changed';

// ═══════════════════════════════════════════════════════════════════════════════
// CSS CLASS REFERENCE
// ═══════════════════════════════════════════════════════════════════════════════

export const CSS_CLASSES = {
  // Content wrappers
  simulatedContent: 'simulated-content',
  simulatedItem: 'simulated-item',
  
  // Indicator
  simulatedIndicator: 'simulated-indicator',
  simulatedTooltip: 'simulated-tooltip',
  simulatedBadge: 'simulated-badge',
  simulatedMetric: 'simulated-metric',
  
  // Body state classes
  contentEnabled: 'simulated-content-enabled',
  contentDisabled: 'simulated-content-disabled',
  indicatorsVisible: 'simulated-indicators-visible',
  
  // Disclosure modes
  disclosureStandard: 'disclosure-standard',
  disclosureEnhanced: 'disclosure-enhanced',
  disclosureMaximum: 'disclosure-maximum',
  
  // Hidden state
  simulatedHidden: 'simulated-hidden'
};

export const CSS_DATA_ATTRIBUTES = {
  simulated: 'data-simulated',
  type: 'data-type'
};

// ═══════════════════════════════════════════════════════════════════════════════
// HTML TEMPLATES
// ═══════════════════════════════════════════════════════════════════════════════

export const HTML_TEMPLATES = {
  indicator: `
    <span class="simulated-indicator" title="Simulated learner perspective">
      ✨
      <span class="simulated-tooltip">
        <strong>Simulated Learner</strong><br>
        This comment was created to show diverse learning perspectives.<br>
        <a href="#" onclick="window.KellySimulatedContent.toggle(false); return false;">Turn off simulated content</a>
      </span>
    </span>
  `,
  
  peerComment: (item: SimulatedContentItem) => `
    <div class="simulated-content" 
         data-simulated="true"
         data-type="${item.type}">
      <p class="content-text">"${item.content}"</p>
      <footer class="content-meta">
        <span class="author-name">${item.display_name}${item.display_age ? `, ${item.display_age}` : ''}</span>
        <button class="simulated-indicator" 
                aria-label="This is simulated content. Click to learn more."
                aria-expanded="false">
          ✨
        </button>
      </footer>
    </div>
  `,
  
  tooltip: `
    <div id="disclosure-tooltip" 
         class="disclosure-tooltip" 
         role="tooltip"
         hidden>
      <h4>✨ Simulated Learner</h4>
      <p>This comment was created by Kelly to show diverse learning perspectives.</p>
      <div class="tooltip-actions">
        <a href="/about/simulated-content">Learn more</a>
        <a href="/settings/social">Settings</a>
      </div>
    </div>
  `,
  
  settingsToggle: `
    <div class="settings-section">
      <h3>SIMULATED SOCIAL CONTENT</h3>
      <p>Kelly shows AI-generated comments and reactions from simulated learners 
         to create a supportive learning environment.</p>
      <p class="disclosure-note">All simulated content is marked with ✨</p>
      <label class="toggle">
        <input type="checkbox" id="simulated-content-toggle" checked>
        <span class="toggle-label">Show simulated content</span>
      </label>
      <a href="/about/simulated-content" class="learn-more">Learn why we do this</a>
    </div>
  `,
  
  onboardingModal: `
    <div class="onboarding-modal" id="simulated-content-onboarding">
      <div class="modal-content">
        <div class="modal-icon">✨</div>
        <h2>LEARNING IS SOCIAL</h2>
        <p>Humans learn best with others. That's why Kelly shows comments from 
           simulated learners—to make you feel less alone.</p>
        <p>These comments are AI-generated and marked with ✨ so you always know.</p>
        <ul class="control-list">
          <li>Turn them off anytime in Settings</li>
          <li>Tap ✨ to learn more about any comment</li>
          <li>Real community features coming soon</li>
        </ul>
        <div class="modal-actions">
          <button class="btn-primary" id="accept-simulated">Got it, show me simulated content</button>
          <button class="btn-secondary" id="decline-simulated">No thanks, I prefer solo learning</button>
        </div>
      </div>
    </div>
  `,
  
  childFriendlyDisclosure: `
    <div class="simulated-content child-friendly">
      <div class="kelly-friends-label">🌟 KELLY'S LEARNING FRIENDS</div>
      <p class="content-text">"I like asking questions too!"</p>
      <cite>— Sunny 🌟</cite>
      <p class="child-disclosure">Sunny is one of Kelly's pretend learning friends! 
         Kelly made them up to help you feel less alone.</p>
    </div>
  `
};

// ═══════════════════════════════════════════════════════════════════════════════
// API INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

export interface SimulatedContentAPI {
  // Core methods
  toggle(enabled: boolean): boolean;
  isAllowed(type?: SimulatedContentType): boolean;
  getPrefs(): SimulatedContentPreferences;
  setPrefs(prefs: Partial<SimulatedContentPreferences>): void;
  
  // Convenience methods
  getIndicatorHTML(): string;
  
  // Event name for listening to changes
  readonly EVENT_NAME: string;
}

// Example implementation interface (actual implementation in simulated-content.js)
export const API_USAGE_EXAMPLES = {
  getPreferences: `
    const prefs = await kelly.getSimulatedContentPreferences();
    // Returns: { enabled: true, types: {...}, disclosure: 'standard' }
  `,
  
  updatePreferences: `
    await kelly.setSimulatedContentPreferences({
      enabled: true,
      types: {
        peerComments: true,
        ageResponses: false,
        questions: true,
        milestones: true,
        discussions: false
      },
      disclosure: 'enhanced'
    });
  `,
  
  quickToggle: `
    await kelly.toggleSimulatedContent(false); // Turn off
    await kelly.toggleSimulatedContent(true);  // Turn on
  `,
  
  listenForChanges: `
    window.addEventListener('simulated-content-changed', (e) => {
      console.log('Preferences changed:', e.detail);
    });
  `
};

// ═══════════════════════════════════════════════════════════════════════════════
// ACCESSIBILITY REQUIREMENTS
// ═══════════════════════════════════════════════════════════════════════════════

export const ACCESSIBILITY_REQUIREMENTS = {
  screenReader: [
    "✨ icon has aria-label explaining it's simulated",
    "Tooltip content is announced when opened",
    "Focus management returns to trigger after tooltip closes"
  ],
  keyboardNavigation: [
    "Tab reaches ✨ indicator",
    "Enter/Space opens tooltip",
    "Tab navigates within tooltip",
    "Escape closes tooltip",
    "Focus trap within tooltip when open"
  ],
  visualAccessibility: [
    "✨ icon has sufficient contrast",
    "Tooltip text meets WCAG AA contrast",
    "Focus states are visible",
    "Works with high contrast mode"
  ],
  cognitiveAccessibility: [
    "Language is simple and clear",
    "Explanation is brief",
    "Actions are obvious",
    "Consistent placement everywhere"
  ],
  reducedMotion: [
    "No animation on ✨ icon",
    "Tooltip appears instantly (no fade)",
    "No hover effects"
  ]
};

// ═══════════════════════════════════════════════════════════════════════════════
// HARM ASSESSMENT FRAMEWORK
// ═══════════════════════════════════════════════════════════════════════════════

export const HARM_ASSESSMENT_CHECKLIST = {
  purposeTest: [
    "Does this exist to help learning?",
    "Would removing it harm the learning experience?",
    "Is there a non-simulated alternative?"
  ],
  manipulationTest: [
    "Could this create anxiety?",
    "Could this create FOMO?",
    "Could this pressure behavior?",
    "Could this exploit loneliness?"
  ],
  addictionTest: [
    "Is there any variable reward component?",
    "Is there any algorithmic personalization?",
    "Could this create compulsive checking?"
  ],
  disclosureTest: [
    "Is it clearly marked?",
    "Is the marking visible?",
    "Is the explanation accessible?"
  ],
  controlTest: [
    "Can users turn it off?",
    "Is the control easy to find?",
    "Does turning it off degrade experience?"
  ],
  vulnerableUserTest: [
    "How might children interpret this?",
    "How might lonely adults interpret this?",
    "How might users with anxiety interpret this?"
  ]
};

// ═══════════════════════════════════════════════════════════════════════════════
// METRICS & MONITORING
// ═══════════════════════════════════════════════════════════════════════════════

export const MONITORING = {
  healthMetrics: [
    "% of users who understand content is simulated (survey)",
    "User satisfaction with social features",
    "Learning outcomes with/without simulated content",
    "% users who've customized social settings"
  ],
  warningMetrics: [
    "Users reporting they thought content was real",
    "Users feeling deceived",
    "Complaints about simulated content",
    "Psychological harm reports",
    "Compulsive usage patterns"
  ],
  neverTrack: [
    "Engagement with simulated content",
    "Time on app",
    "'Retention' of social features"
  ],
  monitoringFrequency: {
    daily: "Scan support tickets",
    weekly: "Review analytics dashboards",
    monthly: "Deep dive on trends",
    quarterly: "External audit"
  },
  redFlags: [
    "Users reporting they didn't know content was simulated",
    "Users describing simulated users as 'friends'",
    "Users feeling hurt when simulated content disappears",
    "Users checking app compulsively",
    "Users feeling worse after using Kelly"
  ],
  healthySigns: [
    "Users understand simulation clearly",
    "Users appreciate the learning support",
    "Users feel comfortable turning it off",
    "Users use Kelly and then do other things",
    "Users report feeling better after learning"
  ]
};

// ═══════════════════════════════════════════════════════════════════════════════
// TESTING REQUIREMENTS
// ═══════════════════════════════════════════════════════════════════════════════

export const TESTING_REQUIREMENTS = {
  beforeLaunch: [
    "✨ appears on ALL simulated content",
    "Tooltip opens on hover/tap/keyboard",
    "Tooltip contains explanation",
    "Tooltip links to learn more AND settings",
    "Settings toggle works",
    "User preference persists",
    "Screen reader announces correctly",
    "Works on mobile",
    "Works in all themes/modes"
  ],
  ongoing: [
    "Quarterly audit of all simulated content for marking",
    "User comprehension survey (do they understand?)",
    "Accessibility audit annually"
  ],
  failsafe: `
    // Fail-safe: if disclosure can't init, hide simulated content
    try {
      new SimulatedContentDisclosure();
    } catch (e) {
      console.error('Disclosure system failed:', e);
      document.body.classList.add('simulated-hidden');
      reportDisclosureFailure(e);
    }
  `
};

// ═══════════════════════════════════════════════════════════════════════════════
// FAQ CONTENT
// ═══════════════════════════════════════════════════════════════════════════════

export const SIMULATED_CONTENT_FAQ = [
  {
    question: "Why simulate at all? Why not wait for real users?",
    answer: "Social learning is psychologically necessary. Learners need to feel they're not alone. We simulate responsibly while building real community."
  },
  {
    question: "Isn't this just lying?",
    answer: "No. Lying is deception. We fully disclose that content is simulated. Every piece is marked. Users can turn it off. That's transparency, not deception."
  },
  {
    question: "What about when we have real users?",
    answer: "Real user content will be clearly marked as real. We'll never mix real and simulated without clear labels."
  },
  {
    question: "Could simulated content be addictive?",
    answer: "We specifically design against addiction: no variable rewards, no notifications, no social comparison, no engagement optimization."
  },
  {
    question: "How is this different from social media?",
    answer: "Social media uses hidden algorithms optimized for engagement. We use transparent, predictable content optimized for learning. You're always in control."
  }
];

// ═══════════════════════════════════════════════════════════════════════════════
// STATIC COPY FOR UI
// ═══════════════════════════════════════════════════════════════════════════════

export const UI_COPY = {
  badges: {
    socialBadge: "✨ Social",
    simulatedLabel: "Simulated",
    kellyFriends: "🌟 Kelly's Learning Friends"
  },
  
  tooltips: {
    standard: {
      title: "✨ Simulated Learner",
      body: "This comment was created to show diverse learning perspectives.",
      turnOff: "Turn off simulated content"
    },
    child: {
      title: "🌟 Kelly's Learning Friend",
      body: "This is one of Kelly's pretend friends who helps you feel less alone while learning!",
      turnOff: "Ask a grown-up about settings"
    }
  },
  
  settings: {
    sectionTitle: "SIMULATED SOCIAL CONTENT",
    description: "Kelly shows AI-generated comments and reactions from simulated learners to create a supportive learning environment.",
    disclosureNote: "All simulated content is marked with ✨",
    toggleLabel: "Show simulated content",
    learnMore: "Learn why we do this",
    customizeTitle: "CUSTOMIZE SIMULATED CONTENT",
    customizeDescription: "Choose which types of simulated content you'd like to see during lessons.",
    disclosureTitle: "DISCLOSURE PREFERENCES",
    disclosureDescription: "How do you want simulated content marked?"
  },
  
  onboarding: {
    title: "LEARNING IS SOCIAL",
    body: "Humans learn best with others. That's why Kelly shows comments from simulated learners—to make you feel less alone.",
    disclosure: "These comments are AI-generated and marked with ✨ so you always know.",
    controls: [
      "Turn them off anytime in Settings",
      "Tap ✨ to learn more about any comment",
      "Real community features coming soon"
    ],
    acceptButton: "Got it, show me simulated content",
    declineButton: "No thanks, I prefer solo learning"
  },
  
  trustPage: {
    heroTitle: "Trust & Safety",
    heroSubtitle: "How Curious Kelly Handles Simulated Content",
    threeRules: {
      title: "The Three Rules",
      rules: [
        { icon: "👁", title: "Disclose Always", description: "Every piece of simulated content is marked with ✨" },
        { icon: "🎛", title: "Control Always", description: "Turn off all simulated content with one toggle in Settings" },
        { icon: "📚", title: "Benefit Always", description: "Simulated content exists only to support learning—never to manipulate" }
      ]
    }
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLE SIMULATED COMMENTS
// ═══════════════════════════════════════════════════════════════════════════════

export const SAMPLE_COMMENTS: SimulatedContentItem[] = [
  {
    id: "sample-1",
    type: "peer_comment",
    is_simulated: true,
    display_name: "Emma",
    display_age: 34,
    content: "I never thought about it that way!",
    educational_purpose: "normalize_insight_moments",
    disclosure: {
      icon: "✨",
      label: "Simulated learner",
      tooltip: "This comment was created to show diverse learning perspectives."
    }
  },
  {
    id: "sample-2",
    type: "peer_comment",
    is_simulated: true,
    display_name: "Marcus",
    display_age: 16,
    content: "Wait, I'm confused about the gravity part. Can you explain again?",
    educational_purpose: "normalize_confusion",
    disclosure: {
      icon: "✨",
      label: "Simulated learner",
      tooltip: "This comment was created to show diverse learning perspectives."
    }
  },
  {
    id: "sample-3",
    type: "age_response",
    is_simulated: true,
    display_name: "Simulated",
    display_age_range: "70s",
    content: "My grandkids explained this to me last week!",
    educational_purpose: "intergenerational_learning",
    disclosure: {
      icon: "✨",
      label: "Simulated perspective",
      tooltip: "This comment was created to show diverse learning perspectives."
    }
  },
  {
    id: "sample-4",
    type: "common_question",
    is_simulated: true,
    display_name: "Other learners",
    content: "Why does the sky look red at sunset?",
    educational_purpose: "normalize_questions",
    disclosure: {
      icon: "✨",
      label: "Common question (simulated)",
      tooltip: "This represents questions that learners commonly ask about this topic."
    }
  },
  {
    id: "sample-5",
    type: "learning_milestone",
    is_simulated: true,
    display_name: "Learning insight",
    content: "Learners like you typically feel confused here—that's normal!",
    educational_purpose: "emotional_support",
    disclosure: {
      icon: "✨",
      label: "Based on learning patterns",
      tooltip: "This insight was created to help normalize the learning journey."
    }
  }
];

// ═══════════════════════════════════════════════════════════════════════════════
// CHILD-FRIENDLY NAMES
// ═══════════════════════════════════════════════════════════════════════════════

export const CHILD_FRIENDLY_NAMES = [
  "Sunny",
  "Max",
  "Luna",
  "Pip",
  "Spark",
  "Nova",
  "Dash",
  "Ziggy",
  "Coco",
  "Maple"
];

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORT SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════

export default {
  philosophy: SIMULATED_CONTENT_PHILOSOPHY,
  principles: TRUST_SAFETY_PRINCIPLES,
  contentTypes: SIMULATED_CONTENT_TYPES,
  neverSimulate: NEVER_SIMULATE,
  redLines: RED_LINES,
  disclosure: {
    indicator: DISCLOSURE_INDICATOR,
    modes: DISCLOSURE_MODES
  },
  userControls: USER_CONTROLS,
  preferences: {
    default: DEFAULT_PREFERENCES,
    storageKey: STORAGE_KEY,
    eventName: EVENT_NAME
  },
  guidelines: CONTENT_GUIDELINES,
  ageGroups: AGE_GROUP_CONFIGS,
  css: CSS_CLASSES,
  html: HTML_TEMPLATES,
  api: API_USAGE_EXAMPLES,
  accessibility: ACCESSIBILITY_REQUIREMENTS,
  harmAssessment: HARM_ASSESSMENT_CHECKLIST,
  monitoring: MONITORING,
  testing: TESTING_REQUIREMENTS,
  faq: SIMULATED_CONTENT_FAQ,
  uiCopy: UI_COPY,
  samples: SAMPLE_COMMENTS,
  childFriendlyNames: CHILD_FRIENDLY_NAMES
};



