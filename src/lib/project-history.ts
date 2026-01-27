/**
 * PROJECT HISTORY & CONTEXT FOR AI ASSISTANTS
 * 
 * This file documents the complete history, vision, architecture, and evolution
 * of the Curious Kelly project. Future AI assistants should read this to gain
 * full context before making contributions.
 * 
 * Last Updated: December 24, 2025
 */

export const PROJECT_HISTORY = {
  // ============================================================================
  // CHAPTER 1: ORIGINS & IDENTITY
  // ============================================================================
  origins: {
    folderName: "UI-TARS-desktop",
    explanation: `
      The folder name "UI-TARS-desktop" is a legacy artifact from the project's inception.
      UI-TARS was an open-source AI agent/automation project that was cloned as a starting
      point for development infrastructure. The original UI-TARS code has been completely
      replaced—only the folder name and Apache 2.0 license attribution remain.
      
      The actual project is "Curious Kelly" by "Lesson of the Day PBC" (Public Benefit Corporation).
      The license file shows "Copyright 2025 UI-TARS Team" as a carryover from the original fork.
    `,
    
    legalEntity: "Lesson of the Day PBC",
    productName: "Curious Kelly",
    trademark: "✨ Curious Kelly (sparkles symbol LOCKED)",
    authorizedEmail: "hello@curiouskelly.com",
    productionUrl: "https://curiouskelly.com",
    
    founderBackground: `
      Founded by a co-founder of the Open Education movement—the global initiative that
      brought free educational resources to millions. But open resources weren't enough.
      People don't finish courses. They don't form habits. They don't have a teacher who
      knows their name. Kelly was built to finish what Open Education started.
    `,
    
    namedAfter: "Kelly is named after a Deaf teacher, representing 20 years of vision"
  },

  // ============================================================================
  // CHAPTER 2: THE VISION - WHAT KELLY IS
  // ============================================================================
  vision: {
    coreIdentity: `
      Kelly is not an app. Kelly is not a feature. Kelly is not a mascot.
      Kelly is THE OPERATING SYSTEM FOR HUMAN LEARNING.
      
      The calendar is her interface.
      The lessons are her heartbeat.
      The live classes are her presence.
      The iLearn device (2026) will be her body.
    `,
    
    missionStatement: `
      SDG 4: "Ensure inclusive and equitable quality education and promote
      lifelong learning opportunities for ALL." (United Nations, 2015)
      
      The world heard "children." We heard "ALL."
      Kelly teaches 2-year-olds their first words and 92-year-olds their first smartphone.
      Education isn't preparation for life—it IS life.
    `,
    
    differentiators: {
      notDuolingo: "Substance over gamification. Real teacher, not a mascot. Streaks mean something because learning happened.",
      notCoursera: "No choice paralysis. Open today, today's lesson starts. Kelly is always your teacher. No prerequisites.",
      notEdTech: "Not a platform to lock you in. Kelly becomes infrastructure. Human flourishing metrics over engagement metrics."
    },
    
    keyPhilosophy: [
      "One lesson per day for everyone on Earth—same topic, adapted differently for each learner",
      "365 lessons that repeat annually (spiral learning)",
      "Ages 2-102: Kelly herself ages to match the learner",
      "12 archetypes (Explorer, Scientist, Rebel, etc.) × 6 age buckets = 72 possible learning experiences",
      "5-10 minute micro-sessions—consistency over intensity",
      "Free forever promise (actual forever, not until acquired)"
    ]
  },

  // ============================================================================
  // CHAPTER 3: PRODUCT ARCHITECTURE
  // ============================================================================
  architecture: {
    currentState: {
      summary: "Hybrid architecture (Vercel + Supabase) with static HTML/JS frontend",
      frontend: "Static HTML/JS served from public/ directory (learn.html is the main lesson player)",
      backend: "Vercel Edge Functions + Serverless Functions",
      database: "Supabase PostgreSQL (project: tvjalxxsyryjphkforjv)",
      storage: "Supabase Storage (kelly-videos, kelly-audio, lesson-visuals buckets)",
      avatar: "2D HD videos with lip-sync (Unity WebGL legacy exists but not in production)"
    },
    
    keyTables: {
      core_lessons: "365 daily lessons (1 per day)",
      lesson_atoms: "~20,341 phase variants (12 archetypes × 5 phases × 365 days)",
      lesson_shards: "~6,570 age/language variants",
      kelly_video_assets: "Video tracking (312 videos as of Dec 2025)",
      kelly_motion_library: "336 generic motion clips as fallback",
      users: "User profiles (extends auth.users)",
      user_progress: "Learning progress, streaks, completions"
    },
    
    lessonStructure: {
      phases: ["Hook", "Fact1 (Q1)", "Fact2 (Q2)", "Fact3 (Q3)", "Wisdom"],
      phasesExplained: "Each lesson has 5 phases. Q phases have interactive choices (A/B/C options).",
      archetypes: [
        "The Explorer", "The Rebel", "The Scientist", "The Architect",
        "The Mystic", "The Provider", "The Artist", "The Scholar",
        "The Warrior", "The Sage", "The Dreamer", "The Builder"
      ],
      ageBuckets: ["2-5", "6-12", "13-17", "18-35", "36-60", "61-102"]
    },
    
    dataFlowCascade: `
      Lesson data loads in this priority order:
      1. URL/localStorage overrides (for testing)
      2. Local JS packs (public/data/day-XXX-complete.js) - 365 files
      3. Seed JSON (public/lessons/day-N.json) - 366 files
      4. Supabase (core_lessons + lesson_atoms + lesson_shards) - Primary production source
      5. Cloudflare D1 / Local API fallback
      6. Emergency hardcoded fallback
    `
  },

  // ============================================================================
  // CHAPTER 4: KEY INTEGRATIONS
  // ============================================================================
  integrations: {
    voice: {
      primary: "ElevenLabs",
      voiceId: "wAdymQH5YucAkXwmrdL0 (Kelly's trained voice)",
      model: "eleven_multilingual_v2",
      rule: "NEVER use browser TTS. ElevenLabs only, or Piper if explicitly configured."
    },
    
    video: {
      generation: "Sync Labs lipsync-2-pro + MiniMax Video-01",
      avatar: "Flux + Kelly LoRA (CuriousKellycom/curious-kelly-lora on HuggingFace)",
      deprecated: "HeyGen (queue issues), Unity WebGL (too heavy)"
    },
    
    payments: {
      provider: "Stripe",
      tiers: {
        daily: "$1.99/day",
        monthly: "$7.99/month",
        annual: "$49.99/year",
        lifetime: "$149 (permanent access)"
      },
      affiliate: "Every subscriber becomes an affiliate automatically (20%/15%/10% commission)"
    },
    
    databases: {
      primary: "Supabase PostgreSQL",
      auth: "Supabase Auth",
      storage: "Supabase Storage (primary), Cloudflare R2 (legacy Unity assets)"
    }
  },

  // ============================================================================
  // CHAPTER 5: RELATED PRODUCTS (ECOSYSTEM)
  // ============================================================================
  ecosystem: {
    curiousKelly: {
      description: "The main product—daily AI teacher for ages 2-102",
      status: "Production (curiouskelly.com)",
      target: "Launched December 17, 2025"
    },
    
    theDailyLesson: {
      description: "Alternative branding/domain for the same core product",
      domain: "thedailylesson.com",
      relationship: "Same product, different marketing angle"
    },
    
    reinmaker: {
      description: "Mobile RPG game experience that integrates with lesson content",
      status: "Draft/Planning (API contracts defined in docs/reinmaker/)",
      integration: "Shares PhaseDNA, audio, animation assets with main platform"
    },
    
    iLearnDevice: {
      description: "Dedicated hardware device for distraction-free learning",
      timeline: "2026 Q3-Q4 launch target",
      specs: "8\" E-ink + Color OLED, Kelly OS pre-installed, no social media/distractions",
      pricePoint: "$299 base, $199 education",
      goal: "Apple Store distribution by 2027"
    }
  },

  // ============================================================================
  // CHAPTER 6: KEY FILES & NAVIGATION
  // ============================================================================
  keyFiles: {
    governance: {
      "CLAUDE.md": "Operating rules for AI contributions (READ THIS FIRST)",
      "KELLY_OS_VISION.md": "Complete vision document for Kelly as an OS",
      "docs/strategy/PHILANTHROPIC_STRATEGY.md": "Philanthropic positioning and SDG 4 alignment"
    },
    
    architecture: {
      "ARCHITECTURE_TRUTH.md": "Definitive content loading and video playback architecture",
      "SYSTEM_ARCHITECTURE.md": "Complete technical audit (database, APIs, deployment)",
      "MIGRATION_PLAN.md": "Vercel-only migration plan (from hybrid Cloudflare)"
    },
    
    execution: {
      "docs/_archive/CURIOUS_KELLLY_EXECUTION_PLAN.md": "Original 12-week execution roadmap",
      "vom/UNIFIED_LESSON_FACTORY_FINAL.md": "Master prompt for lesson generation"
    },
    
    content: {
      "src/lib/static-content.ts": "Privacy, Terms, About, FAQ content",
      "public/learn.html": "Main lesson player (monolithic ~20K lines)",
      "public/index.html": "Landing page with age gate",
      "public/lessons/": "366 seed JSON files for offline/fallback"
    },
    
    pipelines: {
      "scripts/kelly-video-factory/": "HD video generation pipeline",
      "scripts/kelly-phase-visuals/": "Infographic/visual generation",
      "scripts/lesson-factory/": "Unified lesson generation"
    }
  },

  // ============================================================================
  // CHAPTER 7: NON-NEGOTIABLE RULES
  // ============================================================================
  goldenRules: {
    invariants: [
      "Languages are PRECOMPUTED in every DNA/content file (EN + ES/FR). No runtime language generation.",
      "Minimum 60 minutes training audio per voice model. Never downsample or shrink datasets.",
      "NEVER use browser TTS. ElevenLabs only.",
      "Target 60 FPS for applicable media. Respect rate limits. Batch requests. Cache assets.",
      "NEVER use the word 'free' in marketing copy. Use 'yours', 'included', '7 days to explore'."
    ],
    
    forbidden: [
      "Creating new lesson players or pages",
      "Interest-driven lesson selection (everyone gets the same daily topic)",
      "Learner 'learning-style' classification",
      "Degrading or shrinking training datasets",
      "Deleting/moving content without explicit approval",
      "Using unauthorized email addresses (only hello@curiouskelly.com)"
    ],
    
    trustAndSafety: [
      "ALL simulated social content MUST be marked with ✨ indicator",
      "Never claim simulated users are real people",
      "Never use simulated content to manipulate emotions",
      "Never add variable rewards or addiction mechanics",
      "Never show fake metrics as real",
      "Always use 'Lesson of the Day PBC' (not 'Curious Kelly PBC') in legal contexts"
    ],
    
    quality: [
      "Curious Kelly is TIMELESS and PERFECT before launch",
      "Never suggest 'good enough' content. Never recommend lazy options.",
      "If it's not perfect, we don't launch at all",
      "Quality over speed, always"
    ]
  },

  // ============================================================================
  // CHAPTER 8: TIMELINE & MILESTONES
  // ============================================================================
  timeline: {
    prehistory: "~20 years of vision development by founder (Open Education co-founder)",
    
    "2025-10": "Project initiated in UI-TARS-desktop folder structure",
    "2025-10-29": "Execution plan merged from UI-TARS workspace + CK Production Requirements",
    "2025-12-10": "Major implementation work, HeyGen integration attempts",
    "2025-12-12": "Kelly OS Vision document finalized",
    "2025-12-17": "Target genesis launch (v1.0)",
    "2025-12-24": "Project history documentation created for future AI context",
    
    future: {
      "2026 Q1": "Voice tuning, ASL foundation",
      "2026 Q2": "Unity 2D/3D toggle, Community features",
      "2026 Q2-Q3": "Real-time voice dialogue, Practice mode",
      "2026 Q3-Q4": "iLearn hardware prototype and launch",
      "2027 Q1": "Apple Store partnership goal",
      "2027+": "10+ languages, global deployment, Kelly API/SDK"
    }
  },

  // ============================================================================
  // CHAPTER 9: CONTEXT FOR AI CONTRIBUTORS
  // ============================================================================
  aiGuidance: {
    beforeYouStart: [
      "Read CLAUDE.md completely—it governs assistant behavior",
      "Check if there's an approved plan reference for any change",
      "Verify languages are precomputed and schemas validated",
      "Confirm no cost increases without approval",
      "Run tests, linters, and validators locally before proposing changes"
    ],
    
    commonMistakes: [
      "Jumping to implement without fully understanding existing codebase",
      "Creating duplicate implementations instead of finding existing ones",
      "Using localStorage/sessionStorage keys inconsistently",
      "Suggesting 'quick fixes' that compromise quality",
      "Assuming something is broken when it might be intentional design"
    ],
    
    searchStrategy: `
      1. Search codebase for existing solutions FIRST
      2. Read relevant documentation/checklists
      3. Understand the deployment architecture
      4. Verify the fix is actually needed
      5. Only implement if steps 1-4 confirm necessity
      
      Time spent investigating saves 10x more than fixing duplicate work.
    `,
    
    deploymentNotes: {
      vercel: "Node.js must be 22.x (not 18.x, 20.x, or 24.x)",
      packageJson: 'engines.node should be "22.x"',
      nvmrc: 'Should contain "22"',
      vercelJson: 'Functions runtime should be "@vercel/node@3.2.29"'
    }
  },

  // ============================================================================
  // CHAPTER 10: THE SOUL
  // ============================================================================
  soul: {
    whyThisMatters: `
      Kelly isn't a product—she's infrastructure for humanity. The first truly global
      teacher who never sleeps, never burns out, and never gives up on a learner.
      A public utility disguised as a startup.
      
      Social media hijacked social learning. Traditional education struggles with
      personalization. AI tutors risk replacing human thought. Kelly is the ethical
      middle ground—a transparent, predictable, growth-oriented AI companion that
      amplifies learning without replacing the learner.
    `,
    
    kellysGreeting: "I'm here. Every day. For everyone.",
    
    aboutKelly: `
      Hi—I'm Kelly. I'm a digital teacher designed for daily lessons. Every day,
      I find something wonderful and I can't wait to share it with you. I adapt
      each lesson to how you learn—your age, your pace, your curiosity. The same
      topic, explained differently for each learner.
      
      I'm honest about what I am—just 0s and 1s, a timeless scattering of pixels—
      but I'm here to learn alongside you, not above you.
      
      Want to come along?
    `,
    
    coreBeliefs: [
      "Education is a human right, not a privilege",
      "Learning should be daily, not occasional",
      "Every person deserves a teacher who adapts to them",
      "Technology should serve humans, not exploit them",
      "AI should make education more human, not less"
    ]
  }
};

/**
 * Quick reference for the most critical facts
 */
export const QUICK_CONTEXT = {
  project: "Curious Kelly (folder name 'UI-TARS-desktop' is legacy)",
  company: "Lesson of the Day PBC",
  url: "curiouskelly.com",
  email: "hello@curiouskelly.com",
  database: "Supabase (tvjalxxsyryjphkforjv)",
  voice: "ElevenLabs (voice ID: wAdymQH5YucAkXwmrdL0)",
  node: "22.x required",
  mission: "SDG 4: Quality Education for ALL (ages 2-102)",
  dailyLesson: "One topic per day for everyone on Earth",
  
  readFirst: ["CLAUDE.md", "ARCHITECTURE_TRUTH.md", "KELLY_OS_VISION.md"],
  
  neverDo: [
    "Use browser TTS",
    "Create new lesson players",
    "Say 'free' in marketing",
    "Use emails other than hello@curiouskelly.com",
    "Suggest 'good enough' quality"
  ]
};



