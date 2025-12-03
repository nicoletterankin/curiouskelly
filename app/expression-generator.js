/**
 * AI-Powered Expression Generation System for Curious Kelly
 * 
 * Generates facial expressions and gestures from:
 * - ElevenLabs API response (timing, emphasis, pauses)
 * - Lesson text content (emotions, questions, excitement)
 * - User state (age, archetype, tone, language)
 * - Phase type (welcome, q1, q2, q3, wisdom)
 * 
 * @module expression-generator
 */

// =============================================================================
// ARCHETYPE EXPRESSION PROFILES
// =============================================================================

/**
 * The 12 Curious Kelly archetypes with their expression and gesture styles.
 * Each archetype has distinct facial expressions, gestures, and movement patterns.
 */
export const ARCHETYPE_PROFILES = {
  'The Scientist': {
    id: 'scientist',
    traits: ['Analytical', 'Abstract'],
    expressionStyle: 'subtle',
    gestureIntensity: 0.5,
    defaultExpressions: {
      neutral: { smile: 20, eyebrowRaise: 10 },
      thinking: { eyebrowRaise: 40, eyeSquint: 20, lipsPursed: 30 },
      explaining: { eyebrowRaise: 35, smile: 25, eyesWide: 15 },
      curious: { eyebrowRaise: 60, eyesWide: 40, headTilt: 20 },
      satisfied: { smile: 45, eyesClosed: 20, nod: 30 },
    },
    gestureLibrary: [
      { name: 'chin_touch', frequency: 'high', duration: [1.5, 3.0], contexts: ['thinking', 'analyzing'] },
      { name: 'glasses_adjust', frequency: 'medium', duration: [0.8, 1.2], contexts: ['transition', 'emphasis'] },
      { name: 'finger_point_precise', frequency: 'medium', duration: [1.0, 2.0], contexts: ['explaining', 'key_point'] },
      { name: 'hands_steepled', frequency: 'high', duration: [2.0, 4.0], contexts: ['thinking', 'conclusion'] },
      { name: 'palm_up_single', frequency: 'low', duration: [1.5, 2.5], contexts: ['presenting', 'offering_idea'] },
      { name: 'nod_slow', frequency: 'medium', duration: [1.0, 1.5], contexts: ['agreement', 'confirmation'] },
    ],
    eyeMovement: { gazeShifts: 'frequent', blinkRate: 'normal', squintOnThink: true },
    headMovement: { tilts: 'subtle', nods: 'deliberate', speed: 0.8 },
  },

  'The Explorer': {
    id: 'explorer',
    traits: ['Energetic', 'Abstract'],
    expressionStyle: 'animated',
    gestureIntensity: 0.85,
    defaultExpressions: {
      neutral: { smile: 40, eyebrowRaise: 20, eyesWide: 25 },
      excited: { smile: 90, eyebrowRaise: 70, eyesWide: 80, mouthOpen: 30 },
      curious: { eyebrowRaise: 75, eyesWide: 70, headTilt: 35, smile: 50 },
      explaining: { smile: 55, eyebrowRaise: 50, gestureActive: true },
      amazed: { eyesWide: 90, eyebrowRaise: 85, mouthOpen: 50, smile: 40 },
    },
    gestureLibrary: [
      { name: 'point_up_dramatic', frequency: 'high', duration: [0.8, 1.5], contexts: ['discovery', 'insight'] },
      { name: 'arms_wide_open', frequency: 'high', duration: [1.5, 2.5], contexts: ['expansion', 'possibility'] },
      { name: 'reaching_forward', frequency: 'medium', duration: [1.0, 2.0], contexts: ['invitation', 'exploration'] },
      { name: 'hands_clasp_excited', frequency: 'medium', duration: [0.8, 1.2], contexts: ['excitement', 'anticipation'] },
      { name: 'sweep_gesture', frequency: 'high', duration: [1.2, 2.0], contexts: ['showing_scope', 'panoramic'] },
      { name: 'bounce_hop', frequency: 'low', duration: [0.5, 0.8], contexts: ['peak_excitement', 'celebration'] },
    ],
    eyeMovement: { gazeShifts: 'dynamic', blinkRate: 'fast', squintOnThink: false },
    headMovement: { tilts: 'expressive', nods: 'enthusiastic', speed: 1.2 },
  },

  'The Storyteller': {
    id: 'storyteller',
    traits: ['Expressive', 'Abstract'],
    expressionStyle: 'dramatic',
    gestureIntensity: 0.9,
    defaultExpressions: {
      neutral: { smile: 35, eyebrowRaise: 25, expressiveMouth: true },
      dramatic: { eyebrowRaise: 80, eyesWide: 60, mouthExpressive: 70 },
      mysterious: { eyebrowRaise: 20, eyeSquint: 40, smile: 30, headTilt: 25 },
      joyful: { smile: 95, eyesClosed: 30, cheekRaise: 70 },
      suspense: { eyesWide: 50, eyebrowRaise: 60, pause: true },
    },
    gestureLibrary: [
      { name: 'hands_flowing', frequency: 'high', duration: [2.0, 4.0], contexts: ['narration', 'description'] },
      { name: 'theatrical_pause', frequency: 'medium', duration: [1.5, 2.5], contexts: ['suspense', 'emphasis'] },
      { name: 'character_mime', frequency: 'medium', duration: [2.0, 3.5], contexts: ['acting', 'demonstration'] },
      { name: 'frame_gesture', frequency: 'low', duration: [1.5, 2.0], contexts: ['scene_setting', 'visualization'] },
      { name: 'heart_touch', frequency: 'low', duration: [1.0, 1.5], contexts: ['emotional_moment', 'connection'] },
      { name: 'expansive_reveal', frequency: 'medium', duration: [1.2, 2.0], contexts: ['reveal', 'climax'] },
    ],
    eyeMovement: { gazeShifts: 'theatrical', blinkRate: 'varied', squintOnThink: false },
    headMovement: { tilts: 'dramatic', nods: 'expressive', speed: 1.0 },
  },

  'The Empath': {
    id: 'empath',
    traits: ['Warm', 'Abstract'],
    expressionStyle: 'gentle',
    gestureIntensity: 0.6,
    defaultExpressions: {
      neutral: { smile: 50, eyebrowRaise: 15, warmGaze: true },
      compassionate: { smile: 60, eyebrowRaise: 30, eyesClosed: 15, headTilt: 20 },
      understanding: { smile: 45, nod: 40, eyebrowRaise: 25 },
      encouraging: { smile: 75, eyebrowRaise: 40, cheekRaise: 50 },
      listening: { eyebrowRaise: 20, smile: 35, headTilt: 30 },
    },
    gestureLibrary: [
      { name: 'heart_open', frequency: 'high', duration: [2.0, 3.5], contexts: ['emotional', 'connection'] },
      { name: 'gentle_reach', frequency: 'medium', duration: [1.5, 2.5], contexts: ['support', 'comfort'] },
      { name: 'hands_together_heart', frequency: 'medium', duration: [1.5, 2.0], contexts: ['gratitude', 'appreciation'] },
      { name: 'soft_nod', frequency: 'high', duration: [1.0, 1.5], contexts: ['understanding', 'acknowledgment'] },
      { name: 'embrace_gesture', frequency: 'low', duration: [2.0, 3.0], contexts: ['warm_welcome', 'acceptance'] },
      { name: 'palm_up_offering', frequency: 'medium', duration: [1.5, 2.5], contexts: ['invitation', 'openness'] },
    ],
    eyeMovement: { gazeShifts: 'soft', blinkRate: 'slow', squintOnThink: false },
    headMovement: { tilts: 'gentle', nods: 'supportive', speed: 0.7 },
  },

  'The Mystic': {
    id: 'mystic',
    traits: ['Deep', 'Abstract'],
    expressionStyle: 'contemplative',
    gestureIntensity: 0.55,
    defaultExpressions: {
      neutral: { smile: 25, eyebrowRaise: 15, gazeDistance: true },
      contemplative: { eyesClosed: 40, smile: 20, breathDeep: true },
      profound: { eyesWide: 30, eyebrowRaise: 50, smile: 35, pause: true },
      serene: { smile: 40, eyesClosed: 50, headTilt: 15 },
      insightful: { eyebrowRaise: 60, eyesWide: 45, smile: 30 },
    },
    gestureLibrary: [
      { name: 'hands_prayer_position', frequency: 'medium', duration: [2.0, 4.0], contexts: ['reflection', 'wisdom'] },
      { name: 'slow_raise', frequency: 'low', duration: [2.5, 4.0], contexts: ['elevation', 'transcendence'] },
      { name: 'gentle_circle', frequency: 'medium', duration: [2.0, 3.0], contexts: ['wholeness', 'unity'] },
      { name: 'touch_third_eye', frequency: 'low', duration: [1.5, 2.5], contexts: ['insight', 'intuition'] },
      { name: 'palms_up_receiving', frequency: 'medium', duration: [2.5, 4.0], contexts: ['openness', 'receiving'] },
      { name: 'breath_gesture', frequency: 'high', duration: [2.0, 3.5], contexts: ['pause', 'centering'] },
    ],
    eyeMovement: { gazeShifts: 'slow', blinkRate: 'slow', squintOnThink: false },
    headMovement: { tilts: 'minimal', nods: 'deliberate', speed: 0.6 },
  },

  'The Provider': {
    id: 'provider',
    traits: ['Warm', 'Practical'],
    expressionStyle: 'nurturing',
    gestureIntensity: 0.65,
    defaultExpressions: {
      neutral: { smile: 55, eyebrowRaise: 20, warmGaze: true },
      caring: { smile: 70, eyebrowRaise: 35, cheekRaise: 45 },
      encouraging: { smile: 80, eyebrowRaise: 50, nod: 40 },
      proud: { smile: 75, eyesClosed: 20, cheekRaise: 60 },
      protective: { eyebrowRaise: 30, smile: 40, confidenceGaze: true },
    },
    gestureLibrary: [
      { name: 'open_arms_welcome', frequency: 'high', duration: [2.0, 3.0], contexts: ['welcome', 'acceptance'] },
      { name: 'pat_gesture', frequency: 'medium', duration: [1.0, 1.5], contexts: ['encouragement', 'approval'] },
      { name: 'hands_on_hips_proud', frequency: 'low', duration: [1.5, 2.5], contexts: ['pride', 'confidence'] },
      { name: 'gather_gesture', frequency: 'medium', duration: [1.5, 2.5], contexts: ['inclusive', 'community'] },
      { name: 'present_gift', frequency: 'low', duration: [1.5, 2.0], contexts: ['offering', 'sharing'] },
      { name: 'supportive_nod', frequency: 'high', duration: [1.0, 1.5], contexts: ['agreement', 'support'] },
    ],
    eyeMovement: { gazeShifts: 'warm', blinkRate: 'normal', squintOnThink: false },
    headMovement: { tilts: 'nurturing', nods: 'supportive', speed: 0.8 },
  },

  'The Diplomat': {
    id: 'diplomat',
    traits: ['Social', 'Practical'],
    expressionStyle: 'balanced',
    gestureIntensity: 0.6,
    defaultExpressions: {
      neutral: { smile: 45, eyebrowRaise: 15, composedGaze: true },
      agreeable: { smile: 55, nod: 35, eyebrowRaise: 25 },
      considering: { eyebrowRaise: 30, headTilt: 25, smile: 30 },
      pleased: { smile: 65, eyebrowRaise: 30, cheekRaise: 40 },
      negotiating: { eyebrowRaise: 40, smile: 35, handGesture: true },
    },
    gestureLibrary: [
      { name: 'balance_hands', frequency: 'high', duration: [1.5, 2.5], contexts: ['comparing', 'weighing'] },
      { name: 'bridge_gesture', frequency: 'medium', duration: [1.5, 2.0], contexts: ['connecting', 'mediating'] },
      { name: 'open_palm_each_side', frequency: 'medium', duration: [2.0, 3.0], contexts: ['presenting_options', 'fairness'] },
      { name: 'inclusive_sweep', frequency: 'medium', duration: [1.5, 2.5], contexts: ['everyone', 'together'] },
      { name: 'handshake_gesture', frequency: 'low', duration: [1.0, 1.5], contexts: ['agreement', 'deal'] },
      { name: 'thoughtful_chin', frequency: 'medium', duration: [1.5, 2.5], contexts: ['consideration', 'respect'] },
    ],
    eyeMovement: { gazeShifts: 'measured', blinkRate: 'normal', squintOnThink: false },
    headMovement: { tilts: 'considered', nods: 'diplomatic', speed: 0.85 },
  },

  'The Architect': {
    id: 'architect',
    traits: ['Structured', 'Visionary'],
    expressionStyle: 'focused',
    gestureIntensity: 0.65,
    defaultExpressions: {
      neutral: { smile: 25, eyebrowRaise: 15, focusedGaze: true },
      designing: { eyebrowRaise: 45, eyeSquint: 15, handActive: true },
      revealing: { eyesWide: 55, eyebrowRaise: 60, smile: 45 },
      satisfied: { smile: 50, nod: 35, eyesClosed: 15 },
      planning: { eyebrowRaise: 35, focusedGaze: true, lipsPursed: 20 },
    },
    gestureLibrary: [
      { name: 'building_blocks', frequency: 'high', duration: [2.0, 3.5], contexts: ['structuring', 'building'] },
      { name: 'blueprint_trace', frequency: 'medium', duration: [2.0, 3.0], contexts: ['planning', 'designing'] },
      { name: 'precise_placement', frequency: 'high', duration: [1.0, 2.0], contexts: ['positioning', 'arranging'] },
      { name: 'frame_vision', frequency: 'medium', duration: [1.5, 2.5], contexts: ['visualizing', 'imagining'] },
      { name: 'stack_gesture', frequency: 'medium', duration: [1.5, 2.5], contexts: ['layering', 'organizing'] },
      { name: 'connect_points', frequency: 'high', duration: [1.5, 2.0], contexts: ['relating', 'linking'] },
    ],
    eyeMovement: { gazeShifts: 'purposeful', blinkRate: 'normal', squintOnThink: true },
    headMovement: { tilts: 'analytical', nods: 'confirming', speed: 0.9 },
  },

  'The Rebel': {
    id: 'rebel',
    traits: ['Chaotic', 'Practical'],
    expressionStyle: 'bold',
    gestureIntensity: 0.8,
    defaultExpressions: {
      neutral: { smile: 30, eyebrowRaise: 25, smirk: 35 },
      challenging: { eyebrowRaise: 70, smirk: 50, headTilt: 30 },
      triumphant: { smile: 80, eyebrowRaise: 55, fistRaise: true },
      skeptical: { eyebrowRaise: 55, eyeSquint: 40, smirk: 45 },
      energized: { eyesWide: 60, smile: 65, eyebrowRaise: 50 },
    },
    gestureLibrary: [
      { name: 'fist_pump', frequency: 'medium', duration: [0.8, 1.2], contexts: ['victory', 'emphasis'] },
      { name: 'break_chains', frequency: 'low', duration: [1.5, 2.5], contexts: ['freedom', 'breakthrough'] },
      { name: 'dismiss_wave', frequency: 'medium', duration: [1.0, 1.5], contexts: ['rejecting', 'challenging'] },
      { name: 'provocative_point', frequency: 'high', duration: [1.0, 1.5], contexts: ['challenge', 'question'] },
      { name: 'shoulder_shrug', frequency: 'medium', duration: [0.8, 1.2], contexts: ['so_what', 'casual'] },
      { name: 'rock_on', frequency: 'low', duration: [0.8, 1.5], contexts: ['cool', 'excitement'] },
    ],
    eyeMovement: { gazeShifts: 'bold', blinkRate: 'irregular', squintOnThink: true },
    headMovement: { tilts: 'confident', nods: 'defiant', speed: 1.1 },
  },

  'The Strategist': {
    id: 'strategist',
    traits: ['Practical', 'Structured'],
    expressionStyle: 'controlled',
    gestureIntensity: 0.55,
    defaultExpressions: {
      neutral: { smile: 20, eyebrowRaise: 10, composedGaze: true },
      calculating: { eyeSquint: 35, eyebrowRaise: 30, focusedGaze: true },
      pleased: { smile: 45, eyebrowRaise: 25, nod: 25 },
      explaining: { eyebrowRaise: 40, smile: 30, handGesture: true },
      decisive: { smile: 35, nod: 45, confidenceGaze: true },
    },
    gestureLibrary: [
      { name: 'chess_move', frequency: 'medium', duration: [1.5, 2.5], contexts: ['strategic', 'planning'] },
      { name: 'count_fingers', frequency: 'high', duration: [2.0, 3.5], contexts: ['listing', 'organizing'] },
      { name: 'map_gesture', frequency: 'medium', duration: [2.0, 3.0], contexts: ['overview', 'big_picture'] },
      { name: 'precise_point', frequency: 'high', duration: [1.0, 1.5], contexts: ['specific', 'exact'] },
      { name: 'timeline_draw', frequency: 'medium', duration: [2.0, 3.0], contexts: ['sequence', 'process'] },
      { name: 'check_off', frequency: 'medium', duration: [0.8, 1.2], contexts: ['completion', 'progress'] },
    ],
    eyeMovement: { gazeShifts: 'calculated', blinkRate: 'slow', squintOnThink: true },
    headMovement: { tilts: 'strategic', nods: 'decisive', speed: 0.85 },
  },

  'The MacGyver': {
    id: 'macgyver',
    traits: ['Resourceful', 'Analytical'],
    expressionStyle: 'inventive',
    gestureIntensity: 0.75,
    defaultExpressions: {
      neutral: { smile: 35, eyebrowRaise: 20, alertGaze: true },
      eureka: { eyesWide: 80, eyebrowRaise: 75, smile: 70, exciteJump: true },
      tinkering: { eyeSquint: 40, eyebrowRaise: 25, tongueOut: 20 },
      proud: { smile: 65, eyebrowRaise: 45, cheekRaise: 40 },
      problem_solving: { eyebrowRaise: 50, focusedGaze: true, handActive: true },
    },
    gestureLibrary: [
      { name: 'assembling', frequency: 'high', duration: [2.0, 4.0], contexts: ['building', 'creating'] },
      { name: 'light_bulb', frequency: 'medium', duration: [0.8, 1.5], contexts: ['idea', 'solution'] },
      { name: 'tool_mime', frequency: 'medium', duration: [1.5, 2.5], contexts: ['fixing', 'adjusting'] },
      { name: 'improvise_gesture', frequency: 'high', duration: [1.5, 2.5], contexts: ['adapting', 'creating'] },
      { name: 'tada_reveal', frequency: 'low', duration: [1.0, 1.5], contexts: ['completion', 'success'] },
      { name: 'roll_sleeves', frequency: 'low', duration: [1.0, 1.5], contexts: ['getting_started', 'ready'] },
    ],
    eyeMovement: { gazeShifts: 'scanning', blinkRate: 'fast', squintOnThink: true },
    headMovement: { tilts: 'curious', nods: 'quick', speed: 1.0 },
  },

  'The Survivor': {
    id: 'survivor',
    traits: ['Practical', 'Serious'],
    expressionStyle: 'grounded',
    gestureIntensity: 0.5,
    defaultExpressions: {
      neutral: { smile: 15, eyebrowRaise: 10, steadyGaze: true },
      determined: { eyebrowRaise: 30, jawSet: 30, focusedGaze: true },
      reassuring: { smile: 40, eyebrowRaise: 25, nod: 35 },
      alert: { eyesWide: 40, eyebrowRaise: 45, readyStance: true },
      accomplished: { smile: 50, nod: 40, relaxedShoulders: true },
    },
    gestureLibrary: [
      { name: 'grounding_stance', frequency: 'high', duration: [2.0, 4.0], contexts: ['stability', 'foundation'] },
      { name: 'practical_demo', frequency: 'high', duration: [2.0, 3.5], contexts: ['showing', 'teaching'] },
      { name: 'ready_hands', frequency: 'medium', duration: [1.5, 2.5], contexts: ['preparedness', 'alertness'] },
      { name: 'steady_point', frequency: 'medium', duration: [1.0, 2.0], contexts: ['directing', 'focusing'] },
      { name: 'fist_resolve', frequency: 'low', duration: [1.0, 1.5], contexts: ['determination', 'strength'] },
      { name: 'calm_down_motion', frequency: 'medium', duration: [1.5, 2.5], contexts: ['reassurance', 'calming'] },
    ],
    eyeMovement: { gazeShifts: 'scanning', blinkRate: 'normal', squintOnThink: false },
    headMovement: { tilts: 'minimal', nods: 'confirming', speed: 0.75 },
  },
};

// =============================================================================
// TONE EXPRESSION MODIFIERS
// =============================================================================

/**
 * Tone modifiers that adjust base expressions
 */
export const TONE_MODIFIERS = {
  enthusiastic: {
    id: 'enthusiastic',
    multiplier: 1.3,
    baselineShift: {
      smile: 25,
      eyebrowRaise: 20,
      eyesWide: 15,
      energyLevel: 1.3,
    },
    gestureFrequencyMultiplier: 1.4,
    speechRateHint: 'faster',
    peakExpressions: {
      smile: 95,
      eyebrowRaise: 85,
      eyesWide: 80,
    },
    triggers: ['!', 'amazing', 'incredible', 'wow', 'fantastic', 'awesome', 'wonderful'],
  },

  serious: {
    id: 'serious',
    multiplier: 0.7,
    baselineShift: {
      smile: -20,
      eyebrowRaise: -5,
      focusedGaze: 30,
      energyLevel: 0.8,
    },
    gestureFrequencyMultiplier: 0.6,
    speechRateHint: 'slower',
    peakExpressions: {
      eyebrowRaise: 45,
      focusedGaze: 60,
      nod: 50,
    },
    triggers: ['important', 'critical', 'serious', 'careful', 'warning', 'attention'],
  },

  playful: {
    id: 'playful',
    multiplier: 1.2,
    baselineShift: {
      smile: 30,
      eyebrowRaise: 15,
      headBob: 20,
      energyLevel: 1.2,
    },
    gestureFrequencyMultiplier: 1.3,
    speechRateHint: 'varied',
    peakExpressions: {
      smile: 90,
      wink: 80,
      headTilt: 40,
    },
    triggers: ['fun', 'play', 'silly', 'game', 'imagine', 'pretend', 'guess'],
  },

  thoughtful: {
    id: 'thoughtful',
    multiplier: 0.85,
    baselineShift: {
      smile: 5,
      eyebrowRaise: 15,
      gazeShift: 25,
      energyLevel: 0.9,
    },
    gestureFrequencyMultiplier: 0.8,
    speechRateHint: 'deliberate',
    peakExpressions: {
      eyebrowRaise: 55,
      gazeUp: 45,
      headTilt: 35,
    },
    triggers: ['think', 'consider', 'perhaps', 'maybe', 'wonder', 'ponder', 'reflect'],
  },

  warm: {
    id: 'warm',
    multiplier: 1.0,
    baselineShift: {
      smile: 20,
      eyebrowRaise: 10,
      softGaze: 30,
      energyLevel: 1.0,
    },
    gestureFrequencyMultiplier: 0.9,
    speechRateHint: 'gentle',
    peakExpressions: {
      smile: 75,
      eyesClosed: 30,
      cheekRaise: 50,
    },
    triggers: ['love', 'care', 'dear', 'heart', 'wonderful', 'special', 'precious'],
  },

  confident: {
    id: 'confident',
    multiplier: 1.1,
    baselineShift: {
      smile: 15,
      eyebrowRaise: 10,
      confidenceGaze: 35,
      energyLevel: 1.1,
    },
    gestureFrequencyMultiplier: 1.1,
    speechRateHint: 'steady',
    peakExpressions: {
      smile: 60,
      nod: 55,
      eyebrowRaise: 50,
    },
    triggers: ['definitely', 'absolutely', 'certainly', 'know', 'sure', 'exactly'],
  },
};

// =============================================================================
// AGE-BASED EXPRESSION PROFILES
// =============================================================================

/**
 * Age group expression modifiers
 */
export const AGE_PROFILES = {
  '2-5': {
    id: 'toddler',
    label: 'Early Childhood',
    expressionIntensityMultiplier: 1.5,
    movementAmplitude: 1.4,
    gestureFrequencyMultiplier: 1.3,
    expressionDuration: 0.7, // Faster transitions
    characteristics: {
      bouncy: true,
      exaggerated: true,
      highEnergy: true,
    },
    baselineShift: {
      smile: 30,
      eyesWide: 25,
      eyebrowRaise: 20,
    },
    preferredGestures: ['bounce_hop', 'clap_hands', 'arms_wide_open', 'point_up_dramatic'],
    avoidGestures: ['chin_touch', 'hands_steepled', 'thoughtful_chin'],
  },

  '6-12': {
    id: 'child',
    label: 'Childhood',
    expressionIntensityMultiplier: 1.25,
    movementAmplitude: 1.2,
    gestureFrequencyMultiplier: 1.2,
    expressionDuration: 0.85,
    characteristics: {
      animated: true,
      curious: true,
      energetic: true,
    },
    baselineShift: {
      smile: 20,
      eyesWide: 15,
      eyebrowRaise: 15,
    },
    preferredGestures: ['point_up_dramatic', 'hands_clasp_excited', 'nod_enthusiastic'],
    avoidGestures: ['chess_move', 'blueprint_trace'],
  },

  '13-17': {
    id: 'teen',
    label: 'Teenage',
    expressionIntensityMultiplier: 0.9,
    movementAmplitude: 0.85,
    gestureFrequencyMultiplier: 0.85,
    expressionDuration: 1.0,
    characteristics: {
      restrained: true,
      cool: true,
      engaged: true,
    },
    baselineShift: {
      smile: -5,
      eyebrowRaise: 5,
      smirk: 15,
    },
    preferredGestures: ['shoulder_shrug', 'nod_subtle', 'finger_point_precise'],
    avoidGestures: ['bounce_hop', 'clap_hands', 'embrace_gesture'],
  },

  '18-35': {
    id: 'adult',
    label: 'Young Adult',
    expressionIntensityMultiplier: 1.0,
    movementAmplitude: 1.0,
    gestureFrequencyMultiplier: 1.0,
    expressionDuration: 1.0,
    characteristics: {
      balanced: true,
      natural: true,
      professional: true,
    },
    baselineShift: {
      smile: 0,
      eyebrowRaise: 0,
    },
    preferredGestures: [], // All gestures appropriate
    avoidGestures: [],
  },

  '36-60': {
    id: 'mature',
    label: 'Mature Adult',
    expressionIntensityMultiplier: 0.85,
    movementAmplitude: 0.9,
    gestureFrequencyMultiplier: 0.85,
    expressionDuration: 1.15,
    characteristics: {
      subtle: true,
      confident: true,
      measured: true,
    },
    baselineShift: {
      smile: 5,
      eyebrowRaise: -5,
    },
    preferredGestures: ['hands_steepled', 'thoughtful_chin', 'nod_slow'],
    avoidGestures: ['bounce_hop', 'fist_pump'],
  },

  '61-102': {
    id: 'elder',
    label: 'Wisdom Years',
    expressionIntensityMultiplier: 0.75,
    movementAmplitude: 0.7,
    gestureFrequencyMultiplier: 0.7,
    expressionDuration: 1.3,
    characteristics: {
      gentle: true,
      wise: true,
      deliberate: true,
    },
    baselineShift: {
      smile: 10,
      eyebrowRaise: 5,
      warmGaze: 20,
    },
    preferredGestures: ['soft_nod', 'palm_up_offering', 'gentle_reach'],
    avoidGestures: ['fist_pump', 'bounce_hop', 'rock_on', 'provocative_point'],
  },
};

// =============================================================================
// LANGUAGE EXPRESSION PROFILES
// =============================================================================

/**
 * Language-based expression and gesture modifiers
 */
export const LANGUAGE_PROFILES = {
  en: {
    id: 'english',
    gestureIntensityMultiplier: 1.0,
    expressionMultiplier: 1.0,
    pauseFrequency: 'moderate',
    gestureStyle: 'clear',
    culturalGestures: ['thumbs_up', 'okay_sign', 'wave'],
    avoidGestures: [],
  },

  es: {
    id: 'spanish',
    gestureIntensityMultiplier: 1.3,
    expressionMultiplier: 1.15,
    pauseFrequency: 'moderate',
    gestureStyle: 'expressive',
    culturalGestures: ['hands_open_speaking', 'embrace_gesture', 'heart_touch'],
    avoidGestures: [],
    characteristics: {
      moreHandGestures: true,
      expressive: true,
      passionate: true,
    },
  },

  fr: {
    id: 'french',
    gestureIntensityMultiplier: 0.9,
    expressionMultiplier: 1.05,
    pauseFrequency: 'thoughtful',
    gestureStyle: 'elegant',
    culturalGestures: ['subtle_shrug', 'chin_touch', 'expressive_mouth'],
    avoidGestures: ['fist_pump'],
    characteristics: {
      subtleGestures: true,
      elegant: true,
      refined: true,
    },
  },
};

// =============================================================================
// PHASE EXPRESSION PROFILES
// =============================================================================

/**
 * Lesson phase-specific expression defaults
 */
export const PHASE_PROFILES = {
  welcome: {
    id: 'welcome',
    defaultMood: 'warm_greeting',
    energyLevel: 1.1,
    expressionSequence: ['warm', 'excited', 'inviting'],
    baseExpressions: {
      smile: 65,
      eyebrowRaise: 35,
      eyesWide: 25,
    },
    typicalGestures: ['open_arms_welcome', 'wave', 'hands_together_heart'],
    transitionTo: {
      q1: { buildUp: true, anticipation: 0.3 },
    },
  },

  q1: {
    id: 'teaching',
    alias: 'teaching',
    defaultMood: 'curious_exploration',
    energyLevel: 1.05,
    expressionSequence: ['curious', 'explaining', 'questioning'],
    baseExpressions: {
      smile: 45,
      eyebrowRaise: 50,
      eyesWide: 35,
    },
    typicalGestures: ['point_up_dramatic', 'chin_touch', 'hands_open_presenting'],
    transitionTo: {
      q2: { buildUp: true, anticipation: 0.2 },
    },
  },

  q2: {
    id: 'practice',
    alias: 'practice',
    defaultMood: 'engaged_challenge',
    energyLevel: 1.15,
    expressionSequence: ['encouraging', 'attentive', 'supportive'],
    baseExpressions: {
      smile: 55,
      eyebrowRaise: 40,
      nod: 30,
    },
    typicalGestures: ['encouraging_nod', 'balance_hands', 'think_gesture'],
    transitionTo: {
      q3: { buildUp: true, anticipation: 0.25 },
    },
  },

  q3: {
    id: 'synthesis',
    alias: 'practice',
    defaultMood: 'deep_engagement',
    energyLevel: 1.1,
    expressionSequence: ['thoughtful', 'insightful', 'affirming'],
    baseExpressions: {
      smile: 50,
      eyebrowRaise: 45,
      focusedGaze: 35,
    },
    typicalGestures: ['connect_points', 'building_blocks', 'light_bulb'],
    transitionTo: {
      wisdom: { buildUp: true, anticipation: 0.4 },
    },
  },

  wisdom: {
    id: 'wisdom',
    defaultMood: 'profound_conclusion',
    energyLevel: 0.95,
    expressionSequence: ['serene', 'profound', 'warm_closing'],
    baseExpressions: {
      smile: 60,
      eyebrowRaise: 30,
      warmGaze: 40,
    },
    typicalGestures: ['heart_open', 'palms_up_receiving', 'gentle_nod'],
    transitionTo: {},
  },
};

// =============================================================================
// TEXT ANALYSIS ENGINE
// =============================================================================

/**
 * Analyzes text content to detect emotional cues and expression triggers
 */
export class TextAnalyzer {
  constructor() {
    this.emotionPatterns = {
      excitement: {
        patterns: [/!+/g, /amazing/gi, /incredible/gi, /wow/gi, /fantastic/gi, /awesome/gi, /wonderful/gi, /mind[- ]?blow/gi],
        weight: 1.2,
        emotion: 'excited',
      },
      curiosity: {
        patterns: [/\?+/g, /wonder/gi, /curious/gi, /how/gi, /why/gi, /what if/gi, /imagine/gi],
        weight: 1.0,
        emotion: 'curious',
      },
      warmth: {
        patterns: [/love/gi, /heart/gi, /dear/gi, /special/gi, /precious/gi, /care/gi, /beautiful/gi],
        weight: 1.0,
        emotion: 'warm',
      },
      emphasis: {
        patterns: [/VERY/g, /REALLY/g, /SO /gi, /incredibly/gi, /absolutely/gi, /definitely/gi],
        weight: 1.15,
        emotion: 'emphatic',
      },
      humor: {
        patterns: [/haha/gi, /😄|😊|😁|🤣/g, /funny/gi, /silly/gi, /joke/gi, /laugh/gi],
        weight: 1.1,
        emotion: 'amused',
      },
      awe: {
        patterns: [/cosmic/gi, /universe/gi, /infinity/gi, /eternal/gi, /profound/gi, /vast/gi],
        weight: 1.0,
        emotion: 'awed',
      },
      encouragement: {
        patterns: [/you can/gi, /great job/gi, /well done/gi, /excellent/gi, /brilliant/gi, /keep/gi],
        weight: 1.05,
        emotion: 'encouraging',
      },
      challenge: {
        patterns: [/try/gi, /challenge/gi, /think about/gi, /consider/gi, /test/gi],
        weight: 0.95,
        emotion: 'challenging',
      },
    };

    this.pauseIndicators = [
      { pattern: /\.\.\./g, duration: 'long', intensity: 0.7 },
      { pattern: /[.!?]\s+/g, duration: 'medium', intensity: 0.5 },
      { pattern: /,\s+/g, duration: 'short', intensity: 0.3 },
      { pattern: /—|–/g, duration: 'dramatic', intensity: 0.8 },
    ];
  }

  /**
   * Analyze text and return emotion data with timestamps
   * @param {string} text - The text content to analyze
   * @param {number} totalDuration - Total audio duration in seconds
   * @returns {Object} Analysis results with emotions, pauses, and emphasis points
   */
  analyze(text, totalDuration = 60) {
    const sentences = this.splitIntoSentences(text);
    const avgSentenceDuration = totalDuration / Math.max(sentences.length, 1);
    
    const results = {
      emotions: [],
      pauses: [],
      emphasisPoints: [],
      overallMood: null,
      intensityProfile: [],
    };

    let currentTime = 0;
    const emotionScores = {};

    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i];
      const sentenceDuration = this.estimateSentenceDuration(sentence, avgSentenceDuration);
      
      // Detect emotions in sentence
      const sentenceEmotions = this.detectEmotions(sentence);
      
      for (const emotion of sentenceEmotions) {
        results.emotions.push({
          timestamp: currentTime,
          emotion: emotion.type,
          intensity: emotion.intensity,
          duration: sentenceDuration * 0.8,
          trigger: emotion.trigger,
        });
        
        // Track overall emotion scores
        emotionScores[emotion.type] = (emotionScores[emotion.type] || 0) + emotion.intensity;
      }

      // Detect pauses
      const pauses = this.detectPauses(sentence, currentTime, sentenceDuration);
      results.pauses.push(...pauses);

      // Detect emphasis
      const emphasis = this.detectEmphasis(sentence, currentTime, sentenceDuration);
      results.emphasisPoints.push(...emphasis);

      // Build intensity profile
      results.intensityProfile.push({
        timestamp: currentTime,
        intensity: this.calculateIntensity(sentenceEmotions),
        duration: sentenceDuration,
      });

      currentTime += sentenceDuration;
    }

    // Determine overall mood
    results.overallMood = this.determineOverallMood(emotionScores);

    return results;
  }

  splitIntoSentences(text) {
    return text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 0);
  }

  estimateSentenceDuration(sentence, avgDuration) {
    const wordCount = sentence.split(/\s+/).length;
    const avgWordsPerSentence = 15;
    const ratio = wordCount / avgWordsPerSentence;
    return avgDuration * Math.max(0.5, Math.min(2.0, ratio));
  }

  detectEmotions(sentence) {
    const emotions = [];
    
    for (const [name, config] of Object.entries(this.emotionPatterns)) {
      for (const pattern of config.patterns) {
        const matches = sentence.match(pattern);
        if (matches && matches.length > 0) {
          emotions.push({
            type: config.emotion,
            intensity: Math.min(1.0, 0.6 + (matches.length * 0.15)) * config.weight,
            trigger: matches[0],
          });
        }
      }
    }

    // Default emotion if none detected
    if (emotions.length === 0) {
      emotions.push({
        type: 'neutral',
        intensity: 0.5,
        trigger: null,
      });
    }

    return emotions;
  }

  detectPauses(sentence, startTime, duration) {
    const pauses = [];
    
    for (const indicator of this.pauseIndicators) {
      const matches = [...sentence.matchAll(indicator.pattern)];
      for (const match of matches) {
        const relativePosition = match.index / sentence.length;
        pauses.push({
          timestamp: startTime + (duration * relativePosition),
          type: indicator.duration,
          intensity: indicator.intensity,
        });
      }
    }

    return pauses;
  }

  detectEmphasis(sentence, startTime, duration) {
    const emphasisPoints = [];
    const emphasisPatterns = [
      { pattern: /[A-Z]{2,}/g, type: 'strong', intensity: 0.9 },
      { pattern: /\*\*[^*]+\*\*/g, type: 'bold', intensity: 0.8 },
      { pattern: /_[^_]+_/g, type: 'italic', intensity: 0.6 },
    ];

    for (const ep of emphasisPatterns) {
      const matches = [...sentence.matchAll(ep.pattern)];
      for (const match of matches) {
        const relativePosition = match.index / sentence.length;
        emphasisPoints.push({
          timestamp: startTime + (duration * relativePosition),
          type: ep.type,
          intensity: ep.intensity,
          text: match[0],
        });
      }
    }

    return emphasisPoints;
  }

  calculateIntensity(emotions) {
    if (emotions.length === 0) return 0.5;
    const sum = emotions.reduce((acc, e) => acc + e.intensity, 0);
    return Math.min(1.0, sum / emotions.length);
  }

  determineOverallMood(emotionScores) {
    let maxScore = 0;
    let dominantMood = 'neutral';
    
    for (const [emotion, score] of Object.entries(emotionScores)) {
      if (score > maxScore) {
        maxScore = score;
        dominantMood = emotion;
      }
    }
    
    return dominantMood;
  }
}

// =============================================================================
// ELEVENLABS METADATA PROCESSOR
// =============================================================================

/**
 * Process ElevenLabs API response metadata for expression timing
 */
export class ElevenLabsMetadataProcessor {
  /**
   * Parse ElevenLabs audio generation response
   * @param {Object} response - ElevenLabs API response
   * @returns {Object} Processed timing and emphasis data
   */
  process(response) {
    const result = {
      duration: 0,
      wordTimings: [],
      emphasisMarkers: [],
      pauseMarkers: [],
      pitchVariations: [],
    };

    // Handle different response formats
    if (response.alignment) {
      result.wordTimings = this.parseAlignment(response.alignment);
      result.duration = this.calculateDuration(result.wordTimings);
    }

    if (response.normalized_alignment) {
      result.emphasisMarkers = this.extractEmphasis(response.normalized_alignment);
    }

    // Extract from character timings if available
    if (response.characters) {
      const charData = this.parseCharacterTimings(response.characters);
      result.pauseMarkers = charData.pauses;
      result.pitchVariations = charData.pitchVariations;
    }

    // Estimate if no timing data available
    if (result.wordTimings.length === 0 && response.text) {
      result.wordTimings = this.estimateWordTimings(response.text, response.duration || 60);
      result.duration = response.duration || 60;
    }

    return result;
  }

  parseAlignment(alignment) {
    const timings = [];
    
    if (Array.isArray(alignment)) {
      for (const item of alignment) {
        timings.push({
          word: item.word || item.text,
          start: item.start_time || item.start,
          end: item.end_time || item.end,
          confidence: item.confidence || 1.0,
        });
      }
    }

    return timings;
  }

  extractEmphasis(normalizedAlignment) {
    const emphasisMarkers = [];
    
    if (!Array.isArray(normalizedAlignment)) return emphasisMarkers;

    for (const item of normalizedAlignment) {
      // Detect emphasis from various indicators
      if (item.emphasis || item.stress || (item.pitch && item.pitch > 1.1)) {
        emphasisMarkers.push({
          timestamp: item.start_time || item.start,
          intensity: item.emphasis || item.stress || 0.8,
          word: item.word || item.text,
        });
      }
    }

    return emphasisMarkers;
  }

  parseCharacterTimings(characters) {
    const result = {
      pauses: [],
      pitchVariations: [],
    };

    let lastEnd = 0;
    
    for (const char of characters) {
      const gap = (char.start_time || char.start) - lastEnd;
      
      // Detect pauses (gaps > 200ms)
      if (gap > 0.2) {
        result.pauses.push({
          timestamp: lastEnd,
          duration: gap,
          type: gap > 0.5 ? 'long' : 'short',
        });
      }

      // Track pitch variations
      if (char.pitch) {
        result.pitchVariations.push({
          timestamp: char.start_time || char.start,
          pitch: char.pitch,
        });
      }

      lastEnd = char.end_time || char.end || lastEnd + 0.05;
    }

    return result;
  }

  estimateWordTimings(text, totalDuration) {
    const words = text.split(/\s+/);
    const avgWordDuration = totalDuration / words.length;
    const timings = [];
    let currentTime = 0;

    for (const word of words) {
      const wordDuration = avgWordDuration * (word.length / 5); // Adjust by word length
      timings.push({
        word: word,
        start: currentTime,
        end: currentTime + wordDuration,
        confidence: 0.7, // Lower confidence for estimates
      });
      currentTime += wordDuration;
    }

    return timings;
  }

  calculateDuration(wordTimings) {
    if (wordTimings.length === 0) return 0;
    return wordTimings[wordTimings.length - 1].end;
  }
}

// =============================================================================
// MAIN EXPRESSION GENERATOR
// =============================================================================

/**
 * Main AI-powered expression generation system
 */
export class ExpressionGenerator {
  constructor(options = {}) {
    this.textAnalyzer = new TextAnalyzer();
    this.metadataProcessor = new ElevenLabsMetadataProcessor();
    this.options = {
      transitionDuration: 0.3,
      minExpressionDuration: 0.5,
      maxExpressionDuration: 5.0,
      gestureOverlapAllowed: false,
      smoothTransitions: true,
      ...options,
    };
  }

  /**
   * Generate complete expression data for a lesson segment
   * @param {Object} params - Generation parameters
   * @returns {Object} Expression and gesture data with timestamps
   */
  generate(params) {
    const {
      text,
      elevenLabsResponse = null,
      archetype = 'The Scientist',
      tone = 'enthusiastic',
      ageBucket = '18-35',
      language = 'en',
      phase = 'welcome',
      totalDuration = null,
    } = params;

    // Get profiles
    const archetypeProfile = ARCHETYPE_PROFILES[archetype] || ARCHETYPE_PROFILES['The Scientist'];
    const toneModifier = TONE_MODIFIERS[tone] || TONE_MODIFIERS.warm;
    const ageProfile = AGE_PROFILES[ageBucket] || AGE_PROFILES['18-35'];
    const languageProfile = LANGUAGE_PROFILES[language] || LANGUAGE_PROFILES.en;
    const phaseProfile = PHASE_PROFILES[phase] || PHASE_PROFILES.welcome;

    // Process ElevenLabs metadata
    let audioMetadata = {
      duration: totalDuration || 60,
      wordTimings: [],
      emphasisMarkers: [],
      pauseMarkers: [],
    };

    if (elevenLabsResponse) {
      audioMetadata = this.metadataProcessor.process(elevenLabsResponse);
    }

    // Analyze text content
    const textAnalysis = this.textAnalyzer.analyze(text, audioMetadata.duration);

    // Generate expressions
    const expressions = this.generateExpressions({
      textAnalysis,
      audioMetadata,
      archetypeProfile,
      toneModifier,
      ageProfile,
      languageProfile,
      phaseProfile,
    });

    // Generate gestures
    const gestures = this.generateGestures({
      textAnalysis,
      audioMetadata,
      archetypeProfile,
      toneModifier,
      ageProfile,
      languageProfile,
      phaseProfile,
    });

    // Generate blend shapes timeline
    const blendShapeTimeline = this.generateBlendShapeTimeline(expressions, ageProfile);

    return {
      metadata: {
        archetype,
        tone,
        ageBucket,
        language,
        phase,
        totalDuration: audioMetadata.duration,
        generatedAt: new Date().toISOString(),
        version: '1.0.0',
      },
      expressions,
      gestures,
      blendShapeTimeline,
      textAnalysis: {
        overallMood: textAnalysis.overallMood,
        emotionCount: textAnalysis.emotions.length,
        pauseCount: textAnalysis.pauses.length,
      },
    };
  }

  /**
   * Generate expression keyframes
   */
  generateExpressions(params) {
    const {
      textAnalysis,
      audioMetadata,
      archetypeProfile,
      toneModifier,
      ageProfile,
      languageProfile,
      phaseProfile,
    } = params;

    const expressions = [];
    
    // Start with phase-appropriate baseline
    expressions.push({
      timestamp: 0,
      emotion: phaseProfile.expressionSequence[0],
      intensity: 0.6 * ageProfile.expressionIntensityMultiplier,
      blendShapes: this.calculateBlendShapes(
        phaseProfile.baseExpressions,
        archetypeProfile,
        toneModifier,
        ageProfile
      ),
      transitionDuration: this.options.transitionDuration,
    });

    // Process emotions from text analysis
    for (const emotion of textAnalysis.emotions) {
      const baseExpression = archetypeProfile.defaultExpressions[emotion.emotion] ||
                            archetypeProfile.defaultExpressions.neutral;
      
      const adjustedIntensity = emotion.intensity *
        toneModifier.multiplier *
        ageProfile.expressionIntensityMultiplier *
        languageProfile.expressionMultiplier;

      expressions.push({
        timestamp: emotion.timestamp,
        emotion: emotion.emotion,
        intensity: Math.min(1.0, adjustedIntensity),
        blendShapes: this.calculateBlendShapes(
          baseExpression,
          archetypeProfile,
          toneModifier,
          ageProfile
        ),
        trigger: emotion.trigger,
        transitionDuration: this.options.transitionDuration * ageProfile.expressionDuration,
      });
    }

    // Add emphasis-triggered expressions from ElevenLabs data
    for (const emphasis of audioMetadata.emphasisMarkers) {
      // Find if there's already an expression near this timestamp
      const nearbyExpression = expressions.find(
        e => Math.abs(e.timestamp - emphasis.timestamp) < 0.3
      );

      if (nearbyExpression) {
        // Boost existing expression
        nearbyExpression.intensity = Math.min(1.0, nearbyExpression.intensity * 1.2);
      } else {
        // Add emphatic expression
        expressions.push({
          timestamp: emphasis.timestamp,
          emotion: 'emphatic',
          intensity: emphasis.intensity * ageProfile.expressionIntensityMultiplier,
          blendShapes: this.calculateBlendShapes(
            { eyebrowRaise: 60, eyesWide: 40 },
            archetypeProfile,
            toneModifier,
            ageProfile
          ),
          trigger: emphasis.word,
          transitionDuration: this.options.transitionDuration * 0.5,
        });
      }
    }

    // Sort by timestamp and remove duplicates
    return this.cleanupExpressions(expressions);
  }

  /**
   * Generate gesture keyframes
   */
  generateGestures(params) {
    const {
      textAnalysis,
      audioMetadata,
      archetypeProfile,
      toneModifier,
      ageProfile,
      languageProfile,
      phaseProfile,
    } = params;

    const gestures = [];
    const gestureLibrary = archetypeProfile.gestureLibrary;
    
    // Calculate adjusted gesture frequency
    const baseGestureInterval = 5.0; // seconds between gestures
    const adjustedInterval = baseGestureInterval /
      (toneModifier.gestureFrequencyMultiplier *
       ageProfile.gestureFrequencyMultiplier *
       languageProfile.gestureIntensityMultiplier);

    // Add phase-typical opening gesture
    if (phaseProfile.typicalGestures.length > 0) {
      const openingGesture = this.selectGesture(
        phaseProfile.typicalGestures[0],
        gestureLibrary,
        ageProfile
      );
      if (openingGesture) {
        gestures.push({
          timestamp: 0.5,
          gesture: openingGesture.name,
          duration: this.selectGestureDuration(openingGesture, ageProfile),
          intensity: archetypeProfile.gestureIntensity * ageProfile.movementAmplitude,
          context: 'phase_opening',
        });
      }
    }

    // Generate gestures at pause points
    for (const pause of textAnalysis.pauses) {
      if (pause.type === 'long' || pause.type === 'dramatic') {
        const contextGesture = this.selectContextualGesture(
          gestureLibrary,
          'thinking',
          ageProfile,
          languageProfile
        );
        
        if (contextGesture && !this.hasGestureConflict(gestures, pause.timestamp)) {
          gestures.push({
            timestamp: pause.timestamp,
            gesture: contextGesture.name,
            duration: this.selectGestureDuration(contextGesture, ageProfile),
            intensity: archetypeProfile.gestureIntensity * ageProfile.movementAmplitude * 0.8,
            context: 'pause',
          });
        }
      }
    }

    // Generate gestures at emphasis points
    for (const emphasis of audioMetadata.emphasisMarkers) {
      const contextGesture = this.selectContextualGesture(
        gestureLibrary,
        'emphasis',
        ageProfile,
        languageProfile
      );
      
      if (contextGesture && !this.hasGestureConflict(gestures, emphasis.timestamp)) {
        gestures.push({
          timestamp: emphasis.timestamp,
          gesture: contextGesture.name,
          duration: this.selectGestureDuration(contextGesture, ageProfile),
          intensity: archetypeProfile.gestureIntensity * ageProfile.movementAmplitude * emphasis.intensity,
          context: 'emphasis',
        });
      }
    }

    // Fill gaps with periodic gestures
    let currentTime = 2.0;
    const totalDuration = audioMetadata.duration;
    
    while (currentTime < totalDuration - 2.0) {
      if (!this.hasGestureConflict(gestures, currentTime)) {
        const randomGesture = this.selectRandomGesture(gestureLibrary, ageProfile, languageProfile);
        if (randomGesture) {
          gestures.push({
            timestamp: currentTime,
            gesture: randomGesture.name,
            duration: this.selectGestureDuration(randomGesture, ageProfile),
            intensity: archetypeProfile.gestureIntensity * ageProfile.movementAmplitude,
            context: 'periodic',
          });
        }
      }
      currentTime += adjustedInterval * (0.8 + Math.random() * 0.4);
    }

    // Sort and clean up
    return this.cleanupGestures(gestures);
  }

  /**
   * Calculate blend shapes for an expression
   */
  calculateBlendShapes(baseExpression, archetypeProfile, toneModifier, ageProfile) {
    const blendShapes = {};
    const multiplier = ageProfile.expressionIntensityMultiplier;
    
    // Apply base expression values
    for (const [shape, value] of Object.entries(baseExpression)) {
      if (typeof value === 'number') {
        let adjustedValue = value * multiplier;
        
        // Apply tone baseline shift
        if (toneModifier.baselineShift[shape]) {
          adjustedValue += toneModifier.baselineShift[shape];
        }
        
        // Apply age baseline shift
        if (ageProfile.baselineShift[shape]) {
          adjustedValue += ageProfile.baselineShift[shape];
        }
        
        blendShapes[shape] = Math.max(0, Math.min(100, adjustedValue));
      }
    }

    // Ensure minimum expression values
    if (!blendShapes.smile) blendShapes.smile = 0;
    if (!blendShapes.eyebrowRaise) blendShapes.eyebrowRaise = 0;

    return blendShapes;
  }

  /**
   * Generate complete blend shape timeline
   */
  generateBlendShapeTimeline(expressions, ageProfile) {
    const timeline = [];
    
    for (let i = 0; i < expressions.length; i++) {
      const current = expressions[i];
      const next = expressions[i + 1];
      
      timeline.push({
        timestamp: current.timestamp,
        blendShapes: current.blendShapes,
        transitionDuration: current.transitionDuration,
        easing: this.selectEasing(ageProfile),
      });
      
      // Add interpolation keyframes for smooth transitions
      if (next && this.options.smoothTransitions) {
        const midTime = (current.timestamp + next.timestamp) / 2;
        const midBlendShapes = this.interpolateBlendShapes(
          current.blendShapes,
          next.blendShapes,
          0.5
        );
        
        timeline.push({
          timestamp: midTime,
          blendShapes: midBlendShapes,
          transitionDuration: current.transitionDuration,
          easing: 'ease-in-out',
          isInterpolated: true,
        });
      }
    }

    return timeline;
  }

  /**
   * Select appropriate easing function based on age profile
   */
  selectEasing(ageProfile) {
    if (ageProfile.characteristics?.bouncy) return 'ease-out-bounce';
    if (ageProfile.characteristics?.gentle) return 'ease-in-out';
    if (ageProfile.characteristics?.cool) return 'ease-out';
    return 'ease-in-out';
  }

  /**
   * Interpolate between two blend shape states
   */
  interpolateBlendShapes(from, to, t) {
    const result = {};
    const allKeys = new Set([...Object.keys(from), ...Object.keys(to)]);
    
    for (const key of allKeys) {
      const fromValue = from[key] || 0;
      const toValue = to[key] || 0;
      result[key] = fromValue + (toValue - fromValue) * t;
    }
    
    return result;
  }

  // Helper methods
  selectGesture(gestureName, library, ageProfile) {
    // Check if gesture is avoided for this age
    if (ageProfile.avoidGestures.includes(gestureName)) return null;
    
    return library.find(g => g.name === gestureName) ||
           library.find(g => g.contexts.includes('any'));
  }

  selectContextualGesture(library, context, ageProfile, languageProfile) {
    const validGestures = library.filter(g => {
      if (ageProfile.avoidGestures.includes(g.name)) return false;
      return g.contexts.includes(context) || g.contexts.includes('any');
    });
    
    if (validGestures.length === 0) return null;
    
    // Prefer age-appropriate gestures
    const preferred = validGestures.filter(g => 
      ageProfile.preferredGestures.includes(g.name)
    );
    
    if (preferred.length > 0) {
      return preferred[Math.floor(Math.random() * preferred.length)];
    }
    
    return validGestures[Math.floor(Math.random() * validGestures.length)];
  }

  selectRandomGesture(library, ageProfile, languageProfile) {
    const validGestures = library.filter(g => 
      !ageProfile.avoidGestures.includes(g.name) &&
      !languageProfile.avoidGestures?.includes(g.name)
    );
    
    if (validGestures.length === 0) return null;
    
    return validGestures[Math.floor(Math.random() * validGestures.length)];
  }

  selectGestureDuration(gesture, ageProfile) {
    const [min, max] = gesture.duration;
    const baseDuration = min + Math.random() * (max - min);
    return baseDuration * ageProfile.expressionDuration;
  }

  hasGestureConflict(gestures, timestamp) {
    if (this.options.gestureOverlapAllowed) return false;
    
    for (const gesture of gestures) {
      const gestureEnd = gesture.timestamp + gesture.duration;
      if (timestamp >= gesture.timestamp && timestamp <= gestureEnd) {
        return true;
      }
      if (Math.abs(gesture.timestamp - timestamp) < 1.0) {
        return true;
      }
    }
    return false;
  }

  cleanupExpressions(expressions) {
    // Sort by timestamp
    expressions.sort((a, b) => a.timestamp - b.timestamp);
    
    // Remove expressions too close together
    const cleaned = [];
    for (const expr of expressions) {
      const lastExpr = cleaned[cleaned.length - 1];
      if (!lastExpr || expr.timestamp - lastExpr.timestamp >= this.options.minExpressionDuration) {
        cleaned.push(expr);
      } else if (expr.intensity > lastExpr.intensity) {
        // Replace with higher intensity
        cleaned[cleaned.length - 1] = expr;
      }
    }
    
    return cleaned;
  }

  cleanupGestures(gestures) {
    // Sort by timestamp
    gestures.sort((a, b) => a.timestamp - b.timestamp);
    
    // Remove overlapping gestures
    const cleaned = [];
    for (const gesture of gestures) {
      if (!this.hasGestureConflict(cleaned, gesture.timestamp)) {
        cleaned.push(gesture);
      }
    }
    
    return cleaned;
  }
}

// =============================================================================
// BATCH GENERATOR FOR PRE-COMPUTATION
// =============================================================================

/**
 * Batch expression generator for pre-computing lesson expressions
 */
export class BatchExpressionGenerator {
  constructor(options = {}) {
    this.generator = new ExpressionGenerator(options);
    this.cache = new Map();
  }

  /**
   * Generate expressions for all variants of a lesson
   * @param {Object} lessonDNA - Complete lesson DNA with age variants
   * @param {string} archetype - Target archetype
   * @returns {Object} Expression data for all phases and age variants
   */
  generateForLesson(lessonDNA, archetype = 'The Scientist') {
    const results = {
      lessonId: lessonDNA.id,
      title: lessonDNA.title,
      archetype,
      generatedAt: new Date().toISOString(),
      variants: {},
    };

    const ageBuckets = Object.keys(lessonDNA.ageVariants || {});
    
    for (const ageBucket of ageBuckets) {
      const variant = lessonDNA.ageVariants[ageBucket];
      results.variants[ageBucket] = this.generateForVariant(variant, archetype, ageBucket);
    }

    return results;
  }

  /**
   * Generate expressions for a single age variant
   */
  generateForVariant(variant, archetype, ageBucket) {
    const phases = ['welcome', 'q1', 'q2', 'q3', 'wisdom'];
    const variantResults = {};
    
    // Determine tone from variant data
    const tone = this.extractTone(variant.tone);
    const language = variant.language?.en ? 'en' : Object.keys(variant.language || {})[0] || 'en';
    
    for (const phase of phases) {
      const text = this.extractPhaseText(variant, phase);
      if (text) {
        variantResults[phase] = this.generator.generate({
          text,
          archetype,
          tone,
          ageBucket,
          language,
          phase,
        });
      }
    }

    return variantResults;
  }

  /**
   * Extract tone identifier from lesson variant
   */
  extractTone(toneData) {
    if (!toneData) return 'warm';
    
    const emotionalTemp = toneData.emotional_temperature || '';
    
    if (emotionalTemp.includes('high_energy') || emotionalTemp.includes('celebratory')) {
      return 'enthusiastic';
    }
    if (emotionalTemp.includes('calm') || emotionalTemp.includes('precise')) {
      return 'thoughtful';
    }
    if (emotionalTemp.includes('warm') || emotionalTemp.includes('nurturing')) {
      return 'warm';
    }
    
    return 'warm';
  }

  /**
   * Extract text content for a specific phase
   */
  extractPhaseText(variant, phase) {
    const langData = variant.language?.en || {};
    
    switch (phase) {
      case 'welcome':
        return langData.welcome || variant.title;
      case 'q1':
      case 'q2':
      case 'q3':
        return langData.mainContent || variant.description;
      case 'wisdom':
        return langData.wisdomMoment || langData.summary || '';
      default:
        return '';
    }
  }

  /**
   * Generate expressions for all 12 archetypes
   */
  generateAllArchetypes(lessonDNA) {
    const results = {
      lessonId: lessonDNA.id,
      title: lessonDNA.title,
      generatedAt: new Date().toISOString(),
      archetypes: {},
    };

    for (const archetypeName of Object.keys(ARCHETYPE_PROFILES)) {
      results.archetypes[archetypeName] = this.generateForLesson(lessonDNA, archetypeName);
    }

    return results;
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export default ExpressionGenerator;











