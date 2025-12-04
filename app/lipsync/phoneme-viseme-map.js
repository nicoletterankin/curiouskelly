/**
 * Phoneme to Viseme Mapping System for Kelly Lip-Sync
 * 
 * Maps ARPAbet phonemes (from forced alignment) to ARKit-compatible
 * blendshape values for Kelly's 53 facial blendshapes.
 * 
 * @module phoneme-viseme-map
 */

// =============================================================================
// ARPAbet PHONEME SET (CMU Dictionary Standard)
// =============================================================================

/**
 * Complete ARPAbet phoneme inventory
 * Used by Montreal Forced Aligner, CMU Sphinx, etc.
 */
export const ARPABET_PHONEMES = {
  // Vowels (monophthongs)
  'AA': { type: 'vowel', example: 'father', ipa: 'ɑ' },
  'AE': { type: 'vowel', example: 'cat', ipa: 'æ' },
  'AH': { type: 'vowel', example: 'but', ipa: 'ʌ' },
  'AO': { type: 'vowel', example: 'thought', ipa: 'ɔ' },
  'EH': { type: 'vowel', example: 'bed', ipa: 'ɛ' },
  'ER': { type: 'vowel', example: 'bird', ipa: 'ɝ' },
  'IH': { type: 'vowel', example: 'bit', ipa: 'ɪ' },
  'IY': { type: 'vowel', example: 'beat', ipa: 'i' },
  'UH': { type: 'vowel', example: 'book', ipa: 'ʊ' },
  'UW': { type: 'vowel', example: 'boot', ipa: 'u' },
  
  // Vowels (diphthongs)
  'AW': { type: 'diphthong', example: 'cow', ipa: 'aʊ' },
  'AY': { type: 'diphthong', example: 'my', ipa: 'aɪ' },
  'EY': { type: 'diphthong', example: 'say', ipa: 'eɪ' },
  'OW': { type: 'diphthong', example: 'go', ipa: 'oʊ' },
  'OY': { type: 'diphthong', example: 'boy', ipa: 'ɔɪ' },
  
  // Stops
  'P': { type: 'stop', example: 'pat', ipa: 'p', voiced: false },
  'B': { type: 'stop', example: 'bat', ipa: 'b', voiced: true },
  'T': { type: 'stop', example: 'top', ipa: 't', voiced: false },
  'D': { type: 'stop', example: 'dog', ipa: 'd', voiced: true },
  'K': { type: 'stop', example: 'cat', ipa: 'k', voiced: false },
  'G': { type: 'stop', example: 'go', ipa: 'g', voiced: true },
  
  // Fricatives
  'F': { type: 'fricative', example: 'fun', ipa: 'f', voiced: false },
  'V': { type: 'fricative', example: 'van', ipa: 'v', voiced: true },
  'TH': { type: 'fricative', example: 'think', ipa: 'θ', voiced: false },
  'DH': { type: 'fricative', example: 'this', ipa: 'ð', voiced: true },
  'S': { type: 'fricative', example: 'sit', ipa: 's', voiced: false },
  'Z': { type: 'fricative', example: 'zoo', ipa: 'z', voiced: true },
  'SH': { type: 'fricative', example: 'ship', ipa: 'ʃ', voiced: false },
  'ZH': { type: 'fricative', example: 'measure', ipa: 'ʒ', voiced: true },
  'HH': { type: 'fricative', example: 'hat', ipa: 'h', voiced: false },
  
  // Affricates
  'CH': { type: 'affricate', example: 'church', ipa: 'tʃ', voiced: false },
  'JH': { type: 'affricate', example: 'judge', ipa: 'dʒ', voiced: true },
  
  // Nasals
  'M': { type: 'nasal', example: 'mom', ipa: 'm' },
  'N': { type: 'nasal', example: 'no', ipa: 'n' },
  'NG': { type: 'nasal', example: 'sing', ipa: 'ŋ' },
  
  // Liquids
  'L': { type: 'liquid', example: 'let', ipa: 'l' },
  'R': { type: 'liquid', example: 'red', ipa: 'ɹ' },
  
  // Glides/Semivowels
  'W': { type: 'glide', example: 'wet', ipa: 'w' },
  'Y': { type: 'glide', example: 'yes', ipa: 'j' },
  
  // Silence
  'SIL': { type: 'silence', example: '', ipa: '' },
  'SP': { type: 'silence', example: '', ipa: '' },
  'spn': { type: 'silence', example: '', ipa: '' },
};

// =============================================================================
// VISEME CATEGORIES (Preston Blair Standard + ARKit Extensions)
// =============================================================================

/**
 * Standard viseme categories used in animation
 * Based on Preston Blair's simplified phoneme groups
 */
export const VISEME_CATEGORIES = {
  // A: Wide open mouth (AA, AE, AH)
  'A': {
    name: 'Wide Open',
    description: 'Jaw dropped, mouth wide open',
    phonemes: ['AA', 'AE', 'AH'],
  },
  
  // E: Teeth showing, slight smile (EH, EY, IH, IY)
  'E': {
    name: 'Teeth Smile',
    description: 'Lips stretched, teeth visible',
    phonemes: ['EH', 'EY', 'IH', 'IY'],
  },
  
  // I: Narrow mouth (same as E but tighter)
  'I': {
    name: 'Narrow',
    description: 'Lips narrowed, minimal opening',
    phonemes: ['IH', 'IY'],
  },
  
  // O: Rounded lips, medium open (AO, OW, UH)
  'O': {
    name: 'Round Open',
    description: 'Lips rounded, jaw moderately open',
    phonemes: ['AO', 'OW', 'OY'],
  },
  
  // U: Pursed lips, small opening (UW, W)
  'U': {
    name: 'Pursed',
    description: 'Lips pursed forward, small opening',
    phonemes: ['UW', 'UH', 'W'],
  },
  
  // C/D/G/K/N/S/TH/Y/Z: Teeth together or nearly closed
  'C': {
    name: 'Teeth Together',
    description: 'Teeth nearly closed, various tongue positions',
    phonemes: ['S', 'Z', 'T', 'D', 'N', 'L', 'TH', 'DH'],
  },
  
  // F/V: Bottom lip under top teeth
  'F': {
    name: 'Lip Bite',
    description: 'Lower lip under upper teeth',
    phonemes: ['F', 'V'],
  },
  
  // L: Tongue up, mouth slightly open
  'L': {
    name: 'Tongue Up',
    description: 'Tongue touches upper palate',
    phonemes: ['L'],
  },
  
  // M/B/P: Lips pressed together
  'M': {
    name: 'Lips Closed',
    description: 'Lips pressed together',
    phonemes: ['M', 'B', 'P'],
  },
  
  // R: Rounded, slightly pursed
  'R': {
    name: 'R Sound',
    description: 'Lips slightly rounded, tongue back',
    phonemes: ['R', 'ER'],
  },
  
  // SH/CH/JH: Lips pushed forward
  'SH': {
    name: 'Pushed Forward',
    description: 'Lips pushed forward, rounded',
    phonemes: ['SH', 'ZH', 'CH', 'JH'],
  },
  
  // Rest: Neutral closed position
  'REST': {
    name: 'Rest',
    description: 'Neutral, relaxed mouth',
    phonemes: ['SIL', 'SP', 'spn'],
  },
};

// =============================================================================
// PHONEME TO BLENDSHAPE MAPPING
// =============================================================================

/**
 * Maps each ARPAbet phoneme to ARKit-compatible blendshape values
 * Values are 0-100 representing blend weight percentage
 * 
 * Blendshapes based on Apple ARKit Face Tracking standard:
 * - jawOpen: How far the jaw drops
 * - mouthClose: Lips pressed together
 * - mouthFunnel: Lips pushed forward into O shape
 * - mouthPucker: Lips pursed (kiss shape)
 * - mouthLeft/Right: Mouth shifted to side
 * - mouthSmileLeft/Right: Corner of mouth raised
 * - mouthFrownLeft/Right: Corner of mouth lowered
 * - mouthDimpleLeft/Right: Dimple formed
 * - mouthStretchLeft/Right: Mouth corners stretched
 * - mouthRollLower/Upper: Lips rolled inward
 * - mouthShrugLower/Upper: Lips rolled outward
 * - mouthPressLeft/Right: Lips pressed to side
 * - mouthLowerDownLeft/Right: Lower lip pulled down
 * - mouthUpperUpLeft/Right: Upper lip pulled up
 */
export const PHONEME_TO_BLENDSHAPES = {
  // ═══════════════════════════════════════════════════════════════════════════
  // VOWELS - Wide, open mouth shapes
  // ═══════════════════════════════════════════════════════════════════════════
  
  'AA': {
    // "father" - wide open, jaw dropped
    jawOpen: 85,
    mouthOpen: 80,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 10,
    mouthStretchRight: 10,
    tongueOut: 0,
    visemeCategory: 'A',
    duration: { min: 80, typical: 120, max: 200 },
  },
  
  'AE': {
    // "cat" - open with slight stretch
    jawOpen: 65,
    mouthOpen: 60,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 25,
    mouthStretchRight: 25,
    mouthSmileLeft: 15,
    mouthSmileRight: 15,
    visemeCategory: 'A',
    duration: { min: 80, typical: 110, max: 180 },
  },
  
  'AH': {
    // "but" - medium open, relaxed
    jawOpen: 50,
    mouthOpen: 45,
    mouthFunnel: 10,
    mouthPucker: 0,
    mouthStretchLeft: 5,
    mouthStretchRight: 5,
    visemeCategory: 'A',
    duration: { min: 60, typical: 100, max: 160 },
  },
  
  'AO': {
    // "thought" - rounded, medium open
    jawOpen: 60,
    mouthOpen: 55,
    mouthFunnel: 40,
    mouthPucker: 20,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    visemeCategory: 'O',
    duration: { min: 80, typical: 130, max: 200 },
  },
  
  'EH': {
    // "bed" - medium open, slight smile
    jawOpen: 40,
    mouthOpen: 35,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 30,
    mouthStretchRight: 30,
    mouthSmileLeft: 20,
    mouthSmileRight: 20,
    visemeCategory: 'E',
    duration: { min: 60, typical: 100, max: 160 },
  },
  
  'ER': {
    // "bird" - rounded, tongue back
    jawOpen: 30,
    mouthOpen: 25,
    mouthFunnel: 35,
    mouthPucker: 25,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    visemeCategory: 'R',
    duration: { min: 100, typical: 150, max: 250 },
  },
  
  'IH': {
    // "bit" - narrow, teeth visible
    jawOpen: 20,
    mouthOpen: 15,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 40,
    mouthStretchRight: 40,
    mouthSmileLeft: 25,
    mouthSmileRight: 25,
    visemeCategory: 'I',
    duration: { min: 50, typical: 80, max: 130 },
  },
  
  'IY': {
    // "beat" - narrow smile, teeth showing
    jawOpen: 15,
    mouthOpen: 10,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 55,
    mouthStretchRight: 55,
    mouthSmileLeft: 35,
    mouthSmileRight: 35,
    visemeCategory: 'I',
    duration: { min: 80, typical: 120, max: 200 },
  },
  
  'UH': {
    // "book" - slightly rounded
    jawOpen: 25,
    mouthOpen: 20,
    mouthFunnel: 45,
    mouthPucker: 35,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    visemeCategory: 'U',
    duration: { min: 60, typical: 90, max: 150 },
  },
  
  'UW': {
    // "boot" - pursed, rounded
    jawOpen: 20,
    mouthOpen: 15,
    mouthFunnel: 60,
    mouthPucker: 70,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    visemeCategory: 'U',
    duration: { min: 80, typical: 130, max: 200 },
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // DIPHTHONGS - Transitional vowel sounds
  // ═══════════════════════════════════════════════════════════════════════════
  
  'AW': {
    // "cow" - starts open, transitions to rounded
    jawOpen: 55,
    mouthOpen: 50,
    mouthFunnel: 30,
    mouthPucker: 20,
    mouthStretchLeft: 10,
    mouthStretchRight: 10,
    visemeCategory: 'O',
    duration: { min: 120, typical: 180, max: 280 },
    transition: { to: 'UW', at: 0.6 },
  },
  
  'AY': {
    // "my" - open to narrow smile
    jawOpen: 60,
    mouthOpen: 55,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 20,
    mouthStretchRight: 20,
    mouthSmileLeft: 15,
    mouthSmileRight: 15,
    visemeCategory: 'A',
    duration: { min: 120, typical: 180, max: 280 },
    transition: { to: 'IY', at: 0.5 },
  },
  
  'EY': {
    // "say" - medium to narrow smile
    jawOpen: 35,
    mouthOpen: 30,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 40,
    mouthStretchRight: 40,
    mouthSmileLeft: 30,
    mouthSmileRight: 30,
    visemeCategory: 'E',
    duration: { min: 100, typical: 160, max: 250 },
    transition: { to: 'IY', at: 0.6 },
  },
  
  'OW': {
    // "go" - rounded, transitions tighter
    jawOpen: 45,
    mouthOpen: 40,
    mouthFunnel: 55,
    mouthPucker: 45,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    visemeCategory: 'O',
    duration: { min: 100, typical: 160, max: 250 },
    transition: { to: 'UW', at: 0.6 },
  },
  
  'OY': {
    // "boy" - rounded to smile
    jawOpen: 50,
    mouthOpen: 45,
    mouthFunnel: 40,
    mouthPucker: 25,
    mouthStretchLeft: 5,
    mouthStretchRight: 5,
    visemeCategory: 'O',
    duration: { min: 120, typical: 180, max: 280 },
    transition: { to: 'IY', at: 0.5 },
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // STOPS - Brief closures then release
  // ═══════════════════════════════════════════════════════════════════════════
  
  'P': {
    // "pat" - lips pressed together
    jawOpen: 0,
    mouthOpen: 0,
    mouthClose: 100,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthPressLeft: 50,
    mouthPressRight: 50,
    visemeCategory: 'M',
    duration: { min: 20, typical: 60, max: 100 },
  },
  
  'B': {
    // "bat" - lips pressed (voiced)
    jawOpen: 0,
    mouthOpen: 0,
    mouthClose: 100,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthPressLeft: 50,
    mouthPressRight: 50,
    visemeCategory: 'M',
    duration: { min: 30, typical: 70, max: 120 },
  },
  
  'T': {
    // "top" - tongue behind teeth
    jawOpen: 10,
    mouthOpen: 5,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 15,
    mouthStretchRight: 15,
    tongueOut: 0,
    visemeCategory: 'C',
    duration: { min: 20, typical: 50, max: 90 },
  },
  
  'D': {
    // "dog" - tongue behind teeth (voiced)
    jawOpen: 12,
    mouthOpen: 8,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 12,
    mouthStretchRight: 12,
    visemeCategory: 'C',
    duration: { min: 30, typical: 60, max: 100 },
  },
  
  'K': {
    // "cat" - back of tongue
    jawOpen: 15,
    mouthOpen: 10,
    mouthClose: 0,
    mouthFunnel: 5,
    mouthPucker: 0,
    mouthStretchLeft: 8,
    mouthStretchRight: 8,
    visemeCategory: 'C',
    duration: { min: 30, typical: 60, max: 100 },
  },
  
  'G': {
    // "go" - back of tongue (voiced)
    jawOpen: 18,
    mouthOpen: 12,
    mouthClose: 0,
    mouthFunnel: 8,
    mouthPucker: 0,
    mouthStretchLeft: 5,
    mouthStretchRight: 5,
    visemeCategory: 'C',
    duration: { min: 40, typical: 70, max: 120 },
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // FRICATIVES - Air flow through constriction
  // ═══════════════════════════════════════════════════════════════════════════
  
  'F': {
    // "fun" - bottom lip under top teeth
    jawOpen: 8,
    mouthOpen: 5,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthUpperUpLeft: 30,
    mouthUpperUpRight: 30,
    mouthLowerDownLeft: 20,
    mouthLowerDownRight: 20,
    visemeCategory: 'F',
    duration: { min: 60, typical: 100, max: 160 },
  },
  
  'V': {
    // "van" - bottom lip under top teeth (voiced)
    jawOpen: 10,
    mouthOpen: 6,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthUpperUpLeft: 28,
    mouthUpperUpRight: 28,
    mouthLowerDownLeft: 22,
    mouthLowerDownRight: 22,
    visemeCategory: 'F',
    duration: { min: 60, typical: 100, max: 160 },
  },
  
  'TH': {
    // "think" - tongue between teeth
    jawOpen: 12,
    mouthOpen: 8,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    tongueOut: 40,
    mouthStretchLeft: 15,
    mouthStretchRight: 15,
    visemeCategory: 'C',
    duration: { min: 60, typical: 100, max: 160 },
  },
  
  'DH': {
    // "this" - tongue between teeth (voiced)
    jawOpen: 14,
    mouthOpen: 10,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    tongueOut: 35,
    mouthStretchLeft: 12,
    mouthStretchRight: 12,
    visemeCategory: 'C',
    duration: { min: 50, typical: 90, max: 150 },
  },
  
  'S': {
    // "sit" - teeth close, air through
    jawOpen: 5,
    mouthOpen: 3,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 35,
    mouthStretchRight: 35,
    mouthSmileLeft: 15,
    mouthSmileRight: 15,
    visemeCategory: 'C',
    duration: { min: 60, typical: 110, max: 180 },
  },
  
  'Z': {
    // "zoo" - teeth close (voiced)
    jawOpen: 6,
    mouthOpen: 4,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 32,
    mouthStretchRight: 32,
    mouthSmileLeft: 12,
    mouthSmileRight: 12,
    visemeCategory: 'C',
    duration: { min: 60, typical: 100, max: 170 },
  },
  
  'SH': {
    // "ship" - lips pushed forward
    jawOpen: 12,
    mouthOpen: 8,
    mouthClose: 0,
    mouthFunnel: 50,
    mouthPucker: 40,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    visemeCategory: 'SH',
    duration: { min: 80, typical: 130, max: 200 },
  },
  
  'ZH': {
    // "measure" - lips forward (voiced)
    jawOpen: 14,
    mouthOpen: 10,
    mouthClose: 0,
    mouthFunnel: 45,
    mouthPucker: 35,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    visemeCategory: 'SH',
    duration: { min: 70, typical: 120, max: 190 },
  },
  
  'HH': {
    // "hat" - open, air flow
    jawOpen: 30,
    mouthOpen: 25,
    mouthClose: 0,
    mouthFunnel: 10,
    mouthPucker: 0,
    mouthStretchLeft: 10,
    mouthStretchRight: 10,
    visemeCategory: 'A',
    duration: { min: 40, typical: 80, max: 140 },
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // AFFRICATES - Stop followed by fricative
  // ═══════════════════════════════════════════════════════════════════════════
  
  'CH': {
    // "church" - lips pushed forward
    jawOpen: 10,
    mouthOpen: 6,
    mouthClose: 0,
    mouthFunnel: 55,
    mouthPucker: 45,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    visemeCategory: 'SH',
    duration: { min: 80, typical: 120, max: 180 },
  },
  
  'JH': {
    // "judge" - lips forward (voiced)
    jawOpen: 12,
    mouthOpen: 8,
    mouthClose: 0,
    mouthFunnel: 50,
    mouthPucker: 40,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    visemeCategory: 'SH',
    duration: { min: 80, typical: 120, max: 180 },
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // NASALS - Air through nose
  // ═══════════════════════════════════════════════════════════════════════════
  
  'M': {
    // "mom" - lips pressed together
    jawOpen: 0,
    mouthOpen: 0,
    mouthClose: 100,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthPressLeft: 40,
    mouthPressRight: 40,
    visemeCategory: 'M',
    duration: { min: 60, typical: 100, max: 160 },
  },
  
  'N': {
    // "no" - tongue behind teeth
    jawOpen: 8,
    mouthOpen: 5,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 20,
    mouthStretchRight: 20,
    visemeCategory: 'C',
    duration: { min: 50, typical: 90, max: 150 },
  },
  
  'NG': {
    // "sing" - back of tongue raised
    jawOpen: 15,
    mouthOpen: 10,
    mouthClose: 0,
    mouthFunnel: 10,
    mouthPucker: 0,
    mouthStretchLeft: 5,
    mouthStretchRight: 5,
    visemeCategory: 'C',
    duration: { min: 60, typical: 100, max: 160 },
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // LIQUIDS - Continuous sounds
  // ═══════════════════════════════════════════════════════════════════════════
  
  'L': {
    // "let" - tongue touches palate
    jawOpen: 18,
    mouthOpen: 12,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 25,
    mouthStretchRight: 25,
    tongueOut: 15,
    visemeCategory: 'L',
    duration: { min: 50, typical: 90, max: 150 },
  },
  
  'R': {
    // "red" - tongue back, lips slightly rounded
    jawOpen: 22,
    mouthOpen: 16,
    mouthClose: 0,
    mouthFunnel: 30,
    mouthPucker: 20,
    mouthStretchLeft: 5,
    mouthStretchRight: 5,
    visemeCategory: 'R',
    duration: { min: 50, typical: 90, max: 150 },
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // GLIDES/SEMIVOWELS - Transitional sounds
  // ═══════════════════════════════════════════════════════════════════════════
  
  'W': {
    // "wet" - rounded, pursed
    jawOpen: 15,
    mouthOpen: 10,
    mouthClose: 0,
    mouthFunnel: 65,
    mouthPucker: 75,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    visemeCategory: 'U',
    duration: { min: 40, typical: 80, max: 130 },
  },
  
  'Y': {
    // "yes" - narrow smile
    jawOpen: 12,
    mouthOpen: 8,
    mouthClose: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 50,
    mouthStretchRight: 50,
    mouthSmileLeft: 30,
    mouthSmileRight: 30,
    visemeCategory: 'I',
    duration: { min: 40, typical: 70, max: 120 },
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // SILENCE - Rest positions
  // ═══════════════════════════════════════════════════════════════════════════
  
  'SIL': {
    // Silence - neutral face
    jawOpen: 0,
    mouthOpen: 0,
    mouthClose: 20,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    mouthSmileLeft: 15,
    mouthSmileRight: 15,
    visemeCategory: 'REST',
    duration: { min: 50, typical: 200, max: 1000 },
  },
  
  'SP': {
    // Short pause - same as silence
    jawOpen: 0,
    mouthOpen: 0,
    mouthClose: 20,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthStretchLeft: 0,
    mouthStretchRight: 0,
    mouthSmileLeft: 15,
    mouthSmileRight: 15,
    visemeCategory: 'REST',
    duration: { min: 20, typical: 100, max: 500 },
  },
  
  'spn': {
    // Spoken noise - neutral
    jawOpen: 5,
    mouthOpen: 3,
    mouthClose: 10,
    mouthFunnel: 0,
    mouthPucker: 0,
    visemeCategory: 'REST',
    duration: { min: 50, typical: 150, max: 500 },
  },
};

// =============================================================================
// COARTICULATION RULES
// =============================================================================

/**
 * Coarticulation adjustment rules for natural speech
 * Adjacent phonemes influence each other's mouth shapes
 */
export const COARTICULATION_RULES = {
  // Anticipatory coarticulation (upcoming sound affects current)
  anticipatory: {
    // Before rounded vowels, start rounding early
    beforeRounded: {
      targets: ['UW', 'UH', 'OW', 'AO', 'W'],
      adjustment: { mouthFunnel: 15, mouthPucker: 10 },
      frames: 2, // Apply 2 frames before
    },
    // Before high vowels, start raising
    beforeHigh: {
      targets: ['IY', 'IH'],
      adjustment: { mouthSmileLeft: 10, mouthSmileRight: 10 },
      frames: 2,
    },
  },
  
  // Carryover coarticulation (previous sound affects current)
  carryover: {
    // After bilabials, lips slow to separate
    afterBilabial: {
      sources: ['P', 'B', 'M'],
      adjustment: { mouthClose: 15 },
      frames: 1,
    },
    // After rounded vowels, maintain some rounding
    afterRounded: {
      sources: ['UW', 'UH', 'OW', 'AO', 'W'],
      adjustment: { mouthFunnel: 10 },
      frames: 1,
    },
  },
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Get blendshape values for a phoneme
 * @param {string} phoneme - ARPAbet phoneme code
 * @returns {Object} Blendshape values
 */
export function getBlendshapesForPhoneme(phoneme) {
  // Normalize phoneme (remove stress markers like AA1, AA2)
  const normalized = phoneme.replace(/[0-9]/g, '').toUpperCase();
  
  const mapping = PHONEME_TO_BLENDSHAPES[normalized];
  if (mapping) {
    // Return copy without metadata
    const { visemeCategory, duration, transition, ...blendshapes } = mapping;
    return blendshapes;
  }
  
  // Default to silence
  return getBlendshapesForPhoneme('SIL');
}

/**
 * Get viseme category for a phoneme
 * @param {string} phoneme - ARPAbet phoneme code
 * @returns {string} Viseme category name
 */
export function getVisemeCategory(phoneme) {
  const normalized = phoneme.replace(/[0-9]/g, '').toUpperCase();
  return PHONEME_TO_BLENDSHAPES[normalized]?.visemeCategory || 'REST';
}

/**
 * Interpolate between two blendshape states
 * @param {Object} from - Starting blendshapes
 * @param {Object} to - Target blendshapes
 * @param {number} t - Interpolation factor (0-1)
 * @returns {Object} Interpolated blendshapes
 */
export function interpolateBlendshapes(from, to, t) {
  const result = {};
  const allKeys = new Set([...Object.keys(from), ...Object.keys(to)]);
  
  for (const key of allKeys) {
    const fromValue = from[key] || 0;
    const toValue = to[key] || 0;
    result[key] = fromValue + (toValue - fromValue) * t;
  }
  
  return result;
}

/**
 * Apply coarticulation adjustments to a sequence
 * @param {Array} sequence - Array of { phoneme, blendshapes, timestamp, duration }
 * @returns {Array} Adjusted sequence
 */
export function applyCoarticulation(sequence) {
  const adjusted = sequence.map(item => ({
    ...item,
    blendshapes: { ...item.blendshapes },
  }));
  
  // Apply anticipatory coarticulation
  for (let i = 0; i < adjusted.length - 1; i++) {
    const current = adjusted[i];
    const next = adjusted[i + 1];
    
    // Check if next phoneme triggers anticipatory adjustment
    for (const rule of Object.values(COARTICULATION_RULES.anticipatory)) {
      if (rule.targets.includes(next.phoneme.replace(/[0-9]/g, ''))) {
        for (const [key, value] of Object.entries(rule.adjustment)) {
          current.blendshapes[key] = (current.blendshapes[key] || 0) + value;
        }
      }
    }
  }
  
  // Apply carryover coarticulation
  for (let i = 1; i < adjusted.length; i++) {
    const previous = adjusted[i - 1];
    const current = adjusted[i];
    
    // Check if previous phoneme triggers carryover adjustment
    for (const rule of Object.values(COARTICULATION_RULES.carryover)) {
      if (rule.sources.includes(previous.phoneme.replace(/[0-9]/g, ''))) {
        for (const [key, value] of Object.entries(rule.adjustment)) {
          current.blendshapes[key] = (current.blendshapes[key] || 0) + value;
        }
      }
    }
  }
  
  return adjusted;
}

/**
 * Generate smooth blendshape timeline from phoneme sequence
 * @param {Array} phonemeSequence - Array of { phoneme, start, end }
 * @param {number} fps - Frames per second (default 30)
 * @returns {Array} Array of { timestamp, blendshapes } for each frame
 */
export function generateBlendshapeTimeline(phonemeSequence, fps = 30) {
  if (!phonemeSequence || phonemeSequence.length === 0) {
    return [];
  }
  
  const frameInterval = 1 / fps;
  const timeline = [];
  
  // Get total duration
  const totalDuration = phonemeSequence[phonemeSequence.length - 1].end;
  
  // Build intermediate representation with blendshapes
  const phonemesWithBlendshapes = phonemeSequence.map(p => ({
    ...p,
    blendshapes: getBlendshapesForPhoneme(p.phoneme || p.phone),
  }));
  
  // Apply coarticulation
  const adjusted = applyCoarticulation(phonemesWithBlendshapes);
  
  // Generate frame-by-frame timeline
  let currentPhonemeIndex = 0;
  
  for (let time = 0; time <= totalDuration; time += frameInterval) {
    // Find current phoneme
    while (
      currentPhonemeIndex < adjusted.length - 1 &&
      time >= adjusted[currentPhonemeIndex + 1].start
    ) {
      currentPhonemeIndex++;
    }
    
    const current = adjusted[currentPhonemeIndex];
    const next = adjusted[currentPhonemeIndex + 1];
    
    let blendshapes;
    
    if (next && time >= current.start && time < next.start) {
      // Interpolate between current and next
      const phonemeDuration = next.start - current.start;
      const elapsed = time - current.start;
      
      // Use eased interpolation for smoother transitions
      const t = easeInOutQuad(elapsed / phonemeDuration);
      blendshapes = interpolateBlendshapes(current.blendshapes, next.blendshapes, t);
    } else {
      blendshapes = { ...current.blendshapes };
    }
    
    timeline.push({
      timestamp: Math.round(time * 1000) / 1000, // Round to milliseconds
      blendshapes,
    });
  }
  
  return timeline;
}

/**
 * Quadratic ease-in-out function for smooth transitions
 * @param {number} t - Input (0-1)
 * @returns {number} Eased output (0-1)
 */
function easeInOutQuad(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

// =============================================================================
// EXPORTS
// =============================================================================

export default {
  ARPABET_PHONEMES,
  VISEME_CATEGORIES,
  PHONEME_TO_BLENDSHAPES,
  COARTICULATION_RULES,
  getBlendshapesForPhoneme,
  getVisemeCategory,
  interpolateBlendshapes,
  applyCoarticulation,
  generateBlendshapeTimeline,
};

