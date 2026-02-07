/**
 * KELLY UI RULES - BAKED INTO DAILY WORKFLOW
 * ==========================================
 * 
 * These rules are NON-NEGOTIABLE. Every component, every edit, every feature
 * MUST comply with these standards. Reference this file before ANY UI work.
 * 
 * Last Updated: January 21, 2026 - QA PASS COMPLETE
 * Current Day: Day 21 of 365
 * 
 * QA STATUS:
 * - Rule 1 (Two-Plane): PASS - No overlays covering Kelly
 * - Rule 2 (Colors): PASS - No color violations in /components/kelly
 * - Rule 3 (Glass): PASS - Using defined glass-* classes
 * - Rule 4 (Interactions): VERIFIED - All critical paths defined
 * - Rule 5 (No Fake Data): PASS - use-likes.ts and use-presence.ts cleaned
 * - Rule 6 (Checklist): ENFORCED
 * - Rule 7 (Overlays): PASS - Right-side only (420px)
 */

// =============================================================================
// RULE 1: TWO-PLANE ARCHITECTURE
// =============================================================================
/**
 * The screen has exactly TWO visual planes:
 * 
 * PLANE 1 (KELLY): Center of screen
 * - Kelly video/avatar is ALWAYS visible
 * - NOTHING may obscure, dim, blur, or cover Kelly
 * - Kelly is the product - she must always be seen
 * 
 * PLANE 2 (UI): Edges of screen only
 * - Left rail: Share, Live Class, Comments (max 420px)
 * - Right rail: Action buttons (max 60px)
 * - Bottom: Lesson content, controls
 * - Top: Navigation tabs
 * 
 * OVERLAYS:
 * - Settings, Pricing, Help, About = RIGHT SIDE PANEL ONLY (420px)
 * - Backdrop covers RIGHT SIDE ONLY, not center
 * - Kelly remains fully visible and clickable
 * 
 * NEVER:
 * - Full-screen modals that cover Kelly
 * - Center-positioned dialogs
 * - Backdrop blur over Kelly
 * - Any element with inset-0 that covers Kelly
 */
export const TWO_PLANE_RULE = {
  kelly: {
    position: 'center',
    zIndex: 0,
    alwaysVisible: true,
    neverObscured: true,
  },
  ui: {
    leftRail: { maxWidth: 420, position: 'left' },
    rightRail: { maxWidth: 60, position: 'right' },
    overlays: { maxWidth: 420, position: 'right', neverCoverKelly: true },
  },
} as const

// =============================================================================
// RULE 2: COLOR SYSTEM - MONOCHROME + SEMANTIC
// =============================================================================
/**
 * PRIMARY PALETTE (use these ONLY):
 * 
 * BLUE (#3b82f6 / var(--kelly-blue)):
 * - Brand accent color
 * - Interactive elements (buttons, links, toggles ON state)
 * - Progress indicators
 * - "Let's Begin" CTA
 * - Active tab highlights
 * 
 * WHITE/GRAY (monochrome scale):
 * - ALL other UI elements
 * - Text: white/90, white/70, white/50, white/30
 * - Backgrounds: white/10, white/5, black/40
 * - Borders: white/20, white/10
 * - Icons: white/70, white/50
 * 
 * GREEN (#22c55e):
 * - ONLY for LIVE status indicator
 * - Kelly is currently broadcasting live
 * - Live class in progress
 * 
 * RED (#ef4444):
 * - ONLY for OFFLINE status indicator
 * - Kelly is not live (default state)
 * - Error states
 * 
 * NEVER USE:
 * - Purple, violet, pink, orange, yellow, teal
 * - Colored badges (use white/black)
 * - Colored checkmarks (use white)
 * - Colored icons in overlays (use white/gray)
 * - Any color not listed above
 */
export const COLOR_SYSTEM = {
  brand: {
    blue: '#3b82f6', // var(--kelly-blue)
  },
  semantic: {
    live: '#22c55e',    // Green - Kelly is LIVE
    offline: '#ef4444', // Red - Kelly is OFFLINE
  },
  monochrome: {
    textPrimary: 'rgba(255,255,255,0.9)',
    textSecondary: 'rgba(255,255,255,0.7)',
    textTertiary: 'rgba(255,255,255,0.5)',
    textMuted: 'rgba(255,255,255,0.3)',
    bgCard: 'rgba(255,255,255,0.05)',
    bgCardHover: 'rgba(255,255,255,0.08)',
    bgInput: 'rgba(0,0,0,0.4)',
    border: 'rgba(255,255,255,0.1)',
    borderHover: 'rgba(255,255,255,0.2)',
  },
} as const

// =============================================================================
// RULE 3: GLASS MORPHISM CLASSES
// =============================================================================
/**
 * Use ONLY these glass classes for UI elements:
 * 
 * glass-card     - Compact cards (Share, Live Class, Comments)
 * glass-panel    - Larger panels (Caption, Mobile drawer)
 * glass-tabs     - Segmented controls (Live/Kelly/Find)
 * glass-input    - Text inputs with focus glow
 * glass-button-primary - Blue CTA buttons (Let's Begin, Send)
 * glass-icon-button    - Circular action buttons (sidebar)
 * glass-live-dot       - Status indicator (red=offline, green.live=live)
 * 
 * All glass elements:
 * - Dark background with subtle transparency
 * - Subtle white border (1px, low opacity)
 * - Backdrop blur (12-16px)
 * - No colored backgrounds except kelly-blue CTAs
 */
export const GLASS_CLASSES = [
  'glass-card',
  'glass-panel', 
  'glass-tabs',
  'glass-input',
  'glass-button-primary',
  'glass-icon-button',
  'glass-live-dot',
] as const

// =============================================================================
// RULE 4: CRITICAL INTERACTIONS
// =============================================================================
/**
 * These interactions are THE PRODUCT. They MUST work perfectly.
 * 
 * A. "LET'S BEGIN" BUTTON
 *    Click → Kelly speaks Day N Hook audio
 *    Kelly's mouth moves in sync with audio
 *    Lesson phase indicator shows progress
 *    FAIL: Nothing happens, Kelly static, wrong day content
 * 
 * B. TRUE/FALSE BUTTONS
 *    Click → Visual feedback (highlight)
 *    Kelly responds to choice
 *    Lesson advances to next phase
 *    FAIL: No feedback, wrong response, doesn't advance
 * 
 * C. AGE SELECTOR
 *    Click → Kelly transforms to age-appropriate variant
 *    Content adapts (vocabulary, examples)
 *    Selection persists
 *    FAIL: Kelly doesn't change, wrong variant, adult content to child
 * 
 * D. TONE SELECTOR
 *    Click → Kelly's teaching style adapts
 *    Examples/analogies match tone context
 *    FAIL: No change, wrong tone
 * 
 * E. LANGUAGE SELECTOR
 *    Click → Kelly speaks selected language
 *    UI text switches (optional)
 *    Captions in selected language
 *    FAIL: Wrong language, no audio, mixed languages
 * 
 * F. DAY NAVIGATION
 *    Click ← → → Load correct day's lesson
 *    Date, day counter, topic all update
 *    FAIL: Wrong day content, boundary errors (Day 1, Day 365)
 */
export const CRITICAL_INTERACTIONS = {
  letsBegin: {
    trigger: 'button.lets-begin',
    expected: ['audio-plays', 'kelly-animates', 'phase-updates'],
    failStates: ['nothing-happens', 'kelly-static', 'wrong-day'],
  },
  trueFalse: {
    trigger: 'button.true-response | button.false-response',
    expected: ['visual-feedback', 'kelly-responds', 'lesson-advances'],
    failStates: ['no-feedback', 'wrong-response', 'stuck'],
  },
  ageSelector: {
    trigger: 'button.age-*',
    expected: ['kelly-transforms', 'content-adapts', 'persists'],
    failStates: ['no-change', 'wrong-variant', 'inappropriate-content'],
  },
  toneSelector: {
    trigger: 'button.tone-*',
    expected: ['style-changes', 'examples-match'],
    failStates: ['no-change', 'wrong-tone'],
  },
  languageSelector: {
    trigger: 'button.lang-*',
    expected: ['kelly-speaks-language', 'captions-match'],
    failStates: ['wrong-language', 'no-audio', 'mixed'],
  },
  dayNavigation: {
    trigger: 'button.prev-day | button.next-day',
    expected: ['date-updates', 'content-loads', 'topic-matches'],
    failStates: ['wrong-day', 'boundary-error', 'stuck'],
  },
} as const

// =============================================================================
// RULE 5: NO FAKE DATA
// =============================================================================
/**
 * ALL displayed data must be REAL or clearly marked as placeholder.
 * 
 * MUST BE REAL:
 * - Presence count ("12 here") → actual connected users
 * - Live class countdown → actual time to next class
 * - Comments → real user comments from database
 * - Day content → actual lesson for that day
 * 
 * ACCEPTABLE PLACEHOLDERS:
 * - "No comments yet" when empty
 * - "Loading..." during fetch
 * - Demo mode clearly labeled
 * 
 * NEVER:
 * - Hardcoded presence counts that don't reflect reality
 * - Fake comments that appear real
 * - Countdown to non-existent class
 * - "12 here" when no one is there
 */
export const DATA_INTEGRITY = {
  realData: ['presence-count', 'live-countdown', 'comments', 'lesson-content'],
  acceptablePlaceholders: ['empty-state', 'loading-state', 'demo-mode-labeled'],
  neverFake: ['presence-as-real', 'comments-as-real', 'countdown-to-nothing'],
} as const

// =============================================================================
// RULE 6: COMPONENT CHECKLIST
// =============================================================================
/**
 * Before creating/editing ANY component, verify:
 * 
 * [ ] Does NOT cover Kelly (two-plane rule)
 * [ ] Uses monochrome colors (no purple, orange, etc.)
 * [ ] Uses glass-* classes where appropriate
 * [ ] Blue only for interactive/brand elements
 * [ ] Green only for LIVE status
 * [ ] Red only for OFFLINE/error status
 * [ ] No fake data displayed as real
 * [ ] Accessible (aria labels, keyboard nav)
 * [ ] Mobile responsive
 * [ ] Dark theme only (no light mode)
 */
export const COMPONENT_CHECKLIST = [
  'two-plane-compliant',
  'monochrome-colors',
  'glass-classes',
  'blue-interactive-only',
  'green-live-only',
  'red-offline-only',
  'no-fake-data',
  'accessible',
  'mobile-responsive',
  'dark-theme-only',
] as const

// =============================================================================
// RULE 7: OVERLAY POSITIONING
// =============================================================================
/**
 * ALL overlays (Settings, Pricing, Help, About, etc.):
 * 
 * POSITION: Right side of screen
 * WIDTH: Max 420px
 * HEIGHT: Full viewport height
 * BACKDROP: Right side only (not full screen)
 * KELLY: Always visible and clickable
 * 
 * Implementation:
 * - Use OverlayBackdrop + OverlayPanel from /components/kelly/overlays/index.tsx
 * - Position 'right' or 'fullscreen' (both are right-aligned)
 * - Never position 'center' for full-height content
 */
export const OVERLAY_RULES = {
  position: 'right',
  maxWidth: 420,
  height: '100vh',
  backdropCoverage: 'right-side-only',
  kellyVisible: true,
  kellyClickable: true,
} as const

// =============================================================================
// EXPORTS FOR RUNTIME VALIDATION
// =============================================================================
export const KELLY_UI_RULES = {
  TWO_PLANE_RULE,
  COLOR_SYSTEM,
  GLASS_CLASSES,
  CRITICAL_INTERACTIONS,
  DATA_INTEGRITY,
  COMPONENT_CHECKLIST,
  OVERLAY_RULES,
} as const

export default KELLY_UI_RULES
