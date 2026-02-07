/**
 * Uncertainty Filter for Kelly's Statements
 * 
 * Kelly says things that sound factual but may be disputed, uncertain, or oversimplified.
 * This filter identifies statements that need hedging language or disclaimers.
 */

// Patterns that indicate potentially refutable claims
const CERTAINTY_PATTERNS = [
  /your brain is busier/i,
  /your brain does more/i,
  /scientists (have )?(proven|discovered|found that)/i,
  /studies show/i,
  /research (proves|shows|confirms)/i,
  /it('s| is) (a )?fact that/i,
  /we (now )?know that/i,
  /always/i,
  /never/i,
  /100%/i,
  /definitely/i,
  /certainly/i,
  /without (a )?doubt/i,
]

// Hedging phrases to prepend or append
const HEDGING_PHRASES = {
  prepend: [
    "Some scientists believe",
    "Research suggests",
    "Many experts think",
    "One theory is that",
    "It's often said that",
    "Some studies suggest",
  ],
  append: [
    "...though scientists are still learning more!",
    "...but there's always more to discover!",
    "...at least, that's what many researchers think!",
    "...though not everyone agrees!",
  ]
}

export interface UncertaintyResult {
  needsDisclaimer: boolean
  confidence: 'low' | 'medium' | 'high'
  matchedPatterns: string[]
  suggestedHedge?: string
}

/**
 * Analyze text for potentially refutable claims
 */
export function analyzeUncertainty(text: string): UncertaintyResult {
  const matchedPatterns: string[] = []
  
  for (const pattern of CERTAINTY_PATTERNS) {
    if (pattern.test(text)) {
      matchedPatterns.push(pattern.source)
    }
  }
  
  const needsDisclaimer = matchedPatterns.length > 0
  const confidence = matchedPatterns.length === 0 ? 'high' 
    : matchedPatterns.length === 1 ? 'medium' 
    : 'low'
  
  return {
    needsDisclaimer,
    confidence,
    matchedPatterns,
    suggestedHedge: needsDisclaimer 
      ? HEDGING_PHRASES.append[Math.floor(Math.random() * HEDGING_PHRASES.append.length)]
      : undefined
  }
}

/**
 * Add disclaimer UI element for uncertain statements
 */
export function getDisclaimerText(): string {
  return "Kelly shares ideas to spark curiosity - always keep exploring to learn more!"
}

/**
 * Check if a specific claim needs verification
 * Used to flag content for human review
 */
export function flagForReview(text: string): boolean {
  const result = analyzeUncertainty(text)
  return result.confidence === 'low'
}

// Specific claims that are known to be disputed or oversimplified
export const DISPUTED_CLAIMS = [
  {
    claim: "your brain is busier than when you are awake",
    topic: "sleep and brain activity",
    reality: "Brain activity during sleep varies by stage; some stages show less activity than wakefulness, others show different patterns",
    source: "Neuroscience research on sleep stages"
  },
  {
    claim: "we only use 10% of our brain",
    topic: "brain usage",
    reality: "This is a myth - we use virtually all parts of our brain",
    source: "Neuroscience consensus"
  },
  {
    claim: "goldfish have 3 second memory",
    topic: "animal cognition", 
    reality: "Goldfish can remember things for months",
    source: "Animal behavior research"
  }
]
