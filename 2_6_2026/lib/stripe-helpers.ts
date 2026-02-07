// lib/stripe-helpers.ts
// Stripe utility functions for subscription management

/**
 * Pricing plans with Stripe price ID env var mappings
 * Prices defined in the pricing overlay:
 *   Learner (individual): $9.99/mo, $79.99/yr, $199 lifetime
 *   Family: $14.99/mo, $119.99/yr, $349 lifetime
 */
export const PLAN_PRICES: Record<string, Record<string, { env: string; amount: number }>> = {
  individual: {
    monthly:  { env: 'STRIPE_PRICE_MONTHLY',  amount: 999 },
    annual:   { env: 'STRIPE_PRICE_ANNUAL',   amount: 7999 },
    lifetime: { env: 'STRIPE_PRICE_LIFETIME',  amount: 19900 },
  },
  family: {
    monthly:  { env: 'STRIPE_PRICE_FAMILY',    amount: 1499 },
    annual:   { env: 'STRIPE_PRICE_FAMILY_ANNUAL', amount: 11999 },
    lifetime: { env: 'STRIPE_PRICE_FAMILY_LIFETIME', amount: 34900 },
  },
}

/**
 * Get the Stripe price ID from environment variables
 */
export function getStripePriceId(planId: string, billingCycle: string): string | null {
  const plan = PLAN_PRICES[planId]
  if (!plan) return null
  const price = plan[billingCycle]
  if (!price) return null
  return process.env[price.env] || null
}

/**
 * Get price amount in cents for a plan/cycle combo
 */
export function getPriceAmountCents(planId: string, billingCycle: string): number {
  return PLAN_PRICES[planId]?.[billingCycle]?.amount ?? 0
}

/**
 * Affiliate commission rates by tier
 */
const AFFILIATE_RATES: Record<string, number> = {
  starter: 0.20,
  advocate: 0.25,
  ambassador: 0.30,
  champion: 0.35,
  legend: 0.40,
}

/**
 * Calculate affiliate commission for a payment amount
 */
export function calculateAffiliateCommission(amountInCents: number, affiliateTier: string): number {
  const rate = AFFILIATE_RATES[affiliateTier] ?? 0.20
  return Math.round(amountInCents * rate)
}

/**
 * Format price for display
 */
export function formatPrice(amountInCents: number, interval: 'monthly' | 'annual' | 'lifetime' = 'monthly'): string {
  const dollars = amountInCents / 100
  const labels: Record<string, string> = {
    monthly: '/mo',
    annual: '/yr',
    lifetime: ' one-time',
  }
  return `$${dollars.toFixed(2)}${labels[interval]}`
}

/**
 * Get the Stripe price ID for a given tier name (used by embedded checkout)
 * Maps tier names like "learner-monthly" to the correct env var
 */
export function getPriceIdForTier(tierId: string): string | null {
  // tierId can be "individual-monthly", "family-annual", etc.
  const parts = tierId.split('-')
  if (parts.length === 2) {
    return getStripePriceId(parts[0], parts[1])
  }
  // Or it can be a raw price ID already
  if (tierId.startsWith('price_')) return tierId
  return null
}

/**
 * Validate billing cycle input
 */
export function validateBillingCycle(cycle: unknown): 'monthly' | 'annual' | 'lifetime' {
  if (cycle === 'annual' || cycle === 'lifetime') return cycle
  return 'monthly'
}

/**
 * Sanitize ref code to prevent injection
 */
export function sanitizeRefCode(code: unknown): string | null {
  if (typeof code !== 'string') return null
  const cleaned = code.replace(/[^a-zA-Z0-9_-]/g, '')
  return cleaned.length > 0 ? cleaned : null
}
