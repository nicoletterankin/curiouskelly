'use client'

/**
 * KellyOS Pricing Overlay
 * Apple-quality subscription/pricing panel
 * Company: Lesson of the Day
 * Product: iLearn with KellyOS
 */

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel, GlassButton, PillSelector, AnimatedList, AnimatedItem } from './index'
import { Check, Sparkles, Zap, Crown, Gift, X } from 'lucide-react'

// ============================================================================
// TYPES
// ============================================================================

type BillingCycle = 'monthly' | 'annual' | 'lifetime'

interface PricingPlan {
  id: string
  name: string
  description: string
  price: {
    monthly: number
    annual: number
    lifetime: number
  }
  features: string[]
  highlighted?: boolean
  badge?: string
}

interface PricingOverlayProps {
  isOpen: boolean
  onClose: () => void
  currentPlan?: string
  onSelectPlan: (planId: string, billingCycle: BillingCycle) => void
}

// ============================================================================
// DATA
// ============================================================================

const PLANS: PricingPlan[] = [
  {
    id: 'free',
    name: 'Explorer',
    description: 'Start your learning journey',
    price: { monthly: 0, annual: 0, lifetime: 0 },
    features: [
      "Today's lesson only",
      '1 language',
      'Standard Kelly',
      'Basic captions',
    ],
  },
  {
    id: 'individual',
    name: 'Learner',
    description: 'Full access for curious minds',
    price: { monthly: 9.99, annual: 79.99, lifetime: 199 },
    features: [
      '365 daily lessons',
      'All 12 languages',
      'All 12 Kelly archetypes',
      'All age perspectives',
      'Interactive captions',
      'Visual learning aids',
      'Offline downloads',
      'Ad-free experience',
    ],
    highlighted: true,
    badge: 'Most Popular',
  },
  {
    id: 'family',
    name: 'Family',
    description: 'Learning for up to 6 members',
    price: { monthly: 14.99, annual: 119.99, lifetime: 349 },
    features: [
      'Everything in Learner',
      'Up to 6 family members',
      'Individual profiles',
      'Parental controls',
      'Progress tracking',
      'Family discussions',
      'Priority support',
    ],
    badge: 'Best Value',
  },
]

// ============================================================================
// PRICING OVERLAY
// ============================================================================

export function PricingOverlay({
  isOpen,
  onClose,
  currentPlan = 'free',
  onSelectPlan,
}: PricingOverlayProps) {
  const [billingCycle, setBillingCycle] = useState<BillingCycle>('annual')
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null)

  if (!isOpen) return null

  const getPrice = (plan: PricingPlan) => {
    return plan.price[billingCycle]
  }

  const getSavings = (plan: PricingPlan) => {
    if (billingCycle === 'annual' && plan.price.monthly > 0) {
      const monthlyCost = plan.price.monthly * 12
      const annualCost = plan.price.annual
      const savings = Math.round(((monthlyCost - annualCost) / monthlyCost) * 100)
      return savings > 0 ? `Save ${savings}%` : null
    }
    if (billingCycle === 'lifetime' && plan.price.annual > 0) {
      return 'Pay once, learn forever'
    }
    return null
  }

  const handleSelectPlan = (planId: string) => {
    if (planId === 'free') {
      onClose()
      return
    }
    setSelectedPlan(planId)
    onSelectPlan(planId, billingCycle)
  }

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose}>
      <OverlayPanel
        title="Choose Your Plan"
        subtitle="Unlock the full power of KellyOS"
        onClose={onClose}
        size="md"
        position="right"
      >
        <div className="p-3 sm:p-4 space-y-4 overflow-y-auto max-h-[85vh]">
          {/* Billing Toggle - COMPACT */}
          <div className="flex justify-center">
            <div className="inline-flex p-0.5 rounded-full bg-white/10">
              {(['monthly', 'annual', 'lifetime'] as const).map((cycle) => (
                <button
                  key={cycle}
                  onClick={() => setBillingCycle(cycle)}
                  className={cn(
                    'px-3 py-1.5 rounded-full text-xs font-medium transition-all',
                    billingCycle === cycle
                      ? 'bg-white text-black'
                      : 'text-white/60 hover:text-white'
                  )}
                >
                  {cycle === 'monthly' ? 'Mo' : cycle === 'annual' ? 'Year' : 'Life'}
                </button>
              ))}
            </div>
          </div>

          {/* Plans - ULTRA COMPACT cards */}
          <div className="space-y-2">
            {PLANS.map((plan) => {
              const price = getPrice(plan)
              const savings = getSavings(plan)
              const isCurrentPlan = plan.id === currentPlan

              return (
                <button
                  key={plan.id}
                  onClick={() => handleSelectPlan(plan.id)}
                  disabled={isCurrentPlan}
                  className={cn(
                    'w-full text-left rounded-xl p-3 transition-all',
                    plan.highlighted
                      ? 'bg-white/15 border border-white/30'
                      : 'bg-white/5 hover:bg-white/10 border border-white/10',
                    isCurrentPlan && 'opacity-50'
                  )}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-white text-sm">{plan.name}</span>
                      {plan.badge && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-white/20 text-white/80">
                          {plan.badge}
                        </span>
                      )}
                    </div>
                    <div className="text-right">
                      <span className="font-bold text-white">
                        {price === 0 ? 'Free' : `$${price}`}
                      </span>
                      {price > 0 && billingCycle !== 'lifetime' && (
                        <span className="text-[10px] text-white/50">/{billingCycle === 'monthly' ? 'mo' : 'yr'}</span>
                      )}
                    </div>
                  </div>
                  <p className="text-[10px] text-white/50 mb-2">{plan.description}</p>
                  {/* Show only top 3 features */}
                  <div className="flex flex-wrap gap-1">
                    {plan.features.slice(0, 3).map((f, i) => (
                      <span key={i} className="text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-white/60">
                        {f}
                      </span>
                    ))}
                    {plan.features.length > 3 && (
                      <span className="text-[9px] text-white/40">+{plan.features.length - 3} more</span>
                    )}
                  </div>
                  {savings && (
                    <div className="mt-1 text-[10px] text-emerald-400">{savings}</div>
                  )}
                </button>
              )
            })}
          </div>

          {/* Footer - COMPACT */}
          <div className="text-center text-[10px] text-white/30 pt-2 border-t border-white/10">
            <p>Cancel anytime. Powered by Stripe.</p>
          </div>
        </div>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}

// ============================================================================
// CHECKOUT OVERLAY
// ============================================================================

interface CheckoutOverlayProps {
  isOpen: boolean
  onClose: () => void
  planId: string
  billingCycle: BillingCycle
  onComplete: () => void
}

export function CheckoutOverlay({
  isOpen,
  onClose,
  planId,
  billingCycle,
  onComplete,
}: CheckoutOverlayProps) {
  const [loading, setLoading] = useState(false)
  const [email, setEmail] = useState('')

  const plan = PLANS.find(p => p.id === planId)
  if (!plan || !isOpen) return null

  const price = plan.price[billingCycle]

  const handleCheckout = async () => {
    if (!email) return
    
    setLoading(true)
    try {
      const res = await fetch('/api/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          planId,
          billingCycle,
          email,
          successUrl: `${window.location.origin}/success`,
          cancelUrl: window.location.href,
        }),
      })
      
      const data = await res.json()
      
      if (data.url) {
        // Redirect to Stripe checkout
        window.location.href = data.url
      } else {
        console.error('[Checkout] No URL returned:', data)
        setLoading(false)
      }
    } catch (error) {
      console.error('[Checkout] Error:', error)
      setLoading(false)
    }
  }

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose}>
      <OverlayPanel
        title="Complete Your Order"
        onClose={onClose}
        size="md"
        position="right"
      >
        <div className="p-6 space-y-6">
          {/* Order Summary */}
          <div className="bg-muted/30 rounded-2xl p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="font-semibold">{plan.name}</h4>
                <p className="text-sm text-muted-foreground">
                  {billingCycle === 'monthly' ? 'Monthly subscription' :
                   billingCycle === 'annual' ? 'Annual subscription' :
                   'Lifetime access'}
                </p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold">${price}</div>
                {billingCycle !== 'lifetime' && (
                  <div className="text-sm text-muted-foreground">
                    /{billingCycle === 'monthly' ? 'month' : 'year'}
                  </div>
                )}
              </div>
            </div>

            {billingCycle === 'annual' && (
              <div className="flex items-center gap-2 text-sm text-primary">
                <Sparkles className="h-4 w-4" />
                <span>You're saving ${((plan.price.monthly * 12) - plan.price.annual).toFixed(0)}/year</span>
              </div>
            )}
          </div>

          {/* Email Input */}
          <div>
            <label className="block text-sm font-medium mb-2">Email address</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full px-4 py-3 rounded-xl bg-muted/30 border border-border/50 focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          {/* Checkout Button */}
          <button
            onClick={handleCheckout}
            disabled={loading || !email}
            className={cn(
              'w-full py-4 rounded-full font-semibold transition-all',
              'bg-primary text-primary-foreground hover:bg-primary/90',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'flex items-center justify-center gap-2'
            )}
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Zap className="h-5 w-5" />
                Pay ${price} with Stripe
              </>
            )}
          </button>

          {/* Security Note */}
          <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
              <path d="M7 11V7a5 5 0 0 1 10 0v4" />
            </svg>
            <span>Secured by 256-bit SSL encryption</span>
          </div>
        </div>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
