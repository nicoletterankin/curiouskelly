// components/pricing/tier-card.tsx
'use client'

import { useState, useCallback } from 'react'
import { SUBSCRIPTION_TIERS, type SubscriptionTier } from '@/lib/products'
import { formatPrice, calculateAffiliateCommission } from '@/lib/stripe-helpers'
import { ChevronDown } from 'lucide-react'

interface TierCardProps {
  tier: SubscriptionTier
  affiliateName?: string
  refCode?: string
  onCheckoutClick: (tierId: string, interval: 'monthly' | 'yearly' | 'lifetime') => void
  isLoading?: boolean
}

export function TierCard({ tier, affiliateName, refCode, onCheckoutClick, isLoading }: TierCardProps) {
  const [selectedInterval, setSelectedInterval] = useState<'monthly' | 'yearly' | 'lifetime'>('monthly')
  const isPopular = tier.id === 'learner'
  const isFree = tier.id === 'explorer'

  const price = tier.prices[selectedInterval]
  const affiliateEarnings = tier.affiliateCommissionRate > 0 ? calculateAffiliateCommission(price.amount, tier.id) : 0

  const handleCheckout = useCallback(() => {
    if (!isFree) {
      onCheckoutClick(tier.id, selectedInterval)
    }
  }, [tier.id, selectedInterval, isFree, onCheckoutClick])

  return (
    <div
      className={`relative flex flex-col rounded-lg border transition-all ${
        isPopular
          ? 'border-white bg-white/5 shadow-lg scale-105'
          : 'border-white/20 bg-black/40 hover:border-white/40'
      }`}
    >
      {isPopular && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full bg-white px-4 py-1 text-xs font-bold text-black">
          Most Popular
        </div>
      )}

      <div className="p-6">
        {/* Header */}
        <h3 className="text-2xl font-bold text-white">{tier.name}</h3>
        <p className="mt-2 text-sm text-white/60">{tier.description}</p>

        {/* Affiliate Context */}
        {refCode && affiliateName && (
          <div className="mt-4 rounded-lg bg-white/5 p-3 text-xs text-white/70">
            Friend referral: <strong className="text-white">{affiliateName}</strong> earns{' '}
            <strong className="text-white">{(tier.affiliateCommissionRate * 100).toFixed(0)}%</strong>
          </div>
        )}

        {/* Price Display */}
        <div className="mt-6 flex items-baseline gap-1">
          <span className="text-4xl font-bold text-white">
            {price.amount === 0 ? 'Free' : formatPrice(price.amount, selectedInterval).split('/')[0]}
          </span>
          {price.amount > 0 && (
            <span className="text-sm text-white/60">{selectedInterval === 'lifetime' ? 'once' : `/${selectedInterval}`}</span>
          )}
        </div>

        {/* Interval Selector */}
        {!isFree && (
          <div className="mt-4 flex gap-2">
            {(['monthly', 'yearly', 'lifetime'] as const).map((interval) => (
              <button
                key={interval}
                onClick={() => setSelectedInterval(interval)}
                className={`flex-1 rounded px-3 py-2 text-xs font-medium transition-colors ${
                  selectedInterval === interval
                    ? 'bg-white text-black'
                    : 'bg-white/10 text-white/60 hover:bg-white/20'
                }`}
              >
                {interval === 'yearly' ? 'Year' : interval.charAt(0).toUpperCase() + interval.slice(1)}
              </button>
            ))}
          </div>
        )}

        {/* Features List */}
        <ul className="mt-6 space-y-2">
          {tier.features.map((feature) => (
            <li key={feature} className="flex items-start gap-3 text-sm text-white/80">
              <span className="mt-0.5 h-1.5 w-1.5 rounded-full bg-white/40 flex-shrink-0" />
              {feature}
            </li>
          ))}
        </ul>

        {/* CTA Button */}
        <button
          onClick={handleCheckout}
          disabled={isLoading}
          className={`mt-6 w-full rounded-lg py-3 font-semibold transition-all ${
            isFree
              ? 'bg-white/10 text-white hover:bg-white/20 disabled:opacity-50'
              : 'bg-white text-black hover:bg-white/90 disabled:opacity-50'
          }`}
        >
          {isLoading ? (
            <span className="flex items-center justify-center gap-2">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
              Loading...
            </span>
          ) : isFree ? (
            'Get Started'
          ) : (
            'Subscribe Now'
          )}
        </button>

        {/* Affiliate Earnings */}
        {affiliateEarnings > 0 && (
          <div className="mt-4 rounded-lg bg-white/5 p-3 text-center text-xs text-white/60">
            You'd earn <strong className="text-white">{formatPrice(affiliateEarnings, 'monthly')}</strong> per conversion
          </div>
        )}
      </div>
    </div>
  )
}
