'use client'

import { useState, useCallback, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { SUBSCRIPTION_TIERS } from '@/lib/products'
import { TierCard } from '@/components/pricing/tier-card'
import { FAQSection } from '@/components/pricing/faq-section'
import Checkout from '@/components/checkout'
import { sanitizeRefCode } from '@/lib/stripe-helpers'

export default function PricingPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [selectedTier, setSelectedTier] = useState<string | null>(null)
  const [selectedInterval, setSelectedInterval] = useState<'monthly' | 'yearly' | 'lifetime'>('monthly')
  const [isLoading, setIsLoading] = useState(false)
  const [affiliateName, setAffiliateN] = useState<string | null>(null)

  // Get ref code from URL
  const refCode = useCallback(() => {
    const raw = searchParams.get('ref')
    return sanitizeRefCode(raw)
  }, [searchParams])

  useEffect(() => {
    // TODO: Fetch affiliate name from API using ref code
    const code = refCode()
    if (code) {
      setAffiliateN(`Affiliate: ${code}`)
    }
  }, [refCode])

  const handleCheckoutClick = useCallback(
    async (tierId: string, interval: 'monthly' | 'yearly' | 'lifetime') => {
      setSelectedTier(tierId)
      setSelectedInterval(interval)
      setIsLoading(true)
    },
    []
  )

  const closeCheckout = useCallback(() => {
    setSelectedTier(null)
    setIsLoading(false)
  }, [])

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <div className="border-b border-white/10 px-6 py-16">
        <div className="mx-auto max-w-6xl">
          <h1 className="text-5xl font-bold text-balance">Choose Your Learning Path</h1>
          <p className="mt-4 text-xl text-white/60">
            Start free, upgrade anytime. All plans include lifetime access to downloaded lessons.
          </p>
        </div>
      </div>

      {/* Pricing Grid */}
      <div className="px-6 py-16">
        <div className="mx-auto max-w-6xl">
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
            {SUBSCRIPTION_TIERS.map((tier) => (
              <TierCard
                key={tier.id}
                tier={tier}
                affiliateName={affiliateName || undefined}
                refCode={refCode() || undefined}
                onCheckoutClick={handleCheckoutClick}
                isLoading={isLoading && selectedTier === tier.id}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Checkout Modal */}
      {selectedTier && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-2xl rounded-lg bg-black border border-white/20 p-8">
            <button
              onClick={closeCheckout}
              className="absolute top-4 right-4 text-white/60 hover:text-white"
              aria-label="Close"
            >
              ✕
            </button>
            <Checkout tierId={selectedTier} interval={selectedInterval} refCode={refCode() || undefined} />
          </div>
        </div>
      )}

      {/* FAQ Section */}
      <FAQSection />

      {/* Affiliate CTA */}
      <div className="border-t border-white/10 px-6 py-16">
        <div className="mx-auto max-w-6xl">
          <div className="rounded-lg bg-white/5 p-8 text-center">
            <h2 className="text-3xl font-bold">Become an Affiliate</h2>
            <p className="mt-2 text-white/60">Earn 25-35% commission per referral</p>
            <button className="mt-6 rounded-lg bg-white px-8 py-3 font-semibold text-black hover:bg-white/90 transition-colors">
              Join Our Program
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
