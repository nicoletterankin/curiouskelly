'use client'

/**
 * Affiliate Tracking Hook
 * 
 * Automatically tracks referrals when visitors land with ?ref=CODE
 * Sets lifetime cookie (10 years) so the referrer earns forever
 * 
 * LEARN, SHARE, EARN model:
 * - Every learner gets an affiliate code automatically
 * - Share links include affiliate code
 * - Lifetime attribution - first touch wins forever
 */

import { useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { getFingerprint } from '@/lib/fingerprint'

interface AffiliateTrackingState {
  tracked: boolean
  affiliateCode: string | null
  isLoading: boolean
  error: string | null
}

export function useAffiliateTracking() {
  const searchParams = useSearchParams()
  const [state, setState] = useState<AffiliateTrackingState>({
    tracked: false,
    affiliateCode: null,
    isLoading: true,
    error: null,
  })

  useEffect(() => {
    const refCode = searchParams.get('ref')
    
    if (!refCode) {
      // No ref code in URL, check if we have existing cookie
      checkExistingReferral()
      return
    }

    // Track the referral
    trackReferral(refCode)
  }, [searchParams])

  const checkExistingReferral = async () => {
    try {
      const res = await fetch('/api/affiliate/track')
      const data = await res.json()
      
      setState({
        tracked: data.hasReferral || false,
        affiliateCode: data.affiliateCode || null,
        isLoading: false,
        error: null,
      })
    } catch (e) {
      setState(prev => ({ ...prev, isLoading: false }))
    }
  }

  const trackReferral = async (refCode: string) => {
    try {
      const visitorId = getFingerprint()
      
      const res = await fetch('/api/affiliate/track', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refCode, visitorId }),
      })
      
      const data = await res.json()
      
      if (res.ok) {
        setState({
          tracked: true,
          affiliateCode: data.affiliateCode || refCode.toUpperCase(),
          isLoading: false,
          error: null,
        })
        
        // Clean the URL (remove ref param) without triggering navigation
        if (typeof window !== 'undefined') {
          const url = new URL(window.location.href)
          url.searchParams.delete('ref')
          window.history.replaceState({}, '', url.toString())
        }
      } else {
        setState({
          tracked: false,
          affiliateCode: null,
          isLoading: false,
          error: data.error || 'Invalid referral code',
        })
      }
    } catch (e) {
      setState({
        tracked: false,
        affiliateCode: null,
        isLoading: false,
        error: 'Failed to track referral',
      })
    }
  }

  return state
}
