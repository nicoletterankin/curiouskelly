// lib/affiliate-tracking.ts
// Affiliate referral tracking utilities

import { sanitizeRefCode } from './stripe-helpers'

/**
 * Parse ref code from URL query string
 */
export function parseRefCode(url: string | null): string | null {
  if (!url) return null
  try {
    const urlObj = new URL(url, 'http://localhost')
    const ref = urlObj.searchParams.get('ref')
    return ref ? sanitizeRefCode(ref) : null
  } catch {
    return null
  }
}

/**
 * Get ref code from search params
 */
export function getRefCodeFromSearchParams(searchParams: URLSearchParams): string | null {
  const ref = searchParams.get('ref')
  return ref ? sanitizeRefCode(ref) : null
}

/**
 * Validate affiliate ref code exists (would check DB in production)
 * For now, returns true if code is valid format
 */
export async function validateRefCode(code: string | null): Promise<boolean> {
  if (!code) return false
  const sanitized = sanitizeRefCode(code)
  // In production, query affiliate_programs table to verify code exists
  // SELECT EXISTS(SELECT 1 FROM affiliate_programs WHERE ref_code = $1 AND status = 'active')
  return sanitized !== null && sanitized.length > 0
}

/**
 * Track ref code click (event logging)
 * Used for analytics - when user lands on pricing page with ref code
 */
export async function trackRefCodeClick(refCode: string): Promise<void> {
  try {
    await fetch('/api/tracking/ref-click', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refCode, timestamp: new Date().toISOString() }),
    })
  } catch (error) {
    console.warn('[affiliate-tracking] Failed to track ref click:', error)
    // Don't throw - analytics failure shouldn't break the app
  }
}

/**
 * Get affiliate info by ref code
 * Returns name, commission rate, etc.
 */
export async function getAffiliateByRefCode(refCode: string): Promise<{
  id: string
  name: string
  commissionRate: number
} | null> {
  try {
    const res = await fetch(`/api/affiliate/info?code=${encodeURIComponent(refCode)}`)
    if (!res.ok) return null
    return res.json()
  } catch (error) {
    console.warn('[affiliate-tracking] Failed to fetch affiliate info:', error)
    return null
  }
}

/**
 * Build ref link for affiliate sharing
 */
export function buildRefLink(baseUrl: string, refCode: string): string {
  const url = new URL(baseUrl)
  url.searchParams.set('ref', refCode)
  return url.toString()
}
