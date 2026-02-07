'use client'

import { useVisitorSession } from '@/hooks/use-visitor-tracking'

// Invisible component that tracks visitor sessions
// Add to root layout to track all visitors
export function VisitorTracker() {
  useVisitorSession()
  return null
}
