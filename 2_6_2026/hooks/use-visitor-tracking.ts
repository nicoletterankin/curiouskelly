'use client'

import { useEffect, useCallback, useRef } from 'react'

// Get or create visitor ID
function getVisitorId(): string {
  if (typeof window === 'undefined') return ''
  
  let visitorId = localStorage.getItem('kelly_visitor_id')
  if (!visitorId) {
    visitorId = crypto.randomUUID()
    localStorage.setItem('kelly_visitor_id', visitorId)
  }
  return visitorId
}

// Track session on page load
export function useVisitorSession() {
  const initialized = useRef(false)
  
  useEffect(() => {
    if (initialized.current) return
    initialized.current = true
    
    const visitorId = getVisitorId()
    if (!visitorId) return
    
    // Fire and forget - never await, never block
    fetch('/api/visitor/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        visitor_id: visitorId,
        referrer: document.referrer || null,
        landing_page: window.location.pathname,
        user_agent: navigator.userAgent,
        timestamp: new Date().toISOString()
      })
    }).catch(() => {}) // Silently fail
  }, [])
}

// Track events
export function useTrackEvent() {
  return useCallback((eventType: string, eventData?: Record<string, unknown>) => {
    const visitorId = getVisitorId()
    if (!visitorId) return
    
    // Fire and forget - never await, never block
    fetch('/api/visitor/event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        visitor_id: visitorId,
        event_type: eventType,
        event_data: eventData || {},
        page: typeof window !== 'undefined' ? window.location.pathname : '/',
        timestamp: new Date().toISOString()
      })
    }).catch(() => {}) // Silently fail
  }, [])
}

// Combined hook for components that need both
export function useVisitorTracking() {
  useVisitorSession()
  const trackEvent = useTrackEvent()
  return { trackEvent }
}
