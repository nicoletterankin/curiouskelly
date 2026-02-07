'use client'

import React, { createContext, useContext, useState, useEffect, useMemo, useCallback, type ReactNode } from 'react'

// Zone configuration - defines the layout regions
export type ZoneName = 
  | 'top-left'    // Presence badge, mode toggle
  | 'left'        // Share, live class, comments
  | 'top-right'   // Video engine thumbnails
  | 'right'       // Action buttons (like, comment, save, playback)
  | 'bottom'      // Caption, voting, CTA, day picker, title
  | 'center'      // Kelly video (background layer)

interface ZoneSizes {
  leftWidth: number
  rightWidth: number
  topHeight: number
  bottomHeight: number
  // Computed safe areas (what's left for center)
  centerTop: number
  centerBottom: number
  centerLeft: number
  centerRight: number
}

interface LayoutContextValue {
  viewport: { width: number; height: number }
  zones: ZoneSizes
  isMobile: boolean
  isTablet: boolean
  isDesktop: boolean
}

const LayoutContext = createContext<LayoutContextValue | null>(null)

export function useLayout() {
  const context = useContext(LayoutContext)
  if (!context) {
    throw new Error('useLayout must be used within LayoutProvider')
  }
  return context
}

// Calculate zone sizes based on viewport
// CRITICAL: Kelly's EYES must remain visible in the safe zone (middle 30% of screen)
// Mobile safe zone: ~25% from top to ~65% from top (where caption bar starts)
function calculateZones(width: number, height: number): ZoneSizes {
  // Mobile-first: COMPACT overlays to maximize Kelly face visibility
  if (width < 640) {
    const leftWidth = Math.min(200, width * 0.45) // Narrower to not block Kelly
    const rightWidth = 52 // Compact buttons w-11 (44px) + padding
    const topHeight = 100 // REDUCED - keep overlays compact
    // Bottom caption bar - SHORTER to show more Kelly
    const bottomHeight = Math.max(Math.min(height * 0.38, 380), 300)
    
    return {
      leftWidth,
      rightWidth,
      topHeight,
      bottomHeight,
      centerTop: topHeight,
      centerBottom: bottomHeight,
      centerLeft: 0, // On mobile, center extends full width
      centerRight: 0,
    }
  }
  
  // Tablet: balanced zones
  if (width < 1024) {
    const leftWidth = Math.min(280, width * 0.32) // Wider to prevent text wrapping
    const rightWidth = 64 // Compact on tablet - 2x2 grid in overlay only
    const topHeight = 180
    // Tablet: 42% of screen or 420px
    const bottomHeight = Math.max(Math.min(height * 0.42, 440), 360)
    
    return {
      leftWidth,
      rightWidth,
      topHeight,
      bottomHeight,
      centerTop: topHeight,
      centerBottom: bottomHeight,
      centerLeft: leftWidth,
      centerRight: rightWidth,
    }
  }
  
  // Desktop: full experience
  const leftWidth = Math.min(340, width * 0.26) // Wider to prevent text wrapping
  const rightWidth = 72 // Compact - 2x2 grid in overlay only
  const topHeight = 180
  // Desktop: 38% of screen or 400px
  const bottomHeight = Math.max(Math.min(height * 0.38, 420), 340)
  
  return {
    leftWidth,
    rightWidth,
    topHeight,
    bottomHeight,
    centerTop: topHeight,
    centerBottom: bottomHeight,
    centerLeft: leftWidth,
    centerRight: rightWidth,
  }
}

interface LayoutProviderProps {
  children: ReactNode
}

export function LayoutProvider({ children }: LayoutProviderProps) {
  const [viewport, setViewport] = useState({ width: 0, height: 0 })
  
  useEffect(() => {
    const update = () => {
      setViewport({
        width: window.innerWidth,
        height: window.innerHeight,
      })
    }
    
    update()
    window.addEventListener('resize', update)
    window.addEventListener('orientationchange', update)
    
    return () => {
      window.removeEventListener('resize', update)
      window.removeEventListener('orientationchange', update)
    }
  }, [])
  
  const value = useMemo<LayoutContextValue>(() => {
    const { width, height } = viewport
    const zones = calculateZones(width, height)
    
    return {
      viewport,
      zones,
      isMobile: width < 640,
      isTablet: width >= 640 && width < 1024,
      isDesktop: width >= 1024,
    }
  }, [viewport])
  
  return (
    <LayoutContext.Provider value={value}>
      {children}
    </LayoutContext.Provider>
  )
}
