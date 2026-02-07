'use client'

import React, { createContext, useContext, useState, useCallback, useMemo, type ReactNode } from 'react'

/**
 * KELLY RULE - Layout Context
 * 
 * All UI elements are state-aware and know what else is open.
 * They position themselves to NEVER overlap Kelly (center) or each other.
 * 
 * Layout Zones:
 * - TOP: Phase tabs, header
 * - LEFT: Control panel, settings selectors
 * - RIGHT: Action buttons, overlays
 * - BOTTOM-LEFT: Lesson metadata, captions
 * - BOTTOM-RIGHT: Visual aids, floating panels
 */

type LayoutZone = 'top' | 'left' | 'right' | 'bottom-left' | 'bottom-right' | 'center'

interface LayoutElement {
  id: string
  zone: LayoutZone
  priority: number // Higher priority elements push others
  height?: number
  width?: number
}

interface LayoutContextType {
  // What's currently open/visible
  activeElements: Map<string, LayoutElement>
  
  // Register/unregister elements
  registerElement: (element: LayoutElement) => void
  unregisterElement: (id: string) => void
  
  // Check if a zone is occupied
  isZoneOccupied: (zone: LayoutZone) => boolean
  getZoneElements: (zone: LayoutZone) => LayoutElement[]
  
  // Get safe position for an element based on what else is open
  getSafePosition: (zone: LayoutZone, preferredOffset?: { x: number; y: number }) => { x: number; y: number }
  
  // Global overlay state
  hasActiveOverlay: boolean
  activeOverlayZone: LayoutZone | null
}

const LayoutContext = createContext<LayoutContextType | null>(null)

export function LayoutProvider({ children }: { children: ReactNode }) {
  const [activeElements, setActiveElements] = useState<Map<string, LayoutElement>>(new Map())

  const registerElement = useCallback((element: LayoutElement) => {
    setActiveElements(prev => {
      const next = new Map(prev)
      next.set(element.id, element)
      return next
    })
  }, [])

  const unregisterElement = useCallback((id: string) => {
    setActiveElements(prev => {
      const next = new Map(prev)
      next.delete(id)
      return next
    })
  }, [])

  const isZoneOccupied = useCallback((zone: LayoutZone): boolean => {
    return Array.from(activeElements.values()).some(el => el.zone === zone)
  }, [activeElements])

  const getZoneElements = useCallback((zone: LayoutZone): LayoutElement[] => {
    return Array.from(activeElements.values())
      .filter(el => el.zone === zone)
      .sort((a, b) => b.priority - a.priority)
  }, [activeElements])

  const getSafePosition = useCallback((zone: LayoutZone, preferredOffset = { x: 0, y: 0 }): { x: number; y: number } => {
    const zoneElements = getZoneElements(zone)
    
    // Calculate offset based on existing elements in the zone
    let totalOffset = 0
    for (const el of zoneElements) {
      if (zone === 'right' || zone === 'bottom-right') {
        totalOffset += (el.width || 380) + 16 // Add gap
      } else if (zone === 'bottom-left') {
        totalOffset += (el.height || 100) + 16
      }
    }
    
    // Return safe position based on zone
    switch (zone) {
      case 'right':
        return { x: preferredOffset.x - totalOffset, y: preferredOffset.y }
      case 'bottom-right':
        return { x: preferredOffset.x, y: preferredOffset.y - totalOffset }
      case 'bottom-left':
        return { x: preferredOffset.x, y: preferredOffset.y - totalOffset }
      default:
        return preferredOffset
    }
  }, [getZoneElements])

  const hasActiveOverlay = useMemo(() => {
    return Array.from(activeElements.values()).some(el => 
      el.zone === 'right' && el.priority >= 100
    )
  }, [activeElements])

  const activeOverlayZone = useMemo((): LayoutZone | null => {
    const overlays = Array.from(activeElements.values())
      .filter(el => el.priority >= 100)
      .sort((a, b) => b.priority - a.priority)
    return overlays[0]?.zone || null
  }, [activeElements])

  const value: LayoutContextType = {
    activeElements,
    registerElement,
    unregisterElement,
    isZoneOccupied,
    getZoneElements,
    getSafePosition,
    hasActiveOverlay,
    activeOverlayZone,
  }

  return (
    <LayoutContext.Provider value={value}>
      {children}
    </LayoutContext.Provider>
  )
}

export function useLayout() {
  const context = useContext(LayoutContext)
  if (!context) {
    throw new Error('useLayout must be used within a LayoutProvider')
  }
  return context
}

// Hook for registering a layout element
export function useLayoutElement(element: LayoutElement | null) {
  const { registerElement, unregisterElement } = useLayout()
  
  React.useEffect(() => {
    if (element) {
      registerElement(element)
      return () => unregisterElement(element.id)
    }
  }, [element, registerElement, unregisterElement])
}
