'use client'

import React, { type ReactNode, type CSSProperties } from 'react'
import { useLayout, type ZoneName } from '@/lib/layout-context'
import { cn } from '@/lib/utils'

interface LayoutGridProps {
  children: ReactNode
  className?: string
}

/**
 * LayoutGrid - The master container that defines non-overlapping zones
 * 
 * Uses CSS Grid with viewport-aware zone sizing.
 * All positioning is handled by the grid - no child should use position: fixed.
 * Each zone clips its content to prevent overflow.
 */
export function LayoutGrid({ children, className }: LayoutGridProps) {
  const { zones, viewport, isMobile } = useLayout()
  
  // CSS custom properties for zone sizes
  const gridStyles: CSSProperties = {
    '--zone-left-width': `${zones.leftWidth}px`,
    '--zone-right-width': `${zones.rightWidth}px`,
    '--zone-top-height': `${zones.topHeight}px`,
    '--zone-bottom-height': `${zones.bottomHeight}px`,
    '--viewport-width': `${viewport.width}px`,
    '--viewport-height': `${viewport.height}px`,
  } as CSSProperties
  
  return (
    <div 
      className={cn('layout-grid', className)}
      style={gridStyles}
    >
      {children}
    </div>
  )
}

interface ZoneProps {
  name: ZoneName
  children: ReactNode
  className?: string
  layer?: 'front' | 'back' // back = behind other zones (for Kelly video)
}

/**
 * Zone - A container that renders content into a specific grid area
 * 
 * Content is automatically clipped to prevent overflow.
 * Use layer="back" for background content like the Kelly video.
 */
export function Zone({ name, children, className, layer = 'front' }: ZoneProps) {
  const { zones, isMobile } = useLayout()
  
  // Zone-specific styles
  const zoneStyles: Record<ZoneName, string> = {
    'top-left': 'flex flex-col gap-2 p-2 overflow-hidden',
    'left': 'flex flex-col gap-2 p-2 overflow-y-auto overflow-x-hidden scrollbar-hide',
    'top-right': 'flex flex-col items-end p-2 overflow-hidden',
    'right': 'flex flex-col items-center justify-center gap-2 py-4 overflow-hidden',
    'bottom': 'flex flex-col items-center justify-end overflow-hidden',
    'center': 'overflow-hidden',
  }
  
  return (
    <div
      data-zone={name}
      data-layer={layer}
      className={cn(
        'zone',
        `zone-${name}`,
        zoneStyles[name],
        layer === 'back' && 'zone-background',
        className
      )}
    >
      {children}
    </div>
  )
}

// Convenience component for scrollable content within a zone
interface ZoneScrollAreaProps {
  children: ReactNode
  className?: string
  maxHeight?: number | string
}

export function ZoneScrollArea({ children, className, maxHeight }: ZoneScrollAreaProps) {
  return (
    <div 
      className={cn('overflow-y-auto overflow-x-hidden scrollbar-hide flex-1 min-h-0', className)}
      style={maxHeight ? { maxHeight } : undefined}
    >
      {children}
    </div>
  )
}

// Export everything
LayoutGrid.Zone = Zone
LayoutGrid.ScrollArea = ZoneScrollArea
