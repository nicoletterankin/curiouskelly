'use client'

import React from "react"

/**
 * KellyOS Visual Aid Overlay
 * Apple-quality floating visual aid panel with drag, zoom, and polish
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel } from './index'
import { 
  X, 
  Minus, 
  Maximize2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Download,
  ChevronLeft,
  ChevronRight,
  Fullscreen,
  ImageIcon as ImageIcon,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { LessonPhase } from '@/lib/types'

interface VisualAidOverlayProps {
  isOpen: boolean
  onClose: () => void
  dayOfYear: number
  phase: LessonPhase
  lessonTitle?: string
}

// Consistent phase labels across the app
const PHASE_LABELS: Record<LessonPhase, string> = {
  hook: 'Hook',
  story: 'Story',
  wonder: 'Wonder',
  action: 'Action',
  wisdom: 'Wisdom',
}

export function VisualAidOverlay({
  isOpen,
  onClose,
  dayOfYear,
  phase,
  lessonTitle,
}: VisualAidOverlayProps) {
  const [zoom, setZoom] = useState(1)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [imageError, setImageError] = useState(false)
  
  // Generate image paths - check for multiple images per phase
  const getImagePath = useCallback((phaseType: LessonPhase, day: number, index: number = 0) => {
    const paddedDay = day.toString().padStart(3, '0')
    if (index === 0) return `/visuals/${phaseType}/day-${paddedDay}.jpg`
    return `/visuals/${phaseType}/day-${paddedDay}-${index + 1}.jpg`
  }, [])

  const currentImagePath = getImagePath(phase, dayOfYear, currentIndex)
  
  // For now, assume 1 image per phase (can expand later)
  const totalImages = 1

  const handleZoomIn = () => setZoom(z => Math.min(3, z + 0.5))
  const handleZoomOut = () => setZoom(z => Math.max(0.5, z - 0.5))
  const handleReset = () => {
    setZoom(1)
    setCurrentIndex(0)
  }

  useEffect(() => {
    setImageError(false)
    setZoom(1)
    setCurrentIndex(0)
  }, [phase, dayOfYear])

  if (!isOpen) return null

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose} side="right">
        <OverlayPanel
        title={`${PHASE_LABELS[phase]} Visual`}
        subtitle={lessonTitle || "Today's Lesson"}
        onClose={onClose}
        size={isFullscreen ? 'full' : 'lg'}
      >
        <div className="flex flex-col h-full">
          {/* Toolbar */}
          <div className="flex items-center justify-between px-4 py-2 border-b bg-muted/30">
            <div className="flex items-center gap-1">
              {/* Zoom controls */}
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={handleZoomOut}
                disabled={zoom <= 0.5}
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="text-xs text-muted-foreground w-14 text-center">
                {Math.round(zoom * 100)}%
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={handleZoomIn}
                disabled={zoom >= 3}
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
              
              <div className="w-px h-5 bg-border mx-2" />
              
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={handleReset}
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>

            <div className="flex items-center gap-1">
              {/* Navigation for multiple images */}
              {totalImages > 1 && (
                <>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setCurrentIndex(i => Math.max(0, i - 1))}
                    disabled={currentIndex === 0}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <span className="text-xs text-muted-foreground px-2">
                    {currentIndex + 1} / {totalImages}
                  </span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setCurrentIndex(i => Math.min(totalImages - 1, i + 1))}
                    disabled={currentIndex === totalImages - 1}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                  
                  <div className="w-px h-5 bg-border mx-2" />
                </>
              )}

              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setIsFullscreen(!isFullscreen)}
              >
                <Fullscreen className="h-4 w-4" />
              </Button>
              
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => {
                  const link = document.createElement('a')
                  link.href = currentImagePath
                  link.download = `kelly-${phase}-day${dayOfYear}.jpg`
                  link.click()
                }}
              >
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Image */}
          <div className="flex-1 overflow-hidden bg-black/5 flex items-center justify-center p-4">
            <div 
              className="transition-transform duration-200 ease-out"
              style={{ transform: `scale(${zoom})` }}
            >
              {imageError ? (
                <div className="w-full max-w-lg aspect-video rounded-2xl bg-muted/50 flex flex-col items-center justify-center text-muted-foreground">
                  <ImageIcon className="h-16 w-16 mb-4 opacity-50" />
                  <p className="font-medium">Visual not available</p>
                  <p className="text-sm">Image for this phase hasn't been generated yet</p>
                </div>
              ) : (
                <img
                  src={currentImagePath || "/placeholder.svg"}
                  alt={`${PHASE_LABELS[phase]} visual for day ${dayOfYear}`}
                  className="max-w-full max-h-[60vh] rounded-2xl shadow-2xl object-contain"
                  onError={() => setImageError(true)}
                />
              )}
            </div>
          </div>

          {/* Phase indicators */}
          <div className="flex justify-center gap-2 p-4 border-t bg-muted/30">
            {(['hook', 'story', 'wonder', 'action', 'wisdom'] as LessonPhase[]).map((p) => (
              <button
                key={p}
                className={cn(
                  'px-3 py-1.5 rounded-full text-xs font-medium transition-all',
                  p === phase
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                )}
                onClick={() => {
                  // This would need to be hooked up to a phase change handler
                }}
              >
                {PHASE_LABELS[p]}
              </button>
            ))}
          </div>
        </div>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}

// Minimized floating version for non-modal usage
export function VisualAidFloating({
  isVisible,
  onClose,
  onMaximize,
  dayOfYear,
  phase,
}: {
  isVisible: boolean
  onClose: () => void
  onMaximize: () => void
  dayOfYear: number
  phase: LessonPhase
}) {
  const [isMinimized, setIsMinimized] = useState(false)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const dragRef = useRef<{ startX: number; startY: number; posX: number; posY: number } | null>(null)

  const imagePath = `/visuals/${phase}/day-${dayOfYear.toString().padStart(3, '0')}.jpg`

  // Drag handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true)
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      posX: position.x,
      posY: position.y,
    }
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !dragRef.current) return
      setPosition({
        x: dragRef.current.posX + (e.clientX - dragRef.current.startX),
        y: dragRef.current.posY + (e.clientY - dragRef.current.startY),
      })
    }

    const handleMouseUp = () => {
      setIsDragging(false)
      dragRef.current = null
    }

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging])

  if (!isVisible) return null

  /**
   * KELLY RULE + STATE-AWARE POSITIONING
   * 
   * Position: bottom-right, ABOVE the caption bar (bottom-44 = ~176px)
   * This ensures no overlap with:
   * - Caption bar (bottom ~100px)
   * - Lesson metadata (bottom-left)
   * - Kelly (center)
   * 
   * Design: Match the overlay system styling exactly
   * - Same border-radius (rounded-2xl = 16px)
   * - Same shadow style
   * - Same header treatment
   */
  
  if (isMinimized) {
    return (
      <div 
        className="fixed bottom-44 right-4 z-40 flex items-center gap-3 px-4 py-2.5 rounded-2xl shadow-lg"
        style={{ 
          transform: `translate(${position.x}px, ${position.y}px)`,
          background: 'oklch(0.98 0.005 250)',
          boxShadow: '0 0 0 1px oklch(0.90 0.01 250 / 0.5), 0 8px 24px -4px oklch(0.15 0.03 250 / 0.15)',
        }}
      >
        <ImageIcon className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">Visual</span>
        <Button variant="ghost" size="icon" className="h-7 w-7 rounded-lg hover:bg-muted" onClick={() => setIsMinimized(false)}>
          <Maximize2 className="h-3.5 w-3.5" />
        </Button>
        <Button variant="ghost" size="icon" className="h-7 w-7 rounded-lg hover:bg-muted" onClick={onClose}>
          <X className="h-3.5 w-3.5" />
        </Button>
      </div>
    )
  }

  return (
    <div 
      className="fixed bottom-44 right-4 z-40 w-[320px] rounded-2xl overflow-hidden"
      style={{ 
        transform: `translate(${position.x}px, ${position.y}px)`,
        background: 'oklch(0.98 0.005 250)',
        boxShadow: '0 0 0 1px oklch(0.90 0.01 250 / 0.5), 0 24px 48px -12px oklch(0.15 0.03 250 / 0.25), 0 12px 24px -8px oklch(0.15 0.03 250 / 0.15)',
      }}
    >
      {/* Header - consistent with other overlays */}
      <div
        className={cn(
          'flex items-center justify-between px-4 py-3 border-b border-border/50',
          'cursor-grab active:cursor-grabbing'
        )}
        onMouseDown={handleMouseDown}
      >
        <span className="text-sm font-semibold text-foreground">Today's Visual</span>
        <div className="flex items-center gap-0.5">
          <Button variant="ghost" size="icon" className="h-7 w-7 rounded-lg hover:bg-muted" onClick={onMaximize}>
            <Maximize2 className="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="icon" className="h-7 w-7 rounded-lg hover:bg-muted" onClick={() => setIsMinimized(true)}>
            <Minus className="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="icon" className="h-7 w-7 rounded-lg hover:bg-muted" onClick={onClose}>
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Image - clean with subtle background */}
      <div className="p-3">
        <div className="rounded-xl overflow-hidden bg-muted/30">
          <img
            src={imagePath || "/placeholder.svg"}
            alt={`${phase} visual`}
            className="w-full aspect-square object-contain"
            onError={(e) => {
              const target = e.target as HTMLImageElement
              target.src = '/images/visual-placeholder.jpg'
            }}
          />
        </div>
      </div>
    </div>
  )
}
