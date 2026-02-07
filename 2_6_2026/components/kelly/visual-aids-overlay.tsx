'use client'

import React from "react"

import { useState, useCallback, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { 
  X, 
  Minus, 
  Maximize2, 
  Minimize2,
  Move,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  ImageIcon,
  Loader2
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { LessonPhase } from '@/lib/types'
import { useVisualAids } from '@/hooks/use-visual-aids'

interface VisualAidsOverlayProps {
  dayOfYear: number
  phase: LessonPhase
  isVisible: boolean
  onClose: () => void
  className?: string
}

type OverlayState = 'full' | 'minimized' | 'hidden'

export function VisualAidsOverlay({
  dayOfYear,
  phase,
  isVisible,
  onClose,
  className,
}: VisualAidsOverlayProps) {
  const [state, setState] = useState<OverlayState>('full')
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [imageIndex, setImageIndex] = useState(0)
  const [imageError, setImageError] = useState(false)
  const dragRef = useRef<{ startX: number; startY: number; startPosX: number; startPosY: number } | null>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const [visualPaths, setVisualPaths] = useState<string[]>([]); // Declare visualPaths variable

  // Fetch visual aids from database
  const { visuals, visualUrl, hasVisuals, isLoading, error } = useVisualAids(dayOfYear, phase)

  // Filter visuals for current phase
  const phaseVisuals = visuals.filter(v => v.phase === phase)
  const currentVisual = phaseVisuals[imageIndex]?.url || visualUrl

  // Handle dragging
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (state === 'minimized') return
    setIsDragging(true)
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startPosX: position.x,
      startPosY: position.y,
    }
  }, [state, position])

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !dragRef.current) return
      const dx = e.clientX - dragRef.current.startX
      const dy = e.clientY - dragRef.current.startY
      setPosition({
        x: dragRef.current.startPosX + dx,
        y: dragRef.current.startPosY + dy,
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

  // Reset position when phase changes
  useEffect(() => {
    setPosition({ x: 0, y: 0 })
    setZoom(1)
    setImageIndex(0)
    setImageError(false)
  }, [phase, dayOfYear])

  if (!isVisible || state === 'hidden') return null

  // Minimized state - small pill at bottom
  if (state === 'minimized') {
    return (
      <div 
        className={cn(
          'fixed bottom-4 left-1/2 -translate-x-1/2 z-40',
          'bg-background/95 backdrop-blur-md border rounded-full',
          'px-4 py-2 flex items-center gap-3 shadow-lg',
          'transition-all duration-300',
          className
        )}
      >
        <span className="text-sm font-medium">Visual Aid</span>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setState('full')}
        >
          <Maximize2 className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={onClose}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  return (
    <div
      ref={overlayRef}
      style={{
        transform: `translate(calc(-50% + ${position.x}px), ${position.y}px)`,
      }}
      className={cn(
        'fixed bottom-8 left-1/2 z-40',
        'w-[90%] max-w-2xl',
        'bg-background/95 backdrop-blur-md border rounded-2xl',
        'shadow-2xl overflow-hidden',
        'transition-all duration-300',
        isDragging && 'cursor-grabbing',
        className
      )}
    >
      {/* Header - Draggable */}
      <div
        className={cn(
          'flex items-center justify-between px-4 py-2 border-b',
          'bg-muted/50 cursor-grab',
          isDragging && 'cursor-grabbing'
        )}
        onMouseDown={handleMouseDown}
      >
        <div className="flex items-center gap-2">
          <Move className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium capitalize">{phase} Visual</span>
        </div>
        
        <div className="flex items-center gap-1">
          {/* Zoom controls */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
            disabled={zoom <= 0.5}
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-xs text-muted-foreground w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setZoom(z => Math.min(2, z + 0.25))}
            disabled={zoom >= 2}
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          
          <div className="w-px h-4 bg-border mx-1" />
          
          {/* Window controls */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setState('minimized')}
          >
            <Minus className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Image Content */}
      <div className="relative overflow-hidden">
        <div 
          className="aspect-video flex items-center justify-center bg-black/5"
          style={{ 
            transform: `scale(${zoom})`,
            transformOrigin: 'center center',
          }}
        >
          {isLoading ? (
            <div className="flex flex-col items-center justify-center text-muted-foreground">
              <Loader2 className="h-8 w-8 animate-spin mb-2" />
              <span className="text-sm">Loading visual...</span>
            </div>
          ) : !hasVisuals || imageError || !currentVisual ? (
            <div className="flex flex-col items-center justify-center text-muted-foreground p-8">
              <ImageIcon className="h-12 w-12 mb-3 opacity-50" />
              <span className="text-sm font-medium">No visual available</span>
              <span className="text-xs text-muted-foreground/70 mt-1">
                Visual for Day {dayOfYear} - {phase} not yet generated
              </span>
            </div>
          ) : (
            <img
              src={currentVisual || "/placeholder.svg"}
              alt={`${phase} visual aid for day ${dayOfYear}`}
              className="w-full h-full object-contain"
              onError={() => setImageError(true)}
            />
          )}
        </div>

        {/* Navigation arrows for multiple images */}
        {phaseVisuals.length > 1 && (
          <>
            <Button
              variant="ghost"
              size="icon"
              className="absolute left-2 top-1/2 -translate-y-1/2 bg-background/80 hover:bg-background"
              onClick={() => setImageIndex(i => Math.max(0, i - 1))}
              disabled={imageIndex === 0}
            >
              <ChevronLeft className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-background/80 hover:bg-background"
              onClick={() => setImageIndex(i => Math.min(phaseVisuals.length - 1, i + 1))}
              disabled={imageIndex === phaseVisuals.length - 1}
            >
              <ChevronRight className="h-5 w-5" />
            </Button>
            
            {/* Image indicators */}
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-1">
              {phaseVisuals.map((_, i) => (
                <button
                  key={i}
                  className={cn(
                    'w-2 h-2 rounded-full transition-colors',
                    i === imageIndex ? 'bg-primary' : 'bg-muted-foreground/50'
                  )}
                  onClick={() => setImageIndex(i)}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
