'use client'

import { useMemo } from 'react'
import { cn } from '@/lib/utils'

interface CaptionBarProps {
  script: string
  currentTime: number
  isPlaying: boolean
  className?: string
}

// Simple script-to-captions converter
// In production, this would use actual timing data from the video
function generateCaptions(script: string): { text: string; startTime: number; endTime: number }[] {
  if (!script) return []

  // Split by sentences
  const sentences = script.match(/[^.!?]+[.!?]+/g) || [script]
  const avgDuration = 3 // seconds per sentence estimate
  
  return sentences.map((sentence, index) => ({
    text: sentence.trim(),
    startTime: index * avgDuration,
    endTime: (index + 1) * avgDuration,
  }))
}

export function CaptionBar({
  script,
  currentTime,
  isPlaying,
  className,
}: CaptionBarProps) {
  // Generate captions from script
  const captions = useMemo(() => generateCaptions(script), [script])

  // Find current caption
  const currentCaption = useMemo(() => {
    return captions.find(
      (c) => currentTime >= c.startTime && currentTime < c.endTime
    )
  }, [captions, currentTime])

  if (!currentCaption && !isPlaying) {
    return null
  }

  return (
    <div
      className={cn(
        'bg-muted/80 backdrop-blur-sm rounded-lg px-4 py-3 min-h-[60px] flex items-center justify-center',
        className
      )}
    >
      {currentCaption ? (
        <p className="text-center text-base md:text-lg font-medium leading-relaxed">
          {currentCaption.text}
        </p>
      ) : (
        <p className="text-center text-muted-foreground">
          {isPlaying ? 'Listening...' : 'Tap play to start'}
        </p>
      )}
    </div>
  )
}

// Alternative: Animated caption that highlights words
export function AnimatedCaptionBar({
  script,
  currentTime,
  isPlaying,
  className,
}: CaptionBarProps) {
  const words = useMemo(() => {
    if (!script) return []
    return script.split(/\s+/).map((word, index) => ({
      word,
      startTime: index * 0.3, // Rough timing
      endTime: (index + 1) * 0.3,
    }))
  }, [script])

  return (
    <div
      className={cn(
        'bg-muted/80 backdrop-blur-sm rounded-lg px-4 py-3 min-h-[60px] flex items-center justify-center',
        className
      )}
    >
      <p className="text-center text-base md:text-lg leading-relaxed">
        {words.map((w, i) => {
          const isActive = currentTime >= w.startTime && currentTime < w.endTime
          const isPast = currentTime >= w.endTime

          return (
            <span
              key={i}
              className={cn(
                'transition-colors duration-150',
                isActive
                  ? 'text-primary font-semibold'
                  : isPast
                  ? 'text-foreground'
                  : 'text-muted-foreground'
              )}
            >
              {w.word}{' '}
            </span>
          )
        })}
      </p>
    </div>
  )
}
