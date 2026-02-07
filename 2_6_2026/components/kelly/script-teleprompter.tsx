'use client'

/**
 * KELLY SCRIPT TELEPROMPTER
 * Word-by-word script display synchronized with audio playback.
 * Each word highlights in sequence like a karaoke/teleprompter experience.
 * 
 * Features:
 * - Auto-advances words based on estimated speaking pace (~150 WPM)
 * - Smooth scroll to keep current word centered
 * - Phase-aware styling (hook/story/wonder/action/wisdom)
 * - Tap any word to look it up (dictionary integration)
 * - Pauses when audio pauses, resumes when audio resumes
 * - Collapsed/expanded modes for different viewing preferences
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { ChevronUp, ChevronDown, BookOpen, Pause, Play, RotateCcw } from 'lucide-react'
import type { LessonPhase } from '@/lib/types'

// Speaking pace: average conversational English is ~150 words per minute
// Kelly speaks slightly slower for clarity = ~130 WPM = ~462ms per word
const DEFAULT_MS_PER_WORD = 462

// Phase-specific styling
const PHASE_STYLES: Record<LessonPhase, {
  activeColor: string
  activeGlow: string
  labelColor: string
  bgGradient: string
}> = {
  hook: {
    activeColor: 'text-amber-300',
    activeGlow: '0 0 20px rgba(251,191,36,0.4)',
    labelColor: 'text-amber-400/80',
    bgGradient: 'from-amber-500/5 to-transparent',
  },
  story: {
    activeColor: 'text-blue-300',
    activeGlow: '0 0 20px rgba(96,165,250,0.4)',
    labelColor: 'text-blue-400/80',
    bgGradient: 'from-blue-500/5 to-transparent',
  },
  wonder: {
    activeColor: 'text-violet-300',
    activeGlow: '0 0 20px rgba(167,139,250,0.4)',
    labelColor: 'text-violet-400/80',
    bgGradient: 'from-violet-500/5 to-transparent',
  },
  action: {
    activeColor: 'text-emerald-300',
    activeGlow: '0 0 20px rgba(52,211,153,0.4)',
    labelColor: 'text-emerald-400/80',
    bgGradient: 'from-emerald-500/5 to-transparent',
  },
  wisdom: {
    activeColor: 'text-rose-300',
    activeGlow: '0 0 20px rgba(251,113,133,0.4)',
    labelColor: 'text-rose-400/80',
    bgGradient: 'from-rose-500/5 to-transparent',
  },
}

// Phase labels for the header
const PHASE_LABELS: Record<LessonPhase, string> = {
  hook: 'Hook',
  story: 'Story',
  wonder: 'Wonder',
  action: 'Action',
  wisdom: 'Wisdom',
}

interface ScriptTeleprompterProps {
  /** The full script text Kelly speaks for this phase */
  script: string
  /** Current lesson phase */
  phase: LessonPhase
  /** Whether audio/video is currently playing */
  isPlaying: boolean
  /** Callback when a word is tapped (for dictionary lookup) */
  onWordTap?: (word: string) => void
  /** Whether to start word animation automatically */
  autoStart?: boolean
  /** Custom speaking speed in ms per word */
  msPerWord?: number
  /** Whether the teleprompter is expanded/visible */
  isExpanded?: boolean
  /** Toggle expanded state */
  onToggleExpand?: () => void
  /** Current audio progress 0-1 (if available, overrides timer-based tracking) */
  audioProgress?: number
  /** Total audio duration in seconds (enables audio-synced word tracking) */
  audioDuration?: number
}

export function ScriptTeleprompter({
  script,
  phase,
  isPlaying,
  onWordTap,
  autoStart = true,
  msPerWord = DEFAULT_MS_PER_WORD,
  isExpanded = true,
  onToggleExpand,
  audioProgress,
  audioDuration,
}: ScriptTeleprompterProps) {
  // Split script into words
  const words = useMemo(() => {
    if (!script) return []
    return script.split(/\s+/).filter(Boolean)
  }, [script])

  // Current word index being highlighted
  const [currentWordIndex, setCurrentWordIndex] = useState(-1)
  const [isAnimating, setIsAnimating] = useState(false)
  const [hasCompleted, setHasCompleted] = useState(false)
  
  // Refs
  const wordRefs = useRef<(HTMLSpanElement | null)[]>([])
  const containerRef = useRef<HTMLDivElement>(null)
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const startTimeRef = useRef<number>(0)

  // Get phase styling
  const style = PHASE_STYLES[phase] || PHASE_STYLES.hook

  // Calculate word index from audio progress (if available)
  useEffect(() => {
    if (audioProgress !== undefined && audioDuration && audioDuration > 0 && words.length > 0) {
      // Estimate word position from audio progress
      const estimatedIndex = Math.floor(audioProgress * words.length)
      setCurrentWordIndex(Math.min(estimatedIndex, words.length - 1))
    }
  }, [audioProgress, audioDuration, words.length])

  // Timer-based word advancement (when no audio progress available)
  useEffect(() => {
    if (audioProgress !== undefined) return // Skip timer if audio-synced

    if (isPlaying && isExpanded && words.length > 0) {
      if (!isAnimating) {
        setIsAnimating(true)
        setHasCompleted(false)
        if (currentWordIndex < 0) {
          setCurrentWordIndex(0)
          startTimeRef.current = Date.now()
        }
      }

      timerRef.current = setInterval(() => {
        setCurrentWordIndex(prev => {
          const next = prev + 1
          if (next >= words.length) {
            clearInterval(timerRef.current!)
            setIsAnimating(false)
            setHasCompleted(true)
            return prev
          }
          return next
        })
      }, msPerWord)

      return () => {
        if (timerRef.current) clearInterval(timerRef.current)
      }
    } else {
      // Pause - clear timer but keep position
      if (timerRef.current) clearInterval(timerRef.current)
      if (!isPlaying) setIsAnimating(false)
    }
  }, [isPlaying, isExpanded, words.length, msPerWord, isAnimating, audioProgress, currentWordIndex])

  // Reset when script changes (new phase or new day)
  useEffect(() => {
    setCurrentWordIndex(-1)
    setIsAnimating(false)
    setHasCompleted(false)
    if (timerRef.current) clearInterval(timerRef.current)
  }, [script, phase])

  // Auto-scroll to keep current word visible
  useEffect(() => {
    if (currentWordIndex >= 0 && wordRefs.current[currentWordIndex] && containerRef.current) {
      const wordEl = wordRefs.current[currentWordIndex]
      const container = containerRef.current
      if (wordEl) {
        const wordRect = wordEl.getBoundingClientRect()
        const containerRect = container.getBoundingClientRect()
        
        // If word is below the visible area, scroll down
        if (wordRect.bottom > containerRect.bottom - 20) {
          wordEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }
        // If word is above the visible area, scroll up
        if (wordRect.top < containerRect.top + 20) {
          wordEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }
      }
    }
  }, [currentWordIndex])

  // Restart from beginning
  const handleRestart = useCallback(() => {
    setCurrentWordIndex(-1)
    setIsAnimating(false)
    setHasCompleted(false)
    if (timerRef.current) clearInterval(timerRef.current)
    // Will auto-start on next play
  }, [])

  // Handle word tap
  const handleWordClick = useCallback((word: string, index: number) => {
    // Clean the word (remove punctuation for lookup)
    const cleanWord = word.replace(/[^a-zA-Z'-]/g, '').toLowerCase()
    if (cleanWord && onWordTap) {
      onWordTap(cleanWord)
    }
  }, [onWordTap])

  if (!script || words.length === 0) return null

  return (
    <div className="w-full max-w-sm pointer-events-auto">
      {/* Header bar - always visible */}
      <button
        onClick={onToggleExpand}
        className={cn(
          "w-full flex items-center justify-between px-4 py-2 rounded-t-2xl transition-all duration-300",
          isExpanded ? "rounded-t-2xl" : "rounded-2xl"
        )}
        style={{
          background: 'linear-gradient(180deg, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.4) 100%)',
          backdropFilter: 'blur(20px)',
          border: '0.5px solid rgba(255,255,255,0.1)',
        }}
      >
        <div className="flex items-center gap-2">
          <BookOpen className="w-3.5 h-3.5 text-white/50" />
          <span className={cn("text-[11px] font-semibold uppercase tracking-[0.15em]", style.labelColor)}>
            Kelly&apos;s {PHASE_LABELS[phase]} Script
          </span>
        </div>
        <div className="flex items-center gap-2">
          {/* Word counter */}
          <span className="text-[10px] text-white/30 font-mono">
            {currentWordIndex >= 0 ? currentWordIndex + 1 : 0}/{words.length}
          </span>
          {isExpanded ? (
            <ChevronDown className="w-3.5 h-3.5 text-white/40" />
          ) : (
            <ChevronUp className="w-3.5 h-3.5 text-white/40" />
          )}
        </div>
      </button>

      {/* Expanded word-by-word display */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div
              ref={containerRef}
              className="px-4 pt-3 pb-4 max-h-[200px] overflow-y-auto scrollbar-none"
              style={{
                background: 'linear-gradient(180deg, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.6) 100%)',
                backdropFilter: 'blur(20px)',
                borderLeft: '0.5px solid rgba(255,255,255,0.1)',
                borderRight: '0.5px solid rgba(255,255,255,0.1)',
                borderBottom: '0.5px solid rgba(255,255,255,0.1)',
                borderRadius: '0 0 16px 16px',
              }}
            >
              {/* Word-by-word text */}
              <p className="text-[14px] leading-[1.8] font-light tracking-wide text-white/30">
                {words.map((word, index) => {
                  const isActive = index === currentWordIndex
                  const isPast = index < currentWordIndex
                  const isFuture = index > currentWordIndex
                  const isNextFew = index > currentWordIndex && index <= currentWordIndex + 3

                  return (
                    <span
                      key={`${index}-${word}`}
                      ref={el => { wordRefs.current[index] = el }}
                      onClick={(e) => {
                        e.stopPropagation()
                        handleWordClick(word, index)
                      }}
                      className={cn(
                        "inline-block mr-[0.3em] py-0.5 rounded-sm cursor-pointer transition-all duration-200",
                        isActive && cn(
                          style.activeColor,
                          "font-medium scale-105 px-0.5 -mx-0.5 bg-white/5 rounded"
                        ),
                        isPast && "text-white/70",
                        isFuture && !isNextFew && "text-white/20",
                        isNextFew && "text-white/35",
                        !isActive && "hover:text-white/60 hover:bg-white/5"
                      )}
                      style={isActive ? { textShadow: style.activeGlow } : undefined}
                    >
                      {word}
                    </span>
                  )
                })}
              </p>

              {/* Controls bar */}
              <div className="flex items-center justify-between mt-3 pt-2 border-t border-white/5">
                {/* Restart button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleRestart()
                  }}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-white/40 hover:text-white/70 hover:bg-white/5 transition-all text-[10px]"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Restart</span>
                </button>

                {/* Status indicator */}
                <div className="flex items-center gap-1.5">
                  {hasCompleted ? (
                    <span className="text-[10px] text-white/40 italic">Script complete</span>
                  ) : isAnimating ? (
                    <span className={cn("text-[10px] font-medium", style.labelColor)}>
                      Following along...
                    </span>
                  ) : currentWordIndex >= 0 ? (
                    <span className="text-[10px] text-white/30">Paused</span>
                  ) : (
                    <span className="text-[10px] text-white/30">Press play to start</span>
                  )}
                </div>

                {/* Speed indicator */}
                <span className="text-[9px] text-white/20 font-mono">
                  {Math.round(60000 / msPerWord)} wpm
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default ScriptTeleprompter
