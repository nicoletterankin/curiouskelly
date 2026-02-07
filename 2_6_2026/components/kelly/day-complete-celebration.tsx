'use client'

/**
 * KellyOS Day Completion Celebration
 * Delightful animation when learner completes all 5 phases
 * Apple-quality celebration with confetti and XP reveal
 */

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { 
  Check, 
  Sparkles, 
  Star, 
  Trophy,
  ChevronRight,
  Flame
} from 'lucide-react'
import { LessonPhase } from '@/lib/types'

interface DayCompleteCelebrationProps {
  isOpen: boolean
  onClose: () => void
  dayOfYear: number
  lessonTitle: string
  xpEarned: number
  currentStreak: number
  streakBonus?: number
  onNextDay?: () => void
  onShare?: () => void
}

const PHASE_LABELS: Record<LessonPhase, string> = {
  hook: 'Hook',
  story: 'Story',
  wonder: 'Wonder',
  action: 'Action',
  wisdom: 'Wisdom'
}

// Confetti particle component
function ConfettiParticle({ 
  index, 
  color 
}: { 
  index: number
  color: string 
}) {
  const randomX = Math.random() * 100
  const randomDelay = Math.random() * 0.5
  const randomDuration = 2 + Math.random() * 2
  const randomRotation = Math.random() * 360

  return (
    <motion.div
      className={cn("absolute w-2 h-2 rounded-sm", color)}
      initial={{ 
        x: `${randomX}vw`, 
        y: -20, 
        rotate: 0,
        scale: 1
      }}
      animate={{ 
        y: '100vh',
        rotate: randomRotation + 360,
        scale: [1, 1.2, 0.8, 1]
      }}
      transition={{ 
        duration: randomDuration,
        delay: randomDelay,
        ease: 'linear'
      }}
      style={{ left: 0 }}
    />
  )
}

export function DayCompleteCelebration({
  isOpen,
  onClose,
  dayOfYear,
  lessonTitle,
  xpEarned,
  currentStreak,
  streakBonus = 0,
  onNextDay,
  onShare
}: DayCompleteCelebrationProps) {
  const [showConfetti, setShowConfetti] = useState(false)
  const [xpCounter, setXpCounter] = useState(0)

  // Trigger confetti and XP counter animation
  useEffect(() => {
    if (isOpen) {
      setShowConfetti(true)
      
      // Animate XP counter
      const totalXp = xpEarned + streakBonus
      const duration = 1500
      const steps = 30
      const increment = totalXp / steps
      let current = 0
      
      const timer = setInterval(() => {
        current += increment
        if (current >= totalXp) {
          setXpCounter(totalXp)
          clearInterval(timer)
        } else {
          setXpCounter(Math.floor(current))
        }
      }, duration / steps)

      // Stop confetti after 3 seconds
      const confettiTimer = setTimeout(() => {
        setShowConfetti(false)
      }, 3000)

      return () => {
        clearInterval(timer)
        clearTimeout(confettiTimer)
      }
    }
  }, [isOpen, xpEarned, streakBonus])

  if (!isOpen) return null

  const confettiColors = [
    'bg-emerald-400',
    'bg-teal-400', 
    'bg-sky-400',
    'bg-amber-400',
    'bg-pink-400',
    'bg-violet-400'
  ]

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center"
      >
        {/* Backdrop */}
        <motion.div
          className="absolute inset-0 bg-black/80 backdrop-blur-sm"
          onClick={onClose}
        />

        {/* Confetti */}
        {showConfetti && (
          <div className="fixed inset-0 overflow-hidden pointer-events-none">
            {Array.from({ length: 50 }).map((_, i) => (
              <ConfettiParticle
                key={i}
                index={i}
                color={confettiColors[i % confettiColors.length]}
              />
            ))}
          </div>
        )}

        {/* Content */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.8, opacity: 0, y: 20 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="relative z-10 w-full max-w-sm mx-4"
        >
          <div className="bg-gradient-to-b from-white/10 to-white/5 backdrop-blur-xl rounded-3xl border border-white/20 overflow-hidden">
            {/* Header with glow */}
            <div className="relative p-8 text-center">
              {/* Radial glow */}
              <div className="absolute inset-0 bg-gradient-radial from-emerald-500/20 via-transparent to-transparent" />
              
              {/* Trophy/Star icon */}
              <motion.div
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ delay: 0.2, type: 'spring', damping: 15 }}
                className="relative inline-flex items-center justify-center mb-4"
              >
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/30">
                  <Trophy className="w-10 h-10 text-white" />
                </div>
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.4 }}
                  className="absolute -top-1 -right-1 w-8 h-8 rounded-full bg-amber-400 flex items-center justify-center shadow-lg"
                >
                  <Check className="w-5 h-5 text-amber-900" />
                </motion.div>
              </motion.div>

              <motion.h2
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="text-2xl font-bold text-white mb-1"
              >
                Day {dayOfYear} Complete!
              </motion.h2>
              
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="text-white/60 text-sm"
              >
                {lessonTitle}
              </motion.p>
            </div>

            {/* XP Earned */}
            <div className="px-8 pb-6">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-white/5 rounded-2xl p-4 border border-white/10"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center">
                      <Sparkles className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <p className="text-xs text-white/50 uppercase tracking-wider">XP Earned</p>
                      <p className="text-2xl font-bold text-white">
                        +{xpCounter}
                      </p>
                    </div>
                  </div>
                  
                  {/* Streak */}
                  {currentStreak > 1 && (
                    <div className="text-right">
                      <div className="flex items-center gap-1 text-orange-400">
                        <Flame className="w-4 h-4" />
                        <span className="text-lg font-bold">{currentStreak}</span>
                      </div>
                      <p className="text-xs text-white/40">day streak</p>
                    </div>
                  )}
                </div>

                {/* Streak bonus callout */}
                {streakBonus > 0 && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    transition={{ delay: 0.8 }}
                    className="mt-3 pt-3 border-t border-white/10"
                  >
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-orange-400 flex items-center gap-1">
                        <Star className="w-4 h-4" />
                        Streak Bonus!
                      </span>
                      <span className="text-orange-400 font-semibold">
                        +{streakBonus} XP
                      </span>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            </div>

            {/* Completed phases */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="px-8 pb-6"
            >
              <div className="flex items-center justify-center gap-2">
                {Object.keys(PHASE_LABELS).map((phase, i) => (
                  <motion.div
                    key={phase}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.7 + i * 0.1 }}
                    className="flex flex-col items-center gap-1"
                  >
                    <div className="w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center">
                      <Check className="w-4 h-4 text-emerald-400" />
                    </div>
                    <span className="text-[10px] text-white/40">
                      {PHASE_LABELS[phase as LessonPhase]}
                    </span>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Actions */}
            <div className="p-4 bg-black/20 border-t border-white/10">
              <div className="flex items-center gap-2">
                {onShare && (
                  <button
                    onClick={onShare}
                    className="flex-1 py-3 px-4 rounded-xl bg-white/5 text-white/70 text-sm font-medium hover:bg-white/10 transition-colors"
                  >
                    Share
                  </button>
                )}
                <button
                  onClick={onNextDay || onClose}
                  className="flex-1 py-3 px-4 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2"
                >
                  {onNextDay ? (
                    <>
                      Next Lesson
                      <ChevronRight className="w-4 h-4" />
                    </>
                  ) : (
                    'Continue'
                  )}
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

export default DayCompleteCelebration
