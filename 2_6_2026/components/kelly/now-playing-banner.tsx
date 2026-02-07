'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Play, Pause, SkipForward, SkipBack, Volume2, VolumeX,
  ChevronUp, ChevronDown, BookOpen, Clock, Sparkles
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { LessonPhase } from '@/lib/types'

const PHASES: LessonPhase[] = ['hook', 'story', 'wonder', 'action', 'wisdom']

const PHASE_LABELS: Record<LessonPhase, string> = {
  hook: 'Hook',
  story: 'Story',
  wonder: 'Wonder',
  action: 'Action',
  wisdom: 'Wisdom',
}

const PHASE_COLORS: Record<LessonPhase, string> = {
  hook: 'from-[#FF6D6D] to-[#FF9B9B]',
  story: 'from-[#FFCD6D] to-[#FFEB9B]',
  wonder: 'from-[#FF6DFF] to-[#FF9BFF]',
  action: 'from-[#6DFF6D] to-[#9BFF9B]',
  wisdom: 'from-[#6D6DFF] to-[#9B9BFF]',
}

export function NowPlayingBanner({
  lessonTitle,
  dayOfYear,
  currentPhase,
  completedPhases,
  isPlaying,
  isMuted,
  onPlayPause,
  onMuteToggle,
  onPrevPhase,
  onNextPhase,
  onExpand,
}: NowPlayingBannerProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  
  const currentPhaseIndex = PHASES.indexOf(currentPhase)
  const progress = ((currentPhaseIndex + 1) / PHASES.length) * 100

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded)
    if (!isExpanded) onExpand?.()
  }

  return (
    <motion.div
      className="fixed bottom-4 left-4 right-4 z-[90] max-w-md mx-auto"
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: 'spring', damping: 25, stiffness: 300 }}
    >
      <motion.div
        className="rounded-[24px] overflow-hidden"
        style={{
          background: 'rgba(20, 20, 22, 0.9)',
          backdropFilter: 'blur(40px) saturate(180%)',
          WebkitBackdropFilter: 'blur(40px) saturate(180%)',
          boxShadow: '0 20px 50px -15px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
        }}
        layout
      >
        {/* Main content */}
        <div className="p-4">
          <div className="flex items-center gap-4">
            {/* Album art / Phase icon */}
            <motion.div 
              className={cn(
                "w-14 h-14 rounded-2xl flex items-center justify-center flex-shrink-0 bg-gradient-to-br shadow-lg",
                PHASE_COLORS[currentPhase]
              )}
              whileTap={{ scale: 0.95 }}
              onClick={toggleExpanded}
            >
              <BookOpen className="w-6 h-6 text-white" />
            </motion.div>

            {/* Lesson info */}
            <div className="flex-1 min-w-0" onClick={toggleExpanded}>
              <div className="text-white font-semibold text-sm truncate">{lessonTitle}</div>
              <div className="text-white/50 text-xs flex items-center gap-2">
                <span>Day {dayOfYear}</span>
                <span className="w-1 h-1 rounded-full bg-white/30" />
                <span className={cn(
                  "px-2 py-0.5 rounded-full text-[10px] font-medium bg-gradient-to-r text-white",
                  PHASE_COLORS[currentPhase]
                )}>
                  {PHASE_LABELS[currentPhase]}
                </span>
              </div>
              
              {/* Progress bar */}
              <div className="mt-2 h-1 bg-white/10 rounded-full overflow-hidden">
                <motion.div 
                  className={cn("h-full rounded-full bg-gradient-to-r", PHASE_COLORS[currentPhase])}
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                />
              </div>
            </div>

            {/* Play/Pause button */}
            <motion.button
              onClick={onPlayPause}
              className="w-12 h-12 rounded-full bg-white flex items-center justify-center flex-shrink-0"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {isPlaying ? (
                <Pause className="w-5 h-5 text-black" />
              ) : (
                <Play className="w-5 h-5 text-black ml-0.5" />
              )}
            </motion.button>
          </div>
        </div>

        {/* Expanded controls */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="px-4 pb-4 pt-0">
                {/* Phase progress dots */}
                <div className="flex items-center justify-center gap-3 mb-4">
                  {PHASES.map((phase, index) => {
                    const isActive = phase === currentPhase
                    const isComplete = completedPhases.includes(phase)
                    return (
                      <div key={phase} className="flex flex-col items-center gap-1">
                        <div className={cn(
                          "w-2.5 h-2.5 rounded-full transition-all",
                          isActive 
                            ? cn("scale-125 bg-gradient-to-r", PHASE_COLORS[phase])
                            : isComplete 
                              ? "bg-white/60"
                              : "bg-white/20"
                        )} />
                        <span className={cn(
                          "text-[9px] uppercase tracking-wider",
                          isActive ? "text-white/80 font-medium" : "text-white/30"
                        )}>
                          {PHASE_LABELS[phase]}
                        </span>
                      </div>
                    )
                  })}
                </div>

                {/* Media controls */}
                <div className="flex items-center justify-center gap-8">
                  <button 
                    onClick={onPrevPhase}
                    className="p-2 text-white/50 hover:text-white transition-colors"
                  >
                    <SkipBack className="w-6 h-6" />
                  </button>
                  
                  <motion.button
                    onClick={onPlayPause}
                    className="w-16 h-16 rounded-full bg-white flex items-center justify-center"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {isPlaying ? (
                      <Pause className="w-7 h-7 text-black" />
                    ) : (
                      <Play className="w-7 h-7 text-black ml-1" />
                    )}
                  </motion.button>
                  
                  <button 
                    onClick={onNextPhase}
                    className="p-2 text-white/50 hover:text-white transition-colors"
                  >
                    <SkipForward className="w-6 h-6" />
                  </button>
                </div>

                {/* Volume control */}
                <div className="flex items-center justify-center mt-4">
                  <button
                    onClick={onMuteToggle}
                    className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 hover:bg-white/15 transition-colors"
                  >
                    {isMuted ? (
                      <VolumeX className="w-4 h-4 text-white/60" />
                    ) : (
                      <Volume2 className="w-4 h-4 text-white/60" />
                    )}
                    <span className="text-white/60 text-sm">
                      {isMuted ? 'Unmute' : 'Mute'}
                    </span>
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Expand handle */}
        <button
          onClick={toggleExpanded}
          className="w-full py-2 flex items-center justify-center border-t border-white/5 hover:bg-white/5 transition-colors"
        >
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-white/30" />
          ) : (
            <ChevronUp className="w-4 h-4 text-white/30" />
          )}
        </button>
      </motion.div>
    </motion.div>
  )
}

// Compact mini player that sticks to Dynamic Island area
export function MiniNowPlaying({
  lessonTitle,
  currentPhase,
  isPlaying,
  onPlayPause,
  onTap,
}: {
  lessonTitle: string
  currentPhase: LessonPhase
  isPlaying: boolean
  onPlayPause: () => void
  onTap?: () => void
}) {
  return (
    <motion.div
      className="fixed top-4 left-1/2 -translate-x-1/2 z-[95]"
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: 'spring', damping: 25, stiffness: 300 }}
    >
      <motion.button
        onClick={onTap}
        className="flex items-center gap-3 px-4 py-2.5 rounded-full"
        style={{
          background: 'rgba(20, 20, 22, 0.9)',
          backdropFilter: 'blur(30px) saturate(180%)',
          WebkitBackdropFilter: 'blur(30px) saturate(180%)',
          boxShadow: '0 10px 30px -10px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.08)'
        }}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        {/* Animated waveform or icon */}
        <div className={cn(
          "w-8 h-8 rounded-full flex items-center justify-center bg-gradient-to-br",
          PHASE_COLORS[currentPhase]
        )}>
          {isPlaying ? (
            <div className="flex items-end gap-0.5 h-3">
              <motion.div
                className="w-0.5 bg-white rounded-full"
                animate={{ height: ['30%', '100%', '60%', '100%', '30%'] }}
                transition={{ duration: 1, repeat: Infinity }}
              />
              <motion.div
                className="w-0.5 bg-white rounded-full"
                animate={{ height: ['100%', '40%', '100%', '60%', '100%'] }}
                transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
              />
              <motion.div
                className="w-0.5 bg-white rounded-full"
                animate={{ height: ['60%', '100%', '30%', '100%', '60%'] }}
                transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
              />
            </div>
          ) : (
            <Play className="w-3.5 h-3.5 text-white ml-0.5" />
          )}
        </div>

        {/* Title - truncated */}
        <span className="text-white/90 text-sm font-medium max-w-[150px] truncate">
          {lessonTitle}
        </span>

        {/* Play/pause */}
        <button
          onClick={(e) => {
            e.stopPropagation()
            onPlayPause()
          }}
          className="w-7 h-7 rounded-full bg-white/10 flex items-center justify-center hover:bg-white/20 transition-colors"
        >
          {isPlaying ? (
            <Pause className="w-3.5 h-3.5 text-white" />
          ) : (
            <Play className="w-3.5 h-3.5 text-white ml-0.5" />
          )}
        </button>
      </motion.button>
    </motion.div>
  )
}
