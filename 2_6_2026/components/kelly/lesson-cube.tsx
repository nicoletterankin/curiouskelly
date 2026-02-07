'use client'

/**
 * LessonCube - 3D Cube Navigation for Curious Kelly
 * 
 * 6 Faces of the Cube:
 * 1. FRONT: Today's Lesson - Current day content
 * 2. RIGHT: The Zig - Ziggurat/Investor perspective  
 * 3. BACK: Tomorrow's Lesson - Preview next day
 * 4. LEFT: Yesterday's Lesson - Review previous day
 * 5. TOP: Calendar View - Month/year overview
 * 6. BOTTOM: Personalized Lesson - Based on user class (investor-class, learner, etc.)
 * 
 * Square is the new round - ZigTok style
 */

import React, { useState, useCallback, useEffect, ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { 
  ChevronLeft, ChevronRight, ChevronUp, ChevronDown,
  Calendar, Sparkles, Building2, Clock, User, ArrowRight
} from 'lucide-react'
import Link from 'next/link'

// Cube faces
type CubeFace = 'front' | 'right' | 'back' | 'left' | 'top' | 'bottom'

// Face metadata
const CUBE_FACES: Record<CubeFace, { 
  label: string
  icon: ReactNode
  description: string
  rotation: string
}> = {
  front: { 
    label: "Today's Lesson", 
    icon: <Sparkles className="w-4 h-4" />,
    description: 'Your daily learning journey',
    rotation: 'rotateY(0deg)'
  },
  right: { 
    label: 'The Zig', 
    icon: <Building2 className="w-4 h-4" />,
    description: 'Strategic overview',
    rotation: 'rotateY(-90deg)'
  },
  back: { 
    label: "Tomorrow's Lesson", 
    icon: <ArrowRight className="w-4 h-4" />,
    description: 'Preview what comes next',
    rotation: 'rotateY(180deg)'
  },
  left: { 
    label: "Yesterday's Lesson", 
    icon: <Clock className="w-4 h-4" />,
    description: 'Review and reflect',
    rotation: 'rotateY(90deg)'
  },
  top: { 
    label: 'Calendar', 
    icon: <Calendar className="w-4 h-4" />,
    description: 'Your learning journey',
    rotation: 'rotateX(-90deg)'
  },
  bottom: { 
    label: 'For You', 
    icon: <User className="w-4 h-4" />,
    description: 'Personalized experience',
    rotation: 'rotateX(90deg)'
  },
}

// Navigation mapping - which face to go to from current face
const NAVIGATION: Record<CubeFace, { left: CubeFace; right: CubeFace; up: CubeFace; down: CubeFace }> = {
  front: { left: 'left', right: 'right', up: 'top', down: 'bottom' },
  right: { left: 'front', right: 'back', up: 'top', down: 'bottom' },
  back: { left: 'right', right: 'left', up: 'top', down: 'bottom' },
  left: { left: 'back', right: 'front', up: 'top', down: 'bottom' },
  top: { left: 'left', right: 'right', up: 'back', down: 'front' },
  bottom: { left: 'left', right: 'right', up: 'front', down: 'back' },
}

interface LessonCubeProps {
  currentDay: number
  userClass?: 'investor' | 'learner' | 'admin' | 'guest'
  children: ReactNode // Today's lesson content
  yesterdayContent?: ReactNode
  tomorrowContent?: ReactNode
  calendarContent?: ReactNode
  personalizedContent?: ReactNode
  zigContent?: ReactNode
  onDayChange?: (day: number) => void
  onFaceChange?: (face: CubeFace) => void
  className?: string
}

export function LessonCube({
  currentDay,
  userClass = 'guest',
  children,
  yesterdayContent,
  tomorrowContent,
  calendarContent,
  personalizedContent,
  zigContent,
  onDayChange,
  onFaceChange,
  className,
}: LessonCubeProps) {
  const [activeFace, setActiveFace] = useState<CubeFace>('front')
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [touchStart, setTouchStart] = useState<{ x: number; y: number } | null>(null)

  // Navigate to a face
  const navigateTo = useCallback((face: CubeFace) => {
    if (isTransitioning || face === activeFace) return
    
    setIsTransitioning(true)
    setActiveFace(face)
    onFaceChange?.(face)
    
    // Handle day changes
    if (face === 'left') onDayChange?.(currentDay - 1)
    if (face === 'back') onDayChange?.(currentDay + 1)
    
    setTimeout(() => setIsTransitioning(false), 600)
  }, [activeFace, isTransitioning, currentDay, onDayChange, onFaceChange])

  // Swipe handling
  const handleTouchStart = (e: React.TouchEvent) => {
    setTouchStart({ x: e.touches[0].clientX, y: e.touches[0].clientY })
  }

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (!touchStart) return
    
    const deltaX = e.changedTouches[0].clientX - touchStart.x
    const deltaY = e.changedTouches[0].clientY - touchStart.y
    const threshold = 50
    
    // Determine swipe direction (prioritize horizontal for day navigation)
    if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > threshold) {
      if (deltaX > 0) navigateTo(NAVIGATION[activeFace].left)
      else navigateTo(NAVIGATION[activeFace].right)
    } else if (Math.abs(deltaY) > threshold) {
      if (deltaY > 0) navigateTo(NAVIGATION[activeFace].up)
      else navigateTo(NAVIGATION[activeFace].down)
    }
    
    setTouchStart(null)
  }

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') navigateTo(NAVIGATION[activeFace].left)
      if (e.key === 'ArrowRight') navigateTo(NAVIGATION[activeFace].right)
      if (e.key === 'ArrowUp') navigateTo(NAVIGATION[activeFace].up)
      if (e.key === 'ArrowDown') navigateTo(NAVIGATION[activeFace].down)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [activeFace, navigateTo])

  // Get rotation for cube based on active face
  const getCubeRotation = () => {
    switch (activeFace) {
      case 'front': return 'rotateY(0deg) rotateX(0deg)'
      case 'right': return 'rotateY(-90deg) rotateX(0deg)'
      case 'back': return 'rotateY(-180deg) rotateX(0deg)'
      case 'left': return 'rotateY(90deg) rotateX(0deg)'
      case 'top': return 'rotateY(0deg) rotateX(90deg)'
      case 'bottom': return 'rotateY(0deg) rotateX(-90deg)'
      default: return 'rotateY(0deg) rotateX(0deg)'
    }
  }

  // Default content for faces
  const renderZigContent = () => zigContent || (
    <div className="w-full h-full bg-black flex flex-col items-center justify-center p-8 text-center">
      <Building2 className="w-16 h-16 text-white/40 mb-6" />
      <h2 className="text-2xl font-black text-white mb-2">The Ziggurat</h2>
      <p className="text-white/60 mb-8 max-w-md">
        Strategic headquarters for education at scale. 92 acres, 1M sqft, designed by William Pereira.
      </p>
      <Link 
        href="/investors" 
        className="px-6 py-3 bg-white text-black font-semibold text-sm hover:bg-white/90 transition-colors"
      >
        Learn More
      </Link>
    </div>
  )

  const renderPersonalizedContent = () => {
    if (personalizedContent) return personalizedContent
    
    // Investor-class gets special actions
    if (userClass === 'investor') {
      return (
        <div className="w-full h-full bg-black flex flex-col items-center justify-center p-8 text-center">
          <div className="mb-8">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/10 border border-white/20 text-white/80 text-xs tracking-widest uppercase mb-4">
              Investor Class
            </div>
            <h2 className="text-3xl font-black text-white mb-2">Your Impact</h2>
            <p className="text-white/60 max-w-md">
              Direct funds to educational infrastructure while building sustainable returns.
            </p>
          </div>
          
          <div className="space-y-4 w-full max-w-sm">
            <Link 
              href="/ziggurat/investors"
              className="block w-full px-6 py-4 bg-white text-black font-semibold text-sm hover:bg-white/90 transition-colors text-center"
            >
              View Financial Models
            </Link>
            <Link 
              href="/investors"
              className="block w-full px-6 py-4 border border-white/20 text-white font-semibold text-sm hover:bg-white/10 transition-colors text-center"
            >
              Investment Opportunities
            </Link>
            <p className="text-white/40 text-xs mt-6">
              Lesson of the Day, PBC is a deposit-only account.<br/>
              Returns structured through separate investment vehicles.
            </p>
          </div>
        </div>
      )
    }
    
    return (
      <div className="w-full h-full bg-black flex flex-col items-center justify-center p-8 text-center">
        <User className="w-16 h-16 text-white/40 mb-6" />
        <h2 className="text-2xl font-black text-white mb-2">Personalized Learning</h2>
        <p className="text-white/60 max-w-md">
          Content tailored to your age, interests, and learning goals.
        </p>
      </div>
    )
  }

  const renderCalendarContent = () => calendarContent || (
    <div className="w-full h-full bg-black flex flex-col items-center justify-center p-8 text-center">
      <Calendar className="w-16 h-16 text-white/40 mb-6" />
      <h2 className="text-2xl font-black text-white mb-2">365 Days</h2>
      <p className="text-white/60 max-w-md">
        Your complete learning journey. One lesson per day, 5 phases each, infinite growth.
      </p>
    </div>
  )

  const renderYesterdayContent = () => yesterdayContent || (
    <div className="w-full h-full bg-black flex flex-col items-center justify-center p-8 text-center">
      <Clock className="w-16 h-16 text-white/40 mb-6" />
      <h2 className="text-2xl font-black text-white mb-2">Yesterday</h2>
      <p className="text-white/60 max-w-md">
        Review what you learned. Reflection deepens understanding.
      </p>
    </div>
  )

  const renderTomorrowContent = () => tomorrowContent || (
    <div className="w-full h-full bg-black flex flex-col items-center justify-center p-8 text-center">
      <ArrowRight className="w-16 h-16 text-white/40 mb-6" />
      <h2 className="text-2xl font-black text-white mb-2">Tomorrow</h2>
      <p className="text-white/60 max-w-md">
        Preview what comes next. Build anticipation for learning.
      </p>
    </div>
  )

  return (
    <div 
      className={cn("relative w-full h-full overflow-hidden", className)}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
    >
      {/* 3D Cube Container */}
      <div 
        className="w-full h-full"
        style={{ perspective: '1500px', perspectiveOrigin: 'center center' }}
      >
        <motion.div
          className="w-full h-full relative"
          style={{ 
            transformStyle: 'preserve-3d',
            transform: getCubeRotation(),
          }}
          animate={{ transform: getCubeRotation() }}
          transition={{ duration: 0.6, ease: [0.32, 0.72, 0, 1] }}
        >
          {/* FRONT - Today's Lesson */}
          <div 
            className="absolute inset-0 backface-hidden"
            style={{ transform: 'translateZ(50vh)' }}
          >
            {children}
          </div>
          
          {/* RIGHT - The Zig */}
          <div 
            className="absolute inset-0 backface-hidden"
            style={{ transform: 'rotateY(90deg) translateZ(50vh)' }}
          >
            {renderZigContent()}
          </div>
          
          {/* BACK - Tomorrow */}
          <div 
            className="absolute inset-0 backface-hidden"
            style={{ transform: 'rotateY(180deg) translateZ(50vh)' }}
          >
            {renderTomorrowContent()}
          </div>
          
          {/* LEFT - Yesterday */}
          <div 
            className="absolute inset-0 backface-hidden"
            style={{ transform: 'rotateY(-90deg) translateZ(50vh)' }}
          >
            {renderYesterdayContent()}
          </div>
          
          {/* TOP - Calendar */}
          <div 
            className="absolute inset-0 backface-hidden"
            style={{ transform: 'rotateX(90deg) translateZ(50vh)' }}
          >
            {renderCalendarContent()}
          </div>
          
          {/* BOTTOM - Personalized */}
          <div 
            className="absolute inset-0 backface-hidden"
            style={{ transform: 'rotateX(-90deg) translateZ(50vh)' }}
          >
            {renderPersonalizedContent()}
          </div>
        </motion.div>
      </div>

      {/* Navigation Indicators */}
      <div className="absolute inset-0 pointer-events-none z-50">
        {/* Current face label */}
        <div className="absolute top-4 left-1/2 -translate-x-1/2">
          <motion.div
            key={activeFace}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-2 px-4 py-2 bg-black/60 backdrop-blur-sm border border-white/10 rounded-full"
          >
            {CUBE_FACES[activeFace].icon}
            <span className="text-white text-sm font-medium">{CUBE_FACES[activeFace].label}</span>
          </motion.div>
        </div>

        {/* Directional hints */}
        <AnimatePresence>
          {!isTransitioning && (
            <>
              {/* Left hint */}
              <motion.button
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 0.6, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                whileHover={{ opacity: 1, scale: 1.1 }}
                onClick={() => navigateTo(NAVIGATION[activeFace].left)}
                className="absolute left-4 top-1/2 -translate-y-1/2 p-3 bg-black/40 backdrop-blur-sm border border-white/10 rounded-full pointer-events-auto"
              >
                <ChevronLeft className="w-5 h-5 text-white" />
              </motion.button>
              
              {/* Right hint */}
              <motion.button
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 0.6, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                whileHover={{ opacity: 1, scale: 1.1 }}
                onClick={() => navigateTo(NAVIGATION[activeFace].right)}
                className="absolute right-4 top-1/2 -translate-y-1/2 p-3 bg-black/40 backdrop-blur-sm border border-white/10 rounded-full pointer-events-auto"
              >
                <ChevronRight className="w-5 h-5 text-white" />
              </motion.button>
              
              {/* Up hint */}
              <motion.button
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 0.4, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                whileHover={{ opacity: 1, scale: 1.1 }}
                onClick={() => navigateTo(NAVIGATION[activeFace].up)}
                className="absolute top-16 left-1/2 -translate-x-1/2 p-2 bg-black/40 backdrop-blur-sm border border-white/10 rounded-full pointer-events-auto"
              >
                <ChevronUp className="w-4 h-4 text-white" />
              </motion.button>
              
              {/* Down hint */}
              <motion.button
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 0.4, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                whileHover={{ opacity: 1, scale: 1.1 }}
                onClick={() => navigateTo(NAVIGATION[activeFace].down)}
                className="absolute bottom-24 left-1/2 -translate-x-1/2 p-2 bg-black/40 backdrop-blur-sm border border-white/10 rounded-full pointer-events-auto"
              >
                <ChevronDown className="w-4 h-4 text-white" />
              </motion.button>
            </>
          )}
        </AnimatePresence>

        {/* Face indicator dots */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2">
          {(['left', 'front', 'right'] as CubeFace[]).map((face) => (
            <button
              key={face}
              onClick={() => navigateTo(face)}
              className={cn(
                "w-2 h-2 rounded-full transition-all duration-300 pointer-events-auto",
                activeFace === face 
                  ? "w-6 bg-white" 
                  : "bg-white/30 hover:bg-white/50"
              )}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

// CSS for backface-hidden
const cubeStyles = `
.backface-hidden {
  backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
}
`

// Inject styles
if (typeof document !== 'undefined') {
  const styleEl = document.createElement('style')
  styleEl.textContent = cubeStyles
  document.head.appendChild(styleEl)
}

export default LessonCube
