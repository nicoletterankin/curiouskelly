'use client'

import React from "react"

// kellyOS Layer - The magical interface layer for Kelly Time Calendar
// Foundation for iLearn hardware - premium iOS-like experience
// Users say: "the kelly-time-calendar app is the only one I need"

import { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform, PanInfo } from 'framer-motion'
import { cn } from '@/lib/utils'
import { Calendar, Clock, Sparkles, Sun, Moon, Star, ChevronLeft, ChevronRight } from 'lucide-react'

interface KellyOSLayerProps {
  children: React.ReactNode
  formattedDate: string
  dayOfYear: number
  lessonTitle: string
  currentPhase: string
  onDateTap?: () => void
  onPrevDay?: () => void
  onNextDay?: () => void
}

// Floating time orb that can be dragged
function TimeOrb({ 
  formattedDate, 
  lessonTitle, 
  onTap,
  onPrev,
  onNext,
}: { 
  formattedDate: string
  lessonTitle: string
  onTap?: () => void
  onPrev?: () => void
  onNext?: () => void
}) {
  const constraintsRef = useRef<HTMLDivElement>(null)
  const [isExpanded, setIsExpanded] = useState(false)
  const [ripples, setRipples] = useState<{ id: number; x: number; y: number }[]>([])
  
  const x = useMotionValue(0)
  const y = useMotionValue(0)
  
  const springConfig = { damping: 25, stiffness: 400, mass: 0.5 }
  const springX = useSpring(x, springConfig)
  const springY = useSpring(y, springConfig)
  
  // Wobble effect on drag
  const rotate = useTransform(springX, [-100, 100], [-8, 8])
  const scale = useTransform(springX, (sx) => {
    const dist = Math.abs(sx)
    return 1 + Math.min(dist * 0.001, 0.08)
  })

  const handleTap = (e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const rippleX = e.clientX - rect.left
    const rippleY = e.clientY - rect.top
    
    setRipples(prev => [...prev, { id: Date.now(), x: rippleX, y: rippleY }])
    setTimeout(() => setRipples(prev => prev.slice(1)), 800)
    
    setIsExpanded(!isExpanded)
    onTap?.()
  }

  // Get time-appropriate greeting and icon
  const hour = new Date().getHours()
  const TimeIcon = hour < 6 ? Moon : hour < 12 ? Sun : hour < 18 ? Sun : hour < 21 ? Star : Moon
  const timeGradient = hour < 6 ? 'from-indigo-500 to-purple-600' 
    : hour < 12 ? 'from-amber-400 to-orange-500'
    : hour < 18 ? 'from-cyan-400 to-blue-500'
    : hour < 21 ? 'from-purple-500 to-pink-500'
    : 'from-indigo-600 to-violet-700'

  return (
    <div ref={constraintsRef} className="absolute inset-0 pointer-events-none overflow-hidden">
      <motion.div
        className="pointer-events-auto absolute top-16 left-1/2 -translate-x-1/2 z-40"
        style={{ x: springX, y: springY }}
        drag
        dragConstraints={constraintsRef}
        dragElastic={0.15}
        dragTransition={{ bounceStiffness: 400, bounceDamping: 25 }}
      >
        <motion.div
          style={{ rotate, scale }}
          className="relative"
        >
          {/* Expanded state - full calendar card */}
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8, y: -10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.8, y: -10 }}
                transition={{ type: 'spring', damping: 20, stiffness: 300 }}
                className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-72"
              >
                <div className="bg-black/80 backdrop-blur-xl rounded-3xl border border-white/10 p-4 shadow-2xl">
                  {/* Navigation */}
                  <div className="flex items-center justify-between mb-3">
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={(e) => { e.stopPropagation(); onPrev?.() }}
                      className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
                    >
                      <ChevronLeft className="w-4 h-4 text-white" />
                    </motion.button>
                    
                    <div className="text-center">
                      <div className="text-white/60 text-xs uppercase tracking-wider">Today's Lesson</div>
                      <div className="text-white font-semibold">{formattedDate}</div>
                    </div>
                    
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={(e) => { e.stopPropagation(); onNext?.() }}
                      className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
                    >
                      <ChevronRight className="w-4 h-4 text-white" />
                    </motion.button>
                  </div>
                  
                  {/* Lesson title */}
                  <div className="bg-gradient-to-br from-white/10 to-white/5 rounded-2xl p-3 border border-white/10">
                    <div className="flex items-center gap-2 mb-1">
                      <Sparkles className="w-4 h-4 text-amber-400" />
                      <span className="text-white/60 text-xs">Learning</span>
                    </div>
                    <div className="text-white font-medium text-sm line-clamp-2">{lessonTitle}</div>
                  </div>
                  
                  {/* Quick calendar preview */}
                  <div className="mt-3 grid grid-cols-7 gap-1">
                    {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((d, i) => (
                      <div key={i} className="text-center text-[10px] text-white/40">{d}</div>
                    ))}
                    {Array.from({ length: 7 }, (_, i) => {
                      const today = new Date()
                      const dayNum = today.getDate() - today.getDay() + i
                      const isToday = i === today.getDay()
                      return (
                        <motion.div
                          key={i}
                          whileHover={{ scale: 1.2 }}
                          className={cn(
                            "text-center text-xs py-1 rounded-full cursor-pointer transition-colors",
                            isToday 
                              ? "bg-white text-black font-bold" 
                              : "text-white/60 hover:bg-white/10"
                          )}
                        >
                          {dayNum > 0 && dayNum <= 31 ? dayNum : ''}
                        </motion.div>
                      )
                    })}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Main orb */}
          <motion.div
            onClick={handleTap}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={cn(
              "relative cursor-grab active:cursor-grabbing",
              "rounded-full bg-gradient-to-br backdrop-blur-xl",
              "border border-white/20 shadow-lg",
              timeGradient,
              isExpanded ? "w-14 h-14" : "w-12 h-12"
            )}
            animate={{
              boxShadow: isExpanded 
                ? '0 0 40px rgba(255,255,255,0.3)' 
                : '0 0 20px rgba(255,255,255,0.15)',
            }}
          >
            {/* Inner shine */}
            <div 
              className="absolute inset-1 rounded-full bg-gradient-to-br from-white/40 to-transparent"
              style={{ clipPath: 'ellipse(70% 50% at 30% 25%)' }}
            />
            
            {/* Icon */}
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.div
                animate={{ rotate: isExpanded ? 360 : 0 }}
                transition={{ duration: 0.5 }}
              >
                {isExpanded ? (
                  <Calendar className="w-5 h-5 text-white drop-shadow-lg" />
                ) : (
                  <TimeIcon className="w-5 h-5 text-white drop-shadow-lg" />
                )}
              </motion.div>
            </div>
            
            {/* Ripples */}
            {ripples.map(ripple => (
              <motion.div
                key={ripple.id}
                className="absolute rounded-full bg-white"
                style={{ left: ripple.x, top: ripple.y }}
                initial={{ width: 4, height: 4, opacity: 0.6, x: -2, y: -2 }}
                animate={{ width: 100, height: 100, opacity: 0, x: -50, y: -50 }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
              />
            ))}
            
            {/* Pulse ring */}
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-white/30"
              animate={{
                scale: [1, 1.3, 1],
                opacity: [0.5, 0, 0.5],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            />
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  )
}

// Floating particle system for magical atmosphere
function MagicParticles() {
  const [particles, setParticles] = useState<Array<{
    id: number
    x: number
    y: number
    size: number
    duration: number
    delay: number
  }>>([])

  useEffect(() => {
    const newParticles = Array.from({ length: 20 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: 2 + Math.random() * 4,
      duration: 3 + Math.random() * 4,
      delay: Math.random() * 2,
    }))
    setParticles(newParticles)
  }, [])

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {particles.map(p => (
        <motion.div
          key={p.id}
          className="absolute rounded-full bg-white/30"
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: p.size,
            height: p.size,
          }}
          animate={{
            y: [0, -30, 0],
            x: [0, Math.random() * 20 - 10, 0],
            opacity: [0.2, 0.6, 0.2],
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: p.duration,
            delay: p.delay,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  )
}

// Main kellyOS layer component
export function KellyOSLayer({
  children,
  formattedDate,
  dayOfYear,
  lessonTitle,
  currentPhase,
  onDateTap,
  onPrevDay,
  onNextDay,
}: KellyOSLayerProps) {
  const [touchStart, setTouchStart] = useState<{ x: number; y: number } | null>(null)
  const [rippleEffects, setRippleEffects] = useState<Array<{ id: number; x: number; y: number }>>([])

  // Create ripple on touch
  const handleTouchStart = useCallback((e: React.TouchEvent | React.MouseEvent) => {
    const point = 'touches' in e ? e.touches[0] : e
    setTouchStart({ x: point.clientX, y: point.clientY })
    
    // Add ripple effect
    const newRipple = { id: Date.now(), x: point.clientX, y: point.clientY }
    setRippleEffects(prev => [...prev, newRipple])
    
    setTimeout(() => {
      setRippleEffects(prev => prev.filter(r => r.id !== newRipple.id))
    }, 1000)
  }, [])

  return (
    <div 
      className="relative w-full min-h-screen"
      onTouchStart={handleTouchStart}
      onMouseDown={handleTouchStart}
    >
      {/* HEADER - Seamless immersive nav bar */}
      <header className="absolute top-0 left-0 right-0 z-[100] pointer-events-auto">
        <div 
          className="border-b border-white/[0.06]"
          style={{
            background: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(40px) saturate(160%)',
            WebkitBackdropFilter: 'blur(40px) saturate(160%)',
          }}
        >
          <div className="max-w-7xl mx-auto px-4 py-2.5 flex items-center justify-between">
            {/* Left: Brand - refined serif + subtle weight */}
            <a href="/" className="flex items-center gap-2.5 group">
              <span className="font-serif text-[13px] text-white/85 tracking-[0.02em] group-hover:text-white transition-colors">The Daily Lesson</span>
              <span className="text-white/20 hidden sm:inline font-light">|</span>
              <span className="text-[11px] text-white/40 italic hidden sm:inline tracking-wide">with Curious Kelly</span>
            </a>
            
            {/* Center: Date pill - frosted glass, precise */}
            <button 
              onClick={onDateTap}
              className="absolute left-1/2 -translate-x-1/2 hidden md:flex items-center gap-2 px-4 py-1.5 rounded-full transition-all hover:bg-white/[0.08]"
              style={{
                background: 'rgba(255, 255, 255, 0.04)',
                border: '0.5px solid rgba(255, 255, 255, 0.08)',
              }}
            >
              <Calendar className="w-3.5 h-3.5 text-white/40" />
              <span className="text-[12px] text-white/60 font-medium tracking-wide">{formattedDate}</span>
            </button>
            
            {/* Center mobile: Compact date */}
            <button 
              onClick={onDateTap}
              className="md:hidden flex items-center gap-1.5 px-3 py-1 rounded-full"
              style={{
                background: 'rgba(255, 255, 255, 0.04)',
                border: '0.5px solid rgba(255, 255, 255, 0.08)',
              }}
            >
              <Calendar className="w-3 h-3 text-white/40" />
              <span className="text-[11px] text-white/50 font-medium">{formattedDate}</span>
            </button>
            
            {/* Right: Nav + Alpha badge */}
            <div className="flex items-center gap-4">
              <div className="hidden sm:flex items-center gap-5 text-[11px] text-white/40 tracking-wide">
                <a href="/about" className="hover:text-white/80 transition-colors">About</a>
                <a href="/strategic" className="hover:text-white/80 transition-colors">Strategic</a>
                <a href="/join" className="hover:text-white/80 transition-colors">Investors</a>
              </div>
              <span 
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[9px] font-semibold uppercase tracking-[0.1em]"
                style={{
                  background: 'rgba(245, 158, 11, 0.08)',
                  border: '0.5px solid rgba(245, 158, 11, 0.15)',
                  color: 'rgba(245, 158, 11, 0.7)',
                }}
              >
                <span className="w-1 h-1 rounded-full bg-amber-500/60 animate-pulse" />
                Alpha
              </span>
            </div>
          </div>
        </div>
      </header>
      
      {/* Ambient particles */}
      <MagicParticles />
      
      {/* Touch ripple effects */}
      <AnimatePresence>
        {rippleEffects.map(ripple => (
          <motion.div
            key={ripple.id}
            className="absolute pointer-events-none z-50"
            style={{ left: ripple.x, top: ripple.y }}
            initial={{ scale: 0, opacity: 0.4 }}
            animate={{ scale: 4, opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 1, ease: 'easeOut' }}
          >
            <div className="w-20 h-20 -ml-10 -mt-10 rounded-full border border-white/30 bg-white/5" />
          </motion.div>
        ))}
      </AnimatePresence>
      
      {/* Main content - with top padding for header */}
      <div className="relative z-10 w-full pt-12">
        {children}
      </div>
    </div>
  )
}
