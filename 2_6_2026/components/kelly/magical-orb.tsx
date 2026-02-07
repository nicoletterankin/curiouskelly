'use client'

import React from "react"

// kellyOS Magical Orb - Draggable floating UI element with physics
// Water droplet behavior with digital magic properties

import { useRef, useState } from 'react'
import { motion, useMotionValue, useSpring, useTransform, PanInfo } from 'framer-motion'
import { cn } from '@/lib/utils'

interface MagicalOrbProps {
  children: React.ReactNode
  className?: string
  size?: 'sm' | 'md' | 'lg'
  color?: 'cyan' | 'purple' | 'gold' | 'rose'
  glow?: boolean
  draggable?: boolean
  onTap?: () => void
  initialPosition?: { x: number; y: number }
}

const SIZES = {
  sm: { base: 40, ring: 48 },
  md: { base: 56, ring: 68 },
  lg: { base: 72, ring: 88 },
}

const COLORS = {
  cyan: {
    bg: 'from-cyan-400/30 to-teal-500/30',
    ring: 'border-cyan-400/50',
    glow: 'shadow-cyan-500/50',
    pulse: 'bg-cyan-400',
  },
  purple: {
    bg: 'from-violet-400/30 to-purple-500/30',
    ring: 'border-violet-400/50',
    glow: 'shadow-violet-500/50',
    pulse: 'bg-violet-400',
  },
  gold: {
    bg: 'from-amber-400/30 to-orange-500/30',
    ring: 'border-amber-400/50',
    glow: 'shadow-amber-500/50',
    pulse: 'bg-amber-400',
  },
  rose: {
    bg: 'from-rose-400/30 to-pink-500/30',
    ring: 'border-rose-400/50',
    glow: 'shadow-rose-500/50',
    pulse: 'bg-rose-400',
  },
}

export function MagicalOrb({
  children,
  className,
  size = 'md',
  color = 'cyan',
  glow = true,
  draggable = true,
  onTap,
  initialPosition = { x: 0, y: 0 },
}: MagicalOrbProps) {
  const constraintsRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [ripples, setRipples] = useState<{ id: number; x: number; y: number }[]>([])
  
  // Spring physics for fluid movement
  const x = useMotionValue(initialPosition.x)
  const y = useMotionValue(initialPosition.y)
  
  const springConfig = { damping: 20, stiffness: 300, mass: 0.8 }
  const springX = useSpring(x, springConfig)
  const springY = useSpring(y, springConfig)
  
  // Scale based on velocity for squash/stretch effect
  const velocityX = useMotionValue(0)
  const velocityY = useMotionValue(0)
  const scale = useTransform(
    [velocityX, velocityY],
    ([vx, vy]: number[]) => {
      const speed = Math.sqrt(vx * vx + vy * vy)
      return 1 + Math.min(speed * 0.001, 0.15)
    }
  )
  
  // Rotation based on horizontal velocity
  const rotate = useTransform(velocityX, [-500, 500], [-15, 15])
  
  const sizeConfig = SIZES[size]
  const colorConfig = COLORS[color]

  const handleDragStart = () => {
    setIsDragging(true)
  }

  const handleDrag = (_: unknown, info: PanInfo) => {
    velocityX.set(info.velocity.x)
    velocityY.set(info.velocity.y)
  }

  const handleDragEnd = () => {
    setIsDragging(false)
    // Decay velocity
    velocityX.set(0)
    velocityY.set(0)
  }

  const handleTap = (e: React.MouseEvent) => {
    // Create ripple effect
    const rect = e.currentTarget.getBoundingClientRect()
    const rippleX = e.clientX - rect.left - rect.width / 2
    const rippleY = e.clientY - rect.top - rect.height / 2
    
    const newRipple = { id: Date.now(), x: rippleX, y: rippleY }
    setRipples(prev => [...prev, newRipple])
    
    setTimeout(() => {
      setRipples(prev => prev.filter(r => r.id !== newRipple.id))
    }, 600)
    
    onTap?.()
  }

  return (
    <div ref={constraintsRef} className="absolute inset-0 pointer-events-none">
      <motion.div
        className={cn(
          "pointer-events-auto absolute cursor-grab active:cursor-grabbing",
          className
        )}
        style={{ x: springX, y: springY }}
        drag={draggable}
        dragConstraints={constraintsRef}
        dragElastic={0.1}
        dragTransition={{ bounceStiffness: 300, bounceDamping: 20 }}
        onDragStart={handleDragStart}
        onDrag={handleDrag}
        onDragEnd={handleDragEnd}
        onClick={handleTap}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <motion.div
          className="relative"
          style={{ scale, rotate }}
        >
          {/* Outer glow ring */}
          {glow && (
            <motion.div
              className={cn(
                "absolute rounded-full border-2 backdrop-blur-sm",
                colorConfig.ring,
                isDragging && "border-opacity-80"
              )}
              style={{
                width: sizeConfig.ring,
                height: sizeConfig.ring,
                left: -(sizeConfig.ring - sizeConfig.base) / 2,
                top: -(sizeConfig.ring - sizeConfig.base) / 2,
              }}
              animate={{
                scale: isDragging ? [1, 1.1, 1] : 1,
                opacity: isDragging ? 0.8 : 0.4,
              }}
              transition={{ duration: 0.3 }}
            />
          )}
          
          {/* Main orb body */}
          <motion.div
            className={cn(
              "relative rounded-full bg-gradient-to-br backdrop-blur-md",
              "flex items-center justify-center",
              "border border-white/20",
              colorConfig.bg,
              glow && `shadow-lg ${colorConfig.glow}`,
            )}
            style={{
              width: sizeConfig.base,
              height: sizeConfig.base,
            }}
          >
            {/* Inner shine */}
            <div 
              className="absolute inset-1 rounded-full bg-gradient-to-br from-white/30 to-transparent"
              style={{ clipPath: 'ellipse(60% 40% at 30% 30%)' }}
            />
            
            {/* Content */}
            <div className="relative z-10 text-white">
              {children}
            </div>
            
            {/* Ripple effects */}
            {ripples.map(ripple => (
              <motion.div
                key={ripple.id}
                className={cn("absolute rounded-full", colorConfig.pulse)}
                style={{
                  left: '50%',
                  top: '50%',
                  x: ripple.x,
                  y: ripple.y,
                }}
                initial={{ width: 4, height: 4, opacity: 0.8, marginLeft: -2, marginTop: -2 }}
                animate={{ width: sizeConfig.base * 2, height: sizeConfig.base * 2, opacity: 0, marginLeft: -sizeConfig.base, marginTop: -sizeConfig.base }}
                transition={{ duration: 0.6, ease: 'easeOut' }}
              />
            ))}
          </motion.div>
          
          {/* Ambient pulse animation */}
          {glow && !isDragging && (
            <motion.div
              className={cn(
                "absolute rounded-full",
                colorConfig.pulse,
                "opacity-20"
              )}
              style={{
                width: sizeConfig.base,
                height: sizeConfig.base,
              }}
              animate={{
                scale: [1, 1.4, 1],
                opacity: [0.2, 0, 0.2],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            />
          )}
        </motion.div>
      </motion.div>
    </div>
  )
}
