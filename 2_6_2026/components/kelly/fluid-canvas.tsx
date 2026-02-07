'use client'

import React from "react"

// kellyOS Fluid Canvas - Water droplet physics with magical digital properties
// Foundation for Apple iLearn hardware experience

import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion'

interface Particle {
  id: number
  x: number
  y: number
  vx: number
  vy: number
  size: number
  opacity: number
  hue: number
  life: number
}

interface FluidCanvasProps {
  children: React.ReactNode
  intensity?: 'subtle' | 'medium' | 'magical'
  colorScheme?: 'aurora' | 'ocean' | 'sunset' | 'cosmic'
}

const COLOR_SCHEMES = {
  aurora: { base: 160, range: 60 },   // Cyan to green
  ocean: { base: 200, range: 40 },    // Blue spectrum
  sunset: { base: 20, range: 50 },    // Orange to pink
  cosmic: { base: 270, range: 80 },   // Purple to blue
}

export function FluidCanvas({ 
  children, 
  intensity = 'medium',
  colorScheme = 'aurora' 
}: FluidCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const particlesRef = useRef<Particle[]>([])
  const mouseRef = useRef({ x: 0, y: 0, isActive: false })
  const animationRef = useRef<number>(0)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  
  // Fluid motion values
  const mouseX = useMotionValue(0)
  const mouseY = useMotionValue(0)
  const springConfig = { damping: 25, stiffness: 150 }
  const smoothX = useSpring(mouseX, springConfig)
  const smoothY = useSpring(mouseY, springConfig)
  
  // Parallax transforms for children
  const childX = useTransform(smoothX, [0, dimensions.width], [-10, 10])
  const childY = useTransform(smoothY, [0, dimensions.height], [-10, 10])
  
  const intensityMultiplier = intensity === 'subtle' ? 0.5 : intensity === 'magical' ? 2 : 1
  const colors = COLOR_SCHEMES[colorScheme]

  // Create particle with water droplet physics
  const createParticle = useCallback((x: number, y: number, burst = false): Particle => {
    const angle = Math.random() * Math.PI * 2
    const speed = burst ? (2 + Math.random() * 4) : (0.5 + Math.random() * 1.5)
    return {
      id: Date.now() + Math.random(),
      x,
      y,
      vx: Math.cos(angle) * speed * intensityMultiplier,
      vy: Math.sin(angle) * speed * intensityMultiplier,
      size: burst ? (3 + Math.random() * 8) : (2 + Math.random() * 4),
      opacity: 0.6 + Math.random() * 0.4,
      hue: colors.base + (Math.random() - 0.5) * colors.range,
      life: 1,
    }
  }, [intensityMultiplier, colors])

  // Particle physics update
  const updateParticle = useCallback((p: Particle, deltaTime: number): Particle => {
    // Apply fluid dynamics
    const drag = 0.98
    const gravity = 0.02
    
    // Mouse attraction/repulsion
    const dx = mouseRef.current.x - p.x
    const dy = mouseRef.current.y - p.y
    const dist = Math.sqrt(dx * dx + dy * dy)
    
    if (mouseRef.current.isActive && dist < 150) {
      const force = (150 - dist) / 150 * 0.5 * intensityMultiplier
      p.vx += (dx / dist) * force
      p.vy += (dy / dist) * force
    }
    
    // Update velocity with fluid simulation
    p.vx *= drag
    p.vy *= drag
    p.vy += gravity * intensityMultiplier
    
    // Update position
    p.x += p.vx * deltaTime
    p.y += p.vy * deltaTime
    
    // Decay life
    p.life -= 0.008 * deltaTime
    p.opacity = p.life * 0.8
    
    return p
  }, [intensityMultiplier])

  // Canvas animation loop
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    let lastTime = performance.now()
    
    const animate = (currentTime: number) => {
      const deltaTime = Math.min((currentTime - lastTime) / 16, 3)
      lastTime = currentTime
      
      // Clear with fade effect for trails
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      
      // Add ambient particles
      if (Math.random() < 0.1 * intensityMultiplier) {
        particlesRef.current.push(
          createParticle(
            Math.random() * canvas.width,
            Math.random() * canvas.height
          )
        )
      }
      
      // Update and render particles
      particlesRef.current = particlesRef.current
        .map(p => updateParticle(p, deltaTime))
        .filter(p => p.life > 0 && p.x > -50 && p.x < canvas.width + 50 && p.y > -50 && p.y < canvas.height + 50)
      
      // Draw particles with glow effect
      particlesRef.current.forEach(p => {
        // Outer glow
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 3)
        gradient.addColorStop(0, `hsla(${p.hue}, 80%, 60%, ${p.opacity * 0.3})`)
        gradient.addColorStop(0.5, `hsla(${p.hue}, 70%, 50%, ${p.opacity * 0.1})`)
        gradient.addColorStop(1, 'transparent')
        
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size * 3, 0, Math.PI * 2)
        ctx.fillStyle = gradient
        ctx.fill()
        
        // Core particle
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fillStyle = `hsla(${p.hue}, 90%, 70%, ${p.opacity})`
        ctx.fill()
      })
      
      // Draw connection lines between nearby particles
      if (intensity === 'magical') {
        ctx.strokeStyle = `hsla(${colors.base}, 60%, 50%, 0.1)`
        ctx.lineWidth = 0.5
        
        for (let i = 0; i < particlesRef.current.length; i++) {
          for (let j = i + 1; j < particlesRef.current.length; j++) {
            const p1 = particlesRef.current[i]
            const p2 = particlesRef.current[j]
            const dx = p1.x - p2.x
            const dy = p1.y - p2.y
            const dist = Math.sqrt(dx * dx + dy * dy)
            
            if (dist < 100) {
              ctx.globalAlpha = (1 - dist / 100) * 0.3
              ctx.beginPath()
              ctx.moveTo(p1.x, p1.y)
              ctx.lineTo(p2.x, p2.y)
              ctx.stroke()
            }
          }
        }
        ctx.globalAlpha = 1
      }
      
      animationRef.current = requestAnimationFrame(animate)
    }
    
    animationRef.current = requestAnimationFrame(animate)
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [createParticle, updateParticle, intensityMultiplier, intensity, colors])

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      setDimensions({ width: window.innerWidth, height: window.innerHeight })
      if (canvasRef.current) {
        canvasRef.current.width = window.innerWidth
        canvasRef.current.height = window.innerHeight
      }
    }
    
    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // Handle touch/mouse interaction
  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    const x = e.clientX
    const y = e.clientY
    
    mouseRef.current = { x, y, isActive: true }
    mouseX.set(x)
    mouseY.set(y)
    
    // Create trail particles on movement
    if (Math.random() < 0.3 * intensityMultiplier) {
      particlesRef.current.push(createParticle(x, y))
    }
  }, [mouseX, mouseY, createParticle, intensityMultiplier])

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    const x = e.clientX
    const y = e.clientY
    
    // Burst effect on tap/click
    for (let i = 0; i < 12 * intensityMultiplier; i++) {
      particlesRef.current.push(createParticle(x, y, true))
    }
  }, [createParticle, intensityMultiplier])

  const handlePointerLeave = useCallback(() => {
    mouseRef.current.isActive = false
  }, [])

  return (
    <div 
      className="relative w-full h-full overflow-hidden"
      onPointerMove={handlePointerMove}
      onPointerDown={handlePointerDown}
      onPointerLeave={handlePointerLeave}
    >
      {/* Fluid particle canvas - bottom layer */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none z-0"
        style={{ mixBlendMode: 'screen' }}
      />
      
      {/* Content layer with parallax */}
      <motion.div 
        className="relative z-10 w-full h-full"
        style={{ x: childX, y: childY }}
      >
        {children}
      </motion.div>
      
      {/* Ambient glow overlay */}
      <div 
        className="absolute inset-0 pointer-events-none z-20 opacity-30"
        style={{
          background: `radial-gradient(circle at ${smoothX.get()}px ${smoothY.get()}px, hsla(${colors.base}, 60%, 50%, 0.15), transparent 40%)`,
        }}
      />
    </div>
  )
}
