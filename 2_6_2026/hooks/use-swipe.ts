'use client'

import React from "react"

import { useCallback, useRef } from 'react'

interface SwipeHandlers {
  onSwipeLeft?: () => void
  onSwipeRight?: () => void
  onSwipeUp?: () => void
  onSwipeDown?: () => void
}

interface SwipeConfig {
  threshold?: number // Min distance for swipe (default 50px)
  allowedTime?: number // Max time for swipe (default 300ms)
}

export function useSwipe(
  handlers: SwipeHandlers,
  config: SwipeConfig = {}
) {
  const { threshold = 50, allowedTime = 300 } = config
  
  const touchStart = useRef<{ x: number; y: number; time: number } | null>(null)
  
  const onTouchStart = useCallback((e: React.TouchEvent) => {
    const touch = e.touches[0]
    touchStart.current = {
      x: touch.clientX,
      y: touch.clientY,
      time: Date.now(),
    }
  }, [])
  
  const onTouchEnd = useCallback((e: React.TouchEvent) => {
    if (!touchStart.current) return
    
    const touch = e.changedTouches[0]
    const deltaX = touch.clientX - touchStart.current.x
    const deltaY = touch.clientY - touchStart.current.y
    const deltaTime = Date.now() - touchStart.current.time
    
    // Reset
    touchStart.current = null
    
    // Check if swipe was fast enough
    if (deltaTime > allowedTime) return
    
    // Determine direction (horizontal vs vertical)
    const absX = Math.abs(deltaX)
    const absY = Math.abs(deltaY)
    
    if (absX > absY && absX > threshold) {
      // Horizontal swipe
      if (deltaX > 0) {
        handlers.onSwipeRight?.()
      } else {
        handlers.onSwipeLeft?.()
      }
    } else if (absY > absX && absY > threshold) {
      // Vertical swipe
      if (deltaY > 0) {
        handlers.onSwipeDown?.()
      } else {
        handlers.onSwipeUp?.()
      }
    }
  }, [handlers, threshold, allowedTime])
  
  return {
    onTouchStart,
    onTouchEnd,
  }
}
