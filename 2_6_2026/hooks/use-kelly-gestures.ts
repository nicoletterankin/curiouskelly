'use client'

import React from "react"

// kellyOS Gesture System - Premium touch interactions
// Foundation for iLearn hardware integration

import { useCallback, useRef, useState, useEffect } from 'react'

interface GestureState {
  isPressed: boolean
  isDragging: boolean
  isPinching: boolean
  velocity: { x: number; y: number }
  direction: 'up' | 'down' | 'left' | 'right' | null
  scale: number
  rotation: number
}

interface GestureHandlers {
  onSwipeUp?: () => void
  onSwipeDown?: () => void
  onSwipeLeft?: () => void
  onSwipeRight?: () => void
  onDoubleTap?: (x: number, y: number) => void
  onLongPress?: (x: number, y: number) => void
  onPinch?: (scale: number) => void
  onRotate?: (angle: number) => void
}

const SWIPE_THRESHOLD = 50
const SWIPE_VELOCITY_THRESHOLD = 0.3
const DOUBLE_TAP_DELAY = 300
const LONG_PRESS_DELAY = 500

export function useKellyGestures(handlers: GestureHandlers = {}) {
  const [gestureState, setGestureState] = useState<GestureState>({
    isPressed: false,
    isDragging: false,
    isPinching: false,
    velocity: { x: 0, y: 0 },
    direction: null,
    scale: 1,
    rotation: 0,
  })

  const startRef = useRef<{ x: number; y: number; time: number } | null>(null)
  const lastTapRef = useRef<number>(0)
  const longPressTimerRef = useRef<NodeJS.Timeout | null>(null)
  const touchesRef = useRef<{ id: number; x: number; y: number }[]>([])
  const initialPinchDistRef = useRef<number>(0)
  const initialRotationRef = useRef<number>(0)

  const clearLongPress = useCallback(() => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current)
      longPressTimerRef.current = null
    }
  }, [])

  const getDistance = (t1: { x: number; y: number }, t2: { x: number; y: number }) => {
    return Math.sqrt(Math.pow(t2.x - t1.x, 2) + Math.pow(t2.y - t1.y, 2))
  }

  const getAngle = (t1: { x: number; y: number }, t2: { x: number; y: number }) => {
    return Math.atan2(t2.y - t1.y, t2.x - t1.x) * (180 / Math.PI)
  }

  const handleTouchStart = useCallback((e: TouchEvent | React.TouchEvent) => {
    const touches = 'touches' in e ? Array.from(e.touches) : []
    
    touchesRef.current = touches.map(t => ({
      id: t.identifier,
      x: t.clientX,
      y: t.clientY,
    }))

    if (touches.length === 1) {
      const touch = touches[0]
      const now = Date.now()
      
      // Double tap detection
      if (now - lastTapRef.current < DOUBLE_TAP_DELAY) {
        handlers.onDoubleTap?.(touch.clientX, touch.clientY)
        lastTapRef.current = 0
      } else {
        lastTapRef.current = now
      }

      startRef.current = {
        x: touch.clientX,
        y: touch.clientY,
        time: now,
      }

      // Long press detection
      longPressTimerRef.current = setTimeout(() => {
        handlers.onLongPress?.(touch.clientX, touch.clientY)
      }, LONG_PRESS_DELAY)

      setGestureState(prev => ({ ...prev, isPressed: true }))
    } else if (touches.length === 2) {
      // Initialize pinch/rotate
      initialPinchDistRef.current = getDistance(
        { x: touches[0].clientX, y: touches[0].clientY },
        { x: touches[1].clientX, y: touches[1].clientY }
      )
      initialRotationRef.current = getAngle(
        { x: touches[0].clientX, y: touches[0].clientY },
        { x: touches[1].clientX, y: touches[1].clientY }
      )
      
      clearLongPress()
      setGestureState(prev => ({ ...prev, isPinching: true }))
    }
  }, [handlers, clearLongPress])

  const handleTouchMove = useCallback((e: TouchEvent | React.TouchEvent) => {
    const touches = 'touches' in e ? Array.from(e.touches) : []
    
    clearLongPress()

    if (touches.length === 1 && startRef.current) {
      const touch = touches[0]
      const deltaX = touch.clientX - startRef.current.x
      const deltaY = touch.clientY - startRef.current.y
      const deltaTime = (Date.now() - startRef.current.time) / 1000

      const velocityX = deltaX / deltaTime
      const velocityY = deltaY / deltaTime

      let direction: GestureState['direction'] = null
      if (Math.abs(deltaX) > Math.abs(deltaY)) {
        direction = deltaX > 0 ? 'right' : 'left'
      } else {
        direction = deltaY > 0 ? 'down' : 'up'
      }

      setGestureState(prev => ({
        ...prev,
        isDragging: true,
        velocity: { x: velocityX, y: velocityY },
        direction,
      }))
    } else if (touches.length === 2) {
      const currentDist = getDistance(
        { x: touches[0].clientX, y: touches[0].clientY },
        { x: touches[1].clientX, y: touches[1].clientY }
      )
      const currentAngle = getAngle(
        { x: touches[0].clientX, y: touches[0].clientY },
        { x: touches[1].clientX, y: touches[1].clientY }
      )

      const scale = currentDist / initialPinchDistRef.current
      const rotation = currentAngle - initialRotationRef.current

      handlers.onPinch?.(scale)
      handlers.onRotate?.(rotation)

      setGestureState(prev => ({
        ...prev,
        scale,
        rotation,
      }))
    }
  }, [handlers, clearLongPress])

  const handleTouchEnd = useCallback((e: TouchEvent | React.TouchEvent) => {
    const touches = 'changedTouches' in e ? Array.from(e.changedTouches) : []
    
    clearLongPress()

    if (touches.length === 1 && startRef.current) {
      const touch = touches[0]
      const deltaX = touch.clientX - startRef.current.x
      const deltaY = touch.clientY - startRef.current.y
      const deltaTime = (Date.now() - startRef.current.time) / 1000

      const velocityX = Math.abs(deltaX / deltaTime)
      const velocityY = Math.abs(deltaY / deltaTime)

      // Detect swipe
      if (Math.abs(deltaY) > SWIPE_THRESHOLD && velocityY > SWIPE_VELOCITY_THRESHOLD) {
        if (deltaY < 0) {
          handlers.onSwipeUp?.()
        } else {
          handlers.onSwipeDown?.()
        }
      } else if (Math.abs(deltaX) > SWIPE_THRESHOLD && velocityX > SWIPE_VELOCITY_THRESHOLD) {
        if (deltaX < 0) {
          handlers.onSwipeLeft?.()
        } else {
          handlers.onSwipeRight?.()
        }
      }
    }

    startRef.current = null
    touchesRef.current = []
    
    setGestureState({
      isPressed: false,
      isDragging: false,
      isPinching: false,
      velocity: { x: 0, y: 0 },
      direction: null,
      scale: 1,
      rotation: 0,
    })
  }, [handlers, clearLongPress])

  // Cleanup
  useEffect(() => {
    return () => {
      clearLongPress()
    }
  }, [clearLongPress])

  const gestureProps = {
    onTouchStart: handleTouchStart,
    onTouchMove: handleTouchMove,
    onTouchEnd: handleTouchEnd,
  }

  return {
    gestureState,
    gestureProps,
  }
}
