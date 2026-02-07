'use client'

import React from "react"

import { useState, useEffect, useCallback } from 'react'

/**
 * UI Scale Modes
 * - compact: 75% - minimal UI, maximum content
 * - standard: 100% - default balanced experience
 * - theater: 125% - immersive, larger video focus
 * - focus: 150% - accessibility mode, maximum readability
 */
export type ScaleMode = 'compact' | 'standard' | 'theater' | 'focus'

interface ScaleConfig {
  mode: ScaleMode
  scale: number
  fontSize: number
  videoScale: number
  captionSize: number
  buttonScale: number
  label: string
}

const SCALE_CONFIGS: Record<ScaleMode, ScaleConfig> = {
  compact: {
    mode: 'compact',
    scale: 0.75,
    fontSize: 12,
    videoScale: 0.85,
    captionSize: 14,
    buttonScale: 0.8,
    label: 'Compact',
  },
  standard: {
    mode: 'standard',
    scale: 1.0,
    fontSize: 14,
    videoScale: 1.0,
    captionSize: 16,
    buttonScale: 1.0,
    label: 'Standard',
  },
  theater: {
    mode: 'theater',
    scale: 1.25,
    fontSize: 16,
    videoScale: 1.15,
    captionSize: 20,
    buttonScale: 1.1,
    label: 'Theater',
  },
  focus: {
    mode: 'focus',
    scale: 1.5,
    fontSize: 20,
    videoScale: 1.0,
    captionSize: 24,
    buttonScale: 1.25,
    label: 'Focus',
  },
}

const STORAGE_KEY = 'kelly-ui-scale'

export function useUIScale() {
  const [mode, setModeState] = useState<ScaleMode>('standard')
  const [isLoaded, setIsLoaded] = useState(false)

  // Load from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored && stored in SCALE_CONFIGS) {
        setModeState(stored as ScaleMode)
      }
      setIsLoaded(true)
    }
  }, [])

  // Save to localStorage when mode changes
  const setMode = useCallback((newMode: ScaleMode) => {
    setModeState(newMode)
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY, newMode)
    }
  }, [])

  // Cycle through modes
  const cycleMode = useCallback(() => {
    const modes: ScaleMode[] = ['compact', 'standard', 'theater', 'focus']
    const currentIndex = modes.indexOf(mode)
    const nextIndex = (currentIndex + 1) % modes.length
    setMode(modes[nextIndex])
  }, [mode, setMode])

  // Get current config
  const config = SCALE_CONFIGS[mode]

  // CSS variables object for easy application
  const cssVars = {
    '--ui-scale': config.scale,
    '--ui-font-size': `${config.fontSize}px`,
    '--ui-video-scale': config.videoScale,
    '--ui-caption-size': `${config.captionSize}px`,
    '--ui-button-scale': config.buttonScale,
  } as React.CSSProperties

  return {
    mode,
    setMode,
    cycleMode,
    config,
    cssVars,
    isLoaded,
    allModes: Object.values(SCALE_CONFIGS),
  }
}
