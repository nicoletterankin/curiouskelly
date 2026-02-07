'use client'

import React from "react"

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { useUIScale, ScaleMode } from '@/hooks/use-ui-scale'
import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'
import { Maximize2, Minimize2, Monitor, Focus, ChevronUp, ChevronDown } from 'lucide-react'

interface ScaleControlProps {
  className?: string
  variant?: 'minimal' | 'expanded' | 'pill'
}

const MODE_ICONS: Record<ScaleMode, typeof Minimize2> = {
  compact: Minimize2,
  standard: Monitor,
  theater: Maximize2,
  focus: Focus,
}

export function ScaleControl({ className, variant = 'pill' }: ScaleControlProps) {
  const { mode, setMode, cycleMode, config, allModes } = useUIScale()
  const [isOpen, setIsOpen] = useState(false)

  const Icon = MODE_ICONS[mode]

  // Minimal - just a button that cycles through modes
  if (variant === 'minimal') {
    return (
      <Button
        variant="ghost"
        size="icon"
        className={cn('h-8 w-8 rounded-full bg-black/20 backdrop-blur-sm', className)}
        onClick={cycleMode}
        title={`Scale: ${config.label}`}
      >
        <Icon className="h-4 w-4 text-white" />
      </Button>
    )
  }

  // Pill - shows current mode with tap to expand
  if (variant === 'pill') {
    return (
      <div className={cn('relative', className)}>
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
              className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-black/90 backdrop-blur-md rounded-2xl p-2 flex flex-col gap-1 min-w-[140px]"
            >
              {allModes.map((m) => {
                const ModeIcon = MODE_ICONS[m.mode]
                const isActive = m.mode === mode
                return (
                  <button
                    key={m.mode}
                    onClick={() => {
                      setMode(m.mode)
                      setIsOpen(false)
                    }}
                    className={cn(
                      'flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition-colors text-left',
                      isActive 
                        ? 'bg-white/20 text-white' 
                        : 'text-white/70 hover:bg-white/10 hover:text-white'
                    )}
                  >
                    <ModeIcon className="h-4 w-4 shrink-0" />
                    <span className="flex-1">{m.label}</span>
                    <span className="text-xs text-white/50">{Math.round(m.scale * 100)}%</span>
                  </button>
                )
              })}
            </motion.div>
          )}
        </AnimatePresence>

        <motion.button
          onClick={() => setIsOpen(!isOpen)}
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded-full',
            'bg-black/20 backdrop-blur-sm border border-white/10',
            'text-white/90 text-xs font-medium',
            'hover:bg-black/30 transition-colors'
          )}
          whileTap={{ scale: 0.95 }}
        >
          <Icon className="h-3.5 w-3.5" />
          <span>{config.label}</span>
          <motion.div
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronUp className="h-3 w-3" />
          </motion.div>
        </motion.button>
      </div>
    )
  }

  // Expanded - full control panel
  return (
    <div className={cn('bg-black/80 backdrop-blur-md rounded-2xl p-4', className)}>
      <div className="text-xs text-white/50 uppercase tracking-wider mb-3">Display Scale</div>
      <div className="grid grid-cols-2 gap-2">
        {allModes.map((m) => {
          const ModeIcon = MODE_ICONS[m.mode]
          const isActive = m.mode === mode
          return (
            <button
              key={m.mode}
              onClick={() => setMode(m.mode)}
              className={cn(
                'flex flex-col items-center gap-2 p-3 rounded-xl transition-all',
                isActive 
                  ? 'bg-white/20 text-white ring-2 ring-white/30' 
                  : 'bg-white/5 text-white/60 hover:bg-white/10 hover:text-white'
              )}
            >
              <ModeIcon className="h-5 w-5" />
              <span className="text-xs font-medium">{m.label}</span>
              <span className="text-[10px] opacity-50">{Math.round(m.scale * 100)}%</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

/**
 * HOC wrapper to apply scale CSS variables to a component tree
 */
export function ScaledContainer({ 
  children, 
  className 
}: { 
  children: React.ReactNode
  className?: string 
}) {
  const { cssVars, isLoaded } = useUIScale()

  if (!isLoaded) return <div className={className}>{children}</div>

  return (
    <div 
      className={className}
      style={cssVars}
    >
      {children}
    </div>
  )
}
