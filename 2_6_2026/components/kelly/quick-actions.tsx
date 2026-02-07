'use client'

import React from "react"

import { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Share2, Bookmark, BookOpen, MessageCircle, Calendar,
  Download, Copy, Heart, Flag, ExternalLink, Sparkles
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface QuickAction {
  id: string
  label: string
  icon: React.ElementType
  color?: string
  destructive?: boolean
}

interface QuickActionsMenuProps {
  isOpen: boolean
  onClose: () => void
  position: { x: number; y: number }
  actions: QuickAction[]
  onAction: (actionId: string) => void
  title?: string
  subtitle?: string
}

// Default lesson quick actions
export const LESSON_QUICK_ACTIONS: QuickAction[] = [
  { id: 'share', label: 'Share Lesson', icon: Share2, color: 'text-blue-400' },
  { id: 'save', label: 'Save to Favorites', icon: Bookmark, color: 'text-amber-400' },
  { id: 'dictionary', label: 'Word Lookup', icon: BookOpen, color: 'text-purple-400' },
  { id: 'comment', label: 'Add Comment', icon: MessageCircle, color: 'text-green-400' },
  { id: 'calendar', label: 'View in Calendar', icon: Calendar, color: 'text-cyan-400' },
  { id: 'download', label: 'Download Transcript', icon: Download, color: 'text-white/70' },
  { id: 'report', label: 'Report Issue', icon: Flag, color: 'text-red-400', destructive: true },
]

export function QuickActionsMenu({
  isOpen,
  onClose,
  position,
  actions,
  onAction,
  title,
  subtitle,
}: QuickActionsMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null)

  // Adjust position to keep menu in viewport
  const [adjustedPosition, setAdjustedPosition] = useState(position)

  useEffect(() => {
    if (isOpen && menuRef.current) {
      const rect = menuRef.current.getBoundingClientRect()
      const padding = 16
      
      let x = position.x
      let y = position.y

      // Adjust horizontal position
      if (x + rect.width > window.innerWidth - padding) {
        x = window.innerWidth - rect.width - padding
      }
      if (x < padding) x = padding

      // Adjust vertical position
      if (y + rect.height > window.innerHeight - padding) {
        y = position.y - rect.height // Show above
      }
      if (y < padding) y = padding

      setAdjustedPosition({ x, y })
    }
  }, [isOpen, position])

  const handleAction = (actionId: string) => {
    onAction(actionId)
    onClose()
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 z-[400]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Menu */}
          <motion.div
            ref={menuRef}
            className="fixed z-[401] min-w-[220px] max-w-[280px]"
            style={{ 
              left: adjustedPosition.x,
              top: adjustedPosition.y,
            }}
            initial={{ opacity: 0, scale: 0.9, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: -10 }}
            transition={{ type: 'spring', damping: 25, stiffness: 400 }}
          >
            <div 
              className="rounded-2xl overflow-hidden"
              style={{
                background: 'rgba(40, 40, 44, 0.95)',
                backdropFilter: 'blur(40px) saturate(180%)',
                WebkitBackdropFilter: 'blur(40px) saturate(180%)',
                boxShadow: '0 20px 60px -15px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
              }}
            >
              {/* Header */}
              {(title || subtitle) && (
                <div className="px-4 py-3 border-b border-white/10">
                  {title && (
                    <div className="text-white font-semibold text-sm truncate">{title}</div>
                  )}
                  {subtitle && (
                    <div className="text-white/50 text-xs truncate">{subtitle}</div>
                  )}
                </div>
              )}

              {/* Actions */}
              <div className="py-1.5">
                {actions.map((action, index) => {
                  const Icon = action.icon
                  const isDestructive = action.destructive
                  
                  return (
                    <motion.button
                      key={action.id}
                      onClick={() => handleAction(action.id)}
                      className={cn(
                        "w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors",
                        isDestructive 
                          ? "hover:bg-red-500/20" 
                          : "hover:bg-white/10"
                      )}
                      whileTap={{ scale: 0.98, backgroundColor: 'rgba(255,255,255,0.15)' }}
                    >
                      <Icon className={cn(
                        "w-5 h-5",
                        isDestructive ? "text-red-400" : action.color || "text-white"
                      )} />
                      <span className={cn(
                        "text-sm font-medium",
                        isDestructive ? "text-red-400" : "text-white"
                      )}>
                        {action.label}
                      </span>
                    </motion.button>
                  )
                })}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

// Hook for long-press detection
export function useLongPress(
  onLongPress: (e: React.TouchEvent | React.MouseEvent) => void,
  onClick?: () => void,
  { delay = 500 }: { delay?: number } = {}
) {
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const targetRef = useRef<EventTarget | null>(null)
  const longPressTriggered = useRef(false)

  const start = useCallback((e: React.TouchEvent | React.MouseEvent) => {
    targetRef.current = e.target
    longPressTriggered.current = false
    
    timeoutRef.current = setTimeout(() => {
      longPressTriggered.current = true
      onLongPress(e)
    }, delay)
  }, [onLongPress, delay])

  const clear = useCallback((shouldTriggerClick = true) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }
    if (shouldTriggerClick && !longPressTriggered.current && onClick) {
      onClick()
    }
  }, [onClick])

  return {
    onMouseDown: start,
    onMouseUp: () => clear(true),
    onMouseLeave: () => clear(false),
    onTouchStart: start,
    onTouchEnd: () => clear(true),
  }
}

// Wrapper component that adds long-press quick actions to children
export function WithQuickActions({
  children,
  actions = LESSON_QUICK_ACTIONS,
  onAction,
  title,
  subtitle,
}: {
  children: React.ReactNode
  actions?: QuickAction[]
  onAction: (actionId: string) => void
  title?: string
  subtitle?: string
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [menuPosition, setMenuPosition] = useState({ x: 0, y: 0 })

  const handleLongPress = (e: React.TouchEvent | React.MouseEvent) => {
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY
    
    setMenuPosition({ x: clientX, y: clientY })
    setMenuOpen(true)
    
    // Haptic feedback on supported devices
    if ('vibrate' in navigator) {
      navigator.vibrate(10)
    }
  }

  const longPressHandlers = useLongPress(handleLongPress)

  return (
    <>
      <div {...longPressHandlers} className="cursor-pointer select-none">
        {children}
      </div>
      
      <QuickActionsMenu
        isOpen={menuOpen}
        onClose={() => setMenuOpen(false)}
        position={menuPosition}
        actions={actions}
        onAction={onAction}
        title={title}
        subtitle={subtitle}
      />
    </>
  )
}

// Preview hint that appears on long-press start
export function QuickActionsHint({ visible }: { visible: boolean }) {
  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          className="fixed inset-x-0 bottom-8 flex justify-center z-[399] pointer-events-none"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
        >
          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-black/80 backdrop-blur-md">
            <Sparkles className="w-4 h-4 text-blue-400" />
            <span className="text-white/80 text-sm">Hold for Quick Actions</span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
