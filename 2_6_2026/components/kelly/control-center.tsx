'use client'

import React from "react"

import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, User, Settings, CreditCard, HelpCircle, 
  Info, FileText, Sparkles, BookOpen, 
  Search, MessageCircle, Calendar,
  Star, Gift, Zap,
  Play, Pause, SkipForward, SkipBack,
  Shield
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface ControlCenterProps {
  isOpen: boolean
  onClose: () => void
  onNavigate: (view: string) => void
  currentDay?: number
  lessonTitle?: string
  currentPhase?: string
  userName?: string
  isPremium?: boolean
  isPlaying?: boolean
  onPlayPause?: () => void
  streak?: number
  language?: string
}

// iPadOS-style tile sizes
type TileSize = 'small' | 'medium' | 'large' | 'wide'

interface ControlTile {
  id: string
  label: string
  icon: React.ElementType
  size: TileSize
  action?: string
  description?: string
  isToggle?: boolean
  value?: string | number
  color?: string
}

const CONTROL_TILES: ControlTile[] = [
  // Row 1 - Quick actions grid (2x2)
  { id: 'profile', label: 'Profile', icon: User, size: 'small', action: 'profile' },
  { id: 'settings', label: 'Settings', icon: Settings, size: 'small', action: 'settings' },
  { id: 'calendar', label: 'Calendar', icon: Calendar, size: 'small', action: 'calendar' },
  { id: 'search', label: 'Search', icon: Search, size: 'small', action: 'search' },
]

const MEDIA_CONTROLS = [
  { id: 'prev', icon: SkipBack },
  { id: 'play', icon: Play },
  { id: 'next', icon: SkipForward },
]

export function ControlCenter({ 
  isOpen, 
  onClose, 
  onNavigate, 
  currentDay = 1,
  lessonTitle = "Today's Lesson",
  currentPhase = 'hook',
  userName = 'Explorer',
  isPremium = false,
  isPlaying = false,
  onPlayPause,
  streak = 0,
  language = 'English'
}: ControlCenterProps) {
  const [activeToggle, setActiveToggle] = useState<string | null>(null)
  
  // Keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  const handleNavigate = (action: string) => {
    onNavigate(action)
    onClose()
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop with blur */}
          <motion.div
            className="fixed inset-0 z-[200]"
            style={{ 
              background: 'rgba(0, 0, 0, 0.4)',
              backdropFilter: 'blur(20px) saturate(180%)',
              WebkitBackdropFilter: 'blur(20px) saturate(180%)'
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
          />

          {/* Control Center Panel - iPadOS style from top-right */}
          <motion.div
            className="fixed top-4 right-4 left-4 md:left-auto md:w-[380px] z-[201] max-h-[85vh] overflow-hidden"
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            transition={{ type: 'spring', damping: 25, stiffness: 400 }}
          >
            {/* Main container with glass effect */}
            <div 
              className="rounded-[28px] overflow-hidden"
              style={{
                background: 'rgba(30, 30, 32, 0.85)',
                backdropFilter: 'blur(40px) saturate(180%)',
                WebkitBackdropFilter: 'blur(40px) saturate(180%)',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
              }}
            >
              {/* Header - Now Playing style */}
              <div className="p-4 border-b border-white/10">
                <div className="flex items-center gap-3">
                  {/* Album art / Lesson visual */}
                  <div className="w-14 h-14 rounded-xl bg-[var(--kelly-blue)] flex items-center justify-center flex-shrink-0 shadow-lg">
                    <BookOpen className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-white font-semibold text-sm truncate">{lessonTitle}</div>
                    <div className="text-white/60 text-xs">Day {currentDay} - {currentPhase.charAt(0).toUpperCase() + currentPhase.slice(1)} Phase</div>
                    {/* Progress bar */}
                    <div className="mt-2 h-1 bg-white/15 rounded-full overflow-hidden">
                      <motion.div 
                        className="h-full bg-[var(--kelly-blue)] rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${(currentDay / 365) * 100}%` }}
                        transition={{ duration: 1, ease: 'easeOut' }}
                      />
                    </div>
                  </div>
                </div>
                
                {/* Media controls */}
                <div className="flex items-center justify-center gap-6 mt-4">
                  <button className="p-2 text-white/60 hover:text-white transition-colors">
                    <SkipBack className="w-5 h-5" />
                  </button>
                  <button 
                    onClick={onPlayPause}
                    className="w-12 h-12 rounded-full bg-white flex items-center justify-center hover:scale-105 transition-transform"
                  >
                    {isPlaying ? (
                      <Pause className="w-5 h-5 text-black" />
                    ) : (
                      <Play className="w-5 h-5 text-black ml-0.5" />
                    )}
                  </button>
                  <button className="p-2 text-white/60 hover:text-white transition-colors">
                    <SkipForward className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {/* Quick Controls Grid - 2x2 */}
              <div className="p-3 grid grid-cols-4 gap-2">
                {CONTROL_TILES.map((tile) => {
                  const Icon = tile.icon
                  return (
                    <motion.button
                      key={tile.id}
                      onClick={() => tile.action && handleNavigate(tile.action)}
                      className="aspect-square rounded-xl bg-white/15 backdrop-blur-sm border border-white/10 hover:bg-white/20 flex flex-col items-center justify-center gap-1 transition-colors"
                      whileTap={{ scale: 0.95 }}
                    >
                      <Icon className="w-5 h-5 text-white" />
                      <span className="text-[10px] text-white/60">{tile.label}</span>
                    </motion.button>
                  )
                })}
              </div>

              {/* Stats Row */}
              <div className="px-3 pb-3 grid grid-cols-3 gap-2">
                <div className="rounded-xl bg-white/15 backdrop-blur-sm border border-white/10 p-3 text-center">
                  <div className="text-xl font-bold text-white">{currentDay}</div>
                  <div className="text-[10px] text-white/40 uppercase tracking-wide">Day</div>
                </div>
                <div className="rounded-xl bg-white/15 backdrop-blur-sm border border-white/10 p-3 text-center">
                  <div className="text-xl font-bold text-white">{streak}</div>
                  <div className="text-[10px] text-white/40 uppercase tracking-wide">Streak</div>
                </div>
                <div className="rounded-xl bg-white/15 backdrop-blur-sm border border-white/10 p-3 text-center">
                  <div className="text-xl font-bold text-white">{Math.round((currentDay / 365) * 100)}%</div>
                  <div className="text-[10px] text-white/40 uppercase tracking-wide">Progress</div>
                </div>
              </div>

              {/* Feature Tiles - Large buttons */}
              <div className="p-3 pt-0 space-y-2">
                {/* Dictionary + Chat row */}
                <div className="grid grid-cols-2 gap-2">
                  <motion.button
                    onClick={() => handleNavigate('dictionary')}
                    className="rounded-xl bg-white/15 backdrop-blur-sm border border-white/10 p-4 text-left hover:bg-white/20 transition-all group"
                    whileTap={{ scale: 0.98 }}
                  >
                    <BookOpen className="w-6 h-6 text-white mb-2 group-hover:scale-110 transition-transform" />
                    <div className="text-white font-medium text-sm">Dictionary</div>
                    <div className="text-white/40 text-xs">Word lookup</div>
                  </motion.button>
                  <motion.button
                    onClick={() => handleNavigate('chat')}
                    className="rounded-xl bg-white/15 backdrop-blur-sm border border-white/10 p-4 text-left hover:bg-white/20 transition-all group"
                    whileTap={{ scale: 0.98 }}
                  >
                    <MessageCircle className="w-6 h-6 text-white mb-2 group-hover:scale-110 transition-transform" />
                    <div className="text-white font-medium text-sm">Ask Kelly</div>
                    <div className="text-white/40 text-xs">AI tutor chat</div>
                  </motion.button>
                </div>

                {/* Upgrade banner - Premium highlight */}
                {!isPremium && (
                  <motion.button
                    onClick={() => handleNavigate('pricing')}
                    className="w-full rounded-xl bg-[var(--kelly-blue)]/20 backdrop-blur-sm border border-[var(--kelly-blue)]/30 p-4 text-left hover:bg-[var(--kelly-blue)]/30 transition-all"
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <Star className="w-5 h-5 text-white fill-white" />
                          <span className="text-white font-semibold">Upgrade to Premium</span>
                        </div>
                        <div className="text-white/60 text-xs mt-1">Unlock all 365 lessons & features</div>
                      </div>
                      <Zap className="w-6 h-6 text-white" />
                    </div>
                  </motion.button>
                )}

                {/* Affiliate */}
                <motion.button
                  onClick={() => handleNavigate('affiliate')}
                  className="w-full rounded-xl bg-white/15 backdrop-blur-sm border border-white/10 p-4 text-left hover:bg-white/20 transition-all"
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <Gift className="w-5 h-5 text-white" />
                        <span className="text-white font-medium">Earn Rewards</span>
                      </div>
                      <div className="text-white/40 text-xs mt-1">Share Kelly & earn 20-50%</div>
                    </div>
                  </div>
                </motion.button>
              </div>

              {/* Quick Links */}
              <div className="p-3 pt-0">
                <div className="rounded-xl bg-white/15 backdrop-blur-sm border border-white/10 divide-y divide-white/10">
                  {[
                    { id: 'subscription', label: 'Subscription', icon: CreditCard },
                    { id: 'about', label: 'About Kelly', icon: Info },
                    { id: 'help', label: 'Help Center', icon: HelpCircle },
                    { id: 'terms', label: 'Terms & Privacy', icon: FileText },
                  ].map((item) => {
                    const Icon = item.icon
                    return (
                      <button
                        key={item.id}
                        onClick={() => handleNavigate(item.id)}
                        className="w-full flex items-center gap-3 p-3 text-left hover:bg-white/10 transition-colors"
                      >
                        <Icon className="w-4 h-4 text-white/60" />
                        <span className="text-white text-sm">{item.label}</span>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Studio Section - Collapsible */}
              <div className="p-3 pt-0">
                <div className="rounded-xl bg-white/15 backdrop-blur-sm border border-white/10 divide-y divide-white/10">
                  <div className="p-3 text-xs text-white/40 uppercase tracking-wider font-medium">
                    Creator Studio
                  </div>
                  {[
                    { id: 'studio', label: 'Content Studio', icon: Sparkles },
                    { id: 'admin', label: 'Admin Panel', icon: Shield },
                    { id: 'zig', label: 'ZIG Model', icon: Zap },
                  ].map((item) => {
                    const Icon = item.icon
                    return (
                      <button
                        key={item.id}
                        onClick={() => handleNavigate(item.id)}
                        className="w-full flex items-center gap-3 p-3 text-left hover:bg-white/10 transition-colors"
                      >
                        <Icon className="w-4 h-4 text-white/60" />
                        <span className="text-white text-sm">{item.label}</span>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Footer */}
              <div className="p-4 pt-2 flex items-center justify-between text-xs text-white/30">
                <span>{userName} - Day {currentDay}</span>
                <span>v1.0</span>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

// Floating Control Center button - Top right corner
// Simple pill button that indicates "control center" without mimicking iOS status bar
export function ControlCenterButton({ onClick, hasNotification = false }: { onClick: () => void; hasNotification?: boolean }) {
  return (
    <motion.button
      onClick={onClick}
      className="fixed top-4 right-4 z-[100] w-10 h-10 rounded-full bg-black/40 backdrop-blur-md border border-white/10 flex items-center justify-center text-white/80 hover:bg-black/60 hover:text-white transition-all shadow-lg"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      aria-label="Open control center"
    >
      {/* Simple 4-dot grid icon - universal "more controls" symbol */}
      <div className="grid grid-cols-2 gap-1">
        <div className="w-1.5 h-1.5 rounded-full bg-current" />
        <div className="w-1.5 h-1.5 rounded-full bg-current" />
        <div className="w-1.5 h-1.5 rounded-full bg-current" />
        <div className="w-1.5 h-1.5 rounded-full bg-current" />
      </div>
      {hasNotification && (
        <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse border border-black/50" />
      )}
    </motion.button>
  )
}

// Spotlight Search Trigger - CMD+K style
export function SpotlightTrigger({ onClick }: { onClick: () => void }) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        onClick()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClick])

  return null // No visual element, just keyboard listener
}
