'use client'

/**
 * KellyOS Unified Overlay System
 * Apple Motion-quality modal system with spring animations, glass effects
 * NO WHITE PANELS - dark glass only
 * Company: Lesson of the Day (The Daily Lesson)
 * Product: Curious Kelly
 * v3 - Fresh rebuild to clear HMR cache
 */

import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { X, ChevronLeft, Minimize2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

// Apple Motion spring physics - tuned for premium feel
const SPRING = {
  // Snappy entrance with slight overshoot
  enter: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
  // Smooth, decelerating exit
  exit: 'cubic-bezier(0.4, 0, 0.2, 1)',
  // Buttery smooth for micro-interactions
  smooth: 'cubic-bezier(0.16, 1, 0.3, 1)',
}

// Animation durations (ms)
const DURATION = {
  backdrop: 250,
  panel: 400,
  content: 300,
  exit: 200,
}

// Framer Motion spring presets
export const springPresets = {
  snappy: { type: 'spring', stiffness: 400, damping: 30 },
  gentle: { type: 'spring', stiffness: 300, damping: 35 },
  bouncy: { type: 'spring', stiffness: 500, damping: 25, mass: 0.8 },
}

// Stagger container animation
export const staggerContainer = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.1,
    },
  },
}

// Stagger item animation
export const staggerItem = {
  hidden: { opacity: 0, y: 12 },
  show: { 
    opacity: 1, 
    y: 0,
    transition: springPresets.gentle,
  },
}

// ============================================================================
// OVERLAY CONTEXT
// ============================================================================

type OverlayType = 
  | 'settings' 
  | 'share' 
  | 'profile' 
  | 'curriculum' 
  | 'pricing' 
  | 'checkout'
  | 'about'
  | 'offline'
  | 'download'
  | 'parent-mode'
  | 'live-class'
  | null

interface OverlayContextType {
  activeOverlay: OverlayType
  overlayData: Record<string, unknown>
  openOverlay: (type: OverlayType, data?: Record<string, unknown>) => void
  closeOverlay: () => void
  isOpen: boolean
}

const OverlayContext = createContext<OverlayContextType | null>(null)

export function useOverlay() {
  const context = useContext(OverlayContext)
  if (!context) {
    throw new Error('useOverlay must be used within an OverlayProvider')
  }
  return context
}

interface OverlayProviderProps {
  children: React.ReactNode
}

export function OverlayProvider({ children }: OverlayProviderProps) {
  const [activeOverlay, setActiveOverlay] = useState<OverlayType>(null)
  const [overlayData, setOverlayData] = useState<Record<string, unknown>>({})

  const openOverlay = useCallback((type: OverlayType, data?: Record<string, unknown>) => {
    setOverlayData(data || {})
    setActiveOverlay(type)
  }, [])

  const closeOverlay = useCallback(() => {
    setActiveOverlay(null)
    // Delay clearing data to allow exit animation
    setTimeout(() => setOverlayData({}), DURATION.exit + 100)
  }, [])

  const isOpen = activeOverlay !== null

  return (
    <OverlayContext.Provider value={{ activeOverlay, overlayData, openOverlay, closeOverlay, isOpen }}>
      {children}
    </OverlayContext.Provider>
  )
}

// ============================================================================
// OVERLAY BACKDROP - Dark blur with click-to-dismiss
// ============================================================================

interface OverlayBackdropProps {
  isOpen: boolean
  onClose: () => void
  children: React.ReactNode
  className?: string
  /** If true, clicking backdrop won't close */
  preventClose?: boolean
}

export function OverlayBackdrop({ 
  isOpen, 
  onClose, 
  children, 
  className,
  preventClose = false
}: OverlayBackdropProps) {
  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return
    
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !preventClose) {
        onClose()
      }
    }
    
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [isOpen, onClose, preventClose])

  // Prevent body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [isOpen])

  return (
    <AnimatePresence mode="wait">
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: DURATION.backdrop / 1000 }}
          className={cn(
            "fixed inset-0 z-50",
            "bg-black/60 backdrop-blur-md",
            className
          )}
          onClick={preventClose ? undefined : onClose}
        >
          {/* Stop propagation on children container */}
          <div 
            className="absolute inset-0 flex items-center justify-center p-4"
            onClick={e => e.stopPropagation()}
          >
            {children}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// ============================================================================
// OVERLAY PANEL - Dark glass panel with Apple-quality animations
// ============================================================================

interface OverlayPanelProps {
  children: React.ReactNode
  className?: string
  /** Panel width preset */
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  /** Show close button */
  showClose?: boolean
  /** Close handler */
  onClose?: () => void
  /** Panel title */
  title?: string
  /** Show back button instead of close */
  showBack?: boolean
  /** Back handler */
  onBack?: () => void
  /** Minimize handler (for drawer behavior) */
  onMinimize?: () => void
  /** Animation origin */
  origin?: 'center' | 'bottom' | 'right' | 'left'
}

const sizeClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-[95vw] max-h-[95vh]',
}

const originAnimations = {
  center: {
    initial: { opacity: 0, scale: 0.95, y: 10 },
    animate: { opacity: 1, scale: 1, y: 0 },
    exit: { opacity: 0, scale: 0.95, y: 10 },
  },
  bottom: {
    initial: { opacity: 0, y: '100%' },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: '100%' },
  },
  right: {
    initial: { opacity: 0, x: '100%' },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: '100%' },
  },
  left: {
    initial: { opacity: 0, x: '-100%' },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: '-100%' },
  },
}

export function OverlayPanel({
  children,
  className,
  size = 'md',
  showClose = true,
  onClose,
  title,
  showBack = false,
  onBack,
  onMinimize,
  origin = 'center',
}: OverlayPanelProps) {
  const animation = originAnimations[origin]

  return (
    <motion.div
      {...animation}
      transition={springPresets.snappy}
      className={cn(
        "w-full",
        sizeClasses[size],
        // Dark glass panel - NO WHITE
        "bg-black/80 backdrop-blur-xl",
        "border border-white/10",
        "rounded-2xl shadow-2xl",
        "overflow-hidden",
        className
      )}
    >
      {/* Header with title and controls */}
      {(title || showClose || showBack || onMinimize) && (
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
          <div className="flex items-center gap-3">
            {showBack && onBack && (
              <Button
                variant="ghost"
                size="icon"
                onClick={onBack}
                className="h-8 w-8 rounded-full bg-white/5 hover:bg-white/10 text-white/70 hover:text-white"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
            )}
            {title && (
              <h2 className="text-lg font-semibold text-white tracking-tight">
                {title}
              </h2>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            {onMinimize && (
              <Button
                variant="ghost"
                size="icon"
                onClick={onMinimize}
                className="h-8 w-8 rounded-full bg-white/5 hover:bg-white/10 text-white/70 hover:text-white"
              >
                <Minimize2 className="h-4 w-4" />
              </Button>
            )}
            {showClose && onClose && (
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-8 w-8 rounded-full bg-white/5 hover:bg-white/10 text-white/70 hover:text-white"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="overflow-y-auto max-h-[calc(90vh-80px)]">
        {children}
      </div>
    </motion.div>
  )
}

// ============================================================================
// OVERLAY CONTENT - Staggered content container
// ============================================================================

interface OverlayContentProps {
  children: React.ReactNode
  className?: string
  /** Disable stagger animation */
  disableStagger?: boolean
}

export function OverlayContent({ children, className, disableStagger = false }: OverlayContentProps) {
  if (disableStagger) {
    return <div className={cn("p-5", className)}>{children}</div>
  }

  return (
    <motion.div
      variants={staggerContainer}
      initial="hidden"
      animate="show"
      className={cn("p-5", className)}
    >
      {children}
    </motion.div>
  )
}

// ============================================================================
// OVERLAY ITEM - Individual staggered item
// ============================================================================

interface OverlayItemProps {
  children: React.ReactNode
  className?: string
}

export function OverlayItem({ children, className }: OverlayItemProps) {
  return (
    <motion.div variants={staggerItem} className={className}>
      {children}
    </motion.div>
  )
}

// ============================================================================
// OVERLAY SECTION - Grouped content section with optional title
// ============================================================================

interface OverlaySectionProps {
  children: React.ReactNode
  title?: string
  description?: string
  className?: string
}

export function OverlaySection({ children, title, description, className }: OverlaySectionProps) {
  return (
    <motion.div variants={staggerItem} className={cn("space-y-3", className)}>
      {(title || description) && (
        <div className="space-y-1">
          {title && (
            <h3 className="text-sm font-medium text-white/90">{title}</h3>
          )}
          {description && (
            <p className="text-xs text-white/50">{description}</p>
          )}
        </div>
      )}
      {children}
    </motion.div>
  )
}

// ============================================================================
// OVERLAY DIVIDER - Visual separator
// ============================================================================

export function OverlayDivider({ className }: { className?: string }) {
  return (
    <motion.div 
      variants={staggerItem}
      className={cn("h-px bg-white/10 my-4", className)} 
    />
  )
}

// ============================================================================
// OVERLAY ROW - Horizontal layout for settings-style items
// ============================================================================

interface OverlayRowProps {
  children: React.ReactNode
  className?: string
  /** Click handler for the entire row */
  onClick?: () => void
  /** Show chevron indicator */
  showChevron?: boolean
  /** Disabled state */
  disabled?: boolean
}

export function OverlayRow({ 
  children, 
  className, 
  onClick, 
  showChevron = false,
  disabled = false
}: OverlayRowProps) {
  const isClickable = !!onClick && !disabled

  return (
    <motion.div
      variants={staggerItem}
      onClick={isClickable ? onClick : undefined}
      className={cn(
        "flex items-center justify-between py-3 px-1",
        "rounded-lg transition-colors duration-150",
        isClickable && "cursor-pointer hover:bg-white/5 active:bg-white/10",
        disabled && "opacity-50 cursor-not-allowed",
        className
      )}
    >
      {children}
      {showChevron && (
        <ChevronLeft className="h-4 w-4 text-white/30 rotate-180 flex-shrink-0 ml-2" />
      )}
    </motion.div>
  )
}

// ============================================================================
// OVERLAY BUTTON GROUP - Horizontal button container
// ============================================================================

interface OverlayButtonGroupProps {
  children: React.ReactNode
  className?: string
  /** Vertical stack on mobile */
  stackOnMobile?: boolean
}

export function OverlayButtonGroup({ children, className, stackOnMobile = true }: OverlayButtonGroupProps) {
  return (
    <motion.div 
      variants={staggerItem}
      className={cn(
        "flex gap-3 pt-4",
        stackOnMobile ? "flex-col sm:flex-row" : "flex-row",
        className
      )}
    >
      {children}
    </motion.div>
  )
}

// ============================================================================
// SCROLL AREA - Scrollable content with fade edges
// ============================================================================

interface OverlayScrollAreaProps {
  children: React.ReactNode
  className?: string
  maxHeight?: string
}

export function OverlayScrollArea({ children, className, maxHeight = '60vh' }: OverlayScrollAreaProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [showTopFade, setShowTopFade] = useState(false)
  const [showBottomFade, setShowBottomFade] = useState(false)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return

    const checkScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = el
      setShowTopFade(scrollTop > 10)
      setShowBottomFade(scrollTop + clientHeight < scrollHeight - 10)
    }

    checkScroll()
    el.addEventListener('scroll', checkScroll)
    return () => el.removeEventListener('scroll', checkScroll)
  }, [])

  return (
    <div className={cn("relative", className)}>
      {/* Top fade */}
      <div 
        className={cn(
          "absolute top-0 left-0 right-0 h-6 bg-gradient-to-b from-black/80 to-transparent z-10 pointer-events-none transition-opacity duration-200",
          showTopFade ? "opacity-100" : "opacity-0"
        )}
      />
      
      {/* Scrollable content */}
      <div 
        ref={scrollRef}
        className="overflow-y-auto scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent"
        style={{ maxHeight }}
      >
        {children}
      </div>

      {/* Bottom fade */}
      <div 
        className={cn(
          "absolute bottom-0 left-0 right-0 h-6 bg-gradient-to-t from-black/80 to-transparent z-10 pointer-events-none transition-opacity duration-200",
          showBottomFade ? "opacity-100" : "opacity-0"
        )}
      />
    </div>
  )
}

// ============================================================================
// ANIMATED LIST - Staggered list container
// ============================================================================

interface AnimatedListProps {
  children: React.ReactNode
  className?: string
}

export function AnimatedList({ children, className }: AnimatedListProps) {
  return (
    <motion.div
      variants={staggerContainer}
      initial="hidden"
      animate="show"
      className={cn("space-y-1", className)}
    >
      {children}
    </motion.div>
  )
}

// ============================================================================
// ANIMATED ITEM - Individual animated list item
// ============================================================================

interface AnimatedItemProps {
  children: React.ReactNode
  className?: string
}

export function AnimatedItem({ children, className }: AnimatedItemProps) {
  return (
    <motion.div variants={staggerItem} className={className}>
      {children}
    </motion.div>
  )
}

// ============================================================================
// LIST ITEM - Clickable list row
// ============================================================================

interface ListItemProps {
  children: React.ReactNode
  className?: string
  onClick?: () => void
  selected?: boolean
  disabled?: boolean
}

export function ListItem({ children, className, onClick, selected, disabled }: ListItemProps) {
  return (
    <motion.div
      variants={staggerItem}
      onClick={disabled ? undefined : onClick}
      className={cn(
        "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors",
        onClick && !disabled && "cursor-pointer hover:bg-white/5 active:bg-white/10",
        selected && "bg-white/10",
        disabled && "opacity-50 cursor-not-allowed",
        className
      )}
    >
      {children}
    </motion.div>
  )
}

// ============================================================================
// SECTION HEADER - Section title with optional action
// ============================================================================

interface SectionHeaderProps {
  title: string
  action?: React.ReactNode
  className?: string
}

export function SectionHeader({ title, action, className }: SectionHeaderProps) {
  return (
    <motion.div 
      variants={staggerItem}
      className={cn("flex items-center justify-between py-2", className)}
    >
      <h3 className="text-xs font-medium text-white/50 uppercase tracking-wider">{title}</h3>
      {action}
    </motion.div>
  )
}

// ============================================================================
// GLASS BUTTON - Frosted glass style button
// ============================================================================

interface GlassButtonProps {
  children: React.ReactNode
  className?: string
  onClick?: () => void
  variant?: 'default' | 'primary' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  fullWidth?: boolean
}

export function GlassButton({ 
  children, 
  className, 
  onClick, 
  variant = 'default',
  size = 'md',
  disabled,
  fullWidth
}: GlassButtonProps) {
  const variantClasses = {
    default: 'bg-white/10 hover:bg-white/15 text-white border-white/10',
    primary: 'bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border-emerald-500/30',
    danger: 'bg-red-500/20 hover:bg-red-500/30 text-red-300 border-red-500/30',
  }
  
  const sizeClasses = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-5 py-2.5 text-base',
  }

  return (
    <button
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      className={cn(
        "rounded-lg font-medium transition-all duration-200",
        "backdrop-blur-sm border",
        variantClasses[variant],
        sizeClasses[size],
        fullWidth && "w-full",
        disabled && "opacity-50 cursor-not-allowed",
        className
      )}
    >
      {children}
    </button>
  )
}

// ============================================================================
// PILL SELECTOR - Horizontal pill-style selector
// ============================================================================

interface PillSelectorProps<T extends string> {
  options: { value: T; label: string }[]
  value: T
  onChange: (value: T) => void
  className?: string
}

export function PillSelector<T extends string>({ 
  options, 
  value, 
  onChange, 
  className 
}: PillSelectorProps<T>) {
  return (
    <div className={cn("flex gap-1 p-1 bg-white/5 rounded-lg", className)}>
      {options.map(option => (
        <button
          key={option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            "px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200",
            value === option.value
              ? "bg-white/15 text-white"
              : "text-white/50 hover:text-white/70"
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

// ============================================================================
// UTILITY EXPORTS
// ============================================================================

export { SPRING, DURATION }
export type { OverlayType }
