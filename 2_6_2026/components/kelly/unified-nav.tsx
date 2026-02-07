'use client'

/**
 * UNIFIED NAVIGATION - Single nav system for entire app
 * 
 * Two modes:
 * - "immersive" (dark): Floating pill on Kelly experience
 * - "page" (light): Traditional header for Strategic/Investors/About
 * 
 * Consistent navigation items, smooth mode transitions
 */

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  Menu, X, Calendar, ArrowLeft, ChevronRight,
  Home, Target, TrendingUp, Building2, Users
} from 'lucide-react'
import { cn } from '@/lib/utils'

// Core navigation structure - same everywhere
const NAV_ITEMS = [
  { id: 'home', href: '/', label: 'Kelly', icon: Home },
  { id: 'about', href: '/about', label: 'About', icon: Users },
  { id: 'strategic', href: '/strategic', label: 'Strategic', icon: Target },
  { id: 'investors', href: '/investors', label: 'Investors', icon: TrendingUp },
  { id: 'coalition', href: '/coalition', label: 'Coalition', icon: Building2 },
]

interface UnifiedNavProps {
  mode?: 'immersive' | 'page'
  formattedDate?: string
  onMenuClick?: () => void
  onCalendarClick?: () => void
  className?: string
}

export function UnifiedNav({
  mode = 'immersive',
  formattedDate,
  onMenuClick,
  onCalendarClick,
  className,
}: UnifiedNavProps) {
  const pathname = usePathname()
  const isKelly = pathname === '/'
  const [isHovered, setIsHovered] = useState(false)

  // IMMERSIVE MODE: Floating pill that expands on hover
  if (mode === 'immersive') {
    return (
      <motion.nav
        className={cn(
          "fixed top-4 left-1/2 -translate-x-1/2 z-50",
          className
        )}
        onHoverStart={() => setIsHovered(true)}
        onHoverEnd={() => setIsHovered(false)}
      >
        <motion.div
          className="flex items-center gap-2 px-3 py-2 rounded-full bg-black/50 backdrop-blur-md border border-white/10"
          animate={{ 
            width: isHovered ? 'auto' : 'auto',
          }}
        >
          {/* Menu button */}
          <button
            onClick={onMenuClick}
            className="w-8 h-8 rounded-full flex items-center justify-center hover:bg-white/10 transition-colors"
          >
            <Menu className="w-4 h-4 text-white" />
          </button>

          {/* Brand + Date - always visible */}
          <Link href="/" className="flex items-center gap-2">
            <span className="text-white/90 font-semibold text-sm">The Daily Lesson</span>
            {formattedDate && (
              <>
                <span className="text-white/30">|</span>
                <span className="text-white/60 text-sm">{formattedDate}</span>
              </>
            )}
          </Link>

          {/* Expanded nav items on hover */}
          <AnimatePresence>
            {isHovered && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                className="flex items-center gap-1 ml-2 overflow-hidden"
              >
                {NAV_ITEMS.filter(item => item.id !== 'home').map((item) => (
                  <Link
                    key={item.id}
                    href={item.href}
                    className={cn(
                      "px-3 py-1.5 rounded-full text-sm transition-colors whitespace-nowrap",
                      pathname === item.href
                        ? "bg-white/20 text-white"
                        : "text-white/60 hover:text-white hover:bg-white/10"
                    )}
                  >
                    {item.label}
                  </Link>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Calendar button */}
          {onCalendarClick && (
            <button
              onClick={onCalendarClick}
              className="w-8 h-8 rounded-full flex items-center justify-center hover:bg-white/10 transition-colors ml-1"
            >
              <Calendar className="w-4 h-4 text-white/70" />
            </button>
          )}

          {/* Alpha badge */}
          <span className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 text-xs font-medium">
            ALPHA
          </span>
        </motion.div>
      </motion.nav>
    )
  }

  // PAGE MODE: Traditional header for business pages
  return (
    <nav className={cn(
      "fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-sm border-b border-black/5",
      className
    )}>
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Left: Back to Kelly + Brand */}
        <div className="flex items-center gap-4">
          <Link 
            href="/"
            className="flex items-center gap-2 text-black/60 hover:text-black transition-colors group"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            <span className="text-sm">Back to Kelly</span>
          </Link>
          <span className="text-black/20">|</span>
          <Link href="/" className="text-xl font-black tracking-tight text-black">
            The Daily Lesson
          </Link>
          <span className="text-black/40 text-sm italic">with Curious Kelly</span>
        </div>

        {/* Center: Date pill */}
        {formattedDate && (
          <div className="hidden md:flex items-center gap-2 px-4 py-1.5 rounded-full bg-black/5">
            <Calendar className="w-4 h-4 text-black/40" />
            <span className="text-sm text-black/60">{formattedDate}</span>
            <span className="px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-600 text-xs font-medium">
              ALPHA
            </span>
          </div>
        )}

        {/* Right: Nav links */}
        <div className="flex items-center gap-1">
          {NAV_ITEMS.filter(item => item.id !== 'home').map((item) => (
            <Link
              key={item.id}
              href={item.href}
              className={cn(
                "px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                pathname === item.href
                  ? "bg-black text-white"
                  : "text-black/60 hover:text-black hover:bg-black/5"
              )}
            >
              {item.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}

// Simple back button for deep pages
export function BackToKelly({ className }: { className?: string }) {
  return (
    <Link 
      href="/"
      className={cn(
        "inline-flex items-center gap-2 text-black/60 hover:text-black transition-colors group",
        className
      )}
    >
      <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
      <span className="text-sm">Back to Kelly</span>
    </Link>
  )
}
