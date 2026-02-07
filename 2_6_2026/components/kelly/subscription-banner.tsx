'use client'

/**
 * Subscription Upgrade Banner
 * Shown to free-tier users after they've explored a few lessons.
 * Subtle, non-intrusive, but clear value prop.
 */

import { useState } from 'react'
import { X, Sparkles, ArrowRight } from 'lucide-react'
import { cn } from '@/lib/utils'

interface SubscriptionBannerProps {
  onUpgrade: () => void
  className?: string
}

export function SubscriptionBanner({ onUpgrade, className }: SubscriptionBannerProps) {
  const [dismissed, setDismissed] = useState(false)

  if (dismissed) return null

  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-2xl mx-3 my-2',
        'bg-gradient-to-r from-[#1a1a2e] to-[#16213e]',
        'border border-white/10',
        className
      )}
    >
      <div className="relative z-10 flex items-center gap-3 p-3">
        <div className="flex-shrink-0 h-10 w-10 rounded-full bg-amber-500/20 flex items-center justify-center">
          <Sparkles className="h-5 w-5 text-amber-400" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-white">Unlock all 365 lessons</p>
          <p className="text-xs text-white/60 truncate">All languages, archetypes & archives</p>
        </div>
        <button
          onClick={onUpgrade}
          className="flex-shrink-0 flex items-center gap-1 px-3 py-1.5 rounded-full bg-white text-black text-xs font-semibold hover:bg-white/90 transition-colors"
        >
          Upgrade
          <ArrowRight className="h-3 w-3" />
        </button>
        <button
          onClick={() => setDismissed(true)}
          className="flex-shrink-0 p-1 text-white/30 hover:text-white/60 transition-colors"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}
