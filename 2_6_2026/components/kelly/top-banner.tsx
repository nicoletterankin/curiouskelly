'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { X, Sparkles, Gift, PartyPopper } from 'lucide-react'

interface TopBannerProps {
  variant: 'free-day' | 'birthday' | 'promo'
  displayName?: string
  formattedDate: string
  onDismiss?: () => void
  className?: string
}

export function TopBanner({
  variant,
  displayName = 'Curious Learner',
  formattedDate,
  onDismiss,
  className,
}: TopBannerProps) {
  const [dismissed, setDismissed] = useState(false)

  if (dismissed) return null

  const handleDismiss = () => {
    setDismissed(true)
    onDismiss?.()
  }

  // Free day banner
  if (variant === 'free-day') {
    return (
      <div
        className={cn(
          'relative bg-gradient-to-r from-primary/20 via-primary/10 to-primary/20 border-b border-primary/20',
          className
        )}
      >
        <div className="max-w-7xl mx-auto px-4 py-2 flex items-center justify-center gap-2 text-sm">
          <Sparkles className="h-4 w-4 text-primary" />
          <span className="text-foreground">
            <strong>{formattedDate}</strong> is always free.
          </span>
          <Button
            variant="link"
            className="text-primary p-0 h-auto font-semibold hover:text-primary/80"
          >
            Start now — no signup needed
          </Button>
          {onDismiss && (
            <Button
              variant="ghost"
              size="icon"
              onClick={handleDismiss}
              className="absolute right-2 top-1/2 -translate-y-1/2 h-6 w-6 text-muted-foreground hover:text-foreground"
            >
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>
    )
  }

  // Birthday banner
  if (variant === 'birthday') {
    return (
      <div
        className={cn(
          'relative bg-[var(--kelly-blue)]/10 border-b border-white/10',
          className
        )}
      >
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-center gap-3 text-sm">
          <PartyPopper className="h-5 w-5 text-[var(--kelly-blue)]" />
          <span className="text-white">
            <strong>Happy Birthday, {displayName}!</strong> Today&apos;s lesson is extra special, just for you.
          </span>
          <Gift className="h-5 w-5 text-[var(--kelly-blue)]" />
          {onDismiss && (
            <Button
              variant="ghost"
              size="icon"
              onClick={handleDismiss}
              className="absolute right-2 top-1/2 -translate-y-1/2 h-6 w-6 text-muted-foreground hover:text-foreground"
            >
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>
    )
  }

  // Promo banner
  return (
    <div
      className={cn(
        'relative bg-gradient-to-r from-accent/20 via-accent/10 to-accent/20 border-b border-accent/20',
        className
      )}
    >
      <div className="max-w-7xl mx-auto px-4 py-2 flex items-center justify-center gap-2 text-sm">
        <span className="text-foreground">
          Join thousands of curious learners.
        </span>
        <Button
          variant="link"
          className="text-accent p-0 h-auto font-semibold hover:text-accent/80"
        >
          Subscribe today
        </Button>
        {onDismiss && (
          <Button
            variant="ghost"
            size="icon"
            onClick={handleDismiss}
            className="absolute right-2 top-1/2 -translate-y-1/2 h-6 w-6 text-muted-foreground hover:text-foreground"
          >
            <X className="h-3 w-3" />
          </Button>
        )}
      </div>
    </div>
  )
}
