'use client'

import { cn } from '@/lib/utils'
import { useLiveClassCountdown } from '@/hooks/use-live-class-countdown'
import type { AgeVariant } from '@/lib/types'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Video,
  Users,
  Bell,
  ChevronRight,
  Radio,
  Calendar,
} from 'lucide-react'

interface LiveClassBannerProps {
  ageVariant?: AgeVariant
  timezone: string
  locale: string
  variant?: 'banner' | 'compact' | 'card'
  onJoinClass?: () => void
  onNotifyMe?: () => void
  className?: string
}

export function LiveClassBanner({
  ageVariant,
  timezone,
  locale,
  variant = 'banner',
  onJoinClass,
  onNotifyMe,
  className,
}: LiveClassBannerProps) {
  const countdown = useLiveClassCountdown(ageVariant, timezone, locale)

  // Loading state
  if (countdown.isLoading) {
    return (
      <div className={cn('bg-card rounded-lg p-4', className)}>
        <div className="flex items-center gap-4">
          <Skeleton className="h-10 w-10 rounded-full" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-3 w-32" />
          </div>
          <Skeleton className="h-9 w-24" />
        </div>
      </div>
    )
  }

  // No upcoming class
  if (countdown.noUpcoming || !countdown.class) {
    if (variant === 'compact') return null
    
    return (
      <div className={cn('bg-card border border-border rounded-lg p-4', className)}>
        <div className="flex items-center gap-3 text-muted-foreground">
          <Calendar className="h-5 w-5" />
          <span className="text-sm">No live classes scheduled. Check back soon!</span>
        </div>
      </div>
    )
  }

  // LIVE NOW state
  if (countdown.isLive) {
    return (
      <div
        className={cn(
          'relative overflow-hidden rounded-lg',
          variant === 'card' && 'bg-card border border-border',
          className
        )}
      >
        {/* Live glow background */}
        <div className="absolute inset-0 bg-gradient-to-r from-live-red/20 via-live-red/10 to-live-red/20 animate-pulse" />
        
        <div className="relative p-4 flex items-center gap-4">
          {/* Live indicator */}
          <div className="relative flex items-center justify-center">
            <div className="absolute w-12 h-12 bg-live-red/30 rounded-full animate-ping" />
            <div className="relative w-10 h-10 bg-live-red rounded-full flex items-center justify-center live-glow">
              <Radio className="h-5 w-5 text-white" />
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Badge variant="destructive" className="text-xs font-bold tracking-wider">
                LIVE NOW
              </Badge>
              <span className="text-sm text-muted-foreground">
                {countdown.class.participant_count} learners
              </span>
            </div>
            <p className="text-sm font-medium text-foreground">
              {countdown.fullText}
            </p>
          </div>

          {/* Join button */}
          <Button onClick={onJoinClass} className="gap-2 bg-live-red hover:bg-live-red/90 text-white">
            <Video className="h-4 w-4" />
            Join Now
          </Button>
        </div>
      </div>
    )
  }

  // Starting soon state (< 5 minutes)
  if (countdown.isStartingSoon) {
    return (
      <div
        className={cn(
          'relative overflow-hidden rounded-lg border border-white/20 bg-white/5',
          className
        )}
      >
        <div className="p-4 flex items-center gap-4">
          {/* Countdown */}
          <div className="flex items-center justify-center w-10 h-10 bg-[var(--kelly-blue)]/20 rounded-full">
            <Video className="h-5 w-5 text-[var(--kelly-blue)]" />
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Badge className="bg-[var(--kelly-blue)]/20 text-[var(--kelly-blue)] text-xs font-semibold">
                Starting Soon
              </Badge>
            </div>
            <p className="text-sm font-medium text-white">
              Live class starts in{' '}
              <span className="text-[var(--kelly-blue)] font-bold countdown-digit">
                {countdown.countdownText}
              </span>
            </p>
            <p className="text-xs text-white/50 mt-0.5">
              {countdown.scheduledAtFormatted}
            </p>
          </div>

          {/* Action */}
          <Button onClick={onJoinClass} variant="outline" className="gap-2 border-[var(--kelly-blue)]/50 text-[var(--kelly-blue)] hover:bg-[var(--kelly-blue)]/10 bg-transparent">
            Get Ready
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    )
  }

  // Upcoming state - Show countdown with FULL scheduled time
  if (variant === 'compact') {
    return (
      <div className={cn('flex items-center gap-3 text-sm', className)}>
        <Video className="h-4 w-4 text-primary" />
        <span className="text-muted-foreground">Next live class in</span>
        <span className="font-semibold text-primary countdown-digit">
          {countdown.countdownText}
        </span>
      </div>
    )
  }

  return (
    <div
      className={cn(
        'bg-card border border-border rounded-lg p-4',
        className
      )}
    >
      <div className="flex flex-col md:flex-row md:items-center gap-4">
        {/* Icon */}
        <div className="flex items-center justify-center w-12 h-12 bg-primary/10 rounded-full shrink-0">
          <Video className="h-6 w-6 text-primary" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 space-y-1">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-foreground">Next Live Class</h3>
            <Badge variant="secondary" className="text-xs">
              {countdown.class.age_bracket}
            </Badge>
          </div>
          
          {/* Countdown - real-time ticking */}
          <div className="flex items-baseline gap-2">
            <span className="text-sm text-muted-foreground">Starts in</span>
            <span className="text-xl font-bold text-primary countdown-digit">
              {countdown.countdownText}
            </span>
          </div>
          
          {/* Full scheduled time - ALWAYS shown */}
          <p className="text-sm text-muted-foreground">
            {countdown.scheduledAtFormatted}
          </p>

          {/* Participants */}
          {countdown.class.participant_count > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground mt-2">
              <Users className="h-3.5 w-3.5" />
              <span>{countdown.class.participant_count} learners registered</span>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={onNotifyMe} className="gap-1.5 bg-transparent">
            <Bell className="h-3.5 w-3.5" />
            Notify Me
          </Button>
          <Button size="sm" onClick={onJoinClass} className="gap-1.5">
            <Video className="h-3.5 w-3.5" />
            Join Class
          </Button>
        </div>
      </div>
    </div>
  )
}

// Countdown display component - shows individual time units
interface CountdownDisplayProps {
  days: number
  hours: number
  minutes: number
  seconds: number
  showLabels?: boolean
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function CountdownDisplay({
  days,
  hours,
  minutes,
  seconds,
  showLabels = true,
  size = 'md',
  className,
}: CountdownDisplayProps) {
  const sizes = {
    sm: { digit: 'text-lg', label: 'text-xs' },
    md: { digit: 'text-2xl', label: 'text-xs' },
    lg: { digit: 'text-4xl', label: 'text-sm' },
  }

  const TimeUnit = ({ value, label }: { value: number; label: string }) => (
    <div className="flex flex-col items-center">
      <span className={cn('font-bold countdown-digit text-primary', sizes[size].digit)}>
        {value.toString().padStart(2, '0')}
      </span>
      {showLabels && (
        <span className={cn('text-muted-foreground uppercase tracking-wide', sizes[size].label)}>
          {label}
        </span>
      )}
    </div>
  )

  const Separator = () => (
    <span className={cn('text-muted-foreground', sizes[size].digit)}>:</span>
  )

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {days > 0 && (
        <>
          <TimeUnit value={days} label="days" />
          <Separator />
        </>
      )}
      <TimeUnit value={hours} label="hrs" />
      <Separator />
      <TimeUnit value={minutes} label="min" />
      <Separator />
      <TimeUnit value={seconds} label="sec" />
    </div>
  )
}

// Inline countdown text - just the text, no styling
interface CountdownTextProps {
  ageVariant?: AgeVariant
  timezone: string
  locale: string
  prefix?: string
}

export function CountdownText({
  ageVariant,
  timezone,
  locale,
  prefix = 'Next live class starts in',
}: CountdownTextProps) {
  const countdown = useLiveClassCountdown(ageVariant, timezone, locale)

  if (countdown.isLoading) {
    return <Skeleton className="h-4 w-32 inline-block" />
  }

  if (countdown.noUpcoming) {
    return <span className="text-muted-foreground">No upcoming classes</span>
  }

  if (countdown.isLive) {
    return (
      <span className="text-live-red font-semibold">
        LIVE NOW with {countdown.class?.participant_count || 0} learners
      </span>
    )
  }

  return (
    <span>
      {prefix}{' '}
      <span className="font-semibold text-primary countdown-digit">
        {countdown.countdownText}
      </span>
    </span>
  )
}
