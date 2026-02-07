'use client'

import { cn } from '@/lib/utils'
import { PHASE_ORDER, PHASE_METADATA } from '@/hooks/use-lesson'
import type { LessonPhase } from '@/lib/types'
import { Sparkles, BookOpen, Lightbulb, Play, Quote, Check } from 'lucide-react'

interface PhaseNavigationProps {
  currentPhase: LessonPhase
  onPhaseChange: (phase: LessonPhase) => void
  completedPhases?: LessonPhase[]
  className?: string
}

const PHASE_ICONS: Record<LessonPhase, typeof Sparkles> = {
  hook: Sparkles,
  story: BookOpen,
  wonder: Lightbulb,
  action: Play,
  wisdom: Quote,
}

export function PhaseNavigation({
  currentPhase,
  onPhaseChange,
  completedPhases = [],
  className,
}: PhaseNavigationProps) {
  return (
    <nav className={cn('flex items-center justify-center gap-2', className)}>
      {PHASE_ORDER.map((phase) => {
        const Icon = PHASE_ICONS[phase]
        const metadata = PHASE_METADATA[phase]
        const isActive = phase === currentPhase
        const isCompleted = completedPhases.includes(phase)

        return (
          <button
            key={phase}
            onClick={() => onPhaseChange(phase)}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all',
              isActive
                ? 'bg-primary text-primary-foreground shadow-md'
                : isCompleted
                ? 'bg-muted text-foreground hover:bg-muted/80'
                : 'bg-transparent text-muted-foreground hover:bg-muted hover:text-foreground'
            )}
            title={metadata.description}
          >
            {isCompleted && !isActive ? (
              <Check className="w-4 h-4" />
            ) : (
              <Icon className="w-4 h-4" />
            )}
            <span className="hidden sm:inline">{metadata.label}</span>
          </button>
        )
      })}
    </nav>
  )
}

// Vertical variant for sidebar
export function PhaseNavigationVertical({
  currentPhase,
  onPhaseChange,
  completedPhases = [],
  className,
}: PhaseNavigationProps) {
  return (
    <nav className={cn('flex flex-col gap-1', className)}>
      {PHASE_ORDER.map((phase, index) => {
        const Icon = PHASE_ICONS[phase]
        const metadata = PHASE_METADATA[phase]
        const isActive = phase === currentPhase
        const isCompleted = completedPhases.includes(phase)

        return (
          <button
            key={phase}
            onClick={() => onPhaseChange(phase)}
            className={cn(
              'flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-all',
              isActive
                ? 'bg-primary text-primary-foreground'
                : isCompleted
                ? 'bg-muted/50 text-foreground hover:bg-muted'
                : 'text-muted-foreground hover:bg-muted hover:text-foreground'
            )}
          >
            <div
              className={cn(
                'flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium',
                isActive
                  ? 'bg-primary-foreground/20'
                  : isCompleted
                  ? 'bg-primary/20 text-primary'
                  : 'bg-muted'
              )}
            >
              {isCompleted && !isActive ? (
                <Check className="w-4 h-4" />
              ) : (
                <Icon className="w-4 h-4" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="font-medium">{metadata.label}</div>
              <div className="text-xs opacity-70 truncate">{metadata.description}</div>
            </div>
          </button>
        )
      })}
    </nav>
  )
}
