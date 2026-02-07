'use client'

import { useState, useCallback } from 'react'
import { cn } from '@/lib/utils'
import { VideoPlayer } from './video-player'
import { PhaseNavigation } from './phase-navigation'
import { InteractiveCaption } from './interactive-caption'
import { VisualAidsOverlay } from './visual-aids-overlay'
import { useVideoUrl, PHASE_ORDER } from '@/hooks/use-lesson'
import type { LessonPhase, KellyArchetype, Lesson, VideoState } from '@/lib/types'
import { Button } from '@/components/ui/button'
import { Sparkles, Play, ImageIcon } from 'lucide-react'
import { CaptionBar } from './caption-bar' // Declare the variable before using it

interface KellyStageProps {
  lesson: Lesson | null
  dayOfYear: number
  kellyAge: number
  archetype: KellyArchetype
  language: string
  onPhaseChange?: (phase: LessonPhase) => void
  className?: string
}

export function KellyStage({
  lesson,
  dayOfYear,
  kellyAge,
  archetype,
  language,
  onPhaseChange,
  className,
}: KellyStageProps) {
  const [currentPhase, setCurrentPhase] = useState<LessonPhase>('hook')
  const [videoState, setVideoState] = useState<VideoState>('idle')
  const [currentTime, setCurrentTime] = useState(0)
  const [hasStarted, setHasStarted] = useState(false)
  const [showVisualAids, setShowVisualAids] = useState(true)
  const [currentLanguage, setCurrentLanguage] = useState(language)

  // Fetch video URL for current phase
  const { videoUrl, fallbackUrl, thumbnailUrl, status } = useVideoUrl(
    dayOfYear,
    currentPhase,
    kellyAge,
    archetype,
    language
  )

  // Handle phase change
  const handlePhaseChange = useCallback((phase: LessonPhase) => {
    setCurrentPhase(phase)
    setVideoState('loading')
    onPhaseChange?.(phase)
  }, [onPhaseChange])

  // Handle phase complete (auto-advance)
  const handlePhaseComplete = useCallback(() => {
    const currentIndex = PHASE_ORDER.indexOf(currentPhase)
    if (currentIndex < PHASE_ORDER.length - 1) {
      const nextPhase = PHASE_ORDER[currentIndex + 1]
      handlePhaseChange(nextPhase)
    }
  }, [currentPhase, handlePhaseChange])

  // Start learning
  const handleStartLearning = useCallback(() => {
    setHasStarted(true)
    setCurrentPhase('hook')
  }, [])

  // Get current script for caption generation
  const getCurrentScript = (): string => {
    if (!lesson) return ''
    switch (currentPhase) {
      case 'hook': return lesson.hook_script || ''
      case 'story': return lesson.story_script || ''
      case 'wonder': return lesson.wonder_script || ''
      case 'action': return lesson.action_script || ''
      case 'wisdom': return lesson.wisdom_script || ''
      default: return ''
    }
  }

  // Initial state - show start screen
  if (!hasStarted) {
    return (
      <div className={cn('flex flex-col', className)}>
        {/* Preview Image with Start Button */}
        <div className="relative aspect-[9/16] md:aspect-video bg-muted rounded-lg overflow-hidden">
          {thumbnailUrl || fallbackUrl ? (
            <img
              src={thumbnailUrl || fallbackUrl || ''}
              alt={lesson?.title || 'Kelly'}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full bg-gradient-to-br from-blue-900 to-blue-700 flex items-center justify-center">
              <div className="text-center text-white">
                <Sparkles className="w-16 h-16 mx-auto mb-4 opacity-80" />
                <p className="text-lg font-medium">Ready to Learn</p>
              </div>
            </div>
          )}

          {/* Overlay with Start Button */}
          <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
            <Button
              size="lg"
              onClick={handleStartLearning}
              className="gap-2 text-lg px-8 py-6 rounded-full bg-white text-black hover:bg-white/90"
            >
              <Play className="w-6 h-6" />
              Start Learning
            </Button>
          </div>
        </div>

        {/* Lesson Info */}
        {lesson && (
          <div className="mt-4 text-center">
            <h2 className="text-xl font-semibold">{lesson.title}</h2>
            {lesson.subtitle && (
              <p className="text-muted-foreground mt-1">{lesson.subtitle}</p>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className={cn('flex flex-col gap-4 relative', className)}>
      {/* Video Player */}
      <div className="relative">
        <VideoPlayer
          videoUrl={videoUrl}
          fallbackImageUrl={fallbackUrl}
          status={status}
          autoPlay={true}
          onStateChange={setVideoState}
          onTimeUpdate={setCurrentTime}
          onPhaseComplete={handlePhaseComplete}
        />
        
        {/* Visual Aid Toggle Button */}
        <Button
          variant="outline"
          size="icon"
          className="absolute top-4 right-4 z-20 bg-background/80 backdrop-blur-sm"
          onClick={() => setShowVisualAids(!showVisualAids)}
        >
          <ImageIcon className="h-4 w-4" />
        </Button>
      </div>

      {/* Interactive Caption with learning features */}
      <InteractiveCaption
        script={getCurrentScript()}
        currentTime={currentTime}
        isPlaying={videoState === 'playing'}
        language={currentLanguage}
        onLanguageChange={setCurrentLanguage}
      />

      {/* Phase Navigation */}
      <PhaseNavigation
        currentPhase={currentPhase}
        onPhaseChange={handlePhaseChange}
        completedPhases={[]}
      />

      {/* Visual Aids Overlay - floats above content */}
      <VisualAidsOverlay
        dayOfYear={dayOfYear}
        phase={currentPhase}
        isVisible={showVisualAids}
        onClose={() => setShowVisualAids(false)}
      />
    </div>
  )
}
