'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Image from 'next/image'
import { cn } from '@/lib/utils'
import { 
  Play, Pause, SkipForward, Volume2, VolumeX, 
  Sparkles, BookOpen, Lightbulb, Zap, Quote,
  Download, CheckCircle, Loader2, Wifi, WifiOff
} from 'lucide-react'

// ═══════════════════════════════════════════════════════════════════════════
// PHASE CONFIGURATION - 100 seconds total, cinematic pacing
// ═══════════════════════════════════════════════════════════════════════════

const PHASES = [
  { id: 'hook', label: 'HOOK', duration: 15, color: '#f59e0b', icon: Sparkles, description: 'Grab attention' },
  { id: 'story', label: 'STORY', duration: 30, color: '#3b82f6', icon: BookOpen, description: 'Narrative journey' },
  { id: 'wonder', label: 'WONDER', duration: 20, color: '#8b5cf6', icon: Lightbulb, description: 'Deep questions' },
  { id: 'action', label: 'ACTION', duration: 20, color: '#10b981', icon: Zap, description: 'Hands-on moment' },
  { id: 'wisdom', label: 'WISDOM', duration: 15, color: '#f43f5e', icon: Quote, description: 'Key takeaway' },
] as const

type Phase = typeof PHASES[number]['id']

// ═══════════════════════════════════════════════════════════════════════════
// AUDIO WAVEFORM VISUALIZATION
// ═══════════════════════════════════════════════════════════════════════════

function AudioWaveform({ isPlaying, color }: { isPlaying: boolean; color: string }) {
  return (
    <div className="flex items-center justify-center gap-[3px] h-8">
      {[...Array(5)].map((_, i) => (
        <motion.div
          key={i}
          className="w-1 rounded-full"
          style={{ backgroundColor: color }}
          animate={{
            height: isPlaying ? [8, 24, 12, 28, 16][i % 5] : 4,
            opacity: isPlaying ? 1 : 0.3,
          }}
          transition={{
            duration: 0.4,
            repeat: isPlaying ? Infinity : 0,
            repeatType: 'reverse',
            delay: i * 0.1,
          }}
        />
      ))}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// PROGRESS RING - Surrounds Kelly avatar
// ═══════════════════════════════════════════════════════════════════════════

function ProgressRing({ progress, color, size = 200 }: { progress: number; color: string; size?: number }) {
  const strokeWidth = 3
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const strokeDashoffset = circumference - (progress / 100) * circumference

  return (
    <svg
      width={size}
      height={size}
      className="absolute -rotate-90 pointer-events-none"
      style={{ transform: 'rotate(-90deg)' }}
    >
      {/* Background ring */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="rgba(255,255,255,0.1)"
        strokeWidth={strokeWidth}
      />
      {/* Progress ring */}
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        initial={{ strokeDashoffset: circumference }}
        animate={{ strokeDashoffset }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        style={{
          strokeDasharray: circumference,
        }}
      />
    </svg>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// SCRIPT DISPLAY - Highlights words as Kelly speaks
// ═══════════════════════════════════════════════════════════════════════════

function ScriptDisplay({ 
  script, 
  currentTime, 
  duration,
  isPlaying 
}: { 
  script: string | null; 
  currentTime: number; 
  duration: number;
  isPlaying: boolean;
}) {
  if (!script) return null
  
  // Simple word highlighting based on time progress
  const words = script.split(' ')
  const progress = duration > 0 ? currentTime / duration : 0
  const highlightedWordIndex = Math.floor(progress * words.length)
  
  return (
    <motion.div 
      className="absolute bottom-24 left-1/2 -translate-x-1/2 w-full max-w-2xl px-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
    >
      <div className="glass-panel p-4 rounded-2xl text-center">
        <p className="text-sm leading-relaxed">
          {words.map((word, i) => (
            <span
              key={i}
              className={cn(
                'transition-all duration-150',
                i < highlightedWordIndex 
                  ? 'text-white' 
                  : i === highlightedWordIndex && isPlaying
                    ? 'text-white font-medium'
                    : 'text-zinc-500'
              )}
            >
              {word}{' '}
            </span>
          ))}
        </p>
      </div>
    </motion.div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT - CINEMATIC LESSON EXPERIENCE
// ═══════════════════════════════════════════════════════════════════════════

interface CinematicLessonProps {
  dayNumber: number
  ageGroup?: string
  language?: string
  archetype?: string
  onComplete?: () => void
}

export function CinematicLesson({
  dayNumber,
  ageGroup = 'adult',
  language = 'en',
  archetype = 'explorer',
  onComplete,
}: CinematicLessonProps) {
  const [currentPhase, setCurrentPhase] = useState<Phase>('hook')
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [isOffline, setIsOffline] = useState(false)
  const [lessonData, setLessonData] = useState<{
    audioUrl: string | null
    script: string | null
    title: string | null
    fallbackUrl: string | null
  }>({ audioUrl: null, script: null, title: null, fallbackUrl: null })
  
  const audioRef = useRef<HTMLAudioElement>(null)
  const phaseIndex = PHASES.findIndex(p => p.id === currentPhase)
  const currentPhaseConfig = PHASES[phaseIndex]
  
  // ═══════════════════════════════════════════════════════════════════════════
  // OFFLINE DETECTION
  // ═══════════════════════════════════════════════════════════════════════════
  
  useEffect(() => {
    const handleOnline = () => setIsOffline(false)
    const handleOffline = () => setIsOffline(true)
    
    setIsOffline(!navigator.onLine)
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])
  
  // ═══════════════════════════════════════════════════════════════════════════
  // FETCH LESSON DATA
  // ═══════════════════════════════════════════════════════════════════════════
  
  useEffect(() => {
    const fetchLesson = async () => {
      setIsLoading(true)
      
      try {
        const res = await fetch(
          `/api/video/url?day=${dayNumber}&phase=${currentPhase}&ageGroup=${ageGroup}&language=${language}&archetype=${archetype}`
        )
        const data = await res.json()
        
        setLessonData({
          audioUrl: data.audioUrl || data.url,
          script: data.script,
          title: data.title,
          fallbackUrl: data.fallbackUrl,
        })
      } catch (err) {
        console.error('[CinematicLesson] Fetch error:', err)
      } finally {
        setIsLoading(false)
      }
    }
    
    fetchLesson()
  }, [dayNumber, currentPhase, ageGroup, language, archetype])
  
  // ═══════════════════════════════════════════════════════════════════════════
  // AUDIO CONTROLS
  // ═══════════════════════════════════════════════════════════════════════════
  
  const togglePlay = useCallback(() => {
    if (!audioRef.current) return
    
    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }, [isPlaying])
  
  const toggleMute = useCallback(() => {
    if (!audioRef.current) return
    audioRef.current.muted = !isMuted
    setIsMuted(!isMuted)
  }, [isMuted])
  
  const nextPhase = useCallback(() => {
    const nextIndex = (phaseIndex + 1) % PHASES.length
    
    // Stop current audio
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    
    setIsPlaying(false)
    setCurrentTime(0)
    setCurrentPhase(PHASES[nextIndex].id)
    
    if (nextIndex === 0) {
      onComplete?.()
    }
  }, [phaseIndex, onComplete])
  
  // Audio event handlers
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    
    const handleTimeUpdate = () => setCurrentTime(audio.currentTime)
    const handleDurationChange = () => setDuration(audio.duration)
    const handleEnded = () => {
      setIsPlaying(false)
      // Auto-advance to next phase after a brief pause
      setTimeout(nextPhase, 1500)
    }
    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    
    audio.addEventListener('timeupdate', handleTimeUpdate)
    audio.addEventListener('durationchange', handleDurationChange)
    audio.addEventListener('ended', handleEnded)
    audio.addEventListener('play', handlePlay)
    audio.addEventListener('pause', handlePause)
    
    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate)
      audio.removeEventListener('durationchange', handleDurationChange)
      audio.removeEventListener('ended', handleEnded)
      audio.removeEventListener('play', handlePlay)
      audio.removeEventListener('pause', handlePause)
    }
  }, [nextPhase])
  
  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════
  
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0
  
  return (
    <div className="relative w-full h-screen bg-black overflow-hidden">
      {/* Hidden audio element */}
      {lessonData.audioUrl && (
        <audio
          ref={audioRef}
          src={lessonData.audioUrl}
          preload="auto"
        />
      )}
      
      {/* Background gradient based on phase */}
      <motion.div
        className="absolute inset-0 opacity-20"
        animate={{
          background: `radial-gradient(circle at 50% 50%, ${currentPhaseConfig.color}40, transparent 70%)`,
        }}
        transition={{ duration: 0.8 }}
      />
      
      {/* Phase indicator bar - top */}
      <div className="absolute top-0 left-0 right-0 p-4 z-20">
        <div className="flex items-center justify-center gap-2">
          {PHASES.map((phase, i) => {
            const isActive = phase.id === currentPhase
            const isPast = i < phaseIndex
            
            return (
              <button
                key={phase.id}
                onClick={() => setCurrentPhase(phase.id)}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-300',
                  isActive 
                    ? 'bg-white/10 backdrop-blur-md border border-white/20' 
                    : isPast
                      ? 'bg-white/5 text-zinc-400'
                      : 'text-zinc-600 hover:text-zinc-400'
                )}
              >
                <phase.icon className="w-4 h-4" style={{ color: isActive ? phase.color : undefined }} />
                <span className={cn('text-xs font-medium', isActive && 'text-white')}>
                  {phase.label}
                </span>
                {isPast && <CheckCircle className="w-3 h-3 text-emerald-500" />}
              </button>
            )
          })}
        </div>
      </div>
      
      {/* Network status */}
      <div className="absolute top-4 right-4 z-20">
        {isOffline ? (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-500/20 text-amber-400 text-xs">
            <WifiOff className="w-3 h-3" />
            <span>Offline</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/20 text-emerald-400 text-xs">
            <Wifi className="w-3 h-3" />
            <span>Connected</span>
          </div>
        )}
      </div>
      
      {/* Day number */}
      <div className="absolute top-4 left-4 z-20">
        <div className="text-xs text-zinc-500 font-mono">
          DAY {String(dayNumber).padStart(3, '0')}
        </div>
        {lessonData.title && (
          <div className="text-sm text-white mt-1 font-medium">
            {lessonData.title}
          </div>
        )}
      </div>
      
      {/* Center stage - Kelly avatar with progress ring */}
      <div className="absolute inset-0 flex items-center justify-center">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentPhase}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.1 }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            className="relative"
          >
            {/* Progress ring */}
            <ProgressRing 
              progress={progress} 
              color={currentPhaseConfig.color}
              size={280}
            />
            
            {/* Kelly avatar */}
            <div className="w-[280px] h-[280px] rounded-full overflow-hidden bg-zinc-900 flex items-center justify-center">
              {isLoading ? (
                <Loader2 className="w-12 h-12 text-zinc-600 animate-spin" />
              ) : lessonData.fallbackUrl ? (
                <Image
                  src={lessonData.fallbackUrl || "/placeholder.svg"}
                  alt="Kelly"
                  width={280}
                  height={280}
                  className="object-cover"
                  priority
                />
              ) : (
                <div className="w-full h-full bg-gradient-to-br from-zinc-800 to-zinc-900 flex items-center justify-center">
                  <span className="text-6xl">K</span>
                </div>
              )}
            </div>
            
            {/* Audio waveform below avatar */}
            <div className="absolute -bottom-12 left-1/2 -translate-x-1/2">
              <AudioWaveform isPlaying={isPlaying} color={currentPhaseConfig.color} />
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
      
      {/* Script display */}
      <AnimatePresence>
        {lessonData.script && isPlaying && (
          <ScriptDisplay
            script={lessonData.script}
            currentTime={currentTime}
            duration={duration}
            isPlaying={isPlaying}
          />
        )}
      </AnimatePresence>
      
      {/* Bottom controls */}
      <div className="absolute bottom-0 left-0 right-0 p-6 z-20">
        <div className="flex items-center justify-center gap-4">
          {/* Mute toggle */}
          <button
            onClick={toggleMute}
            className="p-3 rounded-full bg-white/5 hover:bg-white/10 transition-colors"
          >
            {isMuted ? (
              <VolumeX className="w-5 h-5 text-zinc-400" />
            ) : (
              <Volume2 className="w-5 h-5 text-white" />
            )}
          </button>
          
          {/* Play/pause */}
          <button
            onClick={togglePlay}
            disabled={isLoading || !lessonData.audioUrl}
            className={cn(
              'p-5 rounded-full transition-all duration-300',
              isLoading || !lessonData.audioUrl
                ? 'bg-zinc-800 text-zinc-600'
                : 'bg-white text-black hover:scale-105'
            )}
          >
            {isLoading ? (
              <Loader2 className="w-8 h-8 animate-spin" />
            ) : isPlaying ? (
              <Pause className="w-8 h-8" />
            ) : (
              <Play className="w-8 h-8 ml-1" />
            )}
          </button>
          
          {/* Skip to next phase */}
          <button
            onClick={nextPhase}
            className="p-3 rounded-full bg-white/5 hover:bg-white/10 transition-colors"
          >
            <SkipForward className="w-5 h-5 text-white" />
          </button>
        </div>
        
        {/* Progress bar */}
        <div className="mt-4 max-w-md mx-auto">
          <div className="h-1 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{ backgroundColor: currentPhaseConfig.color }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.1 }}
            />
          </div>
          <div className="flex justify-between mt-2 text-xs text-zinc-500 font-mono">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

function formatTime(seconds: number): string {
  if (!seconds || isNaN(seconds)) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export default CinematicLesson
