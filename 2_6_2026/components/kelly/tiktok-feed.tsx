'use client'

// TikTok-Style Full-Screen Kelly Experience
// kellyOS Foundation - Premium iOS-like experience for iLearn hardware
// - Fluid water droplet animations on touch
// - Swipe up/down for phases (haptic feedback simulation)
// - Tap to play/pause (instant visual response)
// - Double tap to like (heart burst animation)
// - Long press for options
// - Minimal floating UI that DISAPPEARS when playing
// - Kelly fills ENTIRE viewport - she IS the experience

import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform } from 'framer-motion'
import { cn } from '@/lib/utils'
import { 
  Heart, MessageCircle, Share2, Bookmark, Play, Pause,
  Volume2, VolumeX, ChevronUp, ChevronDown,
  User, Settings, Check, X, Loader2, RotateCcw,
  BookOpen, ChevronLeft, ChevronRight
} from 'lucide-react'
import { t } from '@/lib/translations'
import { useVote } from '@/hooks/use-vote'
import { LessonPhase, KellyArchetype } from '@/lib/types'
import { ARCHETYPE_PERSONALITIES, type Archetype } from '@/lib/kelly-assets'
import PhaseInteraction from './phase-interactions'
import { ScaleControl } from './scale-control'
import { useUIScale } from '@/hooks/use-ui-scale'
import HookVotingUIComponent from './hook-voting-ui-component'
import { KellyDebugHUD } from './kelly-debug-hud'
import { ScriptTeleprompter } from './script-teleprompter'

// Phase data - monochromatic Apple glassmorphism aesthetic
const PHASES: { id: LessonPhase; label: string; color: string; gradient: string }[] = [
  { id: 'hook', label: 'Hook', color: 'rgba(255,255,255,0.8)', gradient: 'from-white/10 to-white/5' },
  { id: 'story', label: 'Story', color: 'rgba(255,255,255,0.8)', gradient: 'from-white/10 to-white/5' },
  { id: 'wonder', label: 'Wonder', color: 'rgba(255,255,255,0.8)', gradient: 'from-white/10 to-white/5' },
  { id: 'action', label: 'Action', color: 'rgba(255,255,255,0.8)', gradient: 'from-white/10 to-white/5' },
  { id: 'wisdom', label: 'Wisdom', color: 'rgba(255,255,255,0.8)', gradient: 'from-white/10 to-white/5' },
]

// Phase audio data from /api/lesson/today
export interface PhaseAudioData {
  audio_url: string | null
  duration_seconds: number
}

interface TikTokFeedProps {
  // Content
  lessonTitle: string
  caption: string
  // Hook Phase Cinematic Flow:
  hookFact?: string | null // 1. The TRUE fact statement learner votes on ("Sunlight bounces off...")
  hookText?: string | null // 2. Kelly's engagement question shown AFTER voting ("Have you ever wondered...")
  hookCorrectAnswer?: boolean | null // Always true - all hook facts are TRUE by design
  explanation?: string | null // 3. Full teaching moment revealed after voting (wonder_script)
  visualUrl?: string | null
  videoUrl?: string | null
  audioUrl?: string | null
  audioSource?: 'kelly_assets' | 'heygen' | 'tts' | 'none' | null // Track where audio comes from
  posterUrl?: string | null
  subtitle?: string | null
  theme?: string | null
  topic?: string | null
  summary?: string | null
  quote?: string | null
  quoteAuthor?: string | null
  currentPhase: LessonPhase
  completedPhases: LessonPhase[]
  dayOfYear: number
  formattedDate: string
  // Playback progress (0-1) for syncing with phase bars
  playbackProgress?: number
  // Stats
  likeCount: number
  isLiked: boolean
  commentCount: number
  isSaved: boolean
  // Handlers
  onPhaseChange: (phase: LessonPhase) => void
  onLike: () => void
  onComment: () => void
  onShare: () => void
  onSave: () => void
  onSettings: () => void
  onProfile: () => void
  onDictionary?: (word: string) => void // NEW: Look up words
  onMenu?: () => void // Open navigation drawer
  onCalendar?: () => void // Open lesson calendar picker
  onPrevDay?: () => void // NEW: Previous day handler
  onNextDay?: () => void // NEW: Next day handler
  // Full phase scripts for word-by-word teleprompter
  hookScript?: string | null
  storyScript?: string | null
  wonderScript?: string | null
  actionScript?: string | null
  wisdomScript?: string | null
  // Settings
  kellyAge: number
  archetype: KellyArchetype
  language: string
  onAgeChange: (age: number) => void
  onArchetypeChange: (archetype: KellyArchetype) => void
  onLanguageChange: (lang: string) => void
  // Phase audio from /api/lesson/today (for auto-advance and per-phase audio)
  getPhaseAudio?: (phase: LessonPhase) => PhaseAudioData | null
  // Subtitle text from kellyos_lessons content_text
  subtitleText?: string | null
}

export function TikTokFeed({
  lessonTitle,
  caption,
  hookFact,
  hookText,
  hookCorrectAnswer,
  explanation,
  visualUrl,
  videoUrl,
  audioUrl,
  audioSource,
  posterUrl,
  subtitle,
  theme,
  topic,
  summary,
  quote,
  quoteAuthor,
  currentPhase,
  completedPhases,
  dayOfYear,
  formattedDate,
  playbackProgress = 0,
  likeCount,
  isLiked,
  commentCount,
  isSaved,
  onPhaseChange,
  onLike,
  onComment,
  onShare,
  onSave,
  onSettings,
  onProfile,
  onDictionary,
  onMenu,
  onCalendar,
  onPrevDay,
  onNextDay,
  kellyAge,
  archetype,
  language,
  onAgeChange,
  onArchetypeChange,
  onLanguageChange,
  hookScript,
  storyScript,
  wonderScript,
  actionScript,
  wisdomScript,
  getPhaseAudio,
  subtitleText,
}: TikTokFeedProps) {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  
  // State
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [videoError, setVideoError] = useState(false) // Handle 410/404 video failures
  const [showHeart, setShowHeart] = useState(false)
  const [heartPosition, setHeartPosition] = useState({ x: 50, y: 50 })
  const [lastTap, setLastTap] = useState(0)
  const [touchStart, setTouchStart] = useState<{ y: number; time: number } | null>(null)
  const [showControls, setShowControls] = useState(true)
  
  const [showLessonDetails, setShowLessonDetails] = useState(false) // NEW: Expandable lesson info
  const [swipeDirection, setSwipeDirection] = useState<'up' | 'down' | null>(null)
  const [phaseTransition, setPhaseTransition] = useState<{ active: boolean; direction: 'up' | 'down'; toPhase: LessonPhase | null }>({ active: false, direction: 'up', toPhase: null })
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)
  
  // Hover preview state - Kelly changes INSTANTLY on hover
  const [hoverAge, setHoverAge] = useState<number | null>(null)
  const [hoverArchetype, setHoverArchetype] = useState<string | null>(null)
  
  // Derived values for display (hover takes precedence)
  const displayAge = hoverAge ?? kellyAge
  const displayArchetype = (hoverArchetype ?? archetype) || 'explorer'
  
  // Determine which avatar folder to use
  const getAvatarFolder = (age: number) => {
    if (age < 18) return 'kid'
    if (age >= 60) return 'senior'
    return 'archetypes'
  }
  const displayFolder = getAvatarFolder(displayAge)
  const [touchRipples, setTouchRipples] = useState<{ id: number; x: number; y: number }[]>([])
  const [showSettings, setShowSettings] = useState(false); // Declaration of setShowSettings
  const [teleprompterExpanded, setTeleprompterExpanded] = useState(true);
  
  // Lip-sync mouth animation state (0-1 amplitude, driven by audio analyser)
  const [mouthOpenness, setMouthOpenness] = useState(0)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const animFrameRef = useRef<number>(0)
  const sourceNodeRef = useRef<MediaElementAudioSourceNode | null>(null)
  
  // Phase auto-advance timer
  const autoAdvanceTimerRef = useRef<NodeJS.Timeout | null>(null)
  
  // Subtitle display state (current phase content_text)
  const [currentSubtitle, setCurrentSubtitle] = useState<string | null>(null);
  
  // Get the current phase script for the teleprompter
  const currentScript = useMemo(() => {
    switch (currentPhase) {
      case 'hook': return hookScript || caption || ''
      case 'story': return storyScript || caption || ''
      case 'wonder': return wonderScript || caption || ''
      case 'action': return actionScript || caption || ''
      case 'wisdom': return wisdomScript || caption || ''
      default: return caption || ''
    }
  }, [currentPhase, hookScript, storyScript, wonderScript, actionScript, wisdomScript, caption])
  const [loadingMsgIndex, setLoadingMsgIndex] = useState(0);
  
  // Fun rotating loading messages
  const loadingMessages = useMemo(() => [
    t('loadingMessage', language) || "Pardon while my systems come online...",
    t('loadingMessage1', language) || "Gathering today's adventure...",
    t('loadingMessage2', language) || "Connecting the dots...",
    t('loadingMessage3', language) || "Loading something wonderful...",
    t('loadingMessage4', language) || "Almost ready to learn together!",
    t('loadingMessage5', language) || "Warming up my curiosity...",
  ], [language]);
  
  // Rotate loading message every 2 seconds
  useEffect(() => {
    if (!isLoading) return;
    const interval = setInterval(() => {
      setLoadingMsgIndex(prev => (prev + 1) % loadingMessages.length);
    }, 2000);
    return () => clearInterval(interval);
  }, [isLoading, loadingMessages.length]);
  
  // ═══════════════════════════════════════════════════════════════════
  // PHASE AUDIO: When phase changes, load phase-specific audio
  // ═══════════════════════════════════════════════════════════════════
  useEffect(() => {
    if (!getPhaseAudio) return
    const phaseAudio = getPhaseAudio(currentPhase)
    if (phaseAudio?.audio_url && audioRef.current) {
      // Only update if URL actually changed
      const currentSrc = audioRef.current.src
      const newSrc = phaseAudio.audio_url
      if (currentSrc !== newSrc) {
        audioRef.current.src = newSrc
        audioRef.current.load()
      }
    }
  }, [currentPhase, getPhaseAudio])

  // ═══════════════════════════════════════════════════════════════════
  // AUTO-ADVANCE: After audio ends, auto-advance to next phase
  // ═══════════════════════════════════════════════════════════════════
  useEffect(() => {
    const audio = audioRef.current
    if (!audio || !getPhaseAudio) return
    
    const handleAudioEnded = () => {
      const phases: LessonPhase[] = ['hook', 'story', 'wonder', 'action', 'wisdom']
      const currentIndex = phases.indexOf(currentPhase)
      
      if (currentIndex < phases.length - 1) {
        const nextPhase = phases[currentIndex + 1]
        // Short delay before auto-advance for breathing room
        autoAdvanceTimerRef.current = setTimeout(() => {
          setPhaseTransition({ active: true, direction: 'up', toPhase: nextPhase })
          setTimeout(() => {
            onPhaseChange(nextPhase)
            setTimeout(() => setPhaseTransition({ active: false, direction: 'up', toPhase: null }), 400)
          }, 300)
        }, 1500) // 1.5s pause between phases
      } else {
        // Final phase completed
        setIsPlaying(false)
      }
    }
    
    audio.addEventListener('ended', handleAudioEnded)
    return () => {
      audio.removeEventListener('ended', handleAudioEnded)
      if (autoAdvanceTimerRef.current) {
        clearTimeout(autoAdvanceTimerRef.current)
      }
    }
  }, [currentPhase, getPhaseAudio, onPhaseChange])

  // ═══════════════════════════════════════════════════════════════════
  // SUBTITLE DISPLAY: Show current phase content_text 
  // ═══════════════════════════════════════════════════════════════════
  useEffect(() => {
    // Use subtitleText prop if provided, otherwise derive from scripts
    if (subtitleText !== undefined && subtitleText !== null) {
      setCurrentSubtitle(subtitleText)
    } else {
      setCurrentSubtitle(currentScript || null)
    }
  }, [subtitleText, currentScript])

  // ═══════════════════════════════════════════════════════════════════
  // LIP-SYNC: Analyze audio amplitude for mouth animation
  // ═══════════════════════════════════════════════════════════════════
  useEffect(() => {
    if (!isPlaying || !audioRef.current) {
      setMouthOpenness(0)
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      return
    }
    
    const audio = audioRef.current
    
    try {
      // Create AudioContext and AnalyserNode only once per audio element
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext()
      }
      const ctx = audioContextRef.current
      
      if (!analyserRef.current) {
        analyserRef.current = ctx.createAnalyser()
        analyserRef.current.fftSize = 256
        analyserRef.current.smoothingTimeConstant = 0.8
      }
      const analyser = analyserRef.current
      
      // Connect audio element to analyser (only once)
      if (!sourceNodeRef.current) {
        try {
          sourceNodeRef.current = ctx.createMediaElementSource(audio)
          sourceNodeRef.current.connect(analyser)
          analyser.connect(ctx.destination)
        } catch {
          // Source already connected - ignore
        }
      }
      
      // Resume context if suspended
      if (ctx.state === 'suspended') {
        ctx.resume()
      }
      
      const dataArray = new Uint8Array(analyser.frequencyBinCount)
      
      const updateMouth = () => {
        analyser.getByteFrequencyData(dataArray)
        
        // Calculate average amplitude from lower frequency bins (voice range)
        const voiceBins = dataArray.slice(2, 20) // ~80Hz - 800Hz voice range
        let sum = 0
        for (let i = 0; i < voiceBins.length; i++) {
          sum += voiceBins[i]
        }
        const avg = sum / voiceBins.length / 255 // Normalize to 0-1
        
        // Apply smoothing and threshold
        const openness = Math.min(1, Math.max(0, (avg - 0.05) * 2.5))
        setMouthOpenness(openness)
        
        animFrameRef.current = requestAnimationFrame(updateMouth)
      }
      
      animFrameRef.current = requestAnimationFrame(updateMouth)
    } catch (e) {
      console.warn('[TikTokFeed] Audio analyser setup failed:', e)
    }
    
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
      }
    }
  }, [isPlaying])

  // Cleanup audio context on unmount
  useEffect(() => {
    return () => {
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close()
      }
    }
  }, [])

  // Fluid motion values for water droplet effects
  const mouseX = useMotionValue(0)
  const mouseY = useMotionValue(0)
  const springConfig = { damping: 25, stiffness: 200 }
  const smoothX = useSpring(mouseX, springConfig)
  const smoothY = useSpring(mouseY, springConfig)
  
  // Parallax effects based on touch position
  const bgX = useTransform(smoothX, [0, typeof window !== 'undefined' ? window.innerWidth : 400], [5, -5])
  const bgY = useTransform(smoothY, [0, typeof window !== 'undefined' ? window.innerHeight : 800], [5, -5])
  const glowOpacity = useTransform(smoothY, [0, typeof window !== 'undefined' ? window.innerHeight : 800], [0.3, 0.1])
  
// Reset video error state when URL changes (new phase, day, or settings)
  useEffect(() => {
    setVideoError(false)
  }, [videoUrl])

  // Handle loading state for different content modes
  // If no video URL: stop loading after a short delay (audio-only or static image mode)
  useEffect(() => {
    if (!videoUrl) {
    // No video - stop loading spinner after 1s (let Kelly image + audio settle)
    const timer = setTimeout(() => setIsLoading(false), 1000)
      return () => clearTimeout(timer)
    }
  }, [videoUrl])

  // Show loading state when audio URL changes (new language/age/archetype selected)
  // This provides visual feedback that content is being regenerated
  useEffect(() => {
    if (audioUrl) {
      // Brief loading state to show content update
      setIsLoading(true)
      // Audio ref needs to reload with new source
      if (audioRef.current) {
        audioRef.current.load()
      }
      const timer = setTimeout(() => setIsLoading(false), 800)
      return () => clearTimeout(timer)
    }
  }, [audioUrl])
  
  // Video loading and progress handlers (only applies when videoUrl exists)
  useEffect(() => {
    const video = videoRef.current
    if (!video || !videoUrl) return
    
    const handleLoadStart = () => setIsLoading(true)
    const handleCanPlay = () => setIsLoading(false)
    const handleTimeUpdate = () => {
      setProgress(video.currentTime)
      setDuration(video.duration || 0)
    }
    const handleWaiting = () => setIsLoading(true)
    const handlePlaying = () => setIsLoading(false)
    const handleError = () => setIsLoading(false) // Stop loading on error too
    
    video.addEventListener('loadstart', handleLoadStart)
    video.addEventListener('canplay', handleCanPlay)
    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('waiting', handleWaiting)
    video.addEventListener('playing', handlePlaying)
    video.addEventListener('error', handleError)
    
    return () => {
      video.removeEventListener('loadstart', handleLoadStart)
      video.removeEventListener('canplay', handleCanPlay)
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('waiting', handleWaiting)
      video.removeEventListener('playing', handlePlaying)
      video.removeEventListener('error', handleError)
    }
  }, [videoUrl])
  
  // Auto-hide controls after 2s of playing
  useEffect(() => {
    if (isPlaying) {
      const timer = setTimeout(() => setShowControls(false), 5000)
      return () => clearTimeout(timer)
    } else {
      setShowControls(true)
    }
  }, [isPlaying])
  
  // Play/pause toggle with error handling for unsupported sources
  const togglePlay = useCallback(async () => {
    try {
      if (videoRef.current && videoUrl) {
        if (isPlaying) {
          videoRef.current.pause()
          setIsPlaying(false)
        } else {
          videoRef.current.muted = false
          await videoRef.current.play()
          setIsPlaying(true)
        }
        setIsMuted(false)
      } else if (audioRef.current && audioUrl && audioUrl.startsWith('http')) {
        if (isPlaying) {
          audioRef.current.pause()
          setIsPlaying(false)
        } else {
          audioRef.current.muted = false
          await audioRef.current.play()
          setIsPlaying(true)
        }
        setIsMuted(false)
      }
      setShowControls(true)
      setIsLoading(false) // Always clear loading after successful play/pause
    } catch (err) {
      // Handle play errors (no supported sources, autoplay blocked, etc.)
      console.warn('Play error:', err)
      setIsPlaying(false) // Ensure playing state is reset on error
      setIsLoading(false) // Clear loading spinner on error
    }
  }, [isPlaying, videoUrl, audioUrl])
  
  // Double tap to like with position-based heart
  const handleTap = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    const now = Date.now()
    const target = e.target as HTMLElement
    
    // Get tap position for heart animation
    let x = 50, y = 50
    if ('clientX' in e) {
      x = (e.clientX / window.innerWidth) * 100
      y = (e.clientY / window.innerHeight) * 100
    } else if (e.touches?.[0]) {
      x = (e.touches[0].clientX / window.innerWidth) * 100
      y = (e.touches[0].clientY / window.innerHeight) * 100
    }
    
    if (now - lastTap < 300) {
      // Double tap - like with heart at tap position
      setHeartPosition({ x, y })
      if (!isLiked) {
        onLike()
      }
      setShowHeart(true)
      setTimeout(() => setShowHeart(false), 800)
    } else {
      // Single tap - show controls or toggle play
      if (!target.closest('button') && !target.closest('[role="button"]')) {
        if (showControls) {
          togglePlay()
        } else {
          setShowControls(true)
        }
      }
    }
    setLastTap(now)
  }, [lastTap, isLiked, onLike, togglePlay, showControls])
  
  // Swipe navigation with visual feedback + fluid ripple
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    const touch = e.touches[0]
    setTouchStart({ y: touch.clientY, time: Date.now() })
    setSwipeDirection(null)
    
    // Update fluid motion tracking
    mouseX.set(touch.clientX)
    mouseY.set(touch.clientY)
    
    // Create water droplet ripple effect
    const ripple = { id: Date.now(), x: touch.clientX, y: touch.clientY }
    setTouchRipples(prev => [...prev, ripple])
    setTimeout(() => {
      setTouchRipples(prev => prev.filter(r => r.id !== ripple.id))
    }, 800)
  }, [mouseX, mouseY])
  
  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (!touchStart) return
    const deltaY = touchStart.y - e.touches[0].clientY
    if (Math.abs(deltaY) > 20) {
      setSwipeDirection(deltaY > 0 ? 'up' : 'down')
    }
  }, [touchStart])
  
  const handleTouchEnd = useCallback((e: React.TouchEvent) => {
    if (!touchStart) return
    
    const deltaY = touchStart.y - e.changedTouches[0].clientY
    const deltaTime = Date.now() - touchStart.time
    const velocity = Math.abs(deltaY) / deltaTime
    
    // Require significant swipe (40px or fast velocity)
    if (Math.abs(deltaY) > 40 || velocity > 0.4) {
      const phases: LessonPhase[] = ['hook', 'story', 'wonder', 'action', 'wisdom']
      const currentIndex = phases.indexOf(currentPhase)
      
      if (deltaY > 0 && currentIndex < phases.length - 1) {
        // Swipe up - next phase with transition
        const nextPhase = phases[currentIndex + 1]
        setPhaseTransition({ active: true, direction: 'up', toPhase: nextPhase })
        setTimeout(() => {
          onPhaseChange(nextPhase)
          setTimeout(() => setPhaseTransition({ active: false, direction: 'up', toPhase: null }), 400)
        }, 300)
      } else if (deltaY < 0 && currentIndex > 0) {
        // Swipe down - previous phase with transition
        const prevPhase = phases[currentIndex - 1]
        setPhaseTransition({ active: true, direction: 'down', toPhase: prevPhase })
        setTimeout(() => {
          onPhaseChange(prevPhase)
          setTimeout(() => setPhaseTransition({ active: false, direction: 'down', toPhase: null }), 400)
        }, 300)
      }
    }
    
    setTouchStart(null)
    setSwipeDirection(null)
  }, [touchStart, currentPhase, onPhaseChange])
  
  // Phase index for progress
  const phaseIndex = PHASES.findIndex(p => p.id === currentPhase)
  
  // Progress bar percentage
  const progressPercent = duration > 0 ? (progress / duration) * 100 : 0
  
  return (
    <div
      ref={containerRef}
      className="fixed inset-0 bg-black overflow-hidden touch-none select-none"
      onClick={handleTap}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      {/* KELLY - Full Screen Background with Fluid Parallax - SHE IS THE EXPERIENCE */}
      <motion.div 
        className={cn(
          "absolute inset-0",
          swipeDirection === 'up' && "-translate-y-4",
          swipeDirection === 'down' && "translate-y-4"
        )}
        style={{ x: bgX, y: bgY }}
        transition={{ type: 'spring', damping: 20 }}
      >
        {videoUrl && !videoError ? (
          <video
            ref={videoRef}
            src={videoUrl}
            poster={posterUrl || "/kelly/archetypes/explorer.png"}
            loop
            playsInline
            muted={isMuted}
            className="w-full h-full object-contain"
            onEnded={() => setIsPlaying(false)}
            onError={(e) => {
              // Handle 410 (Gone), 404 (Not Found), or other video load failures
              console.warn('[v0] Video failed to load:', videoUrl, e)
              setVideoError(true)
              setIsLoading(false)
            }}
            onLoadStart={() => setIsLoading(true)}
            onCanPlay={() => setIsLoading(false)}
          />
        ) : (
          <motion.img 
            src={(hoverAge !== null || hoverArchetype !== null) 
              ? `/kelly/${displayFolder}/${displayArchetype}.png`
              : (posterUrl || `/kelly/${displayFolder}/${displayArchetype}.png`)
            } 
            alt="Curious Kelly"
            className={cn(
              "w-full h-full object-contain",
              (hoverAge !== null || hoverArchetype !== null) && "brightness-105 saturate-110"
            )}
            // Organic breathing: scale + vertical drift + slight brightness shift
            animate={{
              scale: isPlaying 
                ? [1, 1.005, 1.003, 1] 
                : [1, 1.004, 1.002, 1],
              y: isPlaying 
                ? [0, -1, 0.5, 0] 
                : [0, -1.5, 0.5, 0],
              filter: isPlaying
                ? [
                    'contrast(1.04) saturate(1.12) brightness(1.02)',
                    'contrast(1.05) saturate(1.15) brightness(1.04)',
                    'contrast(1.04) saturate(1.10) brightness(1.02)',
                    'contrast(1.04) saturate(1.12) brightness(1.02)',
                  ]
                : [
                    'contrast(1.03) saturate(1.08) brightness(1.01)',
                    'contrast(1.04) saturate(1.12) brightness(1.025)',
                    'contrast(1.03) saturate(1.06) brightness(1.005)',
                    'contrast(1.03) saturate(1.08) brightness(1.01)',
                  ],
            }}
            transition={{
              duration: isPlaying ? 2.5 : 5,
              repeat: Number.POSITIVE_INFINITY,
              ease: [0.37, 0, 0.63, 1],
              times: [0, 0.35, 0.65, 1],
            }}
            // Crossfade on archetype/age hover
            initial={{ opacity: 0 }}
            key={`${displayFolder}-${displayArchetype}-${posterUrl}`}
            animate-enter={{ opacity: 1 }}
            whileInView={{ opacity: 1 }}
            onError={(e) => {
              (e.target as HTMLImageElement).src = '/kelly/archetypes/explorer.png'
            }}
          />
        )}
        {/* Lip-sync mouth overlay — basic open/close synced to audio amplitude */}
        {isPlaying && mouthOpenness > 0.05 && !videoUrl && (
          <motion.div
            className="absolute bottom-[38%] left-1/2 -translate-x-1/2 z-10 pointer-events-none"
            style={{
              width: `${28 + mouthOpenness * 12}px`,
              height: `${4 + mouthOpenness * 16}px`,
              borderRadius: '50%',
              background: `radial-gradient(ellipse, rgba(0,0,0,${0.15 + mouthOpenness * 0.25}) 0%, transparent 70%)`,
              filter: 'blur(1px)',
            }}
            animate={{
              scaleY: 0.5 + mouthOpenness * 1.5,
              scaleX: 1 + mouthOpenness * 0.3,
            }}
            transition={{ duration: 0.05 }}
          />
        )}
        
  {audioUrl && !videoUrl && audioUrl.startsWith('http') && (
    <audio ref={audioRef} src={audioUrl} />
  )}
      </motion.div>
      
      {/* PHASE TRANSITION OVERLAY - Beautiful full-screen transition */}
      <AnimatePresence>
        {phaseTransition.active && phaseTransition.toPhase && (
          <motion.div
            className="absolute inset-0 z-50 flex items-center justify-center pointer-events-none"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            {/* Background blur + gradient */}
            <motion.div 
              className={cn(
                "absolute inset-0 bg-gradient-to-b backdrop-blur-md",
                PHASES.find(p => p.id === phaseTransition.toPhase)?.gradient || 'from-white/10 to-black/60'
              )}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            />
            
            {/* Animated phase announcement */}
            <motion.div
              className="relative text-center z-10"
              initial={{ 
                y: phaseTransition.direction === 'up' ? 100 : -100, 
                opacity: 0,
                scale: 0.8
              }}
              animate={{ 
                y: 0, 
                opacity: 1,
                scale: 1
              }}
              exit={{ 
                y: phaseTransition.direction === 'up' ? -100 : 100, 
                opacity: 0,
                scale: 0.8
              }}
              transition={{ 
                type: 'spring', 
                damping: 25, 
                stiffness: 300 
              }}
            >
              {/* Phase number indicator */}
              <motion.div 
                className="w-16 h-16 rounded-full mb-4 flex items-center justify-center"
                style={{ 
                  background: 'linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05))',
                  border: '2px solid rgba(255,255,255,0.2)',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.3)'
                }}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.1, type: 'spring', damping: 15 }}
              >
                <span className="text-2xl font-bold text-white/90">
                  {PHASES.findIndex(p => p.id === phaseTransition.toPhase) + 1}
                </span>
              </motion.div>
              
              {/* Phase label */}
              <motion.h2 
                className="text-4xl md:text-6xl font-black tracking-tight text-white mb-2"
                style={{ 
                  textShadow: `0 0 40px ${PHASES.find(p => p.id === phaseTransition.toPhase)?.color || '#fff'}40`
                }}
                initial={{ letterSpacing: '0.2em', opacity: 0 }}
                animate={{ letterSpacing: '0.05em', opacity: 1 }}
                transition={{ delay: 0.15, duration: 0.3 }}
              >
                {PHASES.find(p => p.id === phaseTransition.toPhase)?.label || 'Next'}
              </motion.h2>
              
              {/* Phase number indicator */}
              <motion.div
                className="flex items-center justify-center gap-2 mt-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 }}
              >
                {PHASES.map((phase, i) => (
                  <div 
                    key={phase.id}
                    className={cn(
                      "w-2 h-2 rounded-full transition-all duration-300",
                      phase.id === phaseTransition.toPhase 
                        ? "w-8 bg-white" 
                        : "bg-white/30"
                    )}
                  />
                ))}
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Touch ripple effects - water droplet magic */}
      <AnimatePresence>
        {touchRipples.map(ripple => (
          <motion.div
            key={ripple.id}
            className="absolute pointer-events-none z-30"
            style={{ left: ripple.x, top: ripple.y }}
            initial={{ scale: 0, opacity: 0.5 }}
            animate={{ scale: 4, opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          >
            <div className="w-20 h-20 -ml-10 -mt-10 rounded-full border-2 border-white/30 bg-white/5" />
          </motion.div>
        ))}
      </AnimatePresence>
      
      {/* Ambient glow that follows touch */}
      <motion.div
        className="absolute inset-0 pointer-events-none z-5"
        style={{
          background: `radial-gradient(circle 300px at ${smoothX.get()}px ${smoothY.get()}px, rgba(255,255,255,0.08), transparent)`,
          opacity: glowOpacity,
        }}
      />
      
      {/* Loading spinner - Premium glass treatment with fun message */}
      <AnimatePresence>
        {isLoading && (
          <motion.div 
            className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-40 gap-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <motion.div 
              className="w-16 h-16 rounded-full bg-black/60 flex items-center justify-center border border-white/10 shadow-2xl"
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', damping: 20 }}
            >
              <div className="spinner-premium w-8 h-8" />
            </motion.div>
            <motion.p
              className="text-white/70 text-sm font-medium text-center text-balance font-normal tracking-wide max-w-[280px]"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              {loadingMessages[loadingMsgIndex]}
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* NO gradient overlays - Kelly is the star, no blur/gradients over her face */}
      
      {/* Double-tap heart animation - at tap position */}
      {showHeart && (
        <div 
          className="absolute pointer-events-none z-50"
          style={{ 
            left: `${heartPosition.x}%`, 
            top: `${heartPosition.y}%`,
            transform: 'translate(-50%, -50%)'
          }}
        >
          <Heart className="w-24 h-24 text-red-500 fill-red-500 animate-[ping_0.6s_ease-out]" />
          <Heart className="w-24 h-24 text-red-500 fill-red-500 absolute inset-0 animate-[scale-up_0.3s_ease-out]" />
        </div>
      )}
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* TOP BAR - Instagram Stories Style Phase Progress + Date Nav */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className={cn(
        "absolute top-0 left-0 right-0 z-40 transition-all duration-300",
        showControls ? "opacity-100" : "opacity-90"
      )}>
        <div className="pt-[env(safe-area-inset-top,12px)] px-4">
          {/* PHASE PROGRESS BARS - Always visible, Instagram Stories Style */}
          <div className="flex gap-1.5 mb-3">
            {PHASES.map((phase, index) => {
              const isActive = phase.id === currentPhase;
              const phaseIndex = PHASES.findIndex(p => p.id === currentPhase);
              const isPast = index < phaseIndex;
              const isFuture = index > phaseIndex;
              
              return (
                <button
                  key={phase.id}
                  onClick={(e) => {
                    e.stopPropagation();
                    const direction = index > phaseIndex ? 'up' : 'down';
                    setPhaseTransition({ active: true, direction, toPhase: phase.id });
                    setTimeout(() => {
                      onPhaseChange(phase.id);
                      setTimeout(() => setPhaseTransition({ active: false, direction: 'up', toPhase: null }), 400);
                    }, 300);
                  }}
                  className={cn(
                    "flex-1 rounded-full overflow-hidden relative cursor-pointer group transition-all duration-200",
                    isActive ? "h-[5px]" : "h-[4px]"
                  )}
                  style={{ 
                    backgroundColor: 'rgba(255,255,255,0.25)',
                    boxShadow: isActive ? `0 0 8px ${phase.color}40` : 'none'
                  }}
                  aria-label={`Go to ${phase.label} phase`}
                >
                  {/* Progress fill */}
                  <motion.div
                    className="absolute inset-y-0 left-0 rounded-full"
                    style={{ 
                      backgroundColor: isActive ? phase.color : isPast ? 'rgba(255,255,255,0.9)' : 'transparent',
                      boxShadow: isActive ? `0 0 6px ${phase.color}` : 'none'
                    }}
                    initial={false}
                    animate={{
                      width: isPast ? '100%' : isActive ? `${Math.max(playbackProgress * 100, 5)}%` : '0%',
                      opacity: isFuture ? 0.4 : 1
                    }}
                    transition={{ duration: 0.1, ease: 'linear' }}
                  />
                  
                  {/* Hover expand */}
                  <div className="absolute inset-0 group-hover:bg-white/20 transition-colors rounded-full" />
                </button>
              );
            })}
          </div>
          
          {/* Row 2: Date Nav + Phase Label + Media Source */}
          <div className="flex items-center justify-between gap-2">
            {/* Left: Date Navigation */}
            <div className="flex items-center gap-0.5">
              <button 
                onClick={(e) => { e.stopPropagation(); onPrevDay?.(); }} 
                disabled={dayOfYear <= 1} 
                className="p-1 text-white/50 hover:text-white active:scale-90 disabled:opacity-30 transition-all"
                aria-label={t('previous', language) || 'Previous day'}
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button 
                onClick={(e) => { e.stopPropagation(); onCalendar?.(); }} 
                className="flex items-center gap-1.5 px-1.5 py-0.5 rounded hover:bg-white/10 active:scale-95 transition-all"
              >
                <span className="text-white/80 text-xs font-medium">{formattedDate}</span>
                <span className="text-white/40 text-[10px]">Day {dayOfYear}</span>
              </button>
              <button 
                onClick={(e) => { e.stopPropagation(); onNextDay?.(); }} 
                disabled={dayOfYear >= 365} 
                className="p-1 text-white/50 hover:text-white active:scale-90 disabled:opacity-30 transition-all"
                aria-label={t('next', language) || 'Next day'}
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
            
            {/* Center: Active Phase Label - No icon */}
            <motion.div 
              key={currentPhase}
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-1.5"
            >
              <span 
                className="text-[10px] font-semibold uppercase tracking-[0.2em]"
                style={{ color: PHASES.find(p => p.id === currentPhase)?.color }}
              >
                {PHASES.find(p => p.id === currentPhase)?.label}
              </span>
            </motion.div>
            
            {/* Right: Enhanced Media Source Badge */}
            <div 
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg"
              style={{
                background: videoUrl 
                  ? 'linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05))' 
                  : audioUrl 
                    ? 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(59,130,246,0.05))'
                    : 'linear-gradient(135deg, rgba(161,161,170,0.1), rgba(161,161,170,0.03))',
                backdropFilter: 'blur(8px)',
                border: videoUrl 
                  ? '1px solid rgba(16,185,129,0.25)' 
                  : audioUrl 
                    ? '1px solid rgba(59,130,246,0.25)'
                    : '1px solid rgba(161,161,170,0.15)',
                boxShadow: videoUrl 
                  ? '0 2px 8px rgba(16,185,129,0.15)' 
                  : audioUrl 
                    ? '0 2px 8px rgba(59,130,246,0.15)'
                    : 'none',
              }}
            >
              {videoUrl ? (
                <>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" style={{ boxShadow: '0 0 6px rgba(16,185,129,0.6)' }} />
                  <span className="text-emerald-300 text-[10px] font-semibold tracking-wide">Video Ready</span>
                </>
              ) : audioUrl ? (
                <>
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" style={{ boxShadow: '0 0 6px rgba(59,130,246,0.6)' }} />
                  <span className="text-blue-300 text-[10px] font-semibold tracking-wide">
                    {audioSource === 'kelly_assets' ? 'Audio' : audioSource === 'tts' ? 'TTS' : 'Audio'}
                  </span>
                </>
              ) : (
                <>
                  <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-pulse" />
                  <span className="text-zinc-400 text-[10px] font-medium">Generating...</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* ═══════════════════════════════════════════════════════════════════
          KELLYOS 2028 - TWO-PLANE FLOATING INTERFACE
          Plane 1: Kelly (wallpaper layer) - Always visible, never obscured
          Plane 2: Controls (overlay layer) - Floating glass panels
          ═══════════════════════════════════════════════════════════════════ */}
      
      {/* RIGHT RAIL - Social Actions */}
      <div className="absolute right-3 bottom-36 z-20 pointer-events-auto flex flex-col gap-3">
        {[
          { icon: Heart, action: onLike, active: isLiked, activeColor: '#f87171', label: 'Like' },
          { icon: MessageCircle, action: onComment, label: 'Comments' },
          { icon: Bookmark, action: onSave, active: isSaved, fill: isSaved, label: 'Save' },
          { icon: Share2, action: onShare, label: 'Share' },
          { icon: BookOpen, action: () => onDictionary?.(''), label: 'Dictionary' },
        ].map(({ icon: Icon, action, active, activeColor, fill, label }, i) => (
          <motion.button
            key={i}
            onClick={(e) => { e.stopPropagation(); action(); }}
            className="action-btn-refined"
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.92 }}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 + i * 0.05, type: 'spring', damping: 20, stiffness: 300 }}
            aria-label={label}
          >
            <Icon 
              className={cn("w-[18px] h-[18px]", active ? "text-white" : "text-white/70")}
              style={active && activeColor ? { color: activeColor } : {}}
              fill={fill ? 'currentColor' : 'none'}
            />
          </motion.button>
        ))}
      </div>

      {/* LEFT RAIL - Profile & Settings */}
      <div className="absolute left-3 bottom-36 z-20 pointer-events-auto flex flex-col gap-3">
        <motion.button
          onClick={(e) => { e.stopPropagation(); onProfile(); }}
          className="action-btn-refined"
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.92 }}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.15, type: 'spring', damping: 20, stiffness: 300 }}
          aria-label="Profile"
        >
          <User className="w-[18px] h-[18px] text-white/70" />
        </motion.button>
        <motion.button
          onClick={(e) => { e.stopPropagation(); onSettings(); }}
          className="action-btn-refined"
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.92 }}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2, type: 'spring', damping: 20, stiffness: 300 }}
          aria-label="Settings"
        >
          <Settings className="w-[18px] h-[18px] text-white/70" />
        </motion.button>
      </div>

      {/* Duplicate progress bar removed - single set lives in top bar above */}

      {/* BOTTOM CONTENT AREA */}
      <div className="absolute inset-x-0 bottom-0 z-20 pointer-events-none flex flex-col items-center gap-3 pb-4 px-4">
        
        {/* Play Button - Premium frosted glass with breathing pulse */}
        <motion.button
          onClick={(e) => { e.stopPropagation(); togglePlay(); }}
          className="pointer-events-auto w-14 h-14 rounded-full flex items-center justify-center relative"
          style={{
            background: isPlaying 
              ? 'rgba(0, 0, 0, 0.35)' 
              : 'rgba(0, 0, 0, 0.45)',
            backdropFilter: 'blur(40px) saturate(180%)',
            WebkitBackdropFilter: 'blur(40px) saturate(180%)',
            boxShadow: isPlaying 
              ? 'inset 0 0.5px 0 rgba(255,255,255,0.1), 0 4px 16px rgba(0,0,0,0.15)'
              : 'inset 0 0.5px 0 rgba(255,255,255,0.18), 0 8px 32px rgba(0,0,0,0.25), 0 0 0 1px rgba(255,255,255,0.06)',
            border: '0.5px solid rgba(255,255,255,0.15)',
          }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {/* Breathing pulse ring when paused - organic, alive */}
          {!isPlaying && !isLoading && (
            <>
              <motion.div
                className="absolute inset-0 rounded-full border border-white/15"
                animate={{ scale: [1, 1.4, 1], opacity: [0.3, 0, 0.3] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
              />
              <motion.div
                className="absolute inset-0 rounded-full border border-white/10"
                animate={{ scale: [1, 1.6, 1], opacity: [0.15, 0, 0.15] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: 0.5 }}
              />
            </>
          )}
          {isLoading ? (
            <div className="w-5 h-5 border-[1.5px] border-white/25 border-t-white rounded-full animate-spin" />
          ) : isPlaying ? (
            <Pause className="w-5 h-5 text-white/90" />
          ) : (
            <Play className="w-5 h-5 text-white/90 ml-0.5" />
          )}
        </motion.button>
        
        {/* LEVEL 1: Archetype Greeting - Elegant italic caption */}
        <motion.div 
          className="flex items-center gap-2 pointer-events-none"
          key={archetype}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        >
          <span 
            className="text-white/70 text-[13px] font-light italic tracking-wide"
            style={{ textShadow: '0 1px 8px rgba(0,0,0,0.6)' }}
          >
            {ARCHETYPE_PERSONALITIES[archetype as Archetype]?.greeting}
          </span>
          <span className="text-sm opacity-80">{ARCHETYPE_PERSONALITIES[archetype as Archetype]?.emoji}</span>
        </motion.div>
        
        {/* LEVEL 2: Age Selector - hover changes background Kelly */}
        <div 
          className="pointer-events-auto selector-pill flex items-center gap-2"
          onMouseLeave={() => setHoverAge(null)}
        >
          {[
            { age: 8, folder: 'kid', label: 'Kid', desc: 'Ages 5-12' }, 
            { age: 25, folder: 'archetypes', label: 'Adult', desc: 'Ages 13-59' }, 
            { age: 65, folder: 'senior', label: 'Elder', desc: 'Ages 60+' }
          ].map(({ age, folder, label, desc }) => {
            const isSelected = (kellyAge < 18 && age === 8) || (kellyAge >= 18 && kellyAge < 60 && age === 25) || (kellyAge >= 60 && age === 65);
            const isHovered = hoverAge === age;
            return (
              <button 
                key={age}
                onClick={() => onAgeChange(age)}
                onMouseEnter={() => setHoverAge(age)}
                className={cn(
                  "flex items-center gap-1.5 transition-all duration-150 relative group/age",
                  isSelected ? "opacity-100" : "opacity-60 hover:opacity-100",
                  isHovered && !isSelected && "scale-105"
                )}
              >
                <div className={cn(
                  "rounded-full overflow-hidden transition-all duration-150",
                  isSelected ? "w-7 h-7 ring-2 ring-white/70" : "w-6 h-6",
                  isHovered && !isSelected && "ring-1 ring-white/40"
                )}>
                  <img src={`/kelly/${folder}/${displayArchetype}.png`} alt={label} className="w-full h-full object-cover" />
                </div>
                <span className={cn(
                  "text-[11px] font-medium transition-colors",
                  isSelected ? "text-white" : "text-white/60"
                )}>
                  {label}
                </span>
                {/* Tooltip */}
                <span className="absolute -top-7 left-1/2 -translate-x-1/2 px-2 py-1 rounded-lg text-[10px] font-medium text-white bg-black/80 backdrop-blur-sm opacity-0 group-hover/age:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50">
                  {desc}
                </span>
              </button>
            );
          })}
        </div>

        {/* LEVEL 3: Language Flags */}
        <div className="pointer-events-auto selector-pill flex items-center gap-1">

          {[
            { code: 'en', flag: '🇺🇸', label: 'English' }, 
            { code: 'es', flag: '🇪🇸', label: 'Español' }, 
            { code: 'fr', flag: '🇫🇷', label: 'Français' }, 
            { code: 'de', flag: '🇩🇪', label: 'Deutsch' }, 
            { code: 'pt', flag: '🇧🇷', label: 'Português' }, 
            { code: 'zh', flag: '🇨🇳', label: '中文' },
            { code: 'ja', flag: '🇯🇵', label: '日本語' },
            { code: 'ko', flag: '🇰🇷', label: '한국어' },
          ].map(({ code, flag, label }) => (
            <button 
              key={code} 
              onClick={() => onLanguageChange(code)} 
              className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center transition-all duration-150 relative group/lang",
                language === code 
                  ? "bg-white/20 scale-105 ring-1 ring-white/30" 
                  : "opacity-60 hover:opacity-100 hover:scale-110 hover:bg-white/10"
              )}
            >
              <span className="text-base">{flag}</span>
              <span className="absolute -top-7 left-1/2 -translate-x-1/2 px-2 py-1 rounded-lg text-[10px] font-medium text-white bg-black/80 backdrop-blur-sm opacity-0 group-hover/lang:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50">
                {label}
              </span>
            </button>
          ))}
        </div>

        {/* LEVEL 4: Archetype/Tone Selector (widest controls) */}
        <div 
          className="pointer-events-auto flex items-center gap-0.5 px-3 py-1.5 rounded-full"
          style={{
            background: 'linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%)',
            backdropFilter: 'blur(16px)',
            boxShadow: 'inset 0 0.5px 0 rgba(255,255,255,0.15)',
            border: '0.5px solid rgba(255,255,255,0.1)',
          }}
          onMouseLeave={() => setHoverArchetype(null)}
        >
          {(['storyteller', 'explorer', 'scientist', 'architect', 'strategist', 'diplomat', 'mystic', 'rebel', 'macgyver', 'empath', 'provider', 'survivor'] as KellyArchetype[]).map((id) => {
            const personality = ARCHETYPE_PERSONALITIES[id as Archetype]
            const isSelected = archetype === id
            const isHovered = hoverArchetype === id
            return (
              <button
                key={id}
                onClick={() => onArchetypeChange(id)}
                onMouseEnter={() => setHoverArchetype(id)}
                className={cn(
                  "w-7 h-7 rounded-full flex items-center justify-center transition-all duration-150 relative group/arch",
                  isSelected ? "bg-white/20 scale-110 ring-1 ring-white/30" : "opacity-50 hover:opacity-100 hover:scale-110",
                  isHovered && !isSelected && "bg-white/10 opacity-100"
                )}
              >
                <span className="text-xs">{personality?.emoji}</span>
                {/* Tooltip with greeting preview */}
                <span className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 rounded-lg text-[10px] font-medium text-white bg-black/80 backdrop-blur-sm opacity-0 group-hover/arch:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50 flex flex-col items-center">
                  <span className="font-semibold">{personality?.label}</span>
                  <span className="text-white/60 text-[9px]">{personality?.greeting?.substring(0, 25)}...</span>
                </span>
              </button>
            )
          })}
        </div>

        {/* LEVEL 5: Phase Content - Title + Word-by-Word Script Teleprompter */}
        <div className="flex flex-col items-center gap-2.5 max-w-[400px] w-full pointer-events-none">
          {/* Lesson Title - Prominent, balanced, timeless */}
          <motion.h2 
            className={cn(
              "text-white text-[16px] sm:text-[18px] font-semibold text-center text-balance tracking-[-0.01em] leading-snug transition-opacity duration-300",
              lessonTitle === 'Loading...' && "animate-pulse opacity-60"
            )}
            style={{ textShadow: '0 2px 12px rgba(0,0,0,0.7), 0 0 2px rgba(0,0,0,0.3)' }}
            key={lessonTitle}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          >
            {lessonTitle}
          </motion.h2>
          
          {/* SUBTITLE BAR — Shows current phase content_text from kellyos_lessons */}
          {isPlaying && currentSubtitle && currentPhase !== 'hook' && (
            <motion.div
              className="w-full px-2"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.3 }}
            >
              <div 
                className="py-2.5 px-4 rounded-2xl text-center max-h-[80px] overflow-y-auto"
                style={{
                  background: 'rgba(0, 0, 0, 0.55)',
                  backdropFilter: 'blur(24px)',
                  WebkitBackdropFilter: 'blur(24px)',
                  border: '0.5px solid rgba(255,255,255,0.08)',
                }}
              >
                <p 
                  className="text-white/90 text-[13px] leading-relaxed font-normal"
                  style={{ textShadow: '0 1px 4px rgba(0,0,0,0.5)' }}
                >
                  {currentSubtitle.length > 200 
                    ? currentSubtitle.slice(0, 200) + '...' 
                    : currentSubtitle}
                </p>
              </div>
            </motion.div>
          )}

          {/* Wisdom phase: show quote above teleprompter */}
          {currentPhase === 'wisdom' && quote && (
            <p 
              className="text-white text-[14px] italic text-center text-balance font-light leading-relaxed pointer-events-none"
              style={{ textShadow: '0 1px 4px rgba(0,0,0,0.5)' }}
            >
              &ldquo;{quote}&rdquo;
              {quoteAuthor && <span className="block text-white/60 text-[12px] not-italic mt-1">&mdash; {quoteAuthor}</span>}
            </p>
          )}

          {/* WORD-BY-WORD SCRIPT TELEPROMPTER - Kelly's exact words for this phase */}
          {currentScript && currentPhase !== 'hook' && (
            <ScriptTeleprompter
              script={currentScript}
              phase={currentPhase}
              isPlaying={isPlaying}
              onWordTap={(word) => onDictionary?.(word)}
              isExpanded={teleprompterExpanded}
              onToggleExpand={() => setTeleprompterExpanded(!teleprompterExpanded)}
            />
          )}
        </div>

        {/* LEVEL 6: True/False Voting (base - widest) */}
        {currentPhase === 'hook' && (
          <div className="pointer-events-auto w-full max-w-sm">
            <HookVotingUIComponent
              dayOfYear={dayOfYear}
              language={language}
              hookFact={hookFact} // The TRUE fact learner votes on
              hookQuestion={hookText} // Kelly's curiosity question shown AFTER voting
              correctAnswer={hookCorrectAnswer ?? undefined}
              teachingMoment={explanation} // Full explanation revealed after voting
            />
          </div>
        )}
      </div>
      
      {/* Swipe direction indicator */}
      {swipeDirection && (
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 flex justify-center pointer-events-none z-50">
          <div className={cn(
            "px-4 py-2 rounded-full bg-black/60 text-white text-sm font-medium",
            swipeDirection === 'up' ? "animate-[slide-up_0.2s_ease-out]" : "animate-[slide-down_0.2s_ease-out]"
          )}>
            {swipeDirection === 'up' 
              ? phaseIndex < PHASES.length - 1 
                ? `Next: ${PHASES[phaseIndex + 1]?.label}` 
                : 'Complete!'
              : phaseIndex > 0 
                ? `Back: ${PHASES[phaseIndex - 1]?.label}` 
                : 'Start'
            }
          </div>
        </div>
      )}
      
      {/* KELLY DEBUG HUD - DO NOT REMOVE UNTIL SHIP */}
      <KellyDebugHUD
        dayOfYear={dayOfYear}
        formattedDate={formattedDate}
        lessonTitle={lessonTitle}
        topic={topic}
        kellyAge={kellyAge}
        language={language}
        archetype={archetype}
        currentPhase={currentPhase}
        videoUrl={videoUrl}
        dbVideoUrl={videoUrl} // Same for now - will separate when pipeline adds lip-sync
        audioUrl={audioUrl}
        dbAudioUrl={audioUrl}
        audioSource={audioSource}
        hookText={hookText}
        storyText={caption} // Caption contains current phase text
        wonderText={explanation}
        actionText={summary}
        wisdomText={quote}
        lessonsFound={!!lessonTitle}
        perspectivesFound={!!subtitle || !!explanation}
        kellyAssetsFound={!!audioUrl || !!videoUrl}
        isVideoElement={!!videoUrl}
        isPlaying={isPlaying}
        errors={[]}
      />
    </div>
  )
}

// Add custom animations to globals.css or use these inline
const customStyles = `
@keyframes scale-up {
  0% { transform: scale(0); opacity: 0; }
  50% { transform: scale(1.2); opacity: 1; }
  100% { transform: scale(1); opacity: 0; }
}
@keyframes fade-out {
  0% { opacity: 1; }
  100% { opacity: 0; }
}
@keyframes slide-up {
  0% { transform: translateY(100%); }
  100% { transform: translateY(0); }
}
@keyframes slide-down {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(0); }
}
`

export default TikTokFeed
