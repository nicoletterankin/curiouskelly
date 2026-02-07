"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import Link from "next/link"
import { motion, AnimatePresence } from "framer-motion"
import { PersonalizationRail } from "./personalization-rail"

// Kelly avatar mapping - age bucket to folder, archetype to image
const getKellyAvatar = (age: string, archetype: string): string => {
  const ageFolder = age === 'child' || age === 'youth' ? 'kid' 
                  : age === 'elder' ? 'senior' 
                  : 'archetypes'
  const archetypeMap: Record<string, string> = {
    'explorer': 'explorer', 'sage': 'scientist', 'hero': 'survivor',
    'caregiver': 'provider', 'creator': 'macgyver', 'rebel': 'rebel',
    'lover': 'empath', 'jester': 'storyteller', 'everyman': 'diplomat',
    'ruler': 'strategist', 'magician': 'mystic', 'innocent': 'empath',
    'architect': 'architect', 'empath': 'empath', 'mystic': 'mystic',
    'provider': 'provider', 'diplomat': 'diplomat', 'maker': 'macgyver',
    'scientist': 'scientist', 'strategist': 'strategist',
  }
  return `/kelly/${ageFolder}/${archetypeMap[archetype] || 'explorer'}.png`
}

// ═══════════════════════════════════════════════════════════════════════════
// FEEDBACK NERVOUS SYSTEM - Kelly learns from every interaction
// ═══════════════════════════════════════════════════════════════════════════

const SUPABASE_URL = 'https://tvrzja1xosyryjphkforjv.supabase.co'

// Get or create persistent student ID
const getStudentId = () => {
  if (typeof window === 'undefined') return 'server'
  let id = localStorage.getItem('kelly_student_id')
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem('kelly_student_id', id)
  }
  return id
}

// Progress tracking
const getProgress = () => {
  if (typeof window === 'undefined') return {}
  return JSON.parse(localStorage.getItem('kelly_progress') || '{}')
}

const updateLocalProgress = (day: number, phase: string) => {
  if (typeof window === 'undefined') return
  const progress = getProgress()
  if (!progress[day]) progress[day] = {}
  progress[day][phase] = true
  localStorage.setItem('kelly_progress', JSON.stringify(progress))
}

const isLessonComplete = (day: number) => {
  const progress = getProgress()
  const phases = ['hook', 'story', 'wonder', 'action', 'wisdom']
  return phases.every(p => progress[day]?.[p])
}

const getStreak = () => {
  if (typeof window === 'undefined') return 0
  return parseInt(localStorage.getItem('kelly_streak') || '0')
}

const updateStreak = () => {
  if (typeof window === 'undefined') return
  const lastLesson = localStorage.getItem('kelly_last_lesson_date')
  const today = new Date().toDateString()
  
  if (lastLesson === today) return
  
  const yesterday = new Date(Date.now() - 86400000).toDateString()
  if (lastLesson === yesterday) {
    const streak = getStreak() + 1
    localStorage.setItem('kelly_streak', streak.toString())
  } else {
    localStorage.setItem('kelly_streak', '1')
  }
  localStorage.setItem('kelly_last_lesson_date', today)
}

type Phase = "hook" | "story" | "wonder" | "action" | "wisdom"
type Archetype = "explorer" | "sage" | "hero" | "caregiver" | "creator" | "rebel" | "lover" | "jester" | "everyman" | "ruler" | "magician" | "innocent"

const PHASES = [
  { id: "hook", label: "HOOK", color: "#f59e0b", icon: "✦" },
  { id: "story", label: "STORY", color: "#3b82f6", icon: "◈" },
  { id: "wonder", label: "WONDER", color: "#8b5cf6", icon: "◇" },
  { id: "action", label: "ACTION", color: "#10b981", icon: "▸" },
  { id: "wisdom", label: "WISDOM", color: "#f43f5e", icon: "❖" },
]

const ARCHETYPES = [
  { id: "explorer", emoji: "🧭", desc: "Discovery and adventure" },
  { id: "sage", emoji: "📚", desc: "Wisdom and understanding" },
  { id: "hero", emoji: "⚔️", desc: "Achievement and mastery" },
  { id: "caregiver", emoji: "💗", desc: "Helping and nurturing" },
  { id: "creator", emoji: "🎨", desc: "Making new things" },
  { id: "rebel", emoji: "⚡", desc: "Independence and change" },
  { id: "lover", emoji: "💕", desc: "Connection and intimacy" },
  { id: "jester", emoji: "🎭", desc: "Joy and humor" },
  { id: "everyman", emoji: "🤝", desc: "Belonging and connection" },
  { id: "ruler", emoji: "👑", desc: "Leadership and order" },
  { id: "magician", emoji: "✨", desc: "Transformation and power" },
  { id: "innocent", emoji: "🌸", desc: "Safety and happiness" },
]

const AGES = [
  { id: "child", emoji: "👒", age: "2-7", label: "Child" },
  { id: "youth", emoji: "🎒", age: "8-14", label: "Youth" },
  { id: "adult", emoji: "💼", age: "15-44", label: "Adult" },
  { id: "mature", emoji: "👔", age: "45-64", label: "Mature" },
  { id: "elder", emoji: "📚", age: "65+", label: "Elder" },
]

const LANGS = [
  { code: "en", flag: "🇺🇸", label: "English", hasAudio: true },
  { code: "es", flag: "🇪🇸", label: "Español", hasAudio: false },
  { code: "hi", flag: "🇮🇳", label: "हिन्दी", hasAudio: false },
  { code: "zh", flag: "🇨🇳", label: "中文", hasAudio: false },
  { code: "pt", flag: "🇧🇷", label: "Português", hasAudio: false },
  { code: "ar", flag: "🇸🇦", label: "العربية", hasAudio: false },
  { code: "fr", flag: "🇫🇷", label: "Français", hasAudio: false },
  { code: "ru", flag: "🇷🇺", label: "Русский", hasAudio: false },
  { code: "ja", flag: "🇯🇵", label: "日本語", hasAudio: false },
  { code: "de", flag: "🇩🇪", label: "Deutsch", hasAudio: false },
  { code: "ko", flag: "🇰🇷", label: "한국어", hasAudio: false },
  { code: "it", flag: "🇮🇹", label: "Italiano", hasAudio: false },
]

interface KellyPlayerProps {
  dayNumber?: number
  lessonTitle?: string
  lessonCaption?: string
  videoUrl?: string
  initialPhase?: Phase
  onPhaseChange?: (phase: Phase) => void
  onAgeChange?: (age: string) => void
  onLangChange?: (lang: string) => void
}

export default function KellyPlayer({
  dayNumber = 27,
  lessonTitle,
  lessonCaption,
  videoUrl,
  initialPhase = "hook",
  onPhaseChange,
  onAgeChange,
  onLangChange,
}: KellyPlayerProps) {
  const [phase, setPhase] = useState<Phase>(initialPhase)
  const [archetype, setArchetype] = useState<Archetype>("explorer")
  const [age, setAge] = useState("adult")
  const [lang, setLang] = useState("en")
  const [liked, setLiked] = useState(false)
  const [saved, setSaved] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [hookVote, setHookVote] = useState<"true" | "false" | null>(null)
  const [hoveredArchetype, setHoveredArchetype] = useState<string | null>(null)
  const [phaseTransition, setPhaseTransition] = useState(false)
  const [audioCompleted, setAudioCompleted] = useState(false)
  const [communityResults, setCommunityResults] = useState<{ true_votes: number; false_votes: number } | null>(null)
  const [streak, setStreak] = useState(0)
  const [showCelebration, setShowCelebration] = useState(false)
  const [completedPhases, setCompletedPhases] = useState<Phase[]>([])
  
  // Lesson data state
  const [lessonData, setLessonData] = useState<{
    title: string
    caption: string
    audioUrl: string | null
    videoUrl: string | null
    fallbackUrl: string | null
    visualUrl: string | null
    script: string | null
    loading: boolean
    error: string | null
  }>({
    title: lessonTitle || "Loading...",
    caption: lessonCaption || "",
    audioUrl: null,
    videoUrl: null,
    fallbackUrl: null,
    visualUrl: null,
    script: null,
    loading: true,
    error: null
  })
  
  // Get phase-specific visual URL with fallback chain
  const getPhaseVisual = useCallback((currentPhase: string, day: number): string | null => {
    const paddedDay = day.toString().padStart(3, '0')
    // Phase visual availability:
    // hook: days 1-90, wonder: days 1-50, story/action/wisdom: sparse
    const phaseVisualLimits: Record<string, number> = {
      hook: 90,
      wonder: 50,
      story: 18, // only day 18
      action: 18,
      wisdom: 18,
    }
    const limit = phaseVisualLimits[currentPhase] || 0
    if (day <= limit || (currentPhase !== 'hook' && currentPhase !== 'wonder' && day === 18)) {
      return `/visuals/${currentPhase}/day-${paddedDay}.jpg`
    }
    // Fallback: use hook visual if available
    if (day <= 90) {
      return `/visuals/hook/day-${paddedDay}.jpg`
    }
    return null
  }, [])
  
  // Refs for heartbeat tracking
  const heartbeatRef = useRef<NodeJS.Timeout | null>(null)
  const sessionStartRef = useRef<number>(Date.now())
  const studentIdRef = useRef<string>('')
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // ═══════════════════════════════════════════════════════════════════════════
  // LESSON DATA FETCHING - Load audio/video for current phase
  // ══════════════════════════���════════════════════════════════════════════════
  
  useEffect(() => {
    const fetchLessonData = async () => {
      setLessonData(prev => ({ ...prev, loading: true, error: null }))
      
      try {
        // Fetch video/audio URL from API
        const res = await fetch(`/api/video/url?day=${dayNumber}&phase=${phase}&ageGroup=${age}&language=${lang}&archetype=${archetype}`)
        const data = await res.json()
        
        if (data.error) {
          setLessonData(prev => ({ ...prev, loading: false, error: data.error }))
          return
        }
        
        setLessonData({
          title: data.title || lessonTitle || `Day ${dayNumber}`,
          caption: data.script?.slice(0, 100) || lessonCaption || "",
          audioUrl: data.audioUrl || data.url,
          videoUrl: data.url,
          fallbackUrl: data.fallbackUrl || getKellyAvatar(age, archetype),
          visualUrl: data.visualUrl || getPhaseVisual(phase, dayNumber),
          script: data.script,
          loading: false,
          error: null
        })
      } catch (err) {
        console.error('[KellyPlayer] Failed to fetch lesson data:', err)
        setLessonData(prev => ({ 
          ...prev, 
          loading: false, 
          error: 'Failed to load lesson content'
        }))
      }
    }
    
    fetchLessonData()
  }, [dayNumber, phase, age, lang, archetype, lessonTitle, lessonCaption])

  // ═══════════════════════════════════════════════════════════════════════════
  // HEARTBEAT TRACKING - Every 10 seconds Kelly knows you're learning
  // ═══════════════════════════════════════════════════════════════════════════
  
  const sendHeartbeat = useCallback(async (event: 'start' | 'watching' | 'end', duration?: number) => {
    const payload = {
      student_id: studentIdRef.current,
      lesson_day: dayNumber,
      phase,
      event,
      duration_seconds: duration,
      timestamp: new Date().toISOString()
    }
    console.log('[Kelly Heartbeat]', event, payload)
    
    // Send to Supabase when endpoint is ready
    // await fetch(`${SUPABASE_URL}/functions/v1/feedback-heartbeat`, {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(payload)
    // })
  }, [dayNumber, phase])

  const startHeartbeat = useCallback(() => {
    sessionStartRef.current = Date.now()
    sendHeartbeat('start')
    
    heartbeatRef.current = setInterval(() => {
      sendHeartbeat('watching')
    }, 10000)
  }, [sendHeartbeat])

  const stopHeartbeat = useCallback(() => {
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current)
      const duration = Math.floor((Date.now() - sessionStartRef.current) / 1000)
      sendHeartbeat('end', duration)
    }
  }, [sendHeartbeat])

  // Initialize on mount
  useEffect(() => {
    studentIdRef.current = getStudentId()
    setStreak(getStreak())
    startHeartbeat()
    
    return () => stopHeartbeat()
  }, [startHeartbeat, stopHeartbeat])

  // Restart heartbeat on phase change
  useEffect(() => {
    stopHeartbeat()
    startHeartbeat()
  }, [phase, startHeartbeat, stopHeartbeat])

  // ═══════════════════════════════════════════════════════════════════════════
  // VOTE SUBMISSION - True/False captures understanding
  // ═══════════════════════════════════════════════════════════════════════════
  
  const submitVote = async (vote: 'true' | 'false') => {
    setHookVote(vote)
    
    const timeToVote = Math.floor((Date.now() - sessionStartRef.current) / 1000)
    const payload = {
      student_id: studentIdRef.current,
      lesson_day: dayNumber,
      phase,
      vote,
      time_to_vote_seconds: timeToVote
    }
    console.log('[Kelly Vote]', payload)
    
    // Simulate community results (will be real API later)
    setTimeout(() => {
      setCommunityResults({
        true_votes: vote === 'true' ? 847 : 423,
        false_votes: vote === 'false' ? 847 : 423
      })
    }, 500)
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // COMPLETION TRACKING - Phase and lesson completion
  // ═══════════════════════════════════════════════════════════════════════════
  
  const completePhase = () => {
    stopHeartbeat()
    updateLocalProgress(dayNumber, phase)
    
    console.log('[Kelly Complete]', { day: dayNumber, phase })
    
    if (isLessonComplete(dayNumber)) {
      updateStreak()
      setStreak(getStreak())
      setShowCelebration(true)
      setTimeout(() => setShowCelebration(false), 3000)
    }
  }

  const handlePhaseChange = (newPhase: Phase) => {
    if (newPhase === phase) return
    
    // Complete current phase before moving
    completePhase()
    
    // Cinematic phase transition: fade to black, change, fade in
    setPhaseTransition(true)
    setTimeout(() => {
      setPhase(newPhase)
      setHookVote(null)
      setCommunityResults(null)
      setAudioCompleted(false)
      onPhaseChange?.(newPhase)
      // Fetch new content (triggers useEffect)
    }, 300) // Hold black for 300ms
    
    setTimeout(() => {
      setPhaseTransition(false)
    }, 450) // Total transition: 300ms fade + 150ms reveal
  }

  const handleAgeChange = (newAge: string) => {
    setAge(newAge)
    onAgeChange?.(newAge)
  }

  const handleLangChange = (newLang: string) => {
    setLang(newLang)
    onLangChange?.(newLang)
  }

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      {/* Header - Old Money Aesthetic */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        {/* Left: Brand identity */}
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2 group">
            <span className="font-serif text-sm text-white group-hover:text-zinc-300 transition-colors">The Daily Lesson</span>
            <span className="text-[8px] text-zinc-600 tracking-widest">PBC</span>
          </Link>
          <span className="text-zinc-700">|</span>
          <span className="text-[10px] tracking-wide text-zinc-500 italic">with Curious Kelly</span>
        </div>
        
        {/* Center: Streak indicator only */}
        {streak > 0 && (
          <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
            <span className="text-[10px] text-amber-500/80 font-medium">{streak} day streak</span>
          </div>
        )}
        
        {/* Right: Navigation */}
        <div className="flex items-center gap-4">
          <Link href="/about" className="text-[10px] text-zinc-600 hover:text-zinc-300 transition-colors tracking-wide">About</Link>
          <Link href="/strategic" className="text-[10px] text-zinc-600 hover:text-zinc-300 transition-colors tracking-wide">Strategic</Link>
          <Link href="/join" className="text-[10px] text-zinc-600 hover:text-zinc-300 transition-colors tracking-wide">Investors</Link>
        </div>
      </header>

      {/* Kelly viewport */}
      <main className="flex-1 relative flex items-center justify-center bg-gradient-to-b from-zinc-900 to-black overflow-hidden">
        {/* Hidden audio element for TTS playback */}
        {lessonData.audioUrl && (
          <audio
            ref={audioRef}
            src={lessonData.audioUrl}
            onEnded={() => {
                setPlaying(false)
                setAudioCompleted(true)
              }}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
          />
        )}
        
        {/* Video or Kelly avatar placeholder */}
        {lessonData.videoUrl && !lessonData.loading ? (
          <video
            src={lessonData.videoUrl}
            className="absolute inset-0 w-full h-full object-cover z-10"
            autoPlay={playing}
            loop
            muted={!!lessonData.audioUrl}
            playsInline
            onPlay={() => {
              if (lessonData.audioUrl && audioRef.current) {
                audioRef.current.play()
              }
            }}
          />
        ) : (
          <div className="relative z-10 w-72 h-72 rounded-full bg-gradient-to-br from-amber-100 via-amber-50 to-white shadow-2xl flex items-center justify-center overflow-hidden">
            {lessonData.loading ? (
              <motion.div
                className="w-16 h-16 border-4 border-amber-200 border-t-amber-500 rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
            ) : lessonData.fallbackUrl ? (
              <motion.img
                src={lessonData.fallbackUrl}
                alt="Kelly"
                className="w-full h-full object-cover"
                initial={{ scale: 1.1, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
                onError={(e) => {
                  // Fallback to default explorer if image fails
                  (e.target as HTMLImageElement).src = '/kelly/archetypes/explorer.png'
                }}
              />
            ) : (
              <span className="text-8xl">👩‍🏫</span>
            )}
          </div>
        )}
        
        {/* Cinematic phase transition overlay */}
        <AnimatePresence>
          {phaseTransition && (
            <motion.div
              className="absolute inset-0 bg-black z-50 flex items-center justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
            >
              <motion.div
                className="text-center"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.15, duration: 0.2 }}
              >
                <span className="text-4xl mb-2 block">
                  {PHASES.find(p => p.id === phase)?.icon}
                </span>
                <span className="text-xs tracking-[0.3em] text-zinc-500 uppercase">
                  {PHASES.find(p => p.id === phase)?.label}
                </span>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Phase Visual Aid - Full-screen cinematic background when playing */}
        <AnimatePresence>
          {lessonData.visualUrl && (
            <motion.div
              className="absolute inset-0 z-0"
              initial={{ opacity: 0, scale: 1.05 }}
              animate={{ opacity: playing ? 0.85 : 0.4, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            >
              <img
                src={lessonData.visualUrl || "/placeholder.svg"}
                alt={`Visual for ${phase} phase`}
                className="w-full h-full object-cover"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none'
                }}
              />
              {/* Gradient overlay for text readability */}
              <div className="absolute inset-0 bg-gradient-to-t from-black via-black/60 to-black/30" />
              {/* Phase label badge */}
              <div className="absolute bottom-2 left-2 px-2 py-1 rounded-full bg-black/60 backdrop-blur-sm">
                <span className="text-xs font-medium text-white/80">
                  {PHASES.find(p => p.id === phase)?.icon} {PHASES.find(p => p.id === phase)?.label}
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Play overlay with anticipation animation */}
        <AnimatePresence>
          {!playing && !lessonData.loading && !phaseTransition && (
            <motion.button 
              onClick={() => {
                setPlaying(true)
                // Play audio if available
                if (audioRef.current && lessonData.audioUrl) {
                  audioRef.current.play().catch(console.error)
                }
              }}
              className="absolute inset-0 flex items-center justify-center cursor-pointer group"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, scale: 1.2 }}
              transition={{ duration: 0.3 }}
            >
              <motion.div 
                className="relative"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                {/* Outer pulse ring */}
                <motion.div 
                  className="absolute inset-0 rounded-full bg-white/20"
                  animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                  style={{ margin: "-10px" }}
                />
                {/* Inner glow */}
                <motion.div 
                  className="absolute inset-0 rounded-full"
                  style={{ 
                    background: `radial-gradient(circle, ${PHASES.find(p => p.id === phase)?.color || '#f59e0b'}40 0%, transparent 70%)`,
                    margin: "-20px"
                  }}
                  animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.6, 0.3] }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                />
                {/* Main button */}
                <div className="w-20 h-20 rounded-full bg-white/90 group-hover:bg-white flex items-center justify-center shadow-2xl transition-all relative z-10">
                  <motion.svg 
                    className="w-8 h-8 text-black ml-1" 
                    viewBox="0 0 24 24" 
                    fill="currentColor"
                    animate={{ x: [0, 2, 0] }}
                    transition={{ duration: 0.8, repeat: Infinity, ease: "easeInOut" }}
                  >
                    <path d="M8 5v14l11-7z"/>
                  </motion.svg>
                </div>
              </motion.div>
            </motion.button>
          )}
        </AnimatePresence>

        {/* Side rail - Right (fixed position, TRUE vertical center accounting for header) */}
        <div className="fixed right-4 top-[calc(50%+24px)] -translate-y-1/2 flex flex-col items-center justify-center gap-2.5 z-40">
          {/* Profile */}
          <motion.button 
            className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-colors"
            whileHover={{ scale: 1.1, rotate: 5 }}
            whileTap={{ scale: 0.9 }}
          >
            <svg className="w-5 h-5 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"/>
            </svg>
          </motion.button>
          
          {/* Like with heart burst */}
          <motion.button 
            onClick={() => setLiked(!liked)} 
            className={`w-12 h-12 rounded-xl border flex items-center justify-center transition-all relative ${liked ? "bg-pink-500/20 border-pink-500/40" : "bg-white/5 border-white/10 hover:bg-white/10"}`}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.85 }}
          >
            <motion.svg 
              className={`w-5 h-5 transition-colors ${liked ? "text-pink-400 fill-pink-400" : "text-zinc-400"}`} 
              viewBox="0 0 24 24" 
              fill={liked ? "currentColor" : "none"} 
              stroke="currentColor" 
              strokeWidth="1.5"
              animate={liked ? { scale: [1, 1.3, 1] } : {}}
              transition={{ duration: 0.3 }}
            >
              <path d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"/>
            </motion.svg>
            {liked && (
              <motion.div 
                className="absolute inset-0 rounded-xl"
                initial={{ boxShadow: "0 0 0 0 rgba(236, 72, 153, 0.7)" }}
                animate={{ boxShadow: "0 0 0 10px rgba(236, 72, 153, 0)" }}
                transition={{ duration: 0.4 }}
              />
            )}
          </motion.button>
          
          {/* Comment */}
          <motion.button 
            className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-colors"
            whileHover={{ scale: 1.1, y: -2 }}
            whileTap={{ scale: 0.9 }}
          >
            <svg className="w-5 h-5 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 20.25c4.97 0 9-3.694 9-8.25s-4.03-8.25-9-8.25S3 7.444 3 12c0 2.104.859 4.023 2.273 5.48.432.447.74 1.04.586 1.641a4.483 4.483 0 01-.923 1.785A5.969 5.969 0 006 21c1.282 0 2.47-.402 3.445-1.087.81.22 1.668.337 2.555.337z"/>
            </svg>
          </motion.button>
          
          {/* Save with bookmark animation */}
          <motion.button 
            onClick={() => setSaved(!saved)} 
            className={`w-12 h-12 rounded-xl border flex items-center justify-center transition-all ${saved ? "bg-blue-500/20 border-blue-500/40" : "bg-white/5 border-white/10 hover:bg-white/10"}`}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.85 }}
          >
            <motion.svg 
              className={`w-5 h-5 transition-colors ${saved ? "text-blue-400 fill-blue-400" : "text-zinc-400"}`} 
              viewBox="0 0 24 24" 
              fill={saved ? "currentColor" : "none"} 
              stroke="currentColor" 
              strokeWidth="1.5"
              animate={saved ? { y: [0, -3, 0] } : {}}
              transition={{ duration: 0.3 }}
            >
              <path d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0111.186 0z"/>
            </motion.svg>
          </motion.button>
          
          {/* Share with ripple */}
          <motion.button 
            className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-colors"
            whileHover={{ scale: 1.1, rotate: -5 }}
            whileTap={{ scale: 0.9 }}
          >
            <svg className="w-5 h-5 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M7.217 10.907a2.25 2.25 0 100 2.186m0-2.186c.18.324.283.696.283 1.093s-.103.77-.283 1.093m0-2.186l9.566-5.314m-9.566 7.5l9.566 5.314m0 0a2.25 2.25 0 103.935 2.186 2.25 2.25 0 00-3.935-2.186zm0-12.814a2.25 2.25 0 103.933-2.185 2.25 2.25 0 00-3.933 2.185z"/>
            </svg>
          </motion.button>
          
          {/* Ziggurat - Home */}
          <motion.button 
            onClick={() => window.location.href = '/'}
            className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-colors"
            whileHover={{ scale: 1.1, boxShadow: "0 0 20px rgba(255,255,255,0.1)" }}
            whileTap={{ scale: 0.9 }}
          >
            <svg className="w-5 h-5 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2.25 21h19.5m-18-18v18m10.5-18v18m6-13.5V21M6.75 6.75h.75m-.75 3h.75m-.75 3h.75m3-6h.75m-.75 3h.75m-.75 3h.75M6.75 21v-3.375c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21M3 3h12m-.75 4.5H21m-3.75 3.75h.008v.008h-.008v-.008zm0 3h.008v.008h-.008v-.008zm0 3h.008v.008h-.008v-.008z"/>
            </svg>
          </motion.button>
          
          {/* Menu */}
          <motion.button 
            className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <svg className="w-5 h-5 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"/>
            </svg>
          </motion.button>
        </div>
      </main>

      {/* Bottom controls */}
      <div className="border-t border-white/5 bg-black/80 backdrop-blur-sm">
        {/* Phases */}
        <div className="flex justify-center gap-2 py-4">
          {PHASES.map((p) => (
            <button 
              key={p.id} 
              onClick={() => handlePhaseChange(p.id as Phase)} 
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-[11px] tracking-wider border transition-all duration-200 ${phase === p.id ? "border-2 scale-105" : "border-white/10 text-zinc-500 hover:text-white hover:bg-white/10 hover:scale-105 hover:border-white/30"}`} 
              style={phase === p.id ? { borderColor: p.color, color: p.color, backgroundColor: `${p.color}20`, boxShadow: `0 0 15px ${p.color}40` } : {}}
            >
              <span 
                className="w-2 h-2 rounded-full transition-all" 
                style={{ backgroundColor: phase === p.id ? p.color : '#52525b' }}
              />
              {p.icon} {p.label}
            </button>
          ))}
        </div>

        {/* Archetypes */}
        <div className="flex justify-center gap-1.5 pb-3 relative">
          {ARCHETYPES.map((a) => (
            <button 
              key={a.id} 
              onClick={() => setArchetype(a.id as Archetype)} 
              onMouseEnter={() => setHoveredArchetype(a.id)}
              onMouseLeave={() => setHoveredArchetype(null)}
              className={`w-10 h-10 rounded-lg flex items-center justify-center text-lg transition-all duration-200 relative ${archetype === a.id ? "bg-white/20 border-2 border-white/50 scale-110" : "bg-white/5 border border-white/10 hover:bg-white/20 hover:scale-110 hover:border-white/30"}`}
              style={archetype === a.id ? { boxShadow: '0 0 20px rgba(255,255,255,0.4)' } : {}}
            >
              {a.emoji}
              {/* Tooltip */}
              {hoveredArchetype === a.id && (
                <div className="absolute -top-10 left-1/2 -translate-x-1/2 px-2 py-1 bg-black/90 border border-white/20 rounded text-[9px] text-zinc-300 whitespace-nowrap z-50 pointer-events-none">
                  {a.desc}
                </div>
              )}
            </button>
          ))}
        </div>

        {/* Age + Language row */}
        <div className="flex justify-center items-center gap-6 pb-4">
          <div className="flex gap-2">
            {AGES.map((a) => (
              <button 
                key={a.id} 
                onClick={() => handleAgeChange(a.id)} 
                className={`w-12 h-12 rounded-full flex items-center justify-center text-xl transition-all duration-200 ${age === a.id ? "bg-white/20 ring-2 ring-white/60 scale-110" : "bg-white/5 hover:bg-white/20 hover:scale-110"}`} 
                title={`${a.label} (${a.age})`}
                style={age === a.id ? { boxShadow: '0 0 15px rgba(255,255,255,0.3)' } : {}}
              >
                {a.emoji}
              </button>
            ))}
          </div>
          <div className="w-px h-8 bg-white/10" />
          <div className="flex gap-2">
            {LANGS.map((l) => (
              <button 
                key={l.code} 
                onClick={() => handleLangChange(l.code)} 
                className={`w-10 h-10 rounded-lg flex items-center justify-center text-lg transition-all duration-200 ${lang === l.code ? "bg-white/20 ring-2 ring-white/60 scale-110" : "bg-white/5 hover:bg-white/20 hover:scale-110"}`}
                title={l.label}
                style={lang === l.code ? { boxShadow: '0 0 15px rgba(255,255,255,0.3)' } : {}}
              >
                {l.flag}
              </button>
            ))}
          </div>
        </div>

        {/* Lesson info */}
        <div className="text-center pb-4">
          <div className="flex items-center justify-center gap-2 mb-1">
            <span className="text-xs text-zinc-600 font-mono">Day {dayNumber}</span>
            <span className="text-zinc-700">|</span>
            <span className="text-xs text-zinc-600 uppercase tracking-wider">{phase}</span>
          </div>
          <h1 className="text-xl font-light" style={{ fontFamily: 'Georgia, serif' }}>
            {lessonData.title}
          </h1>
          {lessonData.script && (
            <p className="text-sm text-zinc-500 italic mt-1 max-w-md mx-auto line-clamp-2">
              {lessonData.script.slice(0, 120)}...
            </p>
          )}
          {lessonData.error && (
            <p className="text-xs text-red-400 mt-2">{lessonData.error}</p>
          )}
        </div>

        {/* Question - Only show during hook phase - Glassmorphism style */}
        {phase === "hook" && (
          <div 
            className="mx-4 mb-4 rounded-2xl overflow-hidden"
            style={{ 
              background: 'rgba(255,255,255,0.03)',
              backdropFilter: 'blur(20px)',
              WebkitBackdropFilter: 'blur(20px)',
              border: '1px solid rgba(255,255,255,0.08)',
              boxShadow: '0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05)'
            }}
          >
            <div className="text-center py-2.5 border-b border-white/[0.06]">
              <p className="text-[10px] tracking-[0.2em] text-white/40 font-medium">IS THIS TRUE?</p>
            </div>
            
            {/* True/False or Community Results */}
            {!hookVote ? (
              <div className="flex gap-3 p-3">
                <button 
                  onClick={() => { 
                    setHookVote("true")
                    const trueVotes = Math.floor(Math.random() * 800) + 200
                    const falseVotes = Math.floor(Math.random() * 500) + 100
                    setTimeout(() => setCommunityResults({ true_votes: trueVotes, false_votes: falseVotes }), 500)
                  }}
                  className="flex-1 py-2.5 rounded-xl text-sm tracking-wider font-semibold transition-all duration-300 bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/25 hover:scale-[1.02] hover:border-emerald-400/50 active:scale-95"
                  style={{ boxShadow: '0 4px 12px rgba(16, 185, 129, 0.15)' }}
                >
                  TRUE
                </button>
                <button 
                  onClick={() => { 
                    setHookVote("false")
                    const trueVotes = Math.floor(Math.random() * 500) + 100
                    const falseVotes = Math.floor(Math.random() * 800) + 200
                    setTimeout(() => setCommunityResults({ true_votes: trueVotes, false_votes: falseVotes }), 500)
                  }}
                  className="flex-1 py-2.5 rounded-xl text-sm tracking-wider font-semibold transition-all duration-300 bg-rose-500/15 border border-rose-500/30 text-rose-400 hover:bg-rose-500/25 hover:scale-[1.02] hover:border-rose-400/50 active:scale-95"
                  style={{ boxShadow: '0 4px 12px rgba(244, 63, 94, 0.15)' }}
                >
                  FALSE
                </button>
              </div>
            ) : (
              <div className="p-4 max-w-md mx-auto">
                {/* Your vote indicator */}
                <div className="text-center mb-4">
                  <span className={`inline-block px-4 py-1 rounded-full text-sm font-medium ${hookVote === "true" ? "bg-emerald-500/20 text-emerald-300" : "bg-red-500/20 text-red-300"}`}>
                    You voted {hookVote.toUpperCase()}
                  </span>
                </div>
                
                {/* Community Results */}
                {communityResults ? (
                  <div className="space-y-4">
                    {/* Stats */}
                    <div className="text-center text-sm text-zinc-400">
                      {communityResults.true_votes + communityResults.false_votes > 0 
                        ? `${Math.round((communityResults.true_votes / (communityResults.true_votes + communityResults.false_votes)) * 100)}% said True`
                        : 'No votes yet'}
                      <span className="mx-2">•</span>
                      {(communityResults.true_votes + communityResults.false_votes).toLocaleString()} learners voted
                    </div>
                    
                    {/* Animated bar chart */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-emerald-400 w-12">TRUE</span>
                        <div className="flex-1 h-6 bg-white/5 rounded overflow-hidden">
                          <motion.div 
                            className="h-full bg-emerald-500/60 rounded"
                            initial={{ width: 0 }}
                            animate={{ width: `${communityResults.true_votes + communityResults.false_votes > 0 ? (communityResults.true_votes / (communityResults.true_votes + communityResults.false_votes)) * 100 : 50}%` }}
                            transition={{ duration: 0.8, ease: "easeOut" }}
                          />
                        </div>
                        <span className="text-xs text-zinc-500 w-12 text-right">{communityResults.true_votes}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-red-400 w-12">FALSE</span>
                        <div className="flex-1 h-6 bg-white/5 rounded overflow-hidden">
                          <motion.div 
                            className="h-full bg-red-500/60 rounded"
                            initial={{ width: 0 }}
                            animate={{ width: `${communityResults.true_votes + communityResults.false_votes > 0 ? (communityResults.false_votes / (communityResults.true_votes + communityResults.false_votes)) * 100 : 50}%` }}
                            transition={{ duration: 0.8, ease: "easeOut", delay: 0.1 }}
                          />
                        </div>
                        <span className="text-xs text-zinc-500 w-12 text-right">{communityResults.false_votes}</span>
                      </div>
                    </div>
                    
                    {/* Comparison message */}
                    <div className="text-center pt-2">
                      {(() => {
                        const totalVotes = communityResults.true_votes + communityResults.false_votes
                        if (totalVotes === 0) {
                          return <p className="text-sm text-zinc-400">Be the first to vote!</p>
                        }
                        const truePercent = communityResults.true_votes / totalVotes
                        const majorityVote = truePercent > 0.5 ? "true" : "false"
                        const matches = hookVote === majorityVote
                        return (
                          <p className={`text-sm ${matches ? "text-zinc-300" : "text-amber-400"}`}>
                            {matches ? "Your answer matches most learners" : "You think differently than most!"}
                          </p>
                        )
                      })()}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-4">
                    <div className="w-6 h-6 border-2 border-white/20 border-t-white/60 rounded-full animate-spin mx-auto" />
                    <p className="text-xs text-zinc-500 mt-2">Loading results...</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer - Corporate */}
      <footer className="flex items-center justify-between px-4 py-2 border-t border-white/5 bg-black">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-zinc-600">Lesson of the Day, PBC</span>
          <span className="text-zinc-700">·</span>
          <span className="text-[10px] text-zinc-700">California Public Benefit Corporation</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[9px] text-zinc-600">Full launch Q2 2026</span>
          <span className="text-[9px] tracking-[0.15em] px-2 py-1 bg-amber-500/10 border border-amber-500/20 text-amber-400">PRIVATE ALPHA</span>
        </div>
      </footer>

      {/* Lesson Complete Celebration Overlay */}
      <AnimatePresence>
        {showCelebration && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-md"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Confetti particles */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              {Array.from({ length: 50 }).map((_, i) => (
                <motion.div
                  key={i}
                  className="absolute w-3 h-3 rounded-full"
                  style={{
                    left: `${Math.random() * 100}%`,
                    backgroundColor: ['#f59e0b', '#3b82f6', '#8b5cf6', '#10b981', '#f43f5e'][i % 5],
                  }}
                  initial={{ y: -20, opacity: 1 }}
                  animate={{
                    y: window?.innerHeight || 800,
                    x: (Math.random() - 0.5) * 200,
                    rotate: Math.random() * 720,
                    opacity: 0,
                  }}
                  transition={{
                    duration: 2 + Math.random() * 2,
                    delay: Math.random() * 0.5,
                    ease: 'easeOut',
                  }}
                />
              ))}
            </div>

            {/* Celebration content */}
            <motion.div
              className="relative z-10 text-center px-8"
              initial={{ scale: 0.8, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              transition={{ type: 'spring', damping: 15 }}
            >
              <motion.div
                className="text-7xl mb-6"
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: 'spring', delay: 0.2 }}
              >
                🎉
              </motion.div>

              <h2 className="text-4xl font-serif text-white mb-4">Lesson Complete!</h2>
              
              <p className="text-zinc-400 mb-6">
                You finished all 5 phases of today's lesson
              </p>

              {/* Streak update */}
              <motion.div
                className="inline-flex items-center gap-3 px-6 py-3 bg-amber-500/20 border border-amber-500/40 rounded-full mb-8"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.4 }}
              >
                <span className="text-2xl">🔥</span>
                <span className="text-lg text-amber-300 font-medium">{streak + 1} Day Streak</span>
              </motion.div>

              {/* Actions */}
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <button
                  onClick={() => {
                    const shareText = `I just finished my daily lesson on Curious Kelly! ${streak + 1} day learning streak`
                    if (navigator.share) {
                      navigator.share({ text: shareText, url: 'https://curiouskelly.com' })
                    } else {
                      navigator.clipboard.writeText(shareText)
                    }
                  }}
                  className="px-6 py-3 bg-white/10 border border-white/20 rounded-lg text-white hover:bg-white/20 transition-colors"
                >
                  Share Your Streak
                </button>
                <Link
                  href={`/play?day=${dayNumber + 1}`}
                  className="px-6 py-3 bg-white text-black font-medium rounded-lg hover:bg-zinc-200 transition-colors"
                  onClick={() => setShowCelebration(false)}
                >
                  Next Lesson
                </Link>
              </div>

              {/* Close button */}
              <button
                onClick={() => setShowCelebration(false)}
                className="mt-8 text-sm text-zinc-500 hover:text-white transition-colors"
              >
                Close
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Personalization Rail - Collapsible settings panel */}
      <PersonalizationRail
        currentPhase={phase}
        currentAge={age}
        currentArchetype={archetype}
        currentLanguage={lang}
        onPhaseChange={(p) => handlePhaseChange(p as Phase)}
        onAgeChange={handleAgeChange}
        onArchetypeChange={(a) => setArchetype(a as Archetype)}
        onLanguageChange={handleLangChange}
      />
    </div>
  )
}
