'use client'

import Link from "next/link"

// Curious Kelly - Daily Learning Adventures
// Build refresh: 2026-02-05-v1 - KellyOS TikTok Lesson View LIVE
// © 2026 Lesson of the Day, PBC. All rights reserved.
// Build: 2026-02-05-DEPLOY Full Kelly TikTok production deploy. KellyOS active.

// MAINTENANCE MODE - Hide frontend during major updates
// STATUS: Database now connected to wispy-resonance-11000301 (lesson_db) with REAL curriculum
// Real topics: "How Electricity Flows" (Day 35), "How Magnets Work" (Day 34), etc.
// See: /docs/CRITICAL-TRACK-PROTECTION.md
// Two tracks: LEARN (priority, 365 daily lessons) vs GROW (AI track, build later)
const MAINTENANCE_MODE = false // SHIPPED! 513 videos ready, 9110 audio assets, full app enabled

import React, { Suspense } from "react"
import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import useSWR from 'swr'
import { LearnerContextProvider, useLearner } from '@/hooks/use-learner-context'
import { LayoutProvider, useLayout } from '@/lib/layout-context'
import { LayoutGrid, Zone } from '@/components/layout/layout-grid'
import { useLesson, useVideoUrl, getPhaseScript, VIDEO_SOURCES, type VideoSource } from '@/hooks/use-lesson'
import { useLiveClassCountdown } from '@/hooks/use-live-class-countdown'
import { useComments } from '@/hooks/use-comments'
import { useVote } from '@/hooks/use-vote'
import { usePresence } from '@/hooks/use-presence'
import { useSearch } from '@/hooks/use-search'
import { useLikes } from '@/hooks/use-likes'
import { useAffiliateTracking } from '@/hooks/use-affiliate-tracking'
import { useUITranslation } from '@/hooks/use-translations'
import { formatFullDate, formatShortDate } from '@/lib/date-utils'
import { getKellyArchetypeImage, getAgeCategory, getHeyGenLookId, KELLY_IMAGES, HEYGEN_AVATAR_GROUPS } from '@/lib/kelly-assets'
import type { LessonPhase, KellyArchetype } from '@/lib/types'
import { cn } from '@/lib/utils'
import { useTranslation } from '@/lib/translations'
import {
  Heart,
  MessageCircle,
  Share2,
  Bookmark,
  Play,
  SkipForward,
  Volume2,
  VolumeX,
  Settings,
  Pause,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Calendar,
  Sparkles,
  Globe,
  User,
  Clock,
  Users,
  X,
  ImageIcon,
  Crown,
  Search,
  Send,
  Loader2,
  Check,
  ThumbsUp,
  ThumbsDown,
  Grid2x2,
  AlertCircle,
  Eye,
  EyeOff,
  Maximize2,
  Link2,
  Twitter,
  Facebook,
} from 'lucide-react'

// KellyOS Overlay System - Apple Education Quality
import { OverlayProvider } from '@/components/kelly/overlays'
import { SettingsOverlay } from '@/components/kelly/overlays/settings-overlay'
import { PricingOverlay, CheckoutOverlay } from '@/components/kelly/overlays/pricing-overlay'
import { ShareOverlay } from '@/components/kelly/overlays/share-overlay'
import { DictionaryOverlay } from '@/components/kelly/overlays/dictionary-overlay'
import { CommentsOverlay } from '@/components/kelly/overlays/comments-overlay'
import { ProfileOverlay } from '@/components/kelly/overlays/profile-overlay'
import { AboutOverlay } from '@/components/kelly/overlays/about-overlay'
// Visual is now integrated into caption bar - no separate overlay needed
import { HelpOverlay } from '@/components/kelly/overlays/help-overlay'
import { SubscriptionOverlay } from '@/components/kelly/overlays/subscription-overlay'
import { SubscriptionBanner } from '@/components/kelly/subscription-banner'
import { AffiliateOverlay } from '@/components/kelly/overlays/affiliate-overlay'
import { AdminOverlay } from '@/components/kelly/overlays/admin-overlay'
import { TermsOverlay } from '@/components/kelly/overlays/terms-overlay'
import { StudioOverlay } from '@/components/kelly/overlays/studio-overlay'

import { SearchOverlay } from '@/components/kelly/overlays/search-overlay'
import { useSubscription, openCustomerPortal, startCheckout } from '@/hooks/use-subscription'
import { ChatOverlay } from '@/components/kelly/overlays/chat-overlay'
import { LessonCalendar } from '@/components/kelly/lesson-calendar'
import { TikTokFeed } from '@/components/kelly/tiktok-feed'
import { KellyOSLayer } from '@/components/kelly/kelly-os-layer'
import { NavigationDrawer, MenuButton } from '@/components/kelly/navigation-drawer'


// ============================================================================
// TEMPORARY PLACEHOLDER COMPONENTS (TODO: Move to separate files)
// ============================================================================

function ControlCenterButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="fixed top-4 right-4 z-40 w-10 h-10 rounded-full glass-panel flex items-center justify-center hover:scale-105 transition-transform"
      aria-label="Control Center"
    >
      <Grid2x2 className="w-5 h-5 text-white" />
    </button>
  )
}

function SpotlightTrigger({ onClick }: { onClick: () => void }) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        onClick()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClick])
  
  return null
}

function ControlCenter({
  isOpen,
  onClose,
  onNavigate,
  currentDay,
  lessonTitle,
  currentPhase,
  userName,
  isPremium,
  isPlaying,
  onPlayPause,
  streak,
  language,
}: {
  isOpen: boolean
  onClose: () => void
  onNavigate: (view: string) => void
  currentDay: number
  lessonTitle: string
  currentPhase: string
  userName: string
  isPremium: boolean
  isPlaying: boolean
  onPlayPause: () => void
  streak: number
  language: string
}) {
  if (!isOpen) return null
  
  return (
    <>
      <div 
        className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="fixed top-20 right-4 z-50 w-80 rounded-2xl glass-panel p-6 space-y-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-white font-semibold text-lg">Control Center</h2>
          <button onClick={onClose} className="text-white/60 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          <button onClick={() => onNavigate('settings')} className="glass-card p-4 text-center hover:bg-white/10">
            <Settings className="w-6 h-6 text-white mx-auto mb-2" />
            <span className="text-white text-xs">Settings</span>
          </button>
          <button onClick={() => onNavigate('profile')} className="glass-card p-4 text-center hover:bg-white/10">
            <User className="w-6 h-6 text-white mx-auto mb-2" />
            <span className="text-white text-xs">Profile</span>
          </button>
          <button onClick={() => onNavigate('calendar')} className="glass-card p-4 text-center hover:bg-white/10">
            <Calendar className="w-6 h-6 text-white mx-auto mb-2" />
            <span className="text-white text-xs">Calendar</span>
          </button>
          <button onClick={() => onNavigate('search')} className="glass-card p-4 text-center hover:bg-white/10">
            <Search className="w-6 h-6 text-white mx-auto mb-2" />
            <span className="text-white text-xs">Search</span>
          </button>
        </div>
      </div>
    </>
  )
}

// ============================================================================
// TYPES
// ============================================================================

interface Comment {
  id: string
  username: string
  text: string
  highlight?: boolean
}

const PHASES: { id: LessonPhase; label: string; description: string }[] = [
  { id: 'hook', label: 'Hook', description: 'Capture your curiosity' },
  { id: 'story', label: 'Story', description: 'The journey begins' },
  { id: 'wonder', label: 'Wonder', description: 'Dive deeper' },
  { id: 'action', label: 'Action', description: 'Put it into practice' },
  { id: 'wisdom', label: 'Wisdom', description: 'What we learned' },
]

const LANGUAGES = [
  { code: 'en', label: 'US English', flag: '🇺🇸' },
  { code: 'es', label: 'Spanish', flag: '🇪🇸' },
  { code: 'fr', label: 'French', flag: '🇫🇷' },
  { code: 'de', label: 'German', flag: '🇩🇪' },
  { code: 'pt', label: 'Portuguese', flag: '🇧🇷' },
  { code: 'zh', label: 'Chinese', flag: '🇨🇳' },
  { code: 'ja', label: 'Japanese', flag: '🇯🇵' },
  { code: 'ko', label: 'Korean', flag: '🇰🇷' },
  { code: 'hi', label: 'Hindi', flag: '🇮🇳' },
  { code: 'ar', label: 'Arabic', flag: '🇸🇦' },
  { code: 'it', label: 'Italian', flag: '🇮🇹' },
  { code: 'ru', label: 'Russian', flag: '🇷🇺' },
]

const ARCHETYPES: { id: KellyArchetype; label: string; hasImage: boolean }[] = [
  { id: 'storyteller', label: 'Storyteller', hasImage: true },
  { id: 'explorer', label: 'Explorer', hasImage: true },
  { id: 'architect', label: 'Architect', hasImage: true },
  { id: 'empath', label: 'Empath', hasImage: true },
  { id: 'rebel', label: 'Rebel', hasImage: true },
  { id: 'mystic', label: 'Mystic', hasImage: true },
  { id: 'provider', label: 'Provider', hasImage: true },
  { id: 'diplomat', label: 'Diplomat', hasImage: true },
  { id: 'macgyver', label: 'MacGyver', hasImage: true },
  { id: 'scientist', label: 'Scientist', hasImage: true },
  { id: 'strategist', label: 'Strategist', hasImage: true },
  { id: 'survivor', label: 'Survivor', hasImage: true },
]

// Simulated lifelong learners currently in the live class
const LEARNER_AVATARS = [
  { name: 'Sarah', avatar: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=100&h=100&fit=crop&crop=face' },
  { name: 'Marcus', avatar: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&h=100&fit=crop&crop=face' },
  { name: 'Priya', avatar: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=100&h=100&fit=crop&crop=face' },
  { name: 'James', avatar: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=100&h=100&fit=crop&crop=face' },
  { name: 'Elena', avatar: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&h=100&fit=crop&crop=face' },
  { name: 'David', avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=100&h=100&fit=crop&crop=face' },
]

// ============================================================================
// TOP BAR - Phase Navigation + Live Class Indicator (COMPACT)
// ============================================================================

function TopBar({
  currentPhase,
  completedPhases,
  onPhaseChange,
  liveClass,
  onLiveClassClick,
  phaseProgress = 0, // 0-100 progress within current phase
  onAdminLogin,
}: {
  currentPhase: LessonPhase
  completedPhases: LessonPhase[]
  onPhaseChange: (phase: LessonPhase) => void
  liveClass: ReturnType<typeof useLiveClassCountdown> | null
  onLiveClassClick?: () => void
  phaseProgress?: number
  onAdminLogin?: () => void
}) {
  const hasLiveClass = liveClass && liveClass.class && !liveClass.hasEnded && !liveClass.noUpcoming
  const [showAdminLogin, setShowAdminLogin] = useState(false)
  const [adminPassword, setAdminPassword] = useState('')
  const [adminError, setAdminError] = useState(false)

  const handleAdminSubmit = () => {
    if (adminPassword === 'rungirlrun') {
      setShowAdminLogin(false)
      setAdminPassword('')
      setAdminError(false)
      onAdminLogin?.()
    } else {
      setAdminError(true)
      setTimeout(() => setAdminError(false), 2000)
    }
  }

  return (
    <div className="top-bar">
      {/* Admin login button - glass-pill style */}
      <div className="w-full flex justify-end px-4 py-2">
        {!showAdminLogin ? (
          <button
            onClick={() => setShowAdminLogin(true)}
            className="glass-pill px-4 py-2 text-white/60 hover:text-white text-xs font-medium transition-all"
          >
            admin
          </button>
        ) : (
          <div className="glass-pill flex items-center gap-2 px-3 py-1.5">
            <input
              type="password"
              value={adminPassword}
              onChange={(e) => setAdminPassword(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAdminSubmit()}
              placeholder="password"
              className={cn(
                "bg-transparent border-none outline-none text-white text-xs w-24 placeholder:text-white/30",
                adminError && "text-red-400 placeholder:text-red-400/50"
              )}
              autoFocus
            />
            <button
              onClick={handleAdminSubmit}
              className="text-white/60 hover:text-white text-xs font-medium"
            >
              →
            </button>
            <button
              onClick={() => { setShowAdminLogin(false); setAdminPassword(''); setAdminError(false); }}
              className="text-white/40 hover:text-white/60 text-xs"
            >
              ✕
            </button>
          </div>
        )}
      </div>
      
      </div>
  )
}

// ============================================================================
// DATE NAVIGATION BAR - Month/Year/Week/Day + Start CTA (BOTTOM)
// ============================================================================

function FloatingDayPill({
  dayOfYear,
  language,
  onDayChange,
  onOpenCalendar,
  lessonTitle,
}: {
  dayOfYear: number
  language: string
  onDayChange: (delta: number) => void
  onOpenCalendar?: () => void
  lessonTitle?: string
}) {
  const { t: tr } = useUITranslation(language)
  // Get formatted date with proper locale
  const currentDate = new Date(new Date().getFullYear(), 0, dayOfYear)
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
  const shortDate = formatShortDate(currentDate, timezone, language)
  const fullDate = formatFullDate(currentDate, timezone, language)
  
  // No fixed positioning - parent handles placement in flex layout
  return (
    <div className="flex flex-col items-center w-full">
      {/* Lesson title - HIDDEN on mobile (shown in caption), visible desktop */}
      {lessonTitle && (
        <h2 className="hidden sm:block text-[24px] text-white font-semibold tracking-tight text-center mb-3 px-4">
          {lessonTitle}
        </h2>
      )}
      
      <div className="glass-pill px-0.5 sm:px-2">
        {/* Previous day - TINY on mobile */}
        <button
          onClick={() => onDayChange(-1)}
          disabled={dayOfYear <= 1}
          className="p-1.5 sm:p-3 text-white/50 hover:text-white disabled:text-white/15 transition-all duration-200 active:scale-90 rounded-lg hover:bg-white/10 disabled:hover:bg-transparent"
          aria-label="Previous day"
        >
          <ChevronLeft className="w-3.5 h-3.5 sm:w-5 sm:h-5" />
        </button>
        
        {/* Center - SUPER compact on mobile */}
        <button 
          onClick={onOpenCalendar}
          className="flex-1 px-1 sm:px-4 py-1 sm:py-2.5 text-center hover:bg-white/5 rounded-xl transition-all duration-200 min-w-[80px] sm:min-w-[220px]"
        >
          {/* Mobile: Ultra-compact "Day X · Jan Y" */}
<div className="sm:hidden">
  <div className="text-[11px] font-semibold text-white tracking-tight whitespace-nowrap">
  {shortDate}
  </div>
  </div>
          {/* Desktop: Full date */}
          <div className="hidden sm:block">
            <div className="text-[16px] font-semibold text-white tracking-tight">
              {fullDate}
            </div>
            <div className="text-[11px] text-white/40 mt-1 flex items-center justify-center gap-1.5 font-medium">
              <Calendar className="w-3 h-3" />
              <span>Select a date</span>
            </div>
          </div>
        </button>
        
        {/* Next day - TINY on mobile */}
        <button
          onClick={() => onDayChange(1)}
          disabled={dayOfYear >= 365}
          className="p-1.5 sm:p-3 text-white/50 hover:text-white disabled:text-white/15 transition-all duration-200 active:scale-90 rounded-lg hover:bg-white/10 disabled:hover:bg-transparent"
          aria-label="Next day"
        >
          <ChevronRight className="w-3.5 h-3.5 sm:w-5 sm:h-5" />
        </button>
      </div>
    </div>
  )
}

// ============================================================================
// LIVE CLASS COUNTDOWN - Next session with Kelly (uses real API data)
// ============================================================================

function LiveClassCountdown() {
  const liveClassData = useLiveClassCountdown()
  
  // Show loading state briefly
  if (liveClassData.isLoading) {
    return <p className="text-xs font-medium text-white/30">...</p>
  }
  
  // Live now - subtle pulsing
  if (liveClassData.isLive) {
    return (
      <p className="text-xs font-semibold text-white/90 animate-pulse">LIVE NOW</p>
    )
  }
  
  // No upcoming classes - subtle styling, no cyan
  if (liveClassData.noUpcoming) {
    return <p className="text-xs font-medium text-white/40">Coming soon</p>
  }
  
  // Starting soon - slightly brighter
  if (liveClassData.isStartingSoon) {
    return (
      <p className="text-xs font-semibold text-white/80">{liveClassData.countdownText}</p>
    )
  }
  
  // Regular countdown
  return (
    <p className="text-xs font-semibold text-white/70">{liveClassData.countdownText}</p>
  )
}

// ============================================================================
// VIDEO SOURCE SELECTOR - A/B Testing for Lip-Sync Engines
// ============================================================================

function VideoSourceSelector({
  activeSource,
  onSourceChange,
  className,
}: {
  activeSource: VideoSource
  onSourceChange: (source: VideoSource) => void
  className?: string
}) {
  const [isExpanded, setIsExpanded] = useState(false)
  
  return (
    <div className={cn("relative", className)}>
      {/* Collapsed: Show active source bubble */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 px-3 py-2 rounded-2xl bg-black/40 hover:bg-black/60 border border-white/10 transition-all"
      >
        <div 
          className="w-3 h-3 rounded-full animate-pulse"
          style={{ backgroundColor: VIDEO_SOURCES.find(s => s.id === activeSource)?.color || '#6366f1' }}
        />
        <span className="text-white/80 text-xs font-medium">
          {activeSource === 'auto' ? 'Auto' : VIDEO_SOURCES.find(s => s.id === activeSource)?.label}
        </span>
        <ChevronDown className={cn("w-3 h-3 text-white/50 transition-transform", isExpanded && "rotate-180")} />
      </button>
      
      {/* Expanded: Show all 4 engines */}
      {isExpanded && (
        <div className="absolute top-full left-0 mt-2 p-2 rounded-2xl bg-black/80 backdrop-blur-xl border border-white/10 z-50 min-w-[180px]">
          <p className="text-[10px] text-white/40 uppercase tracking-wider px-2 mb-2">Video Engine</p>
          
          {VIDEO_SOURCES.map((source) => (
            <button
              key={source.id}
              onClick={() => {
                onSourceChange(source.id)
                setIsExpanded(false)
              }}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2 rounded-xl transition-all",
                activeSource === source.id 
                  ? "bg-white/10" 
                  : "hover:bg-white/5"
              )}
            >
              <div 
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: source.color }}
              />
              <div className="flex-1 text-left">
                <p className="text-sm text-white font-medium">{source.label}</p>
                <p className="text-[10px] text-white/50">{source.description}</p>
              </div>
              {activeSource === source.id && (
                <Check className="w-4 h-4 text-white/70" />
              )}
            </button>
          ))}
          
          <div className="border-t border-white/10 mt-2 pt-2 px-2">
            <p className="text-[10px] text-white/30">Compare lip-sync quality across engines</p>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// VIDEO FINE-TUNING OVERLAY - 4 Transparent Video Players Over Background
// ============================================================================

interface EngineVideoData {
  engine: VideoSource
  label: string
  color: string
  videoUrl: string | null
  audioUrl: string | null
  status: 'ready' | 'generating' | 'pending' | 'error'
}

function VideoFineTuningOverlay({
  day,
  phase,
  isOpen,
  onClose,
  onSelectSource,
}: {
  day: number
  phase: string
  isOpen: boolean
  onClose: () => void
  onSelectSource: (source: VideoSource) => void
}) {
  const [activeEngines, setActiveEngines] = useState<Record<VideoSource, boolean>>({
    heygen: true,
    fal: true,
    sync: true,
    musetalk: true,
    auto: false,
  })
  const [opacities, setOpacities] = useState<Record<VideoSource, number>>({
    heygen: 0.7,
    fal: 0.7,
    sync: 0.7,
    musetalk: 0.7,
    auto: 0.7,
  })
  const [playingAll, setPlayingAll] = useState(false)
  const [videoErrors, setVideoErrors] = useState<Record<string, boolean>>({})
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({})
  
  const { data: compareData } = useSWR<{
    day: number
    phase: string
    engines: EngineVideoData[]
    fallbackImage: string
  }>(
    isOpen ? `/api/video/compare?day=${day}&phase=${phase}` : null,
    (url: string) => fetch(url).then(r => r.json()),
    { revalidateOnFocus: false }
  )
  
  const engines = compareData?.engines || VIDEO_SOURCES.filter(s => s.id !== 'auto').map(s => ({
    engine: s.id,
    label: s.label,
    color: s.color,
    videoUrl: null,
    audioUrl: null,
    status: 'pending' as const,
  }))
  
  const toggleEngine = (engine: VideoSource) => {
    setActiveEngines(prev => ({ ...prev, [engine]: !prev[engine] }))
  }
  
  const playAll = () => {
    Object.entries(videoRefs.current).forEach(([key, video]) => {
      if (video && activeEngines[key as VideoSource]) {
        video.currentTime = 0
        video.play()
      }
    })
    setPlayingAll(true)
  }
  
  const pauseAll = () => {
    Object.values(videoRefs.current).forEach(video => video?.pause())
    setPlayingAll(false)
  }
  
  const handleVideoError = (source: VideoSource) => {
    setVideoErrors(prev => ({ ...prev, [source]: true }))
  }
  
  // Position each video in a quadrant
  const positions: Record<VideoSource, string> = {
    heygen: 'top-0 left-0',
    fal: 'top-0 right-0',
    sync: 'bottom-0 left-0',
    musetalk: 'bottom-0 right-0',
    auto: 'hidden',
  }
  
  if (!isOpen) return null
  
  return (
    <div className="fixed inset-0 z-50 bg-black">
      {/* Background Kelly Image */}
      <div className="absolute inset-0">
        <img 
          src={compareData?.fallbackImage || '/kelly/archetypes/storyteller.png'}
          alt="Kelly"
          className="w-full h-full object-cover"
        />
        {/* Clean overlay - no gradient */}
      </div>
      
      {/* 4 Transparent Video Overlays */}
      {engines.map((engine) => {
        const hasError = videoErrors[engine.engine]
        const isActive = activeEngines[engine.engine]
        const opacity = opacities[engine.engine]
        
        if (!isActive) return null
        
        return (
          <div 
            key={engine.engine}
            className={`absolute w-1/2 h-1/2 ${positions[engine.engine]} p-2`}
          >
            <div 
              className="relative w-full h-full rounded-2xl overflow-hidden border-2 transition-all"
              style={{ 
                borderColor: engine.color,
                boxShadow: `0 0 20px ${engine.color}40`
              }}
            >
              {/* Video Player */}
              {engine.videoUrl && !hasError ? (
                <video
                  ref={(el) => { videoRefs.current[engine.engine] = el }}
                  src={engine.videoUrl}
                  className="w-full h-full object-cover"
                  style={{ opacity }}
                  loop
                  playsInline
                  muted
                  onError={() => handleVideoError(engine.engine)}
                />
              ) : (
                <div 
                  className="w-full h-full flex items-center justify-center bg-black/50"
                  style={{ opacity }}
                >
                  {hasError ? (
                    <AlertCircle className="w-12 h-12 text-red-400" />
                  ) : (
                    <Clock className="w-12 h-12 text-white/40 animate-pulse" />
                  )}
                </div>
              )}
              
              {/* Engine Label */}
              <div 
                className="absolute top-3 left-3 px-3 py-1.5 rounded-full text-sm font-bold text-white flex items-center gap-2"
                style={{ backgroundColor: engine.color }}
              >
                <span>{engine.label}</span>
                {engine.status === 'ready' && <Check className="w-4 h-4" />}
              </div>
              
              {/* Opacity Slider */}
              <div className="absolute bottom-3 left-3 right-3 flex items-center gap-2">
                <span className="text-white/60 text-xs">Opacity</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={opacity}
                  onChange={(e) => setOpacities(prev => ({ 
                    ...prev, 
                    [engine.engine]: parseFloat(e.target.value) 
                  }))}
                  className="flex-1 h-1 bg-white/20 rounded-full appearance-none cursor-pointer"
                />
                <span className="text-white/60 text-xs w-8">{Math.round(opacity * 100)}%</span>
              </div>
              
              {/* Select Button */}
              <button
                onClick={() => {
                  onSelectSource(engine.engine)
                  onClose()
                }}
                className="absolute top-3 right-3 px-3 py-1.5 rounded-full bg-white/20 backdrop-blur-sm text-white text-sm font-medium hover:bg-white/30 transition-colors"
              >
                Select
              </button>
            </div>
          </div>
        )
      })}
      
      {/* Control Panel */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 px-4 py-3 rounded-full bg-black/70 backdrop-blur-xl border border-white/20">
        {/* Engine Toggles */}
        {engines.map((engine) => (
          <button
            key={engine.engine}
            onClick={() => toggleEngine(engine.engine)}
            className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${
              activeEngines[engine.engine] ? 'ring-2 ring-white' : 'opacity-40'
            }`}
            style={{ backgroundColor: engine.color }}
          >
            {activeEngines[engine.engine] ? (
              <Eye className="w-4 h-4 text-white" />
            ) : (
              <EyeOff className="w-4 h-4 text-white" />
            )}
          </button>
        ))}
        
        <div className="w-px h-6 bg-white/20" />
        
        {/* Play/Pause All */}
        <button
          onClick={playingAll ? pauseAll : playAll}
          className="px-4 py-2 rounded-full bg-white/20 text-white text-sm font-medium hover:bg-white/30 transition-colors flex items-center gap-2"
        >
          {playingAll ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {playingAll ? 'Pause All' : 'Play All'}
        </button>
        
        {/* Close */}
        <button
          onClick={onClose}
          className="p-2 rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      
      {/* Title */}
      <div className="absolute top-6 left-1/2 -translate-x-1/2 text-center">
        <h2 className="text-white text-xl font-bold">Video Engine Fine-Tuning</h2>
        <p className="text-white/60 text-sm">Compare lip-sync quality across engines</p>
      </div>
    </div>
  )
}

// ============================================================================
// VIDEO ENGINE ROW - Compact 2x2 Grid for Quick Access
// ============================================================================

function VideoEngineRow({
  day,
  phase,
  language,
  onSelectSource,
  onOpenFineTuning,
  }: {
  day: number
  phase: string
  language: string
  onSelectSource: (source: VideoSource) => void
  onOpenFineTuning: () => void
  }) {
  const { t: tr } = useUITranslation(language)
  const [videoErrors, setVideoErrors] = useState<Record<string, boolean>>({})
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({})
  
  const { data: compareData } = useSWR<{
    day: number
    phase: string
    engines: EngineVideoData[]
    fallbackImage: string
  }>(
    `/api/video/compare?day=${day}&phase=${phase}`,
    (url: string) => fetch(url).then(r => r.json()),
    { revalidateOnFocus: false }
  )
  
  const engines = compareData?.engines || VIDEO_SOURCES.filter(s => s.id !== 'auto').map(s => ({
    engine: s.id,
    label: s.label,
    color: s.color,
    videoUrl: null,
    audioUrl: null,
    status: 'pending' as const,
  }))
  
  const handleVideoError = (source: VideoSource) => {
    setVideoErrors(prev => ({ ...prev, [source]: true }))
  }
  
  // Play/pause individual videos
  const [playingEngine, setPlayingEngine] = useState<string | null>(null)
  
  const togglePlay = (engineId: string) => {
    const video = videoRefs.current[engineId]
    if (!video) return
    
    if (playingEngine === engineId) {
      video.pause()
      setPlayingEngine(null)
    } else {
      // Pause any other playing video
      Object.values(videoRefs.current).forEach(v => v?.pause())
      video.currentTime = 0
      video.play().catch(() => {})
      setPlayingEngine(engineId)
    }
  }
  
  return (
    <div className="w-full max-w-[200px]">
      {/* 2x2 grid - Video Engine Comparison */}
      <div className="grid grid-cols-2 gap-2">
        {engines.map((engine) => {
          const hasError = videoErrors[engine.engine]
          const isPlaying = playingEngine === engine.engine
          const hasVideo = engine.videoUrl && !hasError
          
          return (
            <div
              key={engine.engine}
              className="relative aspect-square rounded-xl overflow-hidden bg-black/40 transition-all cursor-pointer group"
              style={{ border: `2px solid ${engine.color}` }}
            >
              {/* Video or Placeholder Image */}
              {hasVideo ? (
                <video
                  ref={(el) => { videoRefs.current[engine.engine] = el }}
                  src={engine.videoUrl}
                  className="w-full h-full object-cover"
                  loop
                  playsInline
                  muted
                  poster={compareData?.fallbackImage || '/kelly/archetypes/storyteller.png'}
                  onError={() => handleVideoError(engine.engine)}
                  onEnded={() => setPlayingEngine(null)}
                />
              ) : (
                <img 
                  src={compareData?.fallbackImage || '/kelly/archetypes/storyteller.png'}
                  alt="Kelly"
                  className="w-full h-full object-cover opacity-50"
                />
              )}
              
              {/* Play Button Overlay */}
              <button
                onClick={(e) => { 
                  e.stopPropagation()
                  if (hasVideo) {
                    togglePlay(engine.engine)
                  } else {
                    onSelectSource(engine.engine)
                  }
                }}
                className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/40 transition-colors"
              >
                {hasVideo ? (
                  isPlaying ? (
                    <Pause className="w-6 h-6 text-white drop-shadow-lg" />
                  ) : (
                    <Play className="w-6 h-6 text-white drop-shadow-lg" />
                  )
                ) : (
                  <Clock className="w-5 h-5 text-white/40" />
                )}
              </button>
              
              {/* Engine Label - LARGE AND VISIBLE */}
              <div 
                className="absolute bottom-0 left-0 right-0 px-2 py-1 text-[10px] font-bold text-white text-center uppercase tracking-wide"
                style={{ backgroundColor: engine.color }}
              >
                {engine.label}
              </div>
            </div>
          )
        })}
      </div>
      
{/* Compare All Button */}
  <button
  onClick={onOpenFineTuning}
  className="mt-2 w-full py-2 rounded-xl bg-white/10 text-white/80 text-[10px] font-semibold uppercase tracking-wider hover:bg-white/15 hover:text-white transition-all flex items-center justify-center gap-1.5"
  >
  <Grid2x2 className="w-3.5 h-3.5" />
  {tr('compareEngines')}
  </button>
    </div>
  )
}

// ============================================================================
// LEFT RAIL - Live Comments, Chat, Search
// ============================================================================

// LEFT RAIL SELECTOR COMPONENT
function LeftRailSelector({
  icon: Icon,
  label,
  value,
  onClick,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="glass-select flex items-center gap-3 text-left"
    >
      <Icon className="w-5 h-5 opacity-70" />
      <div className="flex-1 min-w-0">
        <div className="font-semibold truncate">{value}</div>
        <div className="text-xs opacity-60">{label}</div>
      </div>
      <ChevronDown className="w-4 h-4 opacity-50" />
    </button>
  )
}

function LeftRail({
  day,
  phase,
  language,
  chatOpen,
  searchOpen,
  onChatToggle,
  onSearchToggle,
  onDayChange,
  onShare,
}: {
  day: number
  phase: string
  language: string
  chatOpen: boolean
  searchOpen: boolean
  onChatToggle: () => void
  onSearchToggle: () => void
  onDayChange: (day: number) => void
  onShare: () => void
}) {
  const tr = useTranslation(language)
  const { comments, postComment, username, isLoading: commentsLoading } = useComments(day, phase)
  const { count: presenceCount } = usePresence(day)
  const { query, search, clearSearch, results, isLoading: searchLoading, hasSearched } = useSearch()
  const [newComment, setNewComment] = useState('')
  const [chatMessage, setChatMessage] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [chatResponse, setChatResponse] = useState('')
  
  const handleSubmitComment = async () => {
    if (!newComment.trim()) return
    await postComment(newComment)
    setNewComment('')
  }
  
  const handleChatSubmit = async (message?: string) => {
    const text = message || chatMessage
    if (!text.trim()) return
    
    setChatLoading(true)
    setChatMessage('')
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: text }],
          day,
          phase
        })
      })
      
      if (response.ok) {
        const reader = response.body?.getReader()
        const decoder = new TextDecoder()
        let result = ''
        
        while (reader) {
          const { done, value } = await reader.read()
          if (done) break
          result += decoder.decode(value)
        }
        setChatResponse(result)
      }
    } catch {
      setChatResponse('Sorry, I had trouble understanding that. Please try again!')
    } finally {
      setChatLoading(false)
    }
  }
  
// Elegant monochromatic avatar styling for Apple glass design
  const displayCount = Math.min(4, Math.max(1, Math.ceil(presenceCount / 4)))
  
  return (
    <div className="left-rail p-3 space-y-3">
      {/* Tabs - Clean navigation */}
      <div className="space-y-2">
        <div className="glass-tabs relative z-50 pointer-events-auto">
          <button
            onClick={() => { if (chatOpen) onChatToggle(); if (searchOpen) onSearchToggle(); }}
            className={cn('glass-tab', !chatOpen && !searchOpen && 'active')}
          >
            <MessageCircle className="w-3.5 h-3.5" />
            Live
          </button>
          <button
            onClick={onChatToggle}
            className={cn('glass-tab', chatOpen && 'active')}
          >
            <Sparkles className="w-3.5 h-3.5" />
            Kelly
          </button>
          <button
            onClick={onSearchToggle}
            className={cn('glass-tab', searchOpen && 'active')}
          >
            <Search className="w-3.5 h-3.5" />
            Find
          </button>
        </div>
      </div>
      
      {/* Share + Live Class - Vibrant glass panels */}
      <div className="space-y-2">
        {/* Share Button */}
        <button 
          onClick={onShare} 
          className="glass-card w-full flex items-center gap-3 py-3 px-4 hover:bg-white/[0.08] transition-all group cursor-pointer"
        >
          <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center group-hover:bg-white/15 transition-colors">
            <Share2 className="w-4 h-4 text-white" />
          </div>
          <span className="text-sm text-white font-medium">Share this lesson</span>
        </button>
        
        {/* Live Class Card */}
        <div className="glass-card p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="relative">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
                <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-red-500/50 animate-ping" />
              </div>
              <span className="text-white text-xs font-semibold uppercase tracking-wide">Live Class</span>
            </div>
            {presenceCount > 0 && (
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-white/10">
                <div className="flex items-center -space-x-1.5">
                  {LEARNER_AVATARS.slice(0, Math.min(presenceCount, 4)).map((learner, i) => (
                    <div
                      key={i}
                      className="w-6 h-6 rounded-full ring-2 ring-black/60 overflow-hidden relative group"
                      title={learner.name}
                    >
                      <img 
                        src={learner.avatar || "/placeholder.svg"} 
                        alt={learner.name}
                        className="w-full h-full object-cover"
                      />
                      {/* Live indicator dot */}
                      <div className="absolute bottom-0 right-0 w-2 h-2 bg-green-500 rounded-full ring-1 ring-black/50" />
                    </div>
                  ))}
                </div>
                <span className="text-white text-[11px] font-medium">{presenceCount} learning</span>
              </div>
            )}
          </div>
          <p className="text-white/70 text-xs mb-1">with Kelly starts in</p>
          <LiveClassCountdown />
        </div>
      </div>
      
      {/* Search Mode */}
      {searchOpen && (
        <div className="glass-card p-3 space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
            <input
              type="text"
              value={query}
              onChange={(e) => search(e.target.value)}
              placeholder="Search 365 lessons..."
              className="glass-input pl-10 pr-10"
            />
            {query && (
              <button onClick={clearSearch} className="absolute right-3 top-1/2 -translate-y-1/2 text-white/40 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
          
          {searchLoading && (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="w-5 h-5 animate-spin text-white/50" />
            </div>
          )}
          
          {hasSearched && !searchLoading && results.length === 0 && (
            <p className="text-white/40 text-sm text-center py-3">No lessons found</p>
          )}
          
          {results.length > 0 && (
            <div className="space-y-2 max-h-56 overflow-y-auto scrollbar-hide">
              {results.map((result) => (
                <button
                  key={result.day}
                  onClick={() => { onDayChange(result.day); onSearchToggle(); clearSearch(); }}
                  className="w-full p-3 rounded-xl bg-white/5 hover:bg-white/10 text-left transition-colors"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-semibold text-white/70">Day {result.day}</span>
                    <span className="text-xs text-white/40">{result.topic}</span>
                  </div>
                  <p className="text-sm text-white/90 font-medium line-clamp-2">{result.title}</p>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      
      {/* Comments Section - Vibrant glass panel */}
      {!chatOpen && !searchOpen && (
        <div className="glass-card p-4 space-y-3">
          {/* Header row */}
          <div className="flex items-center justify-between">
<div className="flex items-center gap-2">
  <MessageCircle className="w-4 h-4 text-white" />
  <span className="text-xs font-semibold text-white uppercase tracking-wide">{tr('comment')}</span>
  </div>
            <span className="text-[10px] text-white/60 font-medium">{comments.length > 0 ? `${comments.length} new` : 'Preview'}</span>
          </div>
          
          {/* Messages */}
          <div className="space-y-2 max-h-36 overflow-y-auto scrollbar-hide">
            {comments.length === 0 && !commentsLoading && (
              <div className="flex flex-col items-center justify-center py-6 text-center">
                <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center mb-2">
                  <MessageCircle className="w-5 h-5 text-white/50" />
                </div>
                <p className="text-white/60 text-xs">Be the first to comment</p>
              </div>
            )}
            {comments.slice(-4).map((comment) => (
              <div key={comment.id} className="group p-2.5 rounded-xl bg-white/[0.06] hover:bg-white/10 transition-colors">
                <div className="flex items-start gap-2">
                  <div className="w-6 h-6 rounded-full bg-[var(--kelly-blue)] flex items-center justify-center flex-shrink-0">
                    <span className="text-[10px] font-bold text-white">{comment.username.charAt(0).toUpperCase()}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <span className="text-white text-[11px] font-semibold">{comment.username}</span>
                    <p className="text-white text-xs leading-relaxed mt-0.5">{comment.message}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Input - Matching caption bar input style */}
          <div className="flex gap-2 pt-2 border-t border-white/[0.08]">
            <div className="flex-1 relative">
              <input
                type="text"
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSubmitComment()}
                placeholder="Add a comment..."
                className="w-full py-2.5 px-3.5 rounded-xl bg-white/[0.06] border border-white/[0.08] text-white/90 text-xs placeholder:text-white/30 focus:outline-none focus:border-white/20 focus:bg-white/[0.08] transition-all"
              />
            </div>
            <button
              onClick={handleSubmitComment}
              disabled={!newComment.trim()}
              className="w-9 h-9 rounded-xl bg-[var(--kelly-blue)] flex items-center justify-center disabled:opacity-30 disabled:bg-white/10 hover:brightness-110 transition-all"
            >
              <Send className="w-3.5 h-3.5 text-white" />
            </button>
          </div>
        </div>
      )}
      
      {/* Kelly Chat Mode - Vibrant glass panel */}
      {chatOpen && (
        <div className="glass-card p-4 space-y-4">
          {/* Header */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-[var(--kelly-blue)] flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div>
              <span className="text-sm font-semibold text-white">Ask Kelly</span>
              <p className="text-[10px] text-white/70">AI-powered lesson assistant</p>
            </div>
          </div>
          
          {/* Chat response area */}
          {chatResponse ? (
            <div className="p-4 rounded-xl bg-[var(--kelly-blue)]/15 border border-[var(--kelly-blue)]/25">
              <p className="text-sm text-white leading-relaxed">{chatResponse}</p>
            </div>
          ) : (
            <div className="flex items-center gap-3 p-3 rounded-xl bg-white/[0.06]">
              <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center">
                <MessageCircle className="w-4 h-4 text-white/60" />
              </div>
              <p className="text-white/80 text-xs">
                Chat with Kelly about today&apos;s lesson!
              </p>
            </div>
          )}
          
          {/* Quick suggestion chips */}
          <div className="flex flex-wrap gap-2">
            <button 
              onClick={() => handleChatSubmit('Explain it simpler')}
              disabled={chatLoading}
              className="px-3 py-1.5 rounded-full bg-white/10 border border-white/10 text-white text-[11px] font-medium hover:bg-white/15 hover:border-white/20 transition-all disabled:opacity-40"
            >
              Explain simpler
            </button>
            <button 
              onClick={() => handleChatSubmit('Give me an example')}
              disabled={chatLoading}
              className="px-3 py-1.5 rounded-full bg-white/10 border border-white/10 text-white text-[11px] font-medium hover:bg-white/15 hover:border-white/20 transition-all disabled:opacity-40"
            >
              Example
            </button>
            <button 
              onClick={() => handleChatSubmit('Why does this matter?')}
              disabled={chatLoading}
              className="px-3 py-1.5 rounded-full bg-white/10 border border-white/10 text-white text-[11px] font-medium hover:bg-white/15 hover:border-white/20 transition-all disabled:opacity-40"
            >
              Why it matters
            </button>
          </div>
          
          {/* Chat input - Premium matching caption bar */}
          <div className="flex gap-2 pt-3 border-t border-white/[0.08]">
            <div className="flex-1 relative">
              <input
                type="text"
                value={chatMessage}
                onChange={(e) => setChatMessage(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleChatSubmit()}
                placeholder="Ask about today's lesson..."
                disabled={chatLoading}
                className="w-full py-2.5 px-3.5 rounded-xl bg-white/[0.06] border border-white/[0.08] text-white/90 text-xs placeholder:text-white/30 focus:outline-none focus:border-white/20 focus:bg-white/[0.08] transition-all disabled:opacity-50"
              />
            </div>
            <button 
              onClick={() => handleChatSubmit()}
              disabled={chatLoading || !chatMessage.trim()}
              className="w-9 h-9 rounded-xl bg-[var(--kelly-blue)] flex items-center justify-center disabled:opacity-30 disabled:bg-white/10 hover:brightness-110 transition-all"
            >
              {chatLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin text-white" /> : <Send className="w-3.5 h-3.5 text-white" />}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// RIGHT RAIL - TikTok Actions
// ============================================================================

function RightRail({
  isPlaying,
  isMuted,
  isLiked,
  isSaved,
  onPlayPause,
  onNext,
  onMuteToggle,
  onSettings,
  likeCount,
  commentCount,
  onLike,
  onComment,
  onSave,
  language = 'en',
  }: {
  isPlaying: boolean
  isMuted: boolean
  isLiked?: boolean
  isSaved?: boolean
  onPlayPause: () => void
  onNext: () => void
  onMuteToggle: () => void
  onSettings: () => void
  likeCount: number
  commentCount: number
  onLike: () => void
  onComment: () => void
  onSave: () => void
  language?: string
  }) {
  const tr = useTranslation(language)
  
  // Glass icon button - COMPACT for mobile, with translated label + premium micro-interactions
  const GlassButton = ({
  onClick,
  icon,
  label,
  isActive,
  count,
  activeColor
  }: {
  onClick: () => void
  icon: React.ReactNode
  label: string
  isActive?: boolean
  count?: number
  activeColor?: string
  }) => {
  const [isPressed, setIsPressed] = useState(false)
  
  return (
  <button
  onClick={() => {
    setIsPressed(true)
    onClick()
    setTimeout(() => setIsPressed(false), 200)
  }}
  className={cn(
    'glass-icon-button flex-col gap-0.5 !w-11 !h-14 hover-lift focus-ring',
    isActive && 'active',
    isPressed && isActive && 'micro-bounce'
  )}
  style={isActive && activeColor ? { background: `${activeColor}20`, borderColor: `${activeColor}40` } : undefined}
  aria-label={label}
  >
  <div className={cn('w-4 h-4 flex items-center justify-center transition-transform duration-200', isPressed && 'scale-110')}>
  {icon}
  </div>
  {count !== undefined && count > 0 ? (
  <span className="text-[8px] font-bold text-white/90 tabular-nums">{count}</span>
  ) : (
  <span className="text-[7px] font-semibold text-white/60 uppercase tracking-wide leading-tight">{label}</span>
  )}
  </button>
  )
  }
  
  return (
  <div className="flex flex-col items-center gap-1.5 py-3 px-1">
  {/* Social: Like, Comment, Save */}
  <GlassButton
  onClick={onLike}
  icon={<Heart className={cn('w-4 h-4', isLiked ? 'fill-red-500 text-red-500' : '')} />}
  label={tr('like')}
  isActive={isLiked}
  count={likeCount}
  activeColor="#ef4444"
  />
  
  <GlassButton
  onClick={onComment}
  icon={<MessageCircle className="w-4 h-4" />}
  label={tr('comment')}
  count={commentCount}
  />
  
  <GlassButton
  onClick={onSave}
  icon={<Bookmark className={cn('w-4 h-4', isSaved ? 'fill-amber-400 text-amber-400' : '')} />}
  label={tr('save')}
  isActive={isSaved}
  activeColor="#f59e0b"
  />
  
  {/* Separator */}
  <div className="w-6 h-px bg-white/15 my-0.5" />
  
  {/* Playback: Play/Pause, Next, Mute */}
  <GlassButton
  onClick={onPlayPause}
  icon={isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
  label={isPlaying ? tr('pause') : tr('play')}
  isActive={isPlaying}
  />
  
  <GlassButton
  onClick={onNext}
  icon={<SkipForward className="w-4 h-4" />}
  label={tr('next')}
  />
  
  <GlassButton
  onClick={onMuteToggle}
  icon={isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
  label={isMuted ? tr('unmute') : tr('mute')}
  />
  
  {/* Separator */}
  <div className="w-6 h-px bg-white/15 my-0.5" />
  
  {/* Settings */}
  <GlassButton
  onClick={onSettings}
  icon={<Settings className="w-4 h-4" />}
  label={tr('settings')}
  />
  </div>
  )
  }

// ============================================================================
// BOTTOM BAR - Lesson Info, Caption, CTA
// ============================================================================

// Age options for picker
const AGE_OPTIONS = [
  { id: 'kid', label: 'Kid', age: 10, image: '/kelly/kid/storyteller.png' },
  { id: 'adult', label: 'Adult', age: 30, image: '/kelly/archetypes/storyteller.png' },
  { id: 'senior', label: 'Senior', age: 68, image: '/kelly/senior/storyteller.png' },
] as const

// Archetype options - EXACT match to database values (case sensitive)
// DB values: explorer, survivor, scientist, rebel, mystic, empath, strategist, macgyver, provider, architect, storyteller
const ARCHETYPE_OPTIONS = [
  { value: 'storyteller', label: 'Storyteller', icon: '📖' },
  { value: 'explorer', label: 'Explorer', icon: '🧭' },
  { value: 'architect', label: 'Architect', icon: '📐' },
  { value: 'empath', label: 'Empath', icon: '💗' },
  { value: 'rebel', label: 'Rebel', icon: '⚡' },
  { value: 'mystic', label: 'Mystic', icon: '🔮' },
  { value: 'provider', label: 'Provider', icon: '🏠' },
  { value: 'macgyver', label: 'MacGyver', icon: '🔧' },
  { value: 'scientist', label: 'Scientist', icon: '🔬' },
  { value: 'strategist', label: 'Strategist', icon: '♟️' },
  { value: 'survivor', label: 'Survivor', icon: '🏕️' },
] as const

  // Language options - EXACT match to database values: en, es, fr, de, pt, zh
  // ALL languages are available - text, audio, and video work in all languages
  const LANGUAGE_OPTIONS = [
    { code: 'en', label: 'English', short: 'US', flag: '🇺🇸' },
    { code: 'es', label: 'Español', short: 'ES', flag: '🇪🇸' },
    { code: 'fr', label: 'Français', short: 'FR', flag: '🇫🇷' },
    { code: 'de', label: 'Deutsch', short: 'DE', flag: '🇩🇪' },
    { code: 'pt', label: 'Português', short: 'PT', flag: '🇧🇷' },
    { code: 'zh', label: '中文', short: 'CN', flag: '🇨🇳' },
  ] as const

function BottomBar({
  day,
  dayNumber,
  lessonTitle,
  caption,
  onStartLearning,
  currentPhase,
  visualUrl,
  kellyAge,
  archetype,
  language,
  onAgeChange,
  onArchetypeChange,
  onLanguageChange,
  }: {
  day: number
  dayNumber: number
  lessonTitle: string
  caption: string
  onStartLearning: () => void
  currentPhase: LessonPhase
  visualUrl?: string
  kellyAge: number
  archetype: KellyArchetype
  language: string
  onAgeChange: (age: number) => void
  onArchetypeChange: (archetype: KellyArchetype) => void
  onLanguageChange: (lang: string) => void
}) {
  const [showTranslation, setShowTranslation] = useState(false)
  const [translatedCaption, setTranslatedCaption] = useState<string | null>(null)
  const [translating, setTranslating] = useState(false)
  const [selectedWord, setSelectedWord] = useState<string | null>(null)
  const [showDebate, setShowDebate] = useState(false)
  
  // UI translations for current language
  const { t } = useUITranslation(language)
  
  // REAL voting data from database
  const { userVote, submitVote, truePercent, totalVotes, votes } = useVote(day, currentPhase)
  
  // Handle translation toggle
  useEffect(() => {
    if (showTranslation && caption && language !== 'en' && !translatedCaption) {
      setTranslating(true)
      // Simple client-side translation simulation - in production use a translation API
      const translations: Record<string, Record<string, string>> = {
        es: {
          'Energy cannot be created or destroyed—it just changes costumes': 
            'La energia no se puede crear ni destruir, solo cambia de disfraz',
        },
        fr: {
          'Energy cannot be created or destroyed—it just changes costumes': 
            'L\'energie ne peut etre creee ni detruite, elle change simplement de costume',
        },
        zh: {
          'Energy cannot be created or destroyed—it just changes costumes': 
            '能量不能被创造或销毁，它只是换了一套衣服',
        },
      }
      
      setTimeout(() => {
        const langTranslations = translations[language] || {}
        setTranslatedCaption(langTranslations[caption] || `[${LANGUAGE_OPTIONS.find(l => l.code === language)?.label || language}] ${caption}`)
        setTranslating(false)
      }, 500)
    }
    if (!showTranslation) {
      setTranslatedCaption(null)
    }
  }, [showTranslation, caption, language, translatedCaption])
  
  // Get real comments with votes for debate view
  const { comments: debateComments } = useComments(day, currentPhase)
  
  // Handle word click for definition
  const handleWordClick = (word: string) => {
    setSelectedWord(selectedWord === word ? null : word)
  }
  
  // Render caption with clickable words and optional translation
  const renderInteractiveCaption = () => {
    if (!caption) return null
    const displayText = showTranslation && translatedCaption ? translatedCaption : caption
    const words = displayText.split(' ')
    return (
      <div className="space-y-1">
        <p className="text-[16px] leading-relaxed font-medium">
          {words.map((word, i) => (
            <span
              key={i}
              onClick={() => handleWordClick(word.replace(/[^a-zA-Z]/g, ''))}
              className="cursor-pointer hover:text-blue-300 hover:underline transition-colors"
            >
              {word}{i < words.length - 1 ? ' ' : ''}
            </span>
          ))}
        </p>
        {showTranslation && translating && (
          <p className="text-xs text-white/50 italic">Translating...</p>
        )}
        {showTranslation && translatedCaption && language !== 'en' && (
          <p className="text-xs text-white/40 mt-1">
            Original: {caption}
          </p>
        )}
      </div>
    )
  }
  
  const trueVotes = debateComments.filter(c => c.vote === 'true').length
  const falseVotes = debateComments.filter(c => c.vote === 'false').length
  const localTruePercent = debateComments.length > 0 ? Math.round((trueVotes / debateComments.length) * 100) : 0
  
  return (
    <div className="bottom-bar gap-3">
      {/* HERO STATE - When no caption, show minimal typography + play */}
      {!caption && !visualUrl && (
        <div className="flex flex-col items-center gap-4 py-8 px-6">
          {/* TITLE - THE HERO - largest, dominant */}
          <h1 className="text-[32px] sm:text-[40px] font-semibold text-white text-center leading-[1.1]" style={{ letterSpacing: '-0.025em' }}>
            {lessonTitle || "What Makes Things Grow"}
          </h1>
          
          {/* Lesson tagline - whisper quiet */}
          <span className="text-[13px] text-white/40 tracking-wide">Tap play to begin your lesson</span>
          
          {/* Let's Begin CTA Button - Solid blue, no gradient */}
          <button
            onClick={onStartLearning}
            className="group flex items-center gap-3 px-8 py-4 rounded-full bg-[var(--kelly-blue)] hover:bg-[var(--kelly-blue-hover)] transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] mt-4 shadow-lg shadow-blue-500/25"
          >
            <Play className="w-5 h-5 text-white fill-white transition-transform group-hover:scale-110" />
            <span className="text-[17px] font-semibold text-white tracking-tight">{t('lets_begin')}</span>
          </button>
        </div>
      )}
      
      {/* Interactive Caption Card - shows when playing - OPTIMIZED for mobile */}
      {(caption || visualUrl) && (
        <div className="max-w-4xl w-full mx-auto px-1.5 sm:px-0">
          <div className="gradient-border rounded-[20px] sm:rounded-[24px] overflow-hidden glass-panel text-white">
            {/* Main content - tighter on mobile */}
            <div className="flex flex-col">
              {/* Caption + Visual row */}
              <div className="flex items-start gap-2.5 p-2.5 sm:gap-3 sm:p-4">
{/* Visual thumbnail - compact on mobile */}
                <div className="flex-shrink-0 w-12 h-12 sm:w-20 sm:h-20 rounded-xl sm:rounded-[16px] overflow-hidden bg-white/[0.06] relative">
                    {visualUrl && (
                      <img
                        src={visualUrl || "/placeholder.svg"}
                        alt="Lesson visual"
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          // On error, show sparkles icon instead
                          const target = e.target as HTMLImageElement
                          target.style.display = 'none'
                        }}
                      />
                    )}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Sparkles className="w-6 h-6 text-white/40" />
                    </div>
                  </div>
                
                {/* Caption text */}
                <div className="flex-1 min-w-0">
                  {renderInteractiveCaption()}
                  
{/* HOOK PHASE: True or False Game */}
  {currentPhase === 'hook' && (
  <div className="mt-2 sm:mt-3 space-y-2 sm:space-y-3">
  {/* Voting buttons - compact on mobile */}
  {!userVote && (
  <div className="flex items-center gap-2 sm:gap-3">
  <span className="text-[10px] sm:text-[11px] uppercase tracking-wider text-white/40 font-medium">{t('isThis')}</span>
  <button
  onClick={() => { submitVote('true'); setShowDebate(true); }}
  className="px-4 sm:px-5 py-1.5 sm:py-2 rounded-full text-xs sm:text-sm font-semibold transition-all duration-200 bg-white/10 text-white/70 hover:bg-white/20 hover:text-white border border-white/10 hover:border-white/25 active:scale-95"
  >
  {t('true')}
  </button>
  <span className="text-white/25 text-[10px] sm:text-xs">{t('or')}</span>
  <button
  onClick={() => { submitVote('false'); setShowDebate(true); }}
  className="px-4 sm:px-5 py-1.5 sm:py-2 rounded-full text-xs sm:text-sm font-semibold transition-all duration-200 bg-white/10 text-white/70 hover:bg-white/20 hover:text-white border border-white/10 hover:border-white/25 active:scale-95"
  >
  {t('false')}
  </button>
  </div>
  )}
                      
                      {/* After voting - REVEAL the answer (always TRUE) */}
                      {userVote && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <div className={cn(
                              "px-3 py-1 rounded-full text-xs font-bold",
                              userVote === 'true' 
                                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30" 
                                : "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                            )}>
                              {userVote === 'true' ? t('correct') : t('notQuite')}
                            </div>
                            <span className="text-[11px] text-white/50">
                              {t('theAnswerIs')} <span className="font-bold text-white/80">{t('true')}</span>
                            </span>
                          </div>
                          
                          {/* Explanation - why it's true */}
                          <div className="p-3 rounded-xl bg-white/[0.04] border border-white/[0.08]">
                            <p className="text-[13px] text-white/70 leading-relaxed">
                              <span className="text-white/90 font-medium">{t('whyItsTrue')}:</span>{' '}
                              {t('explanation_hook')}
                            </p>
                          </div>
                          
                          {/* Vote stats */}
                          <div className="flex items-center gap-3 text-[11px] text-white/40">
                            <span>{truePercent}% {t('saidTrue')}</span>
                            <span>•</span>
                            <span>{totalVotes} {t('votes')}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* Selected word definition */}
                  {selectedWord && (
                    <div className="mt-2 p-2 rounded-lg bg-white/10 text-sm">
                      <span className="text-white/80 font-medium">{selectedWord}</span>
                      <span className="text-white/40 ml-2">- tap to hear pronunciation</span>
                    </div>
                  )}
                </div>
              </div>
              
              {/* Live Debate Comments - shows after voting */}
              {showDebate && userVote && (
                <div className="border-t border-white/10 bg-white/5">
                  {/* Vote results bar */}
                  <div className="px-4 py-2 flex items-center gap-2">
<div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden flex">
          <div
            className="h-full bg-white/40 transition-all duration-500"
            style={{ width: `${localTruePercent}%` }}
          />
          <div
            className="h-full bg-white/15 transition-all duration-500"
            style={{ width: `${100 - localTruePercent}%` }}
          />
        </div>
                    <span className="text-[11px] text-white/60">{localTruePercent}% say TRUE</span>
                  </div>
                  
                  {/* Live comments stream - REAL DATA from database */}
                  <div className="px-4 pb-3 max-h-32 overflow-y-auto scrollbar-hide space-y-1.5">
                    {debateComments.filter(c => c.vote).map((comment) => (
                      <div key={comment.id} className="flex items-start gap-2 text-[13px]">
                        <span className={cn(
                          "flex-shrink-0 w-1.5 h-1.5 rounded-full mt-1.5",
                          comment.vote === 'true' ? "bg-white/60" : "bg-white/30"
                        )} />
                        <span className="text-white/60 font-medium">{comment.username}</span>
                        <span className="text-white/80 flex-1">{comment.message}</span>
                      </div>
                    ))}
                    {debateComments.filter(c => c.vote).length === 0 && (
                      <p className="text-white/40 text-sm text-center py-2">No debate comments yet - be the first!</p>
                    )}
                  </div>
                </div>
              )}
              
{/* Pickers Row - ULTRA compact on mobile */}
        <div className="flex items-center justify-between gap-1.5 sm:gap-4 px-2.5 sm:px-4 py-1.5 sm:py-3 border-t border-white/[0.06]">
          {/* Age Picker - Compact avatars */}
          <div className="flex items-center gap-0.5 sm:gap-2">
            <span className="text-[7px] sm:text-[10px] uppercase tracking-wider text-white/30 font-medium">Age</span>
            {AGE_OPTIONS.map((opt) => {
              const isSelected = getAgeCategory(kellyAge) === opt.id
              return (
                <button
                  key={opt.id}
                  onClick={() => onAgeChange(opt.age)}
                  className={cn(
                    "w-7 h-7 sm:w-9 sm:h-9 rounded-full overflow-hidden transition-all duration-200",
                    isSelected 
                      ? "ring-2 ring-white/40 scale-110" 
                      : "opacity-40 hover:opacity-90 grayscale-[60%] hover:grayscale-0"
                  )}
                  title={opt.label}
                  aria-label={opt.label}
                >
                  <img src={opt.image || "/placeholder.svg"} alt={opt.label} className="w-full h-full object-cover" />
                </button>
              )
            })}
          </div>
          
{/* Archetype/Tone Selector - Compact on mobile */}
  <div className="flex items-center gap-0.5 sm:gap-1 overflow-x-auto scrollbar-hide py-0.5 flex-shrink-0">
  {ARCHETYPE_OPTIONS.slice(0, 6).map((opt) => {
  const isSelected = archetype === opt.value
  return (
  <button
  key={opt.value}
  onClick={() => onArchetypeChange(opt.value as KellyArchetype)}
  className={cn(
  "flex-shrink-0 w-6 h-6 sm:w-7 sm:h-7 rounded sm:rounded-lg flex items-center justify-center text-[10px] sm:text-sm transition-all duration-200",
  isSelected
  ? "bg-white/25 scale-105"
  : "opacity-40 hover:opacity-100"
  )}
  title={opt.label}
  aria-label={opt.label}
  >
  {opt.icon}
  </button>
  )
  })}
  {/* More archetypes - desktop only */}
  {ARCHETYPE_OPTIONS.slice(6).map((opt) => {
  const isSelected = archetype === opt.value
  return (
  <button
  key={opt.value}
  onClick={() => onArchetypeChange(opt.value as KellyArchetype)}
  className={cn(
  "hidden sm:flex flex-shrink-0 w-7 h-7 rounded-lg items-center justify-center text-sm transition-all duration-200",
  isSelected
  ? "bg-white/20 ring-1 ring-white/40 scale-110"
  : "opacity-40 hover:opacity-100"
  )}
  title={opt.label}
  aria-label={opt.label}
  >
  {opt.icon}
  </button>
  )
  })}
  </div>
          
{/* Language Selector - ALL 6 fit on mobile */}
  <div className="flex items-center flex-shrink-0">
  {LANGUAGE_OPTIONS.map((opt) => {
  const isSelected = language === opt.code
  return (
  <button
  key={opt.code}
  onClick={() => onLanguageChange(opt.code)}
  className={cn(
  "px-1 sm:px-2 py-0.5 sm:py-1 rounded text-[8px] sm:text-[11px] font-bold transition-all duration-200",
  isSelected
  ? "bg-white/25 text-white"
  : "text-white/30 hover:text-white/70"
  )}
  title={opt.label}
  aria-label={opt.label}
  >
  {opt.short}
  </button>
  )
  })}
  </div>
        </div>
              
              {/* Translation panel */}
              {showTranslation && (
                <div className="px-4 py-2 bg-white/5 border-t border-white/10">
                  <p className="text-sm italic text-white/60">
                    [Translation in {LANGUAGE_OPTIONS.find(l => l.code === language)?.label || language}]
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      </div>
  )
}

// ============================================================================
// CENTER STAGE - Kelly Video (SACRED - NEVER COVER)
// ============================================================================

function CenterStage({
  videoUrl,
  audioUrl,
  posterUrl,
  isPlaying,
  isMuted,
  currentPhase,
  kellyAge,
  archetype,
  language,
  onVideoClick,
  onMuteToggle,
  videoRef,
  audioRef,
  onSwipeLeft,
  onSwipeRight,
}: {
  videoUrl: string | null
  audioUrl: string | null
  posterUrl: string
  isPlaying: boolean
  isMuted: boolean
  currentPhase: LessonPhase
  kellyAge: number
  archetype: KellyArchetype
  language: string
  onVideoClick: () => void
  onMuteToggle: () => void
  videoRef: React.RefObject<HTMLVideoElement | null>
  audioRef: React.RefObject<HTMLAudioElement | null>
  onSwipeLeft?: () => void
  onSwipeRight?: () => void
}) {
  // Translation hook
  const { t: tr } = useUITranslation(language)
  
  // Get archetype-specific image from kelly-assets library (age-aware)
  // Using imported getAgeCategory from kelly-assets (returns 'kid' | 'adult' | 'senior')
  const ageCategory = getAgeCategory(kellyAge)
  const kellyImage = getKellyArchetypeImage(archetype, ageCategory) || posterUrl
  
  // Track if video failed to load (404, etc) - fall back to image+audio
  const [videoFailed, setVideoFailed] = React.useState(false)
  
  // Track audio/video progress for progress bar
  const [progress, setProgress] = React.useState(0)
  const [isBuffering, setIsBuffering] = React.useState(false)
  
  // Reset video error when URL changes
  React.useEffect(() => {
    setVideoFailed(false)
    setProgress(0)
  }, [videoUrl, audioUrl])
  
  // Track playback progress
  React.useEffect(() => {
    const media = videoRef.current || audioRef.current
    if (!media) return
    
    const handleTimeUpdate = () => {
      if (media.duration) {
        setProgress((media.currentTime / media.duration) * 100)
      }
    }
    const handleWaiting = () => setIsBuffering(true)
    const handlePlaying = () => setIsBuffering(false)
    const handleEnded = () => setProgress(100)
    
    media.addEventListener('timeupdate', handleTimeUpdate)
    media.addEventListener('waiting', handleWaiting)
    media.addEventListener('playing', handlePlaying)
    media.addEventListener('ended', handleEnded)
    
    return () => {
      media.removeEventListener('timeupdate', handleTimeUpdate)
      media.removeEventListener('waiting', handleWaiting)
      media.removeEventListener('playing', handlePlaying)
      media.removeEventListener('ended', handleEnded)
    }
  }, [videoRef, audioRef, videoUrl, audioUrl])
  
  // Actual video availability: has URL AND hasn't failed
  const hasWorkingVideo = !!videoUrl && !videoFailed
  
  // Audio-only mode: no working video but have audio
  const isAudioOnly = !hasWorkingVideo && !!audioUrl
  
  // Touch gesture tracking
  const touchStart = React.useRef<{ x: number; time: number } | null>(null)
  
  const handleTouchStart = (e: React.TouchEvent) => {
    touchStart.current = { x: e.touches[0].clientX, time: Date.now() }
  }
  
  const handleTouchEnd = (e: React.TouchEvent) => {
    if (!touchStart.current) return
    const deltaX = e.changedTouches[0].clientX - touchStart.current.x
    const deltaTime = Date.now() - touchStart.current.time
    touchStart.current = null
    
    // Swipe must be fast (<300ms) and >50px
    if (deltaTime > 300 || Math.abs(deltaX) < 50) return
    
    if (deltaX > 0) {
      onSwipeRight?.() // Swipe right = previous day
    } else {
      onSwipeLeft?.() // Swipe left = next day
    }
  }
  
  return (
    <div 
      className="center-stage" 
      onClick={onVideoClick}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
    >
      {/* Kelly - ANCHORED TOP & BOTTOM, sides expand, NO shadows, pure #FFFFFF bg */}
      {/* Age classes: kelly-child (<= 12), kelly-teen (13-19), kelly-adult (20-35), kelly-middleAge (36-55), kelly-senior (56+) */}
      <div className={cn(
        "kelly-container relative",
        isPlaying && "kelly-speaking",
        kellyAge <= 12 && "kelly-child",
        kellyAge > 12 && kellyAge <= 19 && "kelly-teen",
        kellyAge > 19 && kellyAge <= 35 && "kelly-adult",
        kellyAge > 35 && kellyAge <= 55 && "kelly-middleAge",
        kellyAge > 55 && "kelly-senior"
      )}>
{hasWorkingVideo ? (
  /* VIDEO MODE: Video has baked-in audio */
  <>
  <video
  ref={videoRef}
  src={videoUrl!}
  poster={kellyImage}
  playsInline
  muted={isMuted}
  loop={false}
  autoPlay={isPlaying}
  onError={() => setVideoFailed(true)}
  />
{/* TAP TO HEAR KELLY - lower right corner, glass-pill style */}
  {isPlaying && isMuted && (
  <button
  onClick={(e) => { e.stopPropagation(); onMuteToggle(); }}
  className="absolute bottom-6 right-6 z-20 cursor-pointer group"
  aria-label={tr('tapToHear')}
  >
  <div className="glass-pill flex items-center gap-3 px-5 py-3 transform group-hover:scale-105 transition-transform">
  <div className="w-10 h-10 rounded-full bg-white/15 flex items-center justify-center">
  <Volume2 className="w-5 h-5 text-white" />
  </div>
  <span className="text-white text-sm font-semibold pr-1">{tr('tapToHear')}</span>
  </div>
  </button>
  )}
  </>
  ) : (
          /* IMAGE MODE: Age+Archetype image with TTS play button */
          <>
            <img
              src={kellyImage || "/kelly/archetypes/storyteller.png"}
              alt="Curious Kelly, trademark of Lesson of the Day PBC - Your AI Learning Companion"
              onError={(e) => {
                e.currentTarget.src = '/kelly/archetypes/storyteller.png'
              }}
            />
            {/* TTS PLAY BUTTON - shows when we have audio but no video */}
            {audioUrl ? (
              <button
                onClick={(e) => { e.stopPropagation(); onVideoClick(); }}
                className="absolute bottom-6 right-6 z-20 cursor-pointer group"
                aria-label={isPlaying ? tr('pause') : tr('play')}
              >
                <div className="glass-pill flex items-center gap-3 px-5 py-3 transform group-hover:scale-105 transition-transform">
                  <div className="w-10 h-10 rounded-full bg-white/15 flex items-center justify-center">
                    {isPlaying ? (
                      <Pause className="w-5 h-5 text-white" />
                    ) : (
                      <Play className="w-5 h-5 text-white ml-0.5" />
                    )}
                  </div>
                  <div className="flex flex-col items-start">
                    <span className="text-white text-sm font-semibold">{isPlaying ? tr('pause') : tr('play')}</span>
                    <span className="text-white/60 text-[10px]">ElevenLabs TTS</span>
                  </div>
                </div>
              </button>
            ) : (
              /* No audio - show preview mode status */
              <div className="absolute inset-0 flex items-end justify-center pb-6 pointer-events-none">
                <div className="relative z-10 text-center px-4">
                  <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/10 backdrop-blur-md border border-white/20 mb-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-white/70 animate-pulse" />
                    <span className="text-white/90 text-[11px] font-medium tracking-wider uppercase">Preview Mode</span>
                  </div>
                  <p className="text-white/60 text-[10px] max-w-[240px] leading-relaxed">
                    Kelly&apos;s video lessons are being crafted with care
                  </p>
                </div>
              </div>
            )}
          </>
        )}
        
        {/* Buffering indicator */}
        {isBuffering && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/20">
            <div className="w-8 h-8 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          </div>
        )}
        
        {/* Progress bar - shows audio/video progress */}
        {isPlaying && progress > 0 && (
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-black/20">
            <div 
              className="h-full bg-[var(--kelly-blue)] transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </div>
      
      {/* AUDIO ELEMENT - Always rendered (hidden) as fallback for video failures */}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          muted={isMuted}
          preload="auto"
          className="hidden"
        />
      )}
    </div>
  )
}

// ============================================================================
// CHAT PANEL (slides in from right over Kelly when open) - FULLY FUNCTIONAL
// ============================================================================

type ChatMessage = { id: string; role: 'user' | 'assistant'; content: string }

function ChatPanel({
  isOpen,
  onClose,
  kellyAge,
  archetype,
  day,
  lessonTitle,
  lessonTopic,
}: {
  isOpen: boolean
  onClose: () => void
  kellyAge: number
  archetype: KellyArchetype
  day: number
  lessonTitle?: string
  lessonTopic?: string
}) {
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const archetypeLabel = ARCHETYPES.find(a => a.id === archetype)?.label || 'Storyteller'
  const ageVariant = kellyAge <= 12 ? 'kid' : kellyAge >= 65 ? 'senior' : 'adult'
  
  // Custom chat state (no external dependencies)
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'intro',
      role: 'assistant',
      content: `Hi! I'm Kelly, your curious learning companion. Today is Day ${day} and we're exploring "${lessonTitle || 'How Energy Changes Form'}". What would you like to know about this topic? Ask me anything!`
    }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
  }
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    
    const userMessage: ChatMessage = { id: Date.now().toString(), role: 'user', content: input.trim() }
    const newMessages = [...messages, userMessage]
    setMessages(newMessages)
    setInput('')
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages.map(m => ({ role: m.role, content: m.content })),
          day,
          ageVariant,
          archetype,
          lessonTitle,
          lessonTopic,
          lessonContext: `Day ${day}: ${lessonTitle || 'How Energy Changes Form'} - ${lessonTopic || 'Energy cannot be created or destroyed, it just changes costumes'}`
        })
      })
      
      if (!response.ok) throw new Error('Chat failed')
      
      // Handle streaming response
      const reader = response.body?.getReader()
      if (!reader) throw new Error('No reader')
      
      const assistantMessage: ChatMessage = { id: (Date.now() + 1).toString(), role: 'assistant', content: '' }
      setMessages([...newMessages, assistantMessage])
      
      const decoder = new TextDecoder()
      let fullText = ''
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        const chunk = decoder.decode(value, { stream: true })
        // Parse SSE data chunks (format: "0:\"text\"\n")
        const lines = chunk.split('\n')
        for (const line of lines) {
          if (line.startsWith('0:')) {
            try {
              const text = JSON.parse(line.slice(2))
              fullText += text
              setMessages(prev => {
                const updated = [...prev]
                const lastIdx = updated.length - 1
                if (updated[lastIdx]?.role === 'assistant') {
                  updated[lastIdx] = { ...updated[lastIdx], content: fullText }
                }
                return updated
              })
            } catch {
              // Ignore JSON parse errors in streaming response
            }
          }
        }
      }
    } catch (err) {
      setError('Something went wrong. Please try again.')
      console.error('Chat error:', err)
    } finally {
      setIsLoading(false)
    }
  }
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])
  
  if (!isOpen) return null
  
  return (
    <div className="fixed top-0 right-16 bottom-0 w-96 max-w-[45vw] z-30 flex flex-col" style={{ background: 'oklch(0.14 0.03 250 / 0.95)', backdropFilter: 'blur(24px)' }}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <div>
          <h2 className="text-white font-semibold">Chat with Kelly</h2>
          <p className="text-white/50 text-xs">Age {kellyAge} - {archetypeLabel}</p>
        </div>
        <button onClick={onClose} className="text-white/60 hover:text-white p-1.5 rounded-lg hover:bg-white/10 transition-colors">
          <X className="w-5 h-5" />
        </button>
      </div>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide">
        {messages.map((msg) => (
          <div key={msg.id} className={cn("flex gap-3", msg.role === 'user' && "flex-row-reverse")}>
            {msg.role === 'assistant' ? (
              <div className="w-8 h-8 rounded-full bg-[var(--kelly-blue)] flex items-center justify-center text-white text-sm font-bold flex-shrink-0">
                K
              </div>
            ) : (
              <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center text-white text-sm font-bold flex-shrink-0">
                <User className="w-4 h-4" />
              </div>
            )}
            <div 
              className={cn(
                "flex-1 rounded-2xl px-4 py-3 text-sm leading-relaxed",
                msg.role === 'assistant' 
                  ? "rounded-tl-sm" 
                  : "rounded-tr-sm"
              )}
              style={{ 
                background: msg.role === 'assistant' 
                  ? 'oklch(0.22 0.03 250 / 0.7)' 
                  : 'oklch(0.35 0.08 250 / 0.6)',
                color: 'oklch(0.95 0.01 250)' 
              }}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}
        
        {/* Loading indicator */}
        {isLoading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-[var(--kelly-blue)] flex items-center justify-center text-white text-sm font-bold flex-shrink-0">
              K
            </div>
            <div className="rounded-2xl rounded-tl-sm px-4 py-3" style={{ background: 'oklch(0.22 0.03 250 / 0.7)' }}>
              <div className="flex gap-1.5">
                <span className="w-2 h-2 bg-white/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-white/50 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-white/50 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        
        {/* Error display */}
        {error && (
          <div className="text-red-400 text-sm text-center py-2">
            Something went wrong. Please try again.
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      {/* Suggested questions */}
      {messages.length <= 1 && (
        <div className="px-4 pb-2">
          <p className="text-white/40 text-xs mb-2">Try asking:</p>
          <div className="flex flex-wrap gap-2">
            {[
              "Explain it simpler",
              "Give me an example",
              "Why does this matter?",
              "Tell me more"
            ].map((q) => (
              <button
                key={q}
                onClick={() => {
                  setInput(q)
                  // Submit after state updates
                  setTimeout(() => {
                    const form = document.querySelector('form[data-chat-form]') as HTMLFormElement
                    form?.requestSubmit()
                  }, 100)
                }}
                className="px-3 py-1.5 rounded-full text-xs text-white/70 hover:text-white hover:bg-white/10 transition-colors border border-white/20"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
      
      {/* Input */}
      <form data-chat-form onSubmit={handleSubmit} className="p-4 border-t border-white/10">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={handleInputChange}
            placeholder="Ask Kelly about today's lesson..."
            disabled={isLoading}
            className="flex-1 px-4 py-2.5 rounded-full text-sm text-white placeholder:text-white/40 outline-none disabled:opacity-50"
            style={{ background: 'oklch(0.22 0.03 250 / 0.5)' }}
          />
          <button 
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-4 py-2.5 rounded-full text-sm font-medium disabled:opacity-50 transition-all hover:scale-105 active:scale-95"
            style={{ background: 'var(--kelly-blue)', color: 'white' }}
          >
            {isLoading ? '...' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  )
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Age category function - matches heygen_videos.age_category in database
// Database values: child, teen, adult, middleAge, senior
type DetailedAgeCategory = 'child' | 'teen' | 'adult' | 'middleAge' | 'senior'

function getDetailedAgeCategory(age: number): DetailedAgeCategory {
  if (age <= 12) return 'child'
  if (age <= 19) return 'teen'
  if (age <= 35) return 'adult'
  if (age <= 55) return 'middleAge'
  return 'senior'
}

// Legacy function for live_class_rooms which still uses kid/adult/senior
function getLiveClassAgeCategory(age: number): 'kid' | 'adult' | 'senior' {
  if (age <= 12) return 'kid'
  if (age >= 65) return 'senior'
  return 'adult'
}

// ============================================================================
// ZONE COMPONENTS - Wrappers for grid layout
// ============================================================================

function LeftRailTopSection({
  day,
  chatOpen,
  searchOpen,
  onChatToggle,
  onSearchToggle,
  onShare,
  language,
}: {
  day: number
  chatOpen: boolean
  searchOpen: boolean
  onChatToggle: () => void
  onSearchToggle: () => void
  onShare: () => void
  language: string
}) {
  // Presence counter is now inside Live Class card (LeftRailContent)
  
const { t } = useUITranslation(language)
  
  return (
  <div className="p-3">
  {/* Mode Toggle - Using glass-tabs design system */}
  <div className="glass-tabs">
  <button
  onClick={() => { if (chatOpen) onChatToggle(); if (searchOpen) onSearchToggle(); }}
  className={cn('glass-tab', !chatOpen && !searchOpen && 'active')}
  >
  <div className={cn(
    'w-2 h-2 rounded-full transition-colors',
    !chatOpen && !searchOpen ? 'bg-red-500' : 'bg-white/30'
  )} />
  <span>{t('live')}</span>
  </button>
  <button
  onClick={onChatToggle}
  className={cn('glass-tab', chatOpen && 'active')}
  >
  <Sparkles className="w-3.5 h-3.5" />
  <span>{t('kelly')}</span>
  </button>
  <button
  onClick={onSearchToggle}
  className={cn('glass-tab', searchOpen && 'active')}
  >
  <Search className="w-3.5 h-3.5" />
  <span>{t('find')}</span>
  </button>
  </div>
  </div>
  )
  }

function LeftRailContent({
  day,
  phase,
  language,
  chatOpen,
  searchOpen,
  onDayChange,
  onChatToggle,
  onSearchToggle,
  lessonTitle,
}: {
  day: number
  phase: string
  language: string
  chatOpen: boolean
  searchOpen: boolean
  onDayChange: (day: number) => void
  onChatToggle: () => void
  onSearchToggle: () => void
  lessonTitle: string
}) {
  // Translations
  const tr = useTranslation(language)
  
  // Presence count for Live Class card
  const { count: presenceCount } = usePresence(day)
  
  // Search functionality
  const { query, results, isLoading: searchLoading, hasSearched, search, clearSearch } = useSearch()
  
  // Chat state
  const [chatInput, setChatInput] = useState('')
  const [chatResponse, setChatResponse] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  
  // Share panel state (inline expandable)
  const [shareExpanded, setShareExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  
  const shareUrl = typeof window !== 'undefined' ? window.location.href : 'https://curiouskelly.com'
  const shareText = `${tr('todaysLesson')}: ${lessonTitle}`
  
  const handleShare = async (platform: string) => {
    switch (platform) {
      case 'copy':
        await navigator.clipboard.writeText(shareUrl)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
        break
      case 'twitter':
        window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}`, '_blank')
        break
      case 'facebook':
        window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`, '_blank')
        break
      case 'whatsapp':
        window.open(`https://wa.me/?text=${encodeURIComponent(`${shareText} ${shareUrl}`)}`, '_blank')
        break
    }
  }
  
  return (
    <div className="px-3 space-y-2 overflow-y-auto scrollbar-hide">
      {/* UNIFIED LEFT RAIL - Same glass styling as caption bar */}
      
      {/* Search Mode - Full panel when active */}
      {searchOpen && (
        <div className="glass-card p-4">
          <div className="relative mb-3">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
            <input
              type="text"
              value={query}
              onChange={(e) => search(e.target.value)}
              placeholder={tr('searchLessons')}
              className="glass-input pl-10 pr-10"
              autoFocus
            />
            {query && (
              <button onClick={clearSearch} className="absolute right-3 top-1/2 -translate-y-1/2 text-white/40 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
          
          {searchLoading && (
            <div className="flex items-center justify-center py-6">
              <Loader2 className="w-5 h-5 animate-spin text-white/50" />
            </div>
          )}
          
          {hasSearched && !searchLoading && results.length === 0 && (
            <p className="text-white/50 text-sm text-center py-6">No lessons found for &ldquo;{query}&rdquo;</p>
          )}
          
          {results.length > 0 && (
            <div className="space-y-2 max-h-64 overflow-y-auto scrollbar-hide">
              {results.map((result) => (
                <button
                  key={result.day}
                  onClick={() => { onDayChange(result.day); onSearchToggle(); clearSearch(); }}
                  className="w-full p-3 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] text-left transition-colors"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[10px] font-bold text-white/60 uppercase">{tr('day')} {result.day}</span>
                    <span className="text-[10px] text-white/40">{result.topic}</span>
                  </div>
                  <p className="text-sm text-white font-medium line-clamp-2">{result.title}</p>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      
      {/* Chat Mode - Full panel when active */}
      {chatOpen && (
        <div className="glass-card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-4 h-4 text-[var(--kelly-blue)]" />
            <span className="text-sm font-semibold text-white">{tr('askKelly')}</span>
          </div>
          
          {chatResponse && (
            <div className="p-3 rounded-xl bg-[var(--kelly-blue)]/10 border border-[var(--kelly-blue)]/20 text-sm text-white/90 mb-3">
              {chatResponse}
            </div>
          )}
          
          <div className="flex gap-2">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder={tr('askKelly')}
              className="glass-input flex-1"
              autoFocus
            />
            <button
              disabled={!chatInput.trim() || chatLoading}
              className="glass-icon-button w-10 h-10 disabled:opacity-30"
            >
              {chatLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
        </div>
      )}
      
      {/* Default State - Clean info cards */}
      {!chatOpen && !searchOpen && (
        <>
{/* Live Class - Using glass-card design system */}
  <div className="glass-card p-4">
  <div className="flex items-center justify-between mb-3">
  <div className="flex items-center gap-2.5">
  <div className="glass-live-dot" />
  <span className="text-[11px] font-bold text-white/90 uppercase tracking-wider">{tr('liveClass')}</span>
  </div>
  {presenceCount > 0 && (
  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-white/10 border border-white/10">
  <Users className="w-3 h-3 text-white/60" />
  <span className="text-[10px] font-semibold text-white/70 tabular-nums">{presenceCount}</span>
  </div>
  )}
  </div>
  <div className="text-xl font-bold text-white tabular-nums tracking-tight">
  <LiveClassCountdown />
  </div>
  </div>
          
{/* Share - Expandable inline panel using glass-card */}
  <div className="glass-card overflow-hidden">
  <button
  onClick={() => setShareExpanded(!shareExpanded)}
  className="w-full flex items-center justify-between p-3 hover:bg-white/5 transition-colors cursor-pointer"
  >
  <div className="flex items-center gap-3">
  <div className="glass-icon-button !w-8 !h-8">
  <Share2 className="w-4 h-4" />
  </div>
  <span className="text-sm font-medium text-white/90">{tr('shareLesson')}</span>
  </div>
  <ChevronDown className={cn(
  "w-4 h-4 text-white/50 transition-transform duration-200",
  shareExpanded && "rotate-180"
  )} />
  </button>
            
            {/* Expanded Share Options */}
            {shareExpanded && (
              <div className="px-3 pb-3 pt-1 border-t border-white/[0.06]">
                {/* Lesson Preview */}
                <div className="flex items-center gap-3 p-2 rounded-lg bg-white/[0.03] mb-3">
                  <img 
                    src="/kelly/archetypes/storyteller.png" 
                    alt="Kelly" 
                    className="w-10 h-10 rounded-lg object-cover"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-white truncate">{lessonTitle}</p>
                    <p className="text-[10px] text-white/40 truncate">{shareUrl}</p>
                  </div>
                </div>
                
                {/* Share Buttons - Using glass design system */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleShare('copy')}
                    className={cn(
                      "glass-button flex-1 !py-2 text-xs",
                      copied && "!bg-green-500/20 !border-green-500/30 !text-green-400"
                    )}
                  >
                    {copied ? <Check className="w-3.5 h-3.5" /> : <Link2 className="w-3.5 h-3.5" />}
                    {copied ? tr('copied') : tr('copyLink')}
                  </button>
                  <button
                    onClick={() => handleShare('twitter')}
                    className="glass-icon-button !w-9 !h-9"
                    title="Twitter"
                  >
                    <Twitter className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleShare('facebook')}
                    className="glass-icon-button !w-9 !h-9"
                    title="Facebook"
                  >
                    <Facebook className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleShare('whatsapp')}
                    className="glass-icon-button !w-9 !h-9"
                    title="WhatsApp"
                  >
                    <MessageCircle className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function RightRailButtons({
  isPlaying,
  isMuted,
  isLiked,
  isSaved,
  onPlayPause,
  onNext,
  onMuteToggle,
  onSettings,
  likeCount,
  commentCount,
  onLike,
  onComment,
  onSave,
  language,
  }: {
  isPlaying: boolean
  isMuted: boolean
  isLiked?: boolean
  isSaved?: boolean
  onPlayPause: () => void
  onNext: () => void
  onMuteToggle: () => void
  onSettings: () => void
  likeCount: number
  commentCount: number
  onLike: () => void
  onComment: () => void
  onSave: () => void
  language: string
  }) {
  return (
  <RightRail
  isPlaying={isPlaying}
  isMuted={isMuted}
  isLiked={isLiked}
  isSaved={isSaved}
  onPlayPause={onPlayPause}
  onNext={onNext}
  onMuteToggle={onMuteToggle}
  onSettings={onSettings}
  likeCount={likeCount}
  commentCount={commentCount}
  onLike={onLike}
  onComment={onComment}
  onSave={onSave}
  language={language}
  />
  )
  }

function BottomZoneContent({
  day,
  caption,
  onStartLearning,
  currentPhase,
  visualUrl,
  kellyAge,
  archetype,
  language,
  onAgeChange,
  onArchetypeChange,
  onLanguageChange,
  lessonTitle,
  onDayChange,
  onOpenCalendar,
}: {
  day: number
  caption: string
  onStartLearning: () => void
  currentPhase: LessonPhase
  visualUrl?: string
  kellyAge: number
  archetype: KellyArchetype
  language: string
  onAgeChange: (age: number) => void
  onArchetypeChange: (archetype: KellyArchetype) => void
  onLanguageChange: (lang: string) => void
  lessonTitle: string
  onDayChange: (delta: number) => void
  onOpenCalendar: () => void
}) {
  // BottomZone layout: Scrollable content area, then fixed day picker at bottom
  // This ensures day picker NEVER overlaps with content above
  return (
    <div className="flex flex-col w-full min-h-screen">
      {/* Scrollable content area - caption card, voting, controls, CTA */}
      <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden scrollbar-hide">
        <BottomBar
          day={day}
          dayNumber={day}
          lessonTitle={lessonTitle}
          caption={caption}
          onStartLearning={onStartLearning}
          currentPhase={currentPhase}
          visualUrl={visualUrl}
          kellyAge={kellyAge}
          archetype={archetype}
          language={language}
          onAgeChange={onAgeChange}
          onArchetypeChange={onArchetypeChange}
          onLanguageChange={onLanguageChange}
        />
      </div>
      
      {/* Day picker - always visible for navigation */}
      <div className="flex-shrink-0 pt-2">
        <FloatingDayPill
          dayOfYear={day}
          language={language}
          onDayChange={onDayChange}
          onOpenCalendar={onOpenCalendar}
          lessonTitle={lessonTitle}
        />
      </div>
    </div>
  )
}

// ============================================================================
// MOBILE CONTROLS DRAWER (shows on mobile when left rail is hidden)
// ============================================================================

function MobileControlsDrawer({
  isOpen,
  onClose,
  kellyAge,
  language,
  archetype,
  onAgeChange,
  onLanguageChange,
  onArchetypeChange,
}: {
  isOpen: boolean
  onClose: () => void
  kellyAge: number
  language: string
  archetype: KellyArchetype
  onAgeChange: (age: number) => void
  onLanguageChange: (lang: string) => void
  onArchetypeChange: (arch: KellyArchetype) => void
}) {
  const selectedLang = LANGUAGES.find((l) => l.code === language) || LANGUAGES[0]
  const selectedArch = ARCHETYPES.find((a) => a.id === archetype) || ARCHETYPES[0]
  
  if (!isOpen) return null
  
  return (
    <>
      {/* Backdrop - navy blue, NOT black */}
      <div 
        className="fixed inset-0 z-40 md:hidden"
        style={{ background: 'oklch(0.18 0.04 250 / 0.6)' }}
        onClick={onClose}
      />
      
      {/* Drawer */}
      <div className="fixed bottom-0 left-0 right-0 z-50 md:hidden rounded-t-3xl overflow-hidden">
        <div className="glass-panel p-6 max-h-[70vh] overflow-y-auto">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-white font-semibold text-lg">Customize Kelly</h3>
            <button onClick={onClose} className="text-white/60 hover:text-white">
              <X className="w-5 h-5" />
            </button>
          </div>
          
          {/* Age slider */}
          <div className="mb-6">
            <label className="text-white/60 text-sm mb-2 block">Kelly&apos;s Age: {kellyAge}</label>
            <input
              type="range"
              min={5}
              max={85}
              value={kellyAge}
              onChange={(e) => onAgeChange(Number(e.target.value))}
              className="w-full accent-[var(--kelly-blue)]"
            />
            <div className="flex justify-between text-xs text-white/40 mt-1">
              <span>Kid (5)</span>
              <span>Senior (85)</span>
            </div>
          </div>
          
          {/* Language */}
          <div className="mb-6">
            <label className="text-white/60 text-sm mb-2 block">Language</label>
            <div className="grid grid-cols-3 gap-2">
              {LANGUAGES.slice(0, 6).map((lang) => (
                <button
                  key={lang.code}
                  onClick={() => onLanguageChange(lang.code)}
                  className={cn(
                    'px-3 py-2 rounded-lg text-sm',
                    language === lang.code ? 'bg-[var(--kelly-blue)] text-white' : 'bg-white/10 text-white/80'
                  )}
                >
                  {lang.flag} {lang.label.split(' ')[0]}
                </button>
              ))}
            </div>
          </div>
          
          {/* Archetype */}
          <div>
            <label className="text-white/60 text-sm mb-2 block">Teaching Style</label>
            <div className="grid grid-cols-3 gap-2">
              {ARCHETYPES.slice(0, 6).map((arch) => (
                <button
                  key={arch.id}
                  onClick={() => onArchetypeChange(arch.id)}
                  className={cn(
                    'px-3 py-2 rounded-lg text-sm',
                    archetype === arch.id ? 'bg-[var(--kelly-blue)] text-white' : 'bg-white/10 text-white/80'
                  )}
                >
                  {arch.icon} {arch.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

// ============================================================================
// MAIN APP
// ============================================================================

function CuriousKellyAppInner() {
  const learner = useLearner()
  const { subscription, isPremium, isLifetime } = useSubscription(learner.email)
  const videoRef = useRef<HTMLVideoElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  
  // AFFILIATE TRACKING - Track ?ref= on landing, sets lifetime cookie
  useAffiliateTracking()
  
  // URL Query Param Handling for unified views
  // ?view=profile opens profile, ?view=settings opens settings, etc.
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search)
      const view = params.get('view')
      if (view === 'profile') setProfileOpen(true)
      else if (view === 'settings') setSettingsOpen(true)
      else if (view === 'pricing') setPricingOpen(true)
      else if (view === 'share') setShareOpen(true)
      else if (view === 'affiliate') setAffiliateOpen(true)
      else if (view === 'admin') setAdminOpen(true)
      else if (view === 'terms') setTermsOpen(true)
      else if (view === 'about') setAboutOpen(true)
      else if (view === 'help') setHelpOpen(true)
      else if (view === 'zig') setZigOpen(true)
      else if (view === 'studio') setStudioOpen(true)
    }
  }, [])
  
  // Preferences state
  const [kellyAge, setKellyAge] = useState(30)
  const [language, setLanguage] = useState('en')
  const [archetype, setArchetype] = useState<KellyArchetype>('storyteller')
  const [currentPhase, setCurrentPhase] = useState<LessonPhase>('hook')
  const [completedPhases, setCompletedPhases] = useState<LessonPhase[]>([])
  const [videoSource, setVideoSource] = useState<VideoSource>('auto') // A/B test lip-sync engines
  const [fineTuningOpen, setFineTuningOpen] = useState(false) // Video fine-tuning overlay
  
  // Playback state - START PLAYING for demo (muted for browser autoplay policy)
  const [isPlaying, setIsPlaying] = useState(true)
  const [isMuted, setIsMuted] = useState(true)
  const [hasInteracted, setHasInteracted] = useState(false)
  
  // UI state
  const [chatOpen, setChatOpen] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false) // Main navigation drawer
  
  const [calendarOpen, setCalendarOpen] = useState(false)
  
  // Overlay state - KellyOS Apple-quality modals (EVERY button opens something)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [pricingOpen, setPricingOpen] = useState(false)
  const [checkoutOpen, setCheckoutOpen] = useState(false)
  const [shareOpen, setShareOpen] = useState(false)
  const [commentsOpen, setCommentsOpen] = useState(false)
  const [dictionaryOpen, setDictionaryOpen] = useState(false)
  const [profileOpen, setProfileOpen] = useState(false)
  const [aboutOpen, setAboutOpen] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)
  const [subscriptionOpen, setSubscriptionOpen] = useState(false)
  const [affiliateOpen, setAffiliateOpen] = useState(false)
  const [adminOpen, setAdminOpen] = useState(false)
  const [termsOpen, setTermsOpen] = useState(false)
  const [studioOpen, setStudioOpen] = useState(false)
  const [zigOpen, setZigOpen] = useState(false)
  
  // Visual state removed - visuals are now integrated into caption bar
  const [selectedPlanId, setSelectedPlanId] = useState('')
  const [selectedBillingCycle, setSelectedBillingCycle] = useState<'monthly' | 'annual' | 'lifetime'>('annual')
  const [selectedWord, setSelectedWord] = useState('')
  
  // Engagement state - REAL likes from database
  const { count: likeCount, userLiked: isLiked, toggleLike } = useLikes(learner.dayOfYear)
  const { total: commentCount } = useComments(learner.dayOfYear, currentPhase)
  const [isSaved, setIsSaved] = useState(false)
  
  // Settings state
  const [autoplay, setAutoplay] = useState(true)
  const [notifications, setNotifications] = useState(true)
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('light')
  
  // Network state for offline handling
  const [isOnline, setIsOnline] = useState(true)
  
  useEffect(() => {
    setIsOnline(navigator.onLine)
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])
  
  // Visual Aid state
  // Visual is now integrated into caption bar, no separate state needed
  const [visualAidOpen, setVisualAidOpen] = useState(false)
  
// Sync preferences from learner context ONLY on initial load
  // Don't override user's manual language selection
  const [hasInitialized, setHasInitialized] = useState(false)
  useEffect(() => {
  if (!learner.isLoading && !hasInitialized) {
  setKellyAge(learner.preferredAge)
  setLanguage(learner.preferredLanguage)
  setArchetype(learner.preferredArchetype)
  setHasInitialized(true)
  }
  }, [learner.isLoading, learner.preferredAge, learner.preferredLanguage, learner.preferredArchetype, hasInitialized])
  
  // Lesson data
  const { lesson, perspective, kellyScripts, isLoading: lessonLoading } = useLesson(
    learner.dayOfYear,
    learner.timezone,
    { age: kellyAge, archetype, language }
  )
  
  // Live class countdown - uses legacy age categories (kid/adult/senior) for live_class_rooms table
  const liveClassAgeCategory = getLiveClassAgeCategory(kellyAge)
  const liveClass = useLiveClassCountdown(
    liveClassAgeCategory,
    learner.timezone,
    learner.locale
  )
  
  // Video URL for current phase/age/archetype - with content mode for visual feedback
  const { 
    videoUrl, 
    audioUrl, 
    thumbnailUrl, 
    status: videoStatus,
    contentMode,
    hasVideo,
    hasAudio,
  } = useVideoUrl(
    learner.dayOfYear,
    currentPhase,
    kellyAge,
    archetype,
    language,
    videoSource // A/B test: heygen, fal, sync, musetalk
  )
  
  // Lesson title - MUST be declared before useEffect that uses it
  // Show loading state, then actual title, fallback only if truly no data
  const lessonTitle = lessonLoading 
    ? 'Loading...' 
    : (perspective?.title || lesson?.title || "Today's Lesson")
  
  // Calculate visual URL for current phase from /public/visuals/[phase]/day-NNN.jpg
  const phaseVisualUrl = useMemo(() => {
    if (currentPhase === 'hook') {
      return `/visuals/hook/day-${String(learner.dayOfYear).padStart(3, '0')}.jpg`
    }
    if (currentPhase === 'wonder') {
      return `/visuals/wonder/day-${String(learner.dayOfYear).padStart(3, '0')}.jpg`
    }
    if (currentPhase === 'story') {
      return `/visuals/story/day-${String(learner.dayOfYear).padStart(3, '0')}.jpg`
    }
    if (currentPhase === 'action') {
      return `/visuals/action/day-${String(learner.dayOfYear).padStart(3, '0')}.jpg`
    }
    if (currentPhase === 'wisdom') {
      return `/visuals/wisdom/day-${String(learner.dayOfYear).padStart(3, '0')}.jpg`
    }
    return null
  }, [currentPhase, learner.dayOfYear])
  // Stop all playback helper
  const stopAllPlayback = useCallback(() => {
    if (videoRef.current) videoRef.current.pause()
    if (audioRef.current) audioRef.current.pause()
    setIsPlaying(false)
  }, [])

  // Video/Audio control handlers
  const handlePlayPause = useCallback(() => {
    if (isPlaying) {
      // PAUSE: Stop both (one may be fallback playing)
      stopAllPlayback()
      return
    }
    
    // PLAY: Try video first, fallback to audio
    if (videoRef.current && videoUrl) {
      videoRef.current.muted = false
      videoRef.current.play().catch(() => {
        // Video failed - try audio
        if (audioRef.current && audioUrl) {
          audioRef.current.muted = false
          audioRef.current.play()
        }
      })
      setIsPlaying(true)
      setIsMuted(false)
    } else if (audioRef.current && audioUrl) {
      audioRef.current.muted = false
      audioRef.current.play()
      setIsPlaying(true)
      setIsMuted(false)
    }
  }, [isPlaying, videoUrl, audioUrl, stopAllPlayback])
  
  const handleMuteToggle = useCallback(() => {
    const newMuted = !isMuted
    if (videoRef.current) videoRef.current.muted = newMuted
    if (audioRef.current) audioRef.current.muted = newMuted
    setIsMuted(newMuted)
  }, [isMuted])
  
  const handlePhaseChange = useCallback((phase: LessonPhase) => {
    // Stop current playback before switching
    stopAllPlayback()
    
    const phases: LessonPhase[] = ['hook', 'story', 'wonder', 'action', 'wisdom']
    const currentIndex = phases.indexOf(currentPhase)
    const newIndex = phases.indexOf(phase)
    
    if (newIndex > currentIndex && !completedPhases.includes(currentPhase)) {
      setCompletedPhases((prev) => [...prev, currentPhase])
    }
    setCurrentPhase(phase)
  }, [currentPhase, completedPhases, stopAllPlayback])
  
  const handleNextPhase = useCallback(() => {
    const phases: LessonPhase[] = ['hook', 'story', 'wonder', 'action', 'wisdom']
    const currentIndex = phases.indexOf(currentPhase)
    if (currentIndex < phases.length - 1) {
      handlePhaseChange(phases[currentIndex + 1])
    }
  }, [currentPhase, handlePhaseChange])
  
// Comments and presence are now fetched via hooks in LeftRail component
  // No hardcoded data - everything is real from the database
  
  // Kelly poster from kelly-assets library (archetype + age aware)
  // Uses REACTIVE state (kellyAge, archetype) so it updates when user changes controls
  const ageCategory = getAgeCategory(kellyAge)
  const posterUrl = getKellyArchetypeImage(archetype, ageCategory) || '/kelly/archetypes/storyteller.png'
  
  // Show loading state - Premium shimmer skeleton with Kelly visible
  if (learner.isLoading) {
    return (
      <div className="fixed inset-0 overflow-hidden" style={{ background: '#FFFFFF' }}>
        {/* Show Kelly image even during loading - immediate visual feedback */}
        <div className="center-stage">
          <div className="kelly-container blur-in">
            <img 
              src={posterUrl || "/placeholder.svg"} 
              alt="Curious Kelly - Lesson of the Day"
              className="opacity-90"
            />
          </div>
        </div>
        
        {/* Premium loading overlay - minimal, elegant */}
        <div className="fixed inset-0 flex items-end justify-center pb-32 z-50 pointer-events-none">
          <div className="text-center blur-in">
            {/* Premium spinner */}
            <div className="relative w-10 h-10 mx-auto mb-4">
              <div className="spinner-premium-light w-10 h-10" />
              <div className="absolute inset-0 flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-[var(--kelly-blue)] icon-pop" />
              </div>
            </div>
            
            {/* Shimmer text placeholder */}
            <div className="space-y-2">
              <div className="h-4 w-32 mx-auto rounded-full shimmer-light" />
              <div className="h-3 w-24 mx-auto rounded-full shimmer-light" style={{ animationDelay: '0.1s' }} />
            </div>
          </div>
        </div>
        
        {/* Skeleton UI hints - show layout structure */}
        <div className="fixed bottom-8 left-1/2 -translate-x-1/2 w-full max-w-md px-4 z-40 pointer-events-none">
          <div className="glass-card p-4 opacity-40">
            <div className="h-3 w-3/4 rounded-full shimmer mb-3" />
            <div className="h-3 w-1/2 rounded-full shimmer" style={{ animationDelay: '0.15s' }} />
          </div>
        </div>
      </div>
    )
  }
  
  // Single unified view - Full-screen immersive SPA experience
  // Wrapped in KellyOSLayer for magical water droplet UI and draggable time orb
  // Header is rendered inside KellyOSLayer with old money aesthetic
  return (
    <>
      <KellyOSLayer
        formattedDate={learner.formattedDate}
        dayOfYear={learner.dayOfYear}
        lessonTitle={lessonTitle}
        currentPhase={currentPhase}
        onDateTap={() => setCalendarOpen(true)}
        onPrevDay={() => learner.setDayOverride?.(learner.dayOfYear - 1)}
        onNextDay={() => learner.setDayOverride?.(learner.dayOfYear + 1)}
      >
      <TikTokFeed
        lessonTitle={lessonTitle}
        caption={getPhaseScript(lesson, currentPhase) || ''}
        hookFact={lesson?.hook_fact || null}
        hookText={lesson?.hook_question || null}
        explanation={lesson?.teaching_moment || null}
        hookCorrectAnswer={lesson?.hook_correct_answer ?? null}
        quote={lesson?.quote || null}
        quoteAuthor={lesson?.quote_author || null}
        visualUrl={
          // Use phase-specific visuals from /public/visuals/[phase]/
          // Day 30 hook: /visuals/hook/day-030.jpg (exists for 1-90)
          // Day 30 wonder: /visuals/wonder/day-030.jpg (exists for 1-50)
          // Other phases fallback to hook visual
          phaseVisualUrl || 
          (learner.dayOfYear <= 90 
            ? `/visuals/hook/day-${learner.dayOfYear.toString().padStart(3, '0')}.jpg`
            : `/visuals/hook/day-001.jpg`)
        }
        videoUrl={videoUrl || undefined}
        audioUrl={audioUrl || undefined}
        posterUrl={(thumbnailUrl && thumbnailUrl.startsWith('/kelly/')) ? thumbnailUrl : posterUrl}
        currentPhase={currentPhase}
        completedPhases={completedPhases}
        dayOfYear={learner.dayOfYear}
        formattedDate={learner.formattedDate}
        likeCount={likeCount}
        isLiked={isLiked}
        commentCount={commentCount || 0}
        isSaved={isSaved}
        onPhaseChange={handlePhaseChange}
        onLike={toggleLike}
        onComment={() => setCommentsOpen(true)}
        onShare={() => setShareOpen(true)}
        onSave={() => setIsSaved(!isSaved)}
        onSettings={() => setSettingsOpen(true)}
        onProfile={() => setProfileOpen(true)}
        kellyAge={kellyAge}
        archetype={archetype}
        language={language}
        onAgeChange={setKellyAge}
        onArchetypeChange={setArchetype}
        onLanguageChange={setLanguage}
        onMenu={() => setMenuOpen(true)}
        onDictionary={() => setDictionaryOpen(true)}
        onCalendar={() => setCalendarOpen(true)}
        onPrevDay={() => learner.dayOfYear > 1 && learner.setDayOverride?.(learner.dayOfYear - 1)}
        onNextDay={() => learner.dayOfYear < 365 && learner.setDayOverride?.(learner.dayOfYear + 1)}
        // Word-by-word Kelly scripts per variant
        // Fallback chain: kellyScripts (most specific per age/archetype/lang) -> perspective -> base lesson
        hookScript={kellyScripts?.hook || (perspective as Record<string, unknown>)?.hook_script as string || lesson?.hook_script || null}
        storyScript={kellyScripts?.story || (perspective as Record<string, unknown>)?.story_script as string || lesson?.story_script || null}
        wonderScript={kellyScripts?.wonder || (perspective as Record<string, unknown>)?.wonder_script as string || lesson?.wonder_script || null}
        actionScript={kellyScripts?.action || (perspective as Record<string, unknown>)?.action_script as string || lesson?.action_script || null}
        wisdomScript={kellyScripts?.wisdom || (perspective as Record<string, unknown>)?.wisdom_script as string || lesson?.wisdom_script || null}
      />
      
      {/* ========== LAYER 2: UI CHROME (Overlays) ========== */}
      {/* Menu and Ziggurat buttons now integrated into TikTokFeed right rail */}
      
      {/* Navigation Drawer - ALL navigation goes through here */}
      <NavigationDrawer
        isOpen={menuOpen}
        onClose={() => setMenuOpen(false)}
        onNavigate={(view) => {
          // Handle navigation to each view
          if (view === 'home') { /* already home */ }
          else if (view === 'calendar') setCalendarOpen(true)
          else if (view === 'dictionary') setDictionaryOpen(true)
          else if (view === 'search') setSearchOpen(true)
          else if (view === 'chat') setChatOpen(true)
          else if (view === 'profile') setProfileOpen(true)
          else if (view === 'settings') setSettingsOpen(true)
          else if (view === 'subscription') setSubscriptionOpen(true)
          else if (view === 'pricing') setPricingOpen(true)
          else if (view === 'affiliate') setAffiliateOpen(true)
          else if (view === 'about') setAboutOpen(true)
          else if (view === 'help') setHelpOpen(true)
          else if (view === 'terms') setTermsOpen(true)
          else if (view === 'studio') setStudioOpen(true)
          else if (view === 'admin') setAdminOpen(true)
          else if (view === 'strategic') window.location.href = '/strategic'
          else if (view === 'investors') window.location.href = '/ziggurat/investors'
          else if (view === 'share') setShareOpen(true)
          else if (view === 'comments') setCommentsOpen(true)
        }}
        currentDay={learner.dayOfYear}
        userName={learner.name || 'Explorer'}
        isPremium={isPremium}
      />
      
      {/* Upgrade Banner for free users */}
      {!isPremium && !menuOpen && (
        <SubscriptionBanner onUpgrade={() => setPricingOpen(true)} />
      )}
      
      {/* ========== UNIFIED OVERLAY SYSTEM ========== */}
      
      {/* Settings Overlay */}
      <SettingsOverlay
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        kellyAge={kellyAge}
        language={language}
        archetype={archetype}
        autoplay={autoplay}
        notifications={notifications}
        theme={theme}
        onAgeChange={setKellyAge}
        onLanguageChange={setLanguage}
        onArchetypeChange={setArchetype}
        onAutoplayToggle={setAutoplay}
        onNotificationsToggle={setNotifications}
        onThemeChange={setTheme}
        onSignOut={() => {}}
      />
      
      {/* Share Overlay */}
      <ShareOverlay
        isOpen={shareOpen}
        onClose={() => setShareOpen(false)}
        lessonTitle={lessonTitle}
        lessonDay={learner.dayOfYear}
      />
      
      {/* Comments Overlay */}
      <CommentsOverlay
        isOpen={commentsOpen}
        onClose={() => setCommentsOpen(false)}
        day={learner.dayOfYear}
        phase={currentPhase}
      />
      
      {/* Profile Overlay */}
      <ProfileOverlay
        isOpen={profileOpen}
        onClose={() => setProfileOpen(false)}
        onUpgrade={() => { setProfileOpen(false); setPricingOpen(true); }}
      />
      
      {/* Pricing Overlay */}
      <PricingOverlay
        isOpen={pricingOpen}
        onClose={() => setPricingOpen(false)}
        currentPlan={subscription?.tier || 'free'}
        onSelectPlan={async (planId, cycle) => {
          if (learner.email) {
            setPricingOpen(false)
            try {
              await startCheckout(planId, cycle, learner.email)
            } catch (e) {
              console.error('[Checkout] Failed:', e)
              // Fallback to in-app checkout overlay
              setSelectedPlanId(planId)
              setSelectedBillingCycle(cycle)
              setCheckoutOpen(true)
            }
          } else {
            // No email - use in-app checkout overlay which collects email
            setSelectedPlanId(planId)
            setSelectedBillingCycle(cycle)
            setPricingOpen(false)
            setCheckoutOpen(true)
          }
        }}
      />
      
      {/* Checkout Overlay */}
      <CheckoutOverlay
        isOpen={checkoutOpen}
        onClose={() => setCheckoutOpen(false)}
        planId={selectedPlanId}
        billingCycle={selectedBillingCycle}
        onComplete={() => setCheckoutOpen(false)}
      />
      
      {/* Dictionary Overlay */}
      <DictionaryOverlay
        isOpen={dictionaryOpen}
        onClose={() => setDictionaryOpen(false)}
        initialWord={selectedWord}
        language={language}
      />
      
      {/* About Overlay */}
      <AboutOverlay
        isOpen={aboutOpen}
        onClose={() => setAboutOpen(false)}
      />
      
      {/* Help Overlay */}
      <HelpOverlay
        isOpen={helpOpen}
        onClose={() => setHelpOpen(false)}
      />
      
      {/* Subscription Overlay */}
      <SubscriptionOverlay
        isOpen={subscriptionOpen}
        onClose={() => setSubscriptionOpen(false)}
        currentPlan={
          subscription?.plan === 'lifetime' ? 'lifetime' :
          subscription?.tier === 'family' ? 'family' :
          subscription?.tier === 'individual' ? 'learner' : 'free'
        }
        billingCycle={subscription?.plan === 'annual' ? 'annual' : 'monthly'}
        nextBillingDate={subscription?.periodEnd ? new Date(subscription.periodEnd).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : undefined}
        onUpgrade={() => {
          setSubscriptionOpen(false)
          setPricingOpen(true)
        }}
        onManageBilling={() => {
          if (learner.email) {
            openCustomerPortal(learner.email).catch(e => console.error('[Portal] Failed:', e))
          }
        }}
        onCancelSubscription={() => {
          if (learner.email) {
            openCustomerPortal(learner.email).catch(e => console.error('[Portal] Failed:', e))
          }
        }}
      />
      
      {/* Affiliate Overlay */}
      <AffiliateOverlay
        isOpen={affiliateOpen}
        onClose={() => setAffiliateOpen(false)}
      />
      
      {/* Admin Overlay */}
      <AdminOverlay
        isOpen={adminOpen}
        onClose={() => setAdminOpen(false)}
      />
      
      {/* Terms Overlay */}
      <TermsOverlay
        isOpen={termsOpen}
        onClose={() => setTermsOpen(false)}
      />
      
      {/* Studio Overlay */}
      <StudioOverlay
        isOpen={studioOpen}
        onClose={() => setStudioOpen(false)}
      />
      
      {/* Lesson Calendar */}
      <LessonCalendar
        open={calendarOpen}
        onOpenChange={setCalendarOpen}
        selectedDay={learner.dayOfYear}
        onDaySelect={(day) => learner.setDayOverride(day)}
      />
      
      {/* Search Overlay */}
      <SearchOverlay
        isOpen={searchOpen}
        onClose={() => setSearchOpen(false)}
        onSelectLesson={(day) => {
          learner.setDayOverride(day)
          setSearchOpen(false)
        }}
      />
      
      {/* Chat Overlay - Ask Kelly AI */}
      <ChatOverlay
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
        lessonContext={lessonTitle}
      />
      </KellyOSLayer>
    </>
  )
}

// Wrapper to add Suspense boundary for useSearchParams
function CuriousKellyApp() {
  return (
    <Suspense fallback={
      <div className="fixed inset-0 flex items-center justify-center" style={{ background: '#FFFFFF' }}>
        <div className="text-center">
          <div className="w-12 h-12 border-3 border-[var(--kelly-blue)] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-gray-500 text-sm">Loading...</p>
        </div>
      </div>
    }>
      <CuriousKellyAppInner />
    </Suspense>
  )
}

// ============================================================================
// PAGE EXPORT - v6.1 Clean 2-Layer Architecture
// ============================================================================

// Maintenance Mode Component - Premium pre-launch experience with real Kelly assets
function MaintenancePage() {
  const [email, setEmail] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [hoveredAge, setHoveredAge] = useState<'kid' | 'adult' | 'senior' | null>(null)
  const [hoveredArchetype, setHoveredArchetype] = useState<string | null>(null)
  
  // Video showcase state
  const [isVideoPlaying, setIsVideoPlaying] = useState(false)
  const [selectedVideoDay, setSelectedVideoDay] = useState(35)
  const [selectedVideoPhase, setSelectedVideoPhase] = useState('hook')
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [videoLoading, setVideoLoading] = useState(false)
  const heroVideoRef = useRef<HTMLVideoElement>(null)
  
  // Featured video data - 513 HeyGen videos available!
  const featuredVideos = [
    { day: 35, phase: 'hook', title: 'How Electricity Flows', archetype: 'scientist' },
    { day: 34, phase: 'hook', title: 'How Magnets Work', archetype: 'explorer' },
    { day: 33, phase: 'story', title: 'The Water Cycle', archetype: 'storyteller' },
    { day: 32, phase: 'wonder', title: 'Why the Sky is Blue', archetype: 'mystic' },
  ]
  
  // Fetch video URL on mount and when selection changes
  useEffect(() => {
    const fetchVideo = async () => {
      setVideoLoading(true)
      try {
        const res = await fetch(`/api/video/url?day=${selectedVideoDay}&phase=${selectedVideoPhase}&age=30&archetype=storyteller`)
        const data = await res.json()
        if (data.url) {
          setVideoUrl(data.url)
        }
      } catch (e) {
        console.warn('Video fetch error:', e)
      } finally {
        setVideoLoading(false)
      }
    }
    fetchVideo()
  }, [selectedVideoDay, selectedVideoPhase])

  const handleWaitlist = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!email) return
    
    setIsSubmitting(true)
    try {
      const res = await fetch('/api/waitlist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      })
      if (res.ok) {
        setSubmitted(true)
      }
    } catch (error) {
      console.error('Waitlist error:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  // Kelly age showcase data
  const kellyAges = [
    { id: 'kid' as const, label: 'Sprouts', age: '4-7', description: 'Playful discovery for young minds' },
    { id: 'adult' as const, label: 'Explorers', age: '8-17', description: 'Deep dives for curious learners' },
    { id: 'senior' as const, label: 'Lifelong', age: '18+', description: 'Wisdom for every stage of life' },
  ]

  // Featured archetypes to showcase
  const featuredArchetypes = [
    { id: 'explorer', label: 'Explorer', tagline: 'Ready for an adventure?' },
    { id: 'storyteller', label: 'Storyteller', tagline: 'Let me tell you a story...' },
    { id: 'scientist', label: 'Scientist', tagline: 'Let\'s experiment!' },
    { id: 'mystic', label: 'Mystic', tagline: 'There\'s magic in everything.' },
    { id: 'rebel', label: 'Rebel', tagline: 'Let\'s break some rules!' },
    { id: 'architect', label: 'Architect', tagline: 'Let\'s build something together.' },
  ]

  return (
    <div className="min-h-screen bg-[#f8f6f3] text-[#1a1715] overflow-hidden">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 px-6 py-5 bg-[#f8f6f3]/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img 
              src="/kelly/logo-512.webp" 
              alt="Curious Kelly" 
              className="w-10 h-10 rounded-full object-cover"
            />
            <span className="text-sm font-medium tracking-tight">Lesson of the Day</span>
          </div>
          <div className="hidden sm:flex items-center gap-8 text-sm">
            <a href="/about" className="text-[#1a1715]/60 hover:text-[#1a1715] transition-colors">About</a>
            <a href="/curriculum" className="text-[#1a1715]/60 hover:text-[#1a1715] transition-colors">Curriculum</a>
            <a href="/investors" className="text-[#1a1715]/60 hover:text-[#1a1715] transition-colors">Investors</a>
          </div>
          <a 
            href="/investors" 
            className="px-4 py-2 bg-[#1a1715] text-[#f8f6f3] text-sm font-medium rounded-full hover:bg-[#2d2a27] transition-colors"
          >
            Request Access
          </a>
        </div>
      </nav>

      {/* Hero Section - Split layout with Kelly */}
      <section className="relative min-h-screen flex items-center px-6 pt-24 pb-12">
        <div className="max-w-7xl mx-auto w-full">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-8 items-center">
            {/* Left: Content */}
            <div className="space-y-8 text-center lg:text-left">
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#1a1715]/5 border border-[#1a1715]/10">
                <span className="w-2 h-2 rounded-full bg-[#c4956a] animate-pulse" />
                <span className="text-xs font-medium text-[#1a1715]/70 uppercase tracking-widest">Private Alpha</span>
              </div>

              {/* Main headline */}
              <h1 className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-serif font-normal tracking-tight leading-[1.05] text-balance">
                Twenty years of learning,
                <br />
                <span className="italic">one lesson at a time</span>
              </h1>

              {/* Subheadline */}
              <p className="text-lg text-[#1a1715]/60 max-w-xl mx-auto lg:mx-0 leading-relaxed text-pretty">
                Meet Curious Kelly, your AI-powered guide through 365 days of wonder, discovery, and growth. Built on two decades of curriculum research.
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row items-center lg:items-start justify-center lg:justify-start gap-4 pt-2">
                <a 
                  href="/investors" 
                  className="group inline-flex items-center gap-3 px-8 py-4 bg-[#1a1715] text-[#f8f6f3] text-base font-medium rounded-full hover:bg-[#2d2a27] transition-all hover:scale-[1.02]"
                >
                  Request Early Access
                  <span className="w-6 h-6 rounded-full bg-[#f8f6f3]/20 flex items-center justify-center group-hover:bg-[#f8f6f3]/30 transition-colors">
                    <ChevronRight className="w-3.5 h-3.5" />
                  </span>
                </a>
                <a 
                  href="/about/curriculum" 
                  className="inline-flex items-center gap-2 px-6 py-4 text-[#1a1715]/70 text-base font-medium hover:text-[#1a1715] transition-colors"
                >
                  Explore Curriculum
                  <ChevronRight className="w-4 h-4" />
                </a>
              </div>
            </div>

            {/* Right: Kelly Image with age switcher */}
            <div className="relative flex flex-col items-center">
              {/* Kelly Image - Interactive */}
              <div className="relative w-full max-w-md aspect-square">
                {/* Glow effect */}
                <div className="absolute inset-0 bg-gradient-to-b from-[#b4d5e8]/30 via-transparent to-[#e8d5c4]/30 rounded-full blur-3xl scale-110" />
                
                {/* Kelly portrait */}
                <img 
                  src={hoveredAge 
                    ? `/kelly/${hoveredAge}/explorer.png`
                    : hoveredArchetype
                      ? `/kelly/archetypes/${hoveredArchetype}.png`
                      : '/kelly/archetypes/explorer.png'
                  }
                  alt="Curious Kelly - Your AI Learning Guide"
                  className="relative w-full h-full object-contain transition-all duration-300 drop-shadow-2xl"
                />
              </div>

              {/* Age selector pills */}
              <div className="flex items-center gap-2 mt-6">
                {kellyAges.map((age) => (
                  <button
                    key={age.id}
                    onMouseEnter={() => setHoveredAge(age.id)}
                    onMouseLeave={() => setHoveredAge(null)}
                    className={cn(
                      "px-4 py-2 rounded-full text-sm font-medium transition-all",
                      hoveredAge === age.id
                        ? "bg-[#1a1715] text-[#f8f6f3]"
                        : "bg-[#1a1715]/5 text-[#1a1715]/70 hover:bg-[#1a1715]/10"
                    )}
                  >
                    {age.label}
                  </button>
                ))}
              </div>
              <p className="text-sm text-[#1a1715]/40 mt-2">Hover to see Kelly at different ages</p>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-[#1a1715]/30">
          <span className="text-xs uppercase tracking-widest">Scroll</span>
          <div className="w-px h-8 bg-gradient-to-b from-[#1a1715]/30 to-transparent" />
        </div>
      </section>

      {/* Meet Kelly's Archetypes Section */}
      <section className="py-24 px-6 bg-white">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 space-y-4">
            <span className="text-xs uppercase tracking-widest text-[#1a1715]/50">12 Teaching Styles</span>
            <h2 className="text-4xl sm:text-5xl font-serif font-normal">
              Kelly adapts to <span className="italic">how you learn</span>
            </h2>
            <p className="text-lg text-[#1a1715]/50 max-w-2xl mx-auto">
              Based on Jungian archetypes, Kelly transforms her teaching style to match your personality and learning preferences.
            </p>
          </div>

          {/* Archetype grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            {featuredArchetypes.map((archetype) => (
              <div
                key={archetype.id}
                onMouseEnter={() => { setHoveredArchetype(archetype.id); setHoveredAge(null); }}
                onMouseLeave={() => setHoveredArchetype(null)}
                className="group relative bg-[#f8f6f3] rounded-2xl p-4 text-center cursor-pointer hover:bg-[#1a1715] transition-all duration-300"
              >
                <div className="aspect-square mb-3 rounded-xl overflow-hidden bg-white">
                  <img 
                    src={`/kelly/archetypes/${archetype.id}.png`}
                    alt={`Kelly as ${archetype.label}`}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                </div>
                <p className="font-medium text-sm group-hover:text-[#f8f6f3] transition-colors">{archetype.label}</p>
                <p className="text-xs text-[#1a1715]/50 group-hover:text-[#f8f6f3]/70 mt-1 transition-colors line-clamp-1">{archetype.tagline}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* VIDEO SHOWCASE SECTION - Watch Kelly Teach */}
      <section className="py-24 px-6 bg-[#1a1715] text-[#f8f6f3]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12 space-y-4">
            <span className="text-xs uppercase tracking-widest text-[#f8f6f3]/50">See Kelly in Action</span>
            <h2 className="text-4xl sm:text-5xl font-serif font-normal">
              Watch <span className="italic">Kelly teach</span>
            </h2>
            <p className="text-lg text-[#f8f6f3]/50 max-w-2xl mx-auto">
              AI-generated video lessons powered by HeyGen. 513 videos across 81 days, with more being created daily.
            </p>
          </div>

          {/* Main Video Player */}
          <div className="relative max-w-4xl mx-auto mb-12">
            <div className="relative aspect-video bg-black rounded-2xl overflow-hidden shadow-2xl">
              {videoLoading ? (
                <div className="absolute inset-0 flex items-center justify-center bg-black">
                  <Loader2 className="w-12 h-12 text-white/50 animate-spin" />
                </div>
              ) : videoUrl ? (
                <>
                  <video
                    ref={heroVideoRef}
                    src={videoUrl}
                    poster="/kelly/archetypes/storyteller.png"
                    className="w-full h-full object-contain"
                    loop
                    playsInline
                    onClick={() => {
                      if (heroVideoRef.current) {
                        if (isVideoPlaying) {
                          heroVideoRef.current.pause()
                        } else {
                          heroVideoRef.current.play()
                        }
                        setIsVideoPlaying(!isVideoPlaying)
                      }
                    }}
                    onPlay={() => setIsVideoPlaying(true)}
                    onPause={() => setIsVideoPlaying(false)}
                  />
                  
                  {/* Play/Pause overlay */}
                  {!isVideoPlaying && (
                    <button
                      onClick={() => {
                        if (heroVideoRef.current) {
                          heroVideoRef.current.play()
                          setIsVideoPlaying(true)
                        }
                      }}
                      className="absolute inset-0 flex items-center justify-center bg-black/30 hover:bg-black/20 transition-colors group"
                    >
                      <div className="w-20 h-20 rounded-full bg-white/90 flex items-center justify-center group-hover:scale-110 transition-transform shadow-lg">
                        <Play className="w-8 h-8 text-[#1a1715] ml-1" />
                      </div>
                    </button>
                  )}
                  
                  {/* Video info badge */}
                  <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-2 rounded-full bg-black/50 backdrop-blur-sm">
                    <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                    <span className="text-xs text-white font-medium">AI Video</span>
                  </div>
                </>
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-[#2d2a27] to-[#1a1715]">
                  <img 
                    src="/kelly/archetypes/storyteller.png" 
                    alt="Kelly" 
                    className="w-48 h-48 object-contain opacity-80"
                  />
                  <p className="text-white/50 mt-4">Loading video...</p>
                </div>
              )}
            </div>
            
            {/* Video title */}
            <div className="text-center mt-4">
              <p className="text-lg font-medium text-[#f8f6f3]">
                Day {selectedVideoDay}: {featuredVideos.find(v => v.day === selectedVideoDay)?.title || 'Daily Lesson'}
              </p>
              <p className="text-sm text-[#f8f6f3]/50 mt-1">
                Phase: {selectedVideoPhase.charAt(0).toUpperCase() + selectedVideoPhase.slice(1)}
              </p>
            </div>
          </div>

          {/* Video selector - Featured lessons */}
          <div className="max-w-3xl mx-auto">
            <p className="text-xs uppercase tracking-widest text-[#f8f6f3]/40 mb-4 text-center">Featured Lessons</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {featuredVideos.map((video) => (
                <button
                  key={`${video.day}-${video.phase}`}
                  onClick={() => {
                    setSelectedVideoDay(video.day)
                    setSelectedVideoPhase(video.phase)
                    setIsVideoPlaying(false)
                  }}
                  className={cn(
                    "p-4 rounded-xl text-left transition-all",
                    selectedVideoDay === video.day && selectedVideoPhase === video.phase
                      ? "bg-[#c4956a] text-[#1a1715]"
                      : "bg-[#f8f6f3]/5 hover:bg-[#f8f6f3]/10 text-[#f8f6f3]"
                  )}
                >
                  <p className="text-xs opacity-60 mb-1">Day {video.day}</p>
                  <p className="text-sm font-medium line-clamp-2">{video.title}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Stats */}
          <div className="flex items-center justify-center gap-8 mt-12 pt-8 border-t border-[#f8f6f3]/10">
            <div className="text-center">
              <p className="text-3xl font-serif text-[#c4956a]">513</p>
              <p className="text-xs text-[#f8f6f3]/40 uppercase tracking-wider mt-1">Videos Ready</p>
            </div>
            <div className="w-px h-12 bg-[#f8f6f3]/10" />
            <div className="text-center">
              <p className="text-3xl font-serif text-[#c4956a]">81</p>
              <p className="text-xs text-[#f8f6f3]/40 uppercase tracking-wider mt-1">Days Covered</p>
            </div>
            <div className="w-px h-12 bg-[#f8f6f3]/10" />
            <div className="text-center">
              <p className="text-3xl font-serif text-[#c4956a]">9,110</p>
              <p className="text-xs text-[#f8f6f3]/40 uppercase tracking-wider mt-1">Audio Assets</p>
            </div>
          </div>
        </div>
      </section>

      {/* Ziggurat Campus Section - Full-bleed hero image */}
      <section className="relative">
        <div className="relative h-[60vh] lg:h-[80vh] overflow-hidden">
          <img 
            src="/ziggurat/heroes/kelly-blue-day.jpg" 
            alt="The Ziggurat - Future Home of Curious Kelly"
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#1a1715] via-[#1a1715]/20 to-transparent" />
          
          {/* Overlay content */}
          <div className="absolute bottom-0 left-0 right-0 p-8 lg:p-16">
            <div className="max-w-7xl mx-auto">
              <span className="text-xs uppercase tracking-widest text-[#f8f6f3]/50">Our Future Home</span>
              <h2 className="text-3xl sm:text-4xl lg:text-5xl font-serif font-normal text-[#f8f6f3] mt-2 max-w-2xl">
                The Ziggurat
              </h2>
              <p className="text-lg text-[#f8f6f3]/70 mt-4 max-w-xl">
                A landmark campus in West Covina, California - transforming a brutalist icon into a living, breathing center for lifelong learning.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section - Editorial Grid */}
      <section className="relative py-32 px-6 bg-[#1a1715] text-[#f8f6f3]">
        <div className="max-w-7xl mx-auto">
          {/* Section header */}
          <div className="flex items-center gap-4 mb-16">
            <div className="w-12 h-px bg-[#f8f6f3]/20" />
            <span className="text-xs uppercase tracking-widest text-[#f8f6f3]/50">The Platform</span>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-px bg-[#f8f6f3]/10">
            <div className="bg-[#1a1715] p-8 lg:p-12">
              <p className="text-5xl sm:text-6xl lg:text-7xl font-serif font-light">365</p>
              <p className="text-sm text-[#f8f6f3]/50 mt-4 uppercase tracking-wider">Daily Lessons</p>
            </div>
            <div className="bg-[#1a1715] p-8 lg:p-12">
              <p className="text-5xl sm:text-6xl lg:text-7xl font-serif font-light">3</p>
              <p className="text-sm text-[#f8f6f3]/50 mt-4 uppercase tracking-wider">Age Groups</p>
            </div>
            <div className="bg-[#1a1715] p-8 lg:p-12">
              <p className="text-5xl sm:text-6xl lg:text-7xl font-serif font-light">12</p>
              <p className="text-sm text-[#f8f6f3]/50 mt-4 uppercase tracking-wider">Teaching Archetypes</p>
            </div>
            <div className="bg-[#1a1715] p-8 lg:p-12">
              <p className="text-5xl sm:text-6xl lg:text-7xl font-serif font-light">20</p>
              <p className="text-sm text-[#f8f6f3]/50 mt-4 uppercase tracking-wider">Years of Research</p>
            </div>
          </div>
        </div>
      </section>

      {/* Philosophy Section - With real Kelly images */}
      <section className="py-32 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 lg:gap-24 items-center">
            {/* Left: Kelly Showcase - 3 ages */}
            <div className="relative">
              <div className="grid grid-cols-3 gap-4">
                {/* Kid Kelly */}
                <div className="relative aspect-[3/4] rounded-2xl overflow-hidden bg-[#e8f4f8]">
                  <img 
                    src="/kelly/kid/storyteller.png"
                    alt="Kid Kelly - Ages 4-7"
                    className="w-full h-full object-cover object-top"
                  />
                  <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/60 to-transparent">
                    <p className="text-white text-xs font-medium">Ages 4-7</p>
                  </div>
                </div>
                
                {/* Adult Kelly - Featured */}
                <div className="relative aspect-[3/4] rounded-2xl overflow-hidden bg-[#e8f4f8] -mt-8 shadow-2xl z-10">
                  <img 
                    src="/kelly/archetypes/storyteller.png"
                    alt="Adult Kelly - Ages 8-17"
                    className="w-full h-full object-cover object-top"
                  />
                  <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/60 to-transparent">
                    <p className="text-white text-xs font-medium">Ages 8-17</p>
                  </div>
                </div>
                
                {/* Senior Kelly */}
                <div className="relative aspect-[3/4] rounded-2xl overflow-hidden bg-[#f8f0e8]">
                  <img 
                    src="/kelly/senior/storyteller.png"
                    alt="Senior Kelly - Ages 18+"
                    className="w-full h-full object-cover object-top"
                  />
                  <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/60 to-transparent">
                    <p className="text-white text-xs font-medium">Ages 18+</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Content */}
            <div className="space-y-8">
              <div className="flex items-center gap-4">
                <div className="w-12 h-px bg-[#1a1715]/20" />
                <span className="text-xs uppercase tracking-widest text-[#1a1715]/50">Our Philosophy</span>
              </div>
              
              <h2 className="text-4xl sm:text-5xl font-serif font-normal leading-tight text-balance">
                Learning is not a destination.
                <br />
                <span className="italic">It&apos;s a daily practice.</span>
              </h2>
              
              <p className="text-lg text-[#1a1715]/60 leading-relaxed">
                Since 2006, we&apos;ve been developing a curriculum that meets learners where they are. Kelly adapts her age, personality, and teaching style to create a personalized experience for everyone.
              </p>
              
              <p className="text-lg text-[#1a1715]/60 leading-relaxed">
                Each lesson is crafted to spark curiosity, build on previous knowledge, and create lasting understanding. This is education designed for how humans actually learn.
              </p>

              <div className="pt-4">
                <a 
                  href="/about" 
                  className="inline-flex items-center gap-2 text-[#1a1715] font-medium hover:gap-3 transition-all"
                >
                  Read our story
                  <ChevronRight className="w-4 h-4" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid - With Kelly icons */}
      <section className="py-32 px-6 bg-[#f0ebe5]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 space-y-4">
            <span className="text-xs uppercase tracking-widest text-[#1a1715]/50">What We&apos;re Building</span>
            <h2 className="text-4xl sm:text-5xl font-serif font-normal">
              Education, reimagined
            </h2>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Feature 1 - Daily Lessons */}
            <div className="bg-[#f8f6f3] p-8 rounded-2xl space-y-4 group hover:bg-[#1a1715] transition-all duration-300">
              <div className="w-12 h-12 rounded-xl bg-[#1a1715]/5 group-hover:bg-[#f8f6f3]/10 flex items-center justify-center overflow-hidden transition-colors">
                <Calendar className="w-5 h-5 text-[#1a1715]/70 group-hover:text-[#f8f6f3]/90 transition-colors" />
              </div>
              <h3 className="text-xl font-medium group-hover:text-[#f8f6f3] transition-colors">One Lesson Daily</h3>
              <p className="text-[#1a1715]/60 group-hover:text-[#f8f6f3]/70 leading-relaxed transition-colors">
                Consistent, bite-sized learning that builds over time. Never overwhelming, always engaging.
              </p>
            </div>

            {/* Feature 2 - Age Adaptive */}
            <div className="bg-[#f8f6f3] p-8 rounded-2xl space-y-4 group hover:bg-[#1a1715] transition-all duration-300">
              <div className="w-12 h-12 rounded-xl bg-[#1a1715]/5 group-hover:bg-[#f8f6f3]/10 flex items-center justify-center overflow-hidden transition-colors">
                <Users className="w-5 h-5 text-[#1a1715]/70 group-hover:text-[#f8f6f3]/90 transition-colors" />
              </div>
              <h3 className="text-xl font-medium group-hover:text-[#f8f6f3] transition-colors">Age-Adaptive</h3>
              <p className="text-[#1a1715]/60 group-hover:text-[#f8f6f3]/70 leading-relaxed transition-colors">
                The same topic, presented three ways. Sprouts (4-7), Explorers (8-17), and Lifelong (18+).
              </p>
            </div>

            {/* Feature 3 - Multilingual */}
            <div className="bg-[#f8f6f3] p-8 rounded-2xl space-y-4 group hover:bg-[#1a1715] transition-all duration-300">
              <div className="w-12 h-12 rounded-xl bg-[#1a1715]/5 group-hover:bg-[#f8f6f3]/10 flex items-center justify-center transition-colors">
                <Globe className="w-5 h-5 text-[#1a1715]/70 group-hover:text-[#f8f6f3]/90 transition-colors" />
              </div>
              <h3 className="text-xl font-medium group-hover:text-[#f8f6f3] transition-colors">Multilingual</h3>
              <p className="text-[#1a1715]/60 group-hover:text-[#f8f6f3]/70 leading-relaxed transition-colors">
                Native support for English, Spanish, French, German, Portuguese, and Mandarin.
              </p>
            </div>

            {/* Feature 4 - AI Kelly */}
            <div className="bg-[#f8f6f3] p-8 rounded-2xl space-y-4 group hover:bg-[#1a1715] transition-all duration-300">
              <div className="w-12 h-12 rounded-xl overflow-hidden">
                <img 
                  src="/kelly/logo-512.webp"
                  alt="Curious Kelly"
                  className="w-full h-full object-cover"
                />
              </div>
              <h3 className="text-xl font-medium group-hover:text-[#f8f6f3] transition-colors">AI-Powered Host</h3>
              <p className="text-[#1a1715]/60 group-hover:text-[#f8f6f3]/70 leading-relaxed transition-colors">
                Curious Kelly brings each lesson to life with personality, warmth, and adaptive teaching.
              </p>
            </div>

            {/* Feature 5 - Video First */}
            <div className="bg-[#f8f6f3] p-8 rounded-2xl space-y-4 group hover:bg-[#1a1715] transition-all duration-300">
              <div className="w-12 h-12 rounded-xl bg-[#1a1715]/5 group-hover:bg-[#f8f6f3]/10 flex items-center justify-center transition-colors">
                <Play className="w-5 h-5 text-[#1a1715]/70 group-hover:text-[#f8f6f3]/90 transition-colors" />
              </div>
              <h3 className="text-xl font-medium group-hover:text-[#f8f6f3] transition-colors">Video-First</h3>
              <p className="text-[#1a1715]/60 group-hover:text-[#f8f6f3]/70 leading-relaxed transition-colors">
                Premium video content designed for modern learners. Watch, learn, and grow.
              </p>
            </div>

            {/* Feature 6 - Research Backed */}
            <div className="bg-[#f8f6f3] p-8 rounded-2xl space-y-4 group hover:bg-[#1a1715] transition-all duration-300">
              <div className="w-12 h-12 rounded-xl bg-[#1a1715]/5 group-hover:bg-[#f8f6f3]/10 flex items-center justify-center transition-colors">
                <Heart className="w-5 h-5 text-[#1a1715]/70 group-hover:text-[#f8f6f3]/90 transition-colors" />
              </div>
              <h3 className="text-xl font-medium group-hover:text-[#f8f6f3] transition-colors">Research-Backed</h3>
              <p className="text-[#1a1715]/60 group-hover:text-[#f8f6f3]/70 leading-relaxed transition-colors">
                Two decades of curriculum development, refined through real-world testing and feedback.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Night Vision Ziggurat - Immersive section */}
      <section className="relative h-[70vh] overflow-hidden">
        <img 
          src="/ziggurat/heroes/ocean-beacon-night.jpg" 
          alt="The Ziggurat at Night - Ocean Beacon LED Display"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-[#1a1715]/80 via-transparent to-[#1a1715]/40" />
        
        {/* Content overlay */}
        <div className="absolute inset-0 flex items-center">
          <div className="max-w-7xl mx-auto px-6 w-full">
            <div className="max-w-lg">
              <span className="text-xs uppercase tracking-widest text-[#f8f6f3]/50">Coming 2028</span>
              <h2 className="text-3xl sm:text-4xl lg:text-5xl font-serif font-normal text-[#f8f6f3] mt-4">
                Where learning <span className="italic">comes alive</span>
              </h2>
              <p className="text-lg text-[#f8f6f3]/70 mt-6 leading-relaxed">
                The world&apos;s largest LED facade will transform the Ziggurat into a beacon of education - broadcasting daily lessons across Southern California.
              </p>
              <a 
                href="/ziggurat" 
                className="inline-flex items-center gap-2 mt-8 text-[#f8f6f3] font-medium hover:gap-3 transition-all"
              >
                Explore the Vision
                <ChevronRight className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Waitlist Section - With Kelly */}
      <section className="py-32 px-6 bg-[#c4956a] text-[#1a1715]">
        <div className="max-w-5xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left: Kelly explorer */}
            <div className="hidden lg:block relative">
              <div className="relative w-full max-w-sm mx-auto">
                <div className="absolute inset-0 bg-[#1a1715]/10 rounded-full blur-3xl scale-90" />
                <img 
                  src="/kelly/archetypes/explorer.png"
                  alt="Kelly Explorer - Ready for adventure"
                  className="relative w-full h-auto drop-shadow-2xl"
                />
              </div>
            </div>
            
            {/* Right: Form */}
            <div className="text-center lg:text-left space-y-8">
              <h2 className="text-4xl sm:text-5xl lg:text-6xl font-serif font-normal leading-tight">
                Be among the first
                <br />
                <span className="italic">to experience</span>
              </h2>
              
              <p className="text-lg text-[#1a1715]/70 max-w-xl">
                We&apos;re opening access to a select group of early learners, educators, and families. Join the waitlist to secure your spot.
              </p>

              {!submitted ? (
                <form onSubmit={handleWaitlist} className="flex flex-col sm:flex-row gap-3 max-w-md mx-auto lg:mx-0 pt-4">
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Enter your email"
                    className="flex-1 px-5 py-4 rounded-full bg-[#1a1715]/10 border-0 text-[#1a1715] placeholder:text-[#1a1715]/40 focus:outline-none focus:ring-2 focus:ring-[#1a1715]/20"
                    required
                  />
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="px-8 py-4 bg-[#1a1715] text-[#f8f6f3] font-medium rounded-full hover:bg-[#2d2a27] transition-colors disabled:opacity-50"
                  >
                    {isSubmitting ? 'Joining...' : 'Join Waitlist'}
                  </button>
                </form>
              ) : (
                <div className="flex items-center justify-center lg:justify-start gap-3 py-4">
                  <div className="w-8 h-8 rounded-full bg-[#1a1715] flex items-center justify-center">
                    <Check className="w-4 h-4 text-[#f8f6f3]" />
                  </div>
                  <p className="text-lg font-medium">You&apos;re on the list. We&apos;ll be in touch.</p>
                </div>
              )}

              <p className="text-sm text-[#1a1715]/50">
                No spam, ever. We respect your inbox.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer - With Kelly branding */}
      <footer className="py-16 px-6 bg-[#1a1715] text-[#f8f6f3]">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col lg:flex-row justify-between gap-12">
            {/* Left: Brand with Kelly */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <img 
                  src="/kelly/logo-512.webp"
                  alt="Curious Kelly"
                  className="w-10 h-10 rounded-full object-cover"
                />
                <span className="text-sm font-medium">Lesson of the Day, PBC</span>
              </div>
              <p className="text-sm text-[#f8f6f3]/50 max-w-xs">
                A Public Benefit Corporation dedicated to making quality education accessible to everyone, everywhere.
              </p>
              <p className="text-xs text-[#f8f6f3]/30">
                Est. 2006 &#8212; Dallas, Texas
              </p>
            </div>

            {/* Right: Links */}
            <div className="flex flex-wrap gap-12">
              <div className="space-y-4">
                <p className="text-xs uppercase tracking-widest text-[#f8f6f3]/40">Product</p>
                <div className="space-y-2">
                  <a href="/about/curriculum" className="block text-sm text-[#f8f6f3]/70 hover:text-[#f8f6f3]">Curriculum</a>
                  <a href="/pricing" className="block text-sm text-[#f8f6f3]/70 hover:text-[#f8f6f3]">Pricing</a>
                  <a href="/ziggurat" className="block text-sm text-[#f8f6f3]/70 hover:text-[#f8f6f3]">The Ziggurat</a>
                </div>
              </div>
              <div className="space-y-4">
                <p className="text-xs uppercase tracking-widest text-[#f8f6f3]/40">Company</p>
                <div className="space-y-2">
                  <a href="/about" className="block text-sm text-[#f8f6f3]/70 hover:text-[#f8f6f3]">About</a>
                  <a href="/investors" className="block text-sm text-[#f8f6f3]/70 hover:text-[#f8f6f3]">Investors</a>
                  <a href="/terms" className="block text-sm text-[#f8f6f3]/70 hover:text-[#f8f6f3]">Terms</a>
                </div>
              </div>
              <div className="space-y-4">
                <p className="text-xs uppercase tracking-widest text-[#f8f6f3]/40">Connect</p>
                <div className="space-y-2">
                  <a href="mailto:hello@thedailylesson.com" className="block text-sm text-[#f8f6f3]/70 hover:text-[#f8f6f3]">Contact</a>
                  <a href="/docs" className="block text-sm text-[#f8f6f3]/70 hover:text-[#f8f6f3]">Documentation</a>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom bar */}
          <div className="mt-16 pt-8 border-t border-[#f8f6f3]/10 flex flex-col sm:flex-row justify-between gap-4">
            <p className="text-xs text-[#f8f6f3]/30">
              {new Date().getFullYear()} Lesson of the Day, PBC. All rights reserved.
            </p>
            <p className="text-xs text-[#f8f6f3]/30">
              Content shown during alpha testing is for demonstration purposes only.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default function CuriousKellyPage() {
  // Show maintenance page when enabled
  if (MAINTENANCE_MODE) {
    return <MaintenancePage />
  }
  
  return (
    <LearnerContextProvider>
      <LayoutProvider>
        <OverlayProvider>
          <CuriousKellyApp />
        </OverlayProvider>
      </LayoutProvider>
    </LearnerContextProvider>
  )
}
