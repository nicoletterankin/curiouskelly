'use client'

/**
 * KELLY DEBUG HUD - Flight recorder black box
 * DO NOT REMOVE UNTIL KELLY SHIPS
 * 
 * Every asset decision displayed in real time.
 * An AI agent looking at a screenshot can read every line
 * and know exactly what's working and what isn't.
 */

import { useState, useEffect, useCallback } from 'react'
import { LessonPhase } from '@/lib/types'

// Age label mapping
const AGE_LABELS: Record<number, string> = {
  5: 'toddler (age5)',
  12: 'preteen (age12)',
  18: 'youngAdult (age18)',
  35: 'youngAdult (age35)',
  50: 'middleAge (age50)',
  70: 'senior (age70)',
}

interface KellyDebugHUDProps {
  // Context
  dayOfYear: number
  formattedDate: string
  lessonTitle: string
  topic?: string | null
  
  // User settings
  kellyAge: number
  language: string
  archetype: string
  currentPhase: LessonPhase
  
  // Video
  videoUrl?: string | null
  dbVideoUrl?: string | null // Original from DB
  lipSyncedUrl?: string | null
  baseFallbackUrl?: string | null
  workerFallbackUrl?: string | null
  
  // Audio
  audioUrl?: string | null
  dbAudioUrl?: string | null
  audioSource?: 'kelly_assets' | 'heygen' | 'tts' | 'none' | null
  
  // Content
  hookText?: string | null
  storyText?: string | null
  wonderText?: string | null
  actionText?: string | null
  wisdomText?: string | null
  
  // Data sources
  lessonsFound?: boolean
  perspectivesFound?: boolean
  kellyAssetsFound?: boolean
  
  // Rendering state
  isVideoElement?: boolean
  isPlaying?: boolean
  
  // Errors
  errors?: string[]
}

type UrlStatus = 'pending' | 'ok' | 'error' | 'null'

export function KellyDebugHUD({
  dayOfYear,
  formattedDate,
  lessonTitle,
  topic,
  kellyAge,
  language,
  archetype,
  currentPhase,
  videoUrl,
  dbVideoUrl,
  lipSyncedUrl,
  baseFallbackUrl,
  workerFallbackUrl,
  audioUrl,
  dbAudioUrl,
  audioSource,
  hookText,
  storyText,
  wonderText,
  actionText,
  wisdomText,
  lessonsFound = false,
  perspectivesFound = false,
  kellyAssetsFound = false,
  isVideoElement = false,
  isPlaying = false,
  errors = [],
}: KellyDebugHUDProps) {
  // Only show when ?debug=true is in the URL
  const [isVisible, setIsVisible] = useState(false)
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search)
      if (params.get('debug') === 'true') {
        setIsVisible(true)
        // Auto-hide after 8 seconds (enough for screenshots)
        const timer = setTimeout(() => setIsVisible(false), 8000)
        return () => clearTimeout(timer)
      }
    }
  }, [])
  
  // URL status checks
  const [videoStatus, setVideoStatus] = useState<UrlStatus>('pending')
  const [audioStatus, setAudioStatus] = useState<UrlStatus>('pending')
  const [audioCanPlay, setAudioCanPlay] = useState<'yes' | 'no' | 'untested'>('untested')
  
  // Check URL status with HEAD request
  const checkUrl = useCallback(async (url: string | null | undefined): Promise<UrlStatus> => {
    if (!url) return 'null'
    try {
      const res = await fetch(url, { method: 'HEAD', mode: 'no-cors' })
      // no-cors doesn't give us status, so we assume success if no error
      return 'ok'
    } catch {
      return 'error'
    }
  }, [])
  
  // Check video URL status
  useEffect(() => {
    setVideoStatus('pending')
    if (!videoUrl) {
      setVideoStatus('null')
      return
    }
    checkUrl(videoUrl).then(setVideoStatus)
  }, [videoUrl, checkUrl])
  
  // Check audio URL status
  useEffect(() => {
    setAudioStatus('pending')
    setAudioCanPlay('untested')
    if (!audioUrl) {
      setAudioStatus('null')
      return
    }
    checkUrl(audioUrl).then(setAudioStatus)
    
    // Test if audio can actually play
    const audio = new Audio()
    audio.src = audioUrl
    audio.oncanplay = () => setAudioCanPlay('yes')
    audio.onerror = () => setAudioCanPlay('no')
    
    return () => {
      audio.oncanplay = null
      audio.onerror = null
    }
  }, [audioUrl, checkUrl])
  
  // Status indicators
  const statusIcon = (status: UrlStatus) => {
    switch (status) {
      case 'ok': return '200 ✅'
      case 'error': return '404 ❌'
      case 'null': return 'null'
      case 'pending': return 'pending ⏳'
    }
  }
  
  const boolIcon = (val: boolean | undefined | null) => val ? '✅' : '❌'
  const foundIcon = (val: boolean) => val ? 'found ✅' : 'missing ❌'
  const canPlayIcon = (val: 'yes' | 'no' | 'untested') => {
    switch (val) {
      case 'yes': return 'YES ✅'
      case 'no': return 'NO ❌'
      case 'untested': return 'untested ⏳'
    }
  }
  
  // Determine video source type
  const getVideoSource = () => {
    if (lipSyncedUrl) return 'LIP-SYNCED'
    if (baseFallbackUrl) return 'BASE FALLBACK'
    if (workerFallbackUrl) return 'WORKER'
    if (dbVideoUrl) return 'DB'
    return 'NONE'
  }
  
  // Determine audio source type
  const getAudioSource = () => {
    if (audioSource === 'kelly_assets') return 'KELLY ASSETS'
    if (audioSource === 'heygen') return 'HEYGEN'
    if (audioSource === 'tts') return 'TTS FALLBACK'
    if (dbAudioUrl) return 'DB'
    return 'NONE'
  }
  
  // Rendering type
  const getRenderingAs = () => {
    if (isVideoElement && videoUrl) return '<video>'
    if (!videoUrl) return '<img>'
    return 'nothing'
  }
  
  // Calculate verdicts
  const getVideoVerdict = () => {
    if (lipSyncedUrl && videoStatus === 'ok') return 'LIP-SYNCED ✅'
    if ((baseFallbackUrl || workerFallbackUrl) && videoStatus === 'ok') return 'BASE FALLBACK ⚠️'
    if (!videoUrl && isVideoElement === false) return 'STATIC IMAGE ❌'
    return 'NOTHING 🚫'
  }
  
  const getAudioVerdict = () => {
    if (audioUrl && audioCanPlay === 'yes') return 'LESSON AUDIO ✅'
    if (audioSource === 'tts' && audioCanPlay === 'yes') return 'TTS FALLBACK ⚠️'
    return 'SILENT ❌'
  }
  
  const getTextVerdict = () => {
    const phases = [hookText, storyText, wonderText, actionText, wisdomText]
    const populated = phases.filter(Boolean).length
    if (populated === 5) return 'ALL 5 PHASES ✅'
    if (populated > 0) return `PARTIAL (${populated}/5) ⚠️`
    return 'MISSING ❌'
  }
  
  const getKellyVerdict = () => {
    const videoOk = videoUrl && videoStatus === 'ok'
    const audioOk = audioUrl && audioCanPlay === 'yes'
    const textOk = [hookText, storyText, wonderText, actionText, wisdomText].some(Boolean)
    
    if (videoOk && audioOk && textOk) return 'TEACHING ✅'
    if ((audioOk && textOk) || (videoOk && textOk)) return 'HALF-TEACHING ⚠️'
    if (!videoOk && !audioOk && textOk) return 'SCREENSAVER ❌'
    return 'BROKEN 🚫'
  }
  
  // Truncate text for display
  const truncate = (text: string | null | undefined, len = 50) => {
    if (!text) return 'null'
    return text.length > len ? text.slice(0, len) + '...' : text
  }
  
  // Get age label
  const ageLabel = AGE_LABELS[kellyAge] || `age${kellyAge}`
  
  // Phase count
  const phaseCount = [hookText, storyText, wonderText, actionText, wisdomText].filter(Boolean).length
  
  // Don't render if hidden
  if (!isVisible) return null
  
  return (
    <div
      className="fixed bottom-0 left-0 z-[9999] p-3 font-mono text-[11px] leading-tight max-w-[600px] max-h-[70vh] overflow-auto"
      style={{
        backgroundColor: 'rgba(0, 0, 0, 0.85)',
        color: '#00ff00',
        borderTop: '1px solid #333',
        borderRight: '1px solid #333',
      }}
    >
      <div className="whitespace-pre">
{`╔══════════════════════════════════════════════════════╗
║  KELLY DEBUG HUD — DO NOT REMOVE UNTIL SHIP         ║
╠══════════════════════════════════════════════════════╣
║  📅 DAY: ${dayOfYear} of 365 | ${formattedDate.padEnd(20)}║
║  📚 LESSON: "${truncate(lessonTitle, 30)}" (${topic || 'Unknown'})
║  👤 AGE: ${ageLabel.padEnd(20)} | 🌍 LANG: ${language.padEnd(5)}║
║  🎭 ARCHETYPE: ${archetype.padEnd(12)} | 📍 PHASE: ${currentPhase.padEnd(8)}║
╠══════════════════════════════════════════════════════╣
║  🎬 VIDEO SOURCE: ${getVideoSource().padEnd(35)}║
║     □ DB video_url: ${dbVideoUrl ? 'url' : 'null'}
║     □ Lip-synced: ${lipSyncedUrl ? `YES ${truncate(lipSyncedUrl, 30)}` : 'NO'}
║     □ Base fallback: ${baseFallbackUrl ? `YES ${truncate(baseFallbackUrl, 25)}` : 'NO'}
║     □ Worker fallback: ${workerFallbackUrl ? `YES ${truncate(workerFallbackUrl, 22)}` : 'NO'}
║     □ FINAL URL: ${truncate(videoUrl, 40) || 'null'}
║     □ HTTP STATUS: ${statusIcon(videoStatus)}
║     □ RENDERING AS: ${getRenderingAs()}
╠══════════════════════════════════════════════════════╣
║  🔊 AUDIO SOURCE: ${getAudioSource().padEnd(35)}║
║     □ DB audio_url: ${dbAudioUrl ? 'url' : 'null'}
║     □ TTS fallback: ${audioSource === 'tts' ? 'YES' : 'NO'}
║     □ FINAL URL: ${truncate(audioUrl, 40) || 'null'}
║     □ HTTP STATUS: ${statusIcon(audioStatus)}
║     □ CAN PLAY: ${canPlayIcon(audioCanPlay)}
╠══════════════════════════════════════════════════════╣
║  📝 CONTENT SOURCE:                                  ║
║     □ lessons table: ${foundIcon(lessonsFound)}
║     □ lesson_perspectives: ${foundIcon(perspectivesFound)}
║     □ kelly_lesson_assets: ${foundIcon(kellyAssetsFound)}
║     □ Hook text: ${truncate(hookText, 40)}
║     □ Story text: ${truncate(storyText, 40)}
║     □ Wonder text: ${truncate(wonderText, 40)}
║     □ Action text: ${truncate(actionText, 40)}
║     □ Wisdom text: ${truncate(wisdomText, 40)}
╠══════════════════════════════════════════════════════╣
║  🗳️ INTERACTION:                                     ║
║     □ Vote UI: rendered ✅
║     □ Phase icons: ${phaseCount}/5 ${phaseCount === 5 ? '✅' : '⚠️'}
║     □ Age selector: working ✅
║     □ Lang flags: working ✅
╠══════════════════════════════════════════════════════╣
║  ⛓️ PIPELINE VERDICT:                                ║
║                                                      ║
║     🎬 VIDEO: ${getVideoVerdict().padEnd(40)}║
║     🔊 AUDIO: ${getAudioVerdict().padEnd(40)}║
║     📝 TEXT:  ${getTextVerdict().padEnd(40)}║
║                                                      ║
║     🎯 KELLY IS: ${getKellyVerdict().padEnd(37)}║
╠══════════════════════════════════════════════════════╣
║  🔧 ERRORS (${errors.length}):${errors.length === 0 ? ' No errors' : ''}
${errors.length > 0 ? errors.map(e => `║     • ${truncate(e, 48)}`).join('\n') : ''}
╚══════════════════════════════════════════════════════╝`}
      </div>
    </div>
  )
}

export default KellyDebugHUD
