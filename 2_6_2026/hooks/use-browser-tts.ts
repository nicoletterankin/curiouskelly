'use client'

/**
 * Browser-based TTS using Web Speech API
 * ZERO COST - Works offline, no API credits needed
 * 
 * Fallback when ElevenLabs/cloud TTS is unavailable
 */

import { useState, useCallback, useRef, useEffect } from 'react'

interface UseBrowserTTSOptions {
  voice?: string // Voice name preference
  rate?: number // Speech rate (0.1 - 10, default 1)
  pitch?: number // Pitch (0 - 2, default 1)
  volume?: number // Volume (0 - 1, default 1)
  language?: string // BCP 47 language tag (en-US, es-ES, etc)
}

interface UseBrowserTTSReturn {
  speak: (text: string) => void
  stop: () => void
  pause: () => void
  resume: () => void
  isSupported: boolean
  isSpeaking: boolean
  isPaused: boolean
  voices: SpeechSynthesisVoice[]
  selectedVoice: SpeechSynthesisVoice | null
  setVoice: (voice: SpeechSynthesisVoice) => void
}

// Map our language codes to BCP 47
const LANGUAGE_MAP: Record<string, string> = {
  en: 'en-US',
  es: 'es-ES',
  fr: 'fr-FR',
  de: 'de-DE',
  pt: 'pt-BR',
  zh: 'zh-CN',
  ja: 'ja-JP',
  ko: 'ko-KR',
  it: 'it-IT',
  ru: 'ru-RU',
}

export function useBrowserTTS(options: UseBrowserTTSOptions = {}): UseBrowserTTSReturn {
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([])
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null)
  
  const isSupported = typeof window !== 'undefined' && 'speechSynthesis' in window

  // Load available voices
  useEffect(() => {
    if (!isSupported) return

    const loadVoices = () => {
      const availableVoices = speechSynthesis.getVoices()
      setVoices(availableVoices)
      
      // Select best voice for language
      const lang = LANGUAGE_MAP[options.language || 'en'] || 'en-US'
      
      // Priority: 1) Exact match female voice, 2) Exact match any, 3) Language prefix match
      const preferredVoice = 
        availableVoices.find(v => v.lang === lang && v.name.toLowerCase().includes('female')) ||
        availableVoices.find(v => v.lang === lang && (v.name.includes('Samantha') || v.name.includes('Karen') || v.name.includes('Zira'))) ||
        availableVoices.find(v => v.lang === lang) ||
        availableVoices.find(v => v.lang.startsWith(lang.split('-')[0])) ||
        availableVoices[0]
      
      if (preferredVoice && !selectedVoice) {
        setSelectedVoice(preferredVoice)
      }
    }

    loadVoices()
    speechSynthesis.onvoiceschanged = loadVoices
    
    return () => {
      speechSynthesis.onvoiceschanged = null
    }
  }, [isSupported, options.language, selectedVoice])

  const speak = useCallback((text: string) => {
    if (!isSupported || !text) return
    
    // Stop any current speech
    speechSynthesis.cancel()
    
    const utterance = new SpeechSynthesisUtterance(text)
    utteranceRef.current = utterance
    
    // Apply settings
    if (selectedVoice) utterance.voice = selectedVoice
    utterance.rate = options.rate ?? 0.95 // Slightly slower for learning
    utterance.pitch = options.pitch ?? 1.05 // Slightly higher for friendliness
    utterance.volume = options.volume ?? 1
    
    // Set language
    const lang = LANGUAGE_MAP[options.language || 'en'] || 'en-US'
    utterance.lang = lang
    
    // Event handlers
    utterance.onstart = () => {
      setIsSpeaking(true)
      setIsPaused(false)
    }
    
    utterance.onend = () => {
      setIsSpeaking(false)
      setIsPaused(false)
    }
    
    utterance.onerror = (event) => {
      console.warn('[browser-tts] Error:', event.error)
      setIsSpeaking(false)
      setIsPaused(false)
    }
    
    utterance.onpause = () => setIsPaused(true)
    utterance.onresume = () => setIsPaused(false)
    
    speechSynthesis.speak(utterance)
  }, [isSupported, selectedVoice, options])

  const stop = useCallback(() => {
    if (!isSupported) return
    speechSynthesis.cancel()
    setIsSpeaking(false)
    setIsPaused(false)
  }, [isSupported])

  const pause = useCallback(() => {
    if (!isSupported) return
    speechSynthesis.pause()
  }, [isSupported])

  const resume = useCallback(() => {
    if (!isSupported) return
    speechSynthesis.resume()
  }, [isSupported])

  const setVoice = useCallback((voice: SpeechSynthesisVoice) => {
    setSelectedVoice(voice)
  }, [])

  return {
    speak,
    stop,
    pause,
    resume,
    isSupported,
    isSpeaking,
    isPaused,
    voices,
    selectedVoice,
    setVoice,
  }
}

// Utility: Get script for a phase from lesson data
export function getPhaseScript(lesson: {
  hook_script?: string
  story_script?: string
  teach_script?: string
  action_script?: string
  reflection_script?: string
} | null, phase: string): string {
  if (!lesson) return ''
  
  switch (phase) {
    case 'hook': return lesson.hook_script || ''
    case 'story': return lesson.story_script || ''
    case 'wonder': 
    case 'teach': return lesson.teach_script || ''
    case 'action': return lesson.action_script || ''
    case 'wisdom':
    case 'reflection': return lesson.reflection_script || ''
    default: return ''
  }
}
