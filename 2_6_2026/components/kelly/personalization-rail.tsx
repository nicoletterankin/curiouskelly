"use client"

import { useState, useEffect } from "react"
import { X, Settings } from "lucide-react"

const AGE_OPTIONS = [
  { id: "young", emoji: "🧒", label: "Young (4-8)", desc: "Simple words, playful tone" },
  { id: "tween", emoji: "👦", label: "Tween (9-12)", desc: "Curious explorer mode" },
  { id: "teen", emoji: "🧑", label: "Teen (13-17)", desc: "Real talk, no fluff" },
  { id: "adult", emoji: "👤", label: "Adult (18-64)", desc: "Full depth, nuanced" },
  { id: "senior", emoji: "👴", label: "Senior (65+)", desc: "Wisdom-focused, reflective" },
]

const ARCHETYPES = [
  { id: "explorer", emoji: "🧭", name: "Explorer", short: "Exp", desc: "Curious adventurer" },
  { id: "architect", emoji: "📐", name: "Architect", short: "Arc", desc: "Master builder" },
  { id: "empath", emoji: "💗", name: "Empath", short: "Emp", desc: "Heart-centered" },
  { id: "rebel", emoji: "⚡", name: "Rebel", short: "Reb", desc: "Status quo challenger" },
  { id: "mystic", emoji: "🔮", name: "Mystic", short: "Mys", desc: "Hidden truth seeker" },
  { id: "provider", emoji: "🤲", name: "Provider", short: "Pro", desc: "Generous nurturer" },
  { id: "diplomat", emoji: "🤝", name: "Diplomat", short: "Dip", desc: "Bridge builder" },
  { id: "maker", emoji: "🔧", name: "Maker", short: "Mak", desc: "Creative craftsperson" },
  { id: "scientist", emoji: "🔬", name: "Scientist", short: "Sci", desc: "Truth investigator" },
  { id: "strategist", emoji: "♟️", name: "Strategist", short: "Str", desc: "Visionary planner" },
]

const LANGUAGES = [
  { code: "en", flag: "🇺🇸", label: "English" },
  { code: "es", flag: "🇪🇸", label: "Español" },
  { code: "fr", flag: "🇫🇷", label: "Français" },
  { code: "de", flag: "🇩🇪", label: "Deutsch" },
  { code: "pt", flag: "🇧🇷", label: "Português" },
  { code: "zh", flag: "🇨🇳", label: "中文" },
]

const PHASES = [
  { id: "hook", emoji: "🪝", label: "HOOK", color: "#f59e0b" },
  { id: "story", emoji: "📖", label: "STORY", color: "#3b82f6" },
  { id: "wonder", emoji: "💭", label: "WONDER", color: "#8b5cf6" },
  { id: "action", emoji: "🎯", label: "ACTION", color: "#10b981" },
  { id: "wisdom", emoji: "💎", label: "WISDOM", color: "#f43f5e" },
]

interface PersonalizationRailProps {
  currentPhase: string
  currentAge: string
  currentArchetype: string | null
  currentLanguage: string
  onPhaseChange: (phase: string) => void
  onAgeChange: (age: string) => void
  onArchetypeChange: (archetype: string) => void
  onLanguageChange: (language: string) => void
}

export function PersonalizationRail({
  currentPhase,
  currentAge,
  currentArchetype,
  currentLanguage,
  onPhaseChange,
  onAgeChange,
  onArchetypeChange,
  onLanguageChange,
}: PersonalizationRailProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [touchStart, setTouchStart] = useState<number | null>(null)

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      
      if (e.key === 's' || e.key === 'S') {
        e.preventDefault()
        setIsOpen(prev => !prev)
      }
      
      // Phase shortcuts 1-5
      if (['1', '2', '3', '4', '5'].includes(e.key)) {
        const phaseIndex = parseInt(e.key) - 1
        if (PHASES[phaseIndex]) {
          onPhaseChange(PHASES[phaseIndex].id)
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onPhaseChange])

  // Swipe from right edge to open
  useEffect(() => {
    const handleTouchStart = (e: TouchEvent) => {
      const touch = e.touches[0]
      if (touch.clientX > window.innerWidth - 30) {
        setTouchStart(touch.clientX)
      }
    }

    const handleTouchMove = (e: TouchEvent) => {
      if (touchStart === null) return
      const touch = e.touches[0]
      const diff = touchStart - touch.clientX
      if (diff > 50) {
        setIsOpen(true)
        setTouchStart(null)
      }
    }

    const handleTouchEnd = () => setTouchStart(null)

    window.addEventListener('touchstart', handleTouchStart)
    window.addEventListener('touchmove', handleTouchMove)
    window.addEventListener('touchend', handleTouchEnd)

    return () => {
      window.removeEventListener('touchstart', handleTouchStart)
      window.removeEventListener('touchmove', handleTouchMove)
      window.removeEventListener('touchend', handleTouchEnd)
    }
  }, [touchStart])

  const phaseIndex = PHASES.findIndex(p => p.id === currentPhase)

  return (
    <>
      {/* Settings toggle - always visible */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed right-4 top-1/2 -translate-y-1/2 z-50
                   w-12 h-12 rounded-full bg-white/10 backdrop-blur-sm border border-white/10
                   hover:bg-white/20 hover:scale-110 transition-all duration-200
                   flex items-center justify-center md:top-1/2 bottom-20 md:bottom-auto"
        aria-label="Personalization settings"
      >
        <Settings className="w-5 h-5 text-white/70" />
      </button>

      {/* Rail panel */}
      <div
        className={`
          fixed right-0 top-0 h-full w-80 max-w-[85vw]
          bg-zinc-900/95 backdrop-blur-xl border-l border-white/10
          transform transition-transform duration-300 ease-out
          ${isOpen ? 'translate-x-0' : 'translate-x-full'}
          z-40 overflow-y-auto
        `}
      >
        <div className="p-6">
          {/* Header */}
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-lg font-medium text-white">
              Personalize Your Kelly
            </h2>
            <button
              onClick={() => setIsOpen(false)}
              className="w-8 h-8 rounded-full hover:bg-white/10 flex items-center justify-center transition-colors"
            >
              <X className="w-5 h-5 text-white/60 hover:text-white" />
            </button>
          </div>

          {/* Phase Section */}
          <section className="mb-8">
            <h3 className="text-sm font-medium text-white/60 mb-3 flex items-center gap-2">
              <span>📚</span> PHASE
            </h3>
            <div className="grid grid-cols-5 gap-2 mb-2">
              {PHASES.map((phase) => (
                <button
                  key={phase.id}
                  onClick={() => onPhaseChange(phase.id)}
                  className={`
                    aspect-square rounded-lg flex flex-col items-center justify-center
                    text-xl transition-all duration-200
                    ${currentPhase === phase.id
                      ? 'ring-2 scale-105'
                      : 'bg-white/5 hover:bg-white/10 hover:scale-105'}
                  `}
                  style={currentPhase === phase.id ? {
                    backgroundColor: `${phase.color}20`,
                    ringColor: phase.color,
                    boxShadow: `0 0 15px ${phase.color}40`
                  } : {}}
                  title={phase.label}
                >
                  {phase.emoji}
                  <span className="text-[9px] text-white/60 mt-1">{phase.label.slice(0, 4)}</span>
                </button>
              ))}
            </div>
            <p className="text-xs text-white/40">
              Currently: {PHASES.find(p => p.id === currentPhase)?.label} ({phaseIndex + 1} of 5)
            </p>
          </section>

          <div className="h-px bg-white/10 mb-8" />

          {/* Age Section */}
          <section className="mb-8">
            <h3 className="text-sm font-medium text-white/60 mb-3 flex items-center gap-2">
              <span>🎭</span> WHO IS KELLY FOR YOU?
            </h3>
            <div className="space-y-2">
              {AGE_OPTIONS.map((age) => (
                <button
                  key={age.id}
                  onClick={() => onAgeChange(age.id)}
                  className={`
                    w-full p-3 rounded-lg text-left transition-all duration-200
                    ${currentAge === age.id
                      ? 'bg-blue-500/20 border border-blue-500/50 text-white scale-[1.02]'
                      : 'bg-white/5 hover:bg-white/10 text-white/80 border border-transparent'}
                  `}
                  style={currentAge === age.id ? { boxShadow: '0 0 15px rgba(59, 130, 246, 0.2)' } : {}}
                >
                  <span className="flex items-center justify-between">
                    <span>{age.emoji} {age.label}</span>
                    {currentAge === age.id && <span className="text-blue-400 text-xs">Active</span>}
                  </span>
                  <span className="text-xs text-white/40 mt-1 block">{age.desc}</span>
                </button>
              ))}
            </div>
          </section>

          <div className="h-px bg-white/10 mb-8" />

          {/* Archetype Section */}
          <section className="mb-8">
            <h3 className="text-sm font-medium text-white/60 mb-1 flex items-center gap-2">
              <span>🎨</span> LEARNING STYLE
            </h3>
            <p className="text-xs text-white/40 mb-3">How do you learn best?</p>
            <div className="grid grid-cols-4 gap-2 mb-3">
              {ARCHETYPES.map((arch) => (
                <button
                  key={arch.id}
                  onClick={() => onArchetypeChange(arch.id)}
                  className={`
                    aspect-square rounded-lg flex flex-col items-center justify-center
                    text-2xl transition-all duration-200
                    ${currentArchetype === arch.id
                      ? 'bg-blue-500/20 ring-2 ring-blue-500 scale-105'
                      : 'bg-white/5 hover:bg-white/10 hover:scale-105'}
                  `}
                  style={currentArchetype === arch.id ? { boxShadow: '0 0 15px rgba(59, 130, 246, 0.3)' } : {}}
                  title={`${arch.name}: ${arch.desc}`}
                >
                  {arch.emoji}
                  <span className="text-[9px] text-white/60 mt-1">{arch.short}</span>
                </button>
              ))}
            </div>
            <button className="text-sm text-blue-400 hover:text-blue-300 transition-colors">
              Take the Quiz →
            </button>
          </section>

          <div className="h-px bg-white/10 mb-8" />

          {/* Language Section */}
          <section className="mb-8">
            <h3 className="text-sm font-medium text-white/60 mb-3 flex items-center gap-2">
              <span>🌍</span> LANGUAGE
            </h3>
            <div className="grid grid-cols-6 gap-2">
              {LANGUAGES.map((lang) => (
                <button
                  key={lang.code}
                  onClick={() => onLanguageChange(lang.code)}
                  className={`
                    aspect-square rounded-lg flex flex-col items-center justify-center
                    transition-all duration-200
                    ${currentLanguage === lang.code
                      ? 'bg-blue-500/20 ring-2 ring-blue-500 scale-110'
                      : 'bg-white/5 hover:bg-white/10 hover:scale-105'}
                  `}
                  style={currentLanguage === lang.code ? { boxShadow: '0 0 15px rgba(59, 130, 246, 0.3)' } : {}}
                  title={lang.label}
                >
                  <span className="text-xl">{lang.flag}</span>
                  <span className="text-[9px] text-white/60 mt-1">{lang.code.toUpperCase()}</span>
                </button>
              ))}
            </div>
          </section>

          {/* Keyboard hints */}
          <div className="pt-4 border-t border-white/10">
            <p className="text-[10px] text-white/30 mb-2">KEYBOARD SHORTCUTS</p>
            <div className="grid grid-cols-2 gap-1 text-[10px] text-white/40">
              <span><kbd className="px-1 py-0.5 bg-white/10 rounded text-[9px]">S</kbd> Toggle panel</span>
              <span><kbd className="px-1 py-0.5 bg-white/10 rounded text-[9px]">1-5</kbd> Switch phase</span>
            </div>
          </div>

        </div>
      </div>

      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 transition-opacity"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  )
}

export default PersonalizationRail
