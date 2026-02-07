"use client"

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Info, Sparkles } from 'lucide-react'

// Kelly's 12 Teaching Archetypes with pedagogical descriptions
export const KELLY_TEACHING_STYLES = [
  {
    id: 'storyteller',
    name: 'The Storyteller',
    emoji: '📖',
    tone: 'narrative',
    description: 'Weaves lessons into captivating narratives',
    pedagogy: 'Uses story-based learning to create emotional connections with content. Every fact becomes part of an adventure.',
    strengths: ['Memory retention', 'Emotional engagement', 'Context building'],
    bestFor: 'Visual learners who love examples',
    voice: 'warm, expressive, dramatic pauses',
    color: 'amber',
  },
  {
    id: 'explorer',
    name: 'The Explorer',
    emoji: '🧭',
    tone: 'curious',
    description: 'Discovers knowledge through adventure',
    pedagogy: 'Inquiry-based learning that treats every lesson as an expedition into the unknown.',
    strengths: ['Curiosity cultivation', 'Discovery learning', 'Wonder activation'],
    bestFor: 'Curious minds who ask "why?"',
    voice: 'excited, wondering, adventurous',
    color: 'emerald',
  },
  {
    id: 'architect',
    name: 'The Architect',
    emoji: '📐',
    tone: 'structured',
    description: 'Builds understanding block by block',
    pedagogy: 'Systematic scaffolding that constructs knowledge from foundational principles to complex applications.',
    strengths: ['Logical progression', 'Clear frameworks', 'Building blocks'],
    bestFor: 'Analytical thinkers who need structure',
    voice: 'precise, methodical, clear',
    color: 'blue',
  },
  {
    id: 'empath',
    name: 'The Empath',
    emoji: '💗',
    tone: 'caring',
    description: 'Connects learning to feelings and people',
    pedagogy: 'Social-emotional learning that emphasizes human impact and personal relevance of every concept.',
    strengths: ['Emotional intelligence', 'Personal relevance', 'Supportive environment'],
    bestFor: 'Social learners who need connection',
    voice: 'gentle, understanding, encouraging',
    color: 'pink',
  },
  {
    id: 'rebel',
    name: 'The Rebel',
    emoji: '⚡',
    tone: 'challenging',
    description: 'Questions everything, breaks conventions',
    pedagogy: 'Critical thinking through challenging assumptions and conventional wisdom.',
    strengths: ['Critical thinking', 'Convention breaking', 'Bold questions'],
    bestFor: 'Independent thinkers who challenge norms',
    voice: 'bold, provocative, energetic',
    color: 'red',
  },
  {
    id: 'mystic',
    name: 'The Mystic',
    emoji: '🔮',
    tone: 'philosophical',
    description: 'Explores the deeper meaning of things',
    pedagogy: 'Contemplative learning that connects facts to universal truths and life wisdom.',
    strengths: ['Deep reflection', 'Meaning-making', 'Big picture thinking'],
    bestFor: 'Philosophical minds seeking wisdom',
    voice: 'thoughtful, mysterious, profound',
    color: 'purple',
  },
  {
    id: 'provider',
    name: 'The Provider',
    emoji: '🤲',
    tone: 'nurturing',
    description: 'Gives practical tools for life',
    pedagogy: 'Practical application focus that emphasizes real-world utility and life skills.',
    strengths: ['Practical application', 'Life skills', 'Real-world utility'],
    bestFor: 'Hands-on learners who want usefulness',
    voice: 'warm, practical, helpful',
    color: 'orange',
  },
  {
    id: 'diplomat',
    name: 'The Diplomat',
    emoji: '🕊️',
    tone: 'balanced',
    description: 'Presents all sides fairly',
    pedagogy: 'Perspective-taking that explores multiple viewpoints and fosters understanding.',
    strengths: ['Multiple perspectives', 'Fair presentation', 'Nuanced thinking'],
    bestFor: 'Open-minded learners who see complexity',
    voice: 'fair, balanced, respectful',
    color: 'teal',
  },
  {
    id: 'macgyver',
    name: 'The MacGyver',
    emoji: '🔧',
    tone: 'inventive',
    description: 'Solves problems creatively',
    pedagogy: 'Problem-based learning that emphasizes creative solutions and resourcefulness.',
    strengths: ['Creative problem-solving', 'Resourcefulness', 'Innovation'],
    bestFor: 'Makers and tinkerers',
    voice: 'clever, resourceful, can-do',
    color: 'yellow',
  },
  {
    id: 'scientist',
    name: 'The Scientist',
    emoji: '🔬',
    tone: 'analytical',
    description: 'Tests hypotheses with evidence',
    pedagogy: 'Evidence-based learning that emphasizes the scientific method and data-driven conclusions.',
    strengths: ['Evidence-based thinking', 'Hypothesis testing', 'Data literacy'],
    bestFor: 'Logical minds who need proof',
    voice: 'precise, evidence-focused, curious',
    color: 'cyan',
  },
  {
    id: 'strategist',
    name: 'The Strategist',
    emoji: '♟️',
    tone: 'tactical',
    description: 'Plans moves ahead, sees patterns',
    pedagogy: 'Strategic thinking that emphasizes planning, patterns, and long-term consequences.',
    strengths: ['Pattern recognition', 'Planning ahead', 'Strategic thinking'],
    bestFor: 'Strategic thinkers who plan ahead',
    voice: 'calculated, insightful, forward-thinking',
    color: 'slate',
  },
  {
    id: 'survivor',
    name: 'The Survivor',
    emoji: '🏕️',
    tone: 'resilient',
    description: 'Teaches through challenges',
    pedagogy: 'Growth mindset learning that embraces challenges as opportunities for strength.',
    strengths: ['Resilience building', 'Challenge embracing', 'Grit development'],
    bestFor: 'Determined learners who embrace challenge',
    voice: 'determined, encouraging, tough-loving',
    color: 'green',
  },
] as const

export type TeachingStyle = typeof KELLY_TEACHING_STYLES[number]['id']

interface TeachingStyleSelectorProps {
  selected: TeachingStyle
  onChange: (style: TeachingStyle) => void
  variant?: 'compact' | 'full' | 'dropdown'
  showDescription?: boolean
  className?: string
}

export function TeachingStyleSelector({
  selected,
  onChange,
  variant = 'compact',
  showDescription = false,
  className,
}: TeachingStyleSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [hoveredStyle, setHoveredStyle] = useState<TeachingStyle | null>(null)
  
  const selectedStyle = KELLY_TEACHING_STYLES.find(s => s.id === selected)
  const displayStyle = hoveredStyle 
    ? KELLY_TEACHING_STYLES.find(s => s.id === hoveredStyle) 
    : selectedStyle

  const colorMap: Record<string, string> = {
    amber: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    emerald: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    blue: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    pink: 'bg-pink-500/20 text-pink-300 border-pink-500/30',
    red: 'bg-red-500/20 text-red-300 border-red-500/30',
    purple: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    orange: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
    teal: 'bg-teal-500/20 text-teal-300 border-teal-500/30',
    yellow: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    cyan: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
    slate: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
    green: 'bg-green-500/20 text-green-300 border-green-500/30',
  }

  // Compact: emoji row only
  if (variant === 'compact') {
    return (
      <div className={cn("flex flex-wrap gap-1", className)}>
        {KELLY_TEACHING_STYLES.map(style => (
          <button
            key={style.id}
            onMouseEnter={() => onChange(style.id)}
            onTouchStart={() => onChange(style.id)}
            className={cn(
              "w-8 h-8 rounded-full flex items-center justify-center transition-all",
              selected === style.id 
                ? "bg-white/20 scale-110" 
                : "opacity-50 hover:opacity-80"
            )}
            title={style.name}
          >
            <span className="text-base">{style.emoji}</span>
          </button>
        ))}
      </div>
    )
  }

  // Dropdown: collapsed selector
  if (variant === 'dropdown') {
    return (
      <div className={cn("relative", className)}>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-3 px-4 py-3 bg-white/5 border border-white/10 rounded-xl w-full hover:bg-white/10 transition-colors"
        >
          <span className="text-2xl">{selectedStyle?.emoji}</span>
          <div className="flex-1 text-left">
            <p className="font-medium text-sm">{selectedStyle?.name}</p>
            <p className="text-xs text-white/50">{selectedStyle?.description}</p>
          </div>
          <ChevronDown className={cn("w-4 h-4 transition-transform", isOpen && "rotate-180")} />
        </button>
        
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-full left-0 right-0 mt-2 bg-zinc-900 border border-white/10 rounded-xl shadow-xl z-50 max-h-80 overflow-y-auto"
            >
              {KELLY_TEACHING_STYLES.map(style => (
                <button
                  key={style.id}
                  onClick={() => { onChange(style.id); setIsOpen(false) }}
                  className={cn(
                    "flex items-center gap-3 px-4 py-3 w-full hover:bg-white/10 transition-colors",
                    selected === style.id && "bg-white/5"
                  )}
                >
                  <span className="text-xl">{style.emoji}</span>
                  <div className="flex-1 text-left">
                    <p className="font-medium text-sm">{style.name}</p>
                    <p className="text-xs text-white/50">{style.description}</p>
                  </div>
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    )
  }

  // Full: grid with description panel
  return (
    <div className={cn("space-y-4", className)}>
      {/* Style Grid */}
      <div className="grid grid-cols-4 gap-2">
        {KELLY_TEACHING_STYLES.map(style => (
          <button
            key={style.id}
            onMouseEnter={() => setHoveredStyle(style.id)}
            onMouseLeave={() => setHoveredStyle(null)}
            onClick={() => onChange(style.id)}
            className={cn(
              "flex flex-col items-center gap-1 p-3 rounded-xl border transition-all",
              selected === style.id 
                ? colorMap[style.color]
                : "bg-white/5 border-white/10 hover:bg-white/10"
            )}
          >
            <span className="text-2xl">{style.emoji}</span>
            <span className="text-[10px] font-medium text-center">{style.name.replace('The ', '')}</span>
          </button>
        ))}
      </div>
      
      {/* Description Panel */}
      {showDescription && displayStyle && (
        <motion.div
          key={displayStyle.id}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className={cn("p-4 rounded-xl border", colorMap[displayStyle.color])}
        >
          <div className="flex items-start gap-3 mb-3">
            <span className="text-3xl">{displayStyle.emoji}</span>
            <div>
              <h3 className="font-semibold">{displayStyle.name}</h3>
              <p className="text-sm opacity-80">{displayStyle.description}</p>
            </div>
          </div>
          
          <div className="space-y-3 text-sm">
            <div>
              <p className="text-xs uppercase tracking-wider opacity-60 mb-1">Pedagogy</p>
              <p className="opacity-90">{displayStyle.pedagogy}</p>
            </div>
            
            <div className="flex gap-4">
              <div className="flex-1">
                <p className="text-xs uppercase tracking-wider opacity-60 mb-1">Strengths</p>
                <div className="flex flex-wrap gap-1">
                  {displayStyle.strengths.map(s => (
                    <span key={s} className="text-xs px-2 py-0.5 rounded-full bg-white/10">{s}</span>
                  ))}
                </div>
              </div>
              <div className="flex-1">
                <p className="text-xs uppercase tracking-wider opacity-60 mb-1">Best For</p>
                <p className="opacity-90">{displayStyle.bestFor}</p>
              </div>
            </div>
            
            <div>
              <p className="text-xs uppercase tracking-wider opacity-60 mb-1">Voice Quality</p>
              <p className="opacity-90 italic">"{displayStyle.voice}"</p>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

// Export for investor-class documentation
export function TeachingStylesOverview() {
  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold mb-2">Kelly's 12 Teaching Archetypes</h2>
        <p className="text-white/60 max-w-2xl mx-auto">
          Each archetype represents a distinct pedagogical approach, voice quality, and learning style optimization.
          Learners can switch archetypes at any time to find the teaching style that resonates most.
        </p>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {KELLY_TEACHING_STYLES.map(style => (
          <div
            key={style.id}
            className="p-4 bg-white/5 rounded-xl border border-white/10 hover:bg-white/10 transition-colors"
          >
            <div className="flex items-center gap-3 mb-3">
              <span className="text-3xl">{style.emoji}</span>
              <div>
                <h3 className="font-semibold text-sm">{style.name}</h3>
                <p className="text-xs text-white/50">{style.tone}</p>
              </div>
            </div>
            <p className="text-xs text-white/70 mb-2">{style.description}</p>
            <p className="text-[10px] text-white/40">{style.bestFor}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
