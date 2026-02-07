'use client'

/**
 * KellyOS About Overlay - v6
 * Clean design: no blur over Kelly, positioned to the LEFT
 * Uses Nicolette's real photo, transparent backdrop
 */

import { motion, AnimatePresence } from 'framer-motion'
import { AnimatedList, AnimatedItem, springPresets } from './index'
import { ChevronRight, X } from 'lucide-react'

interface AboutOverlayProps {
  isOpen: boolean
  onClose: () => void
}

const METRICS = [
  { value: '365', label: 'Lessons', sub: 'One per day' },
  { value: '50+', label: 'Languages', sub: 'Accessible' },
  { value: '12', label: 'Archetypes', sub: 'Universal' },
  { value: '8B', label: 'Goal by 2075', sub: 'Mission-Driven' },
]

// Nicolette's actual photo
const NICOLETTE_PHOTO = '/images/nicolette-rankin.jpg'

export function AboutOverlay({ isOpen, onClose }: AboutOverlayProps) {
  if (!isOpen) return null

  return (
    <AnimatePresence mode="wait">
      {isOpen && (
        <>
          {/* Minimal backdrop - NO blur, just slight dim on left side */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50"
            onClick={onClose}
          >
            {/* Gradient that dims left side only, keeps Kelly visible */}
            <div className="absolute inset-0 bg-gradient-to-r from-black/70 via-black/30 to-transparent" />
          </motion.div>

          {/* Panel positioned LEFT - doesn't cover Kelly */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={springPresets.snappy}
            className="fixed left-4 top-20 bottom-4 z-50 w-[340px] max-w-[90vw]"
            onClick={e => e.stopPropagation()}
          >
            <div className="h-full flex flex-col bg-zinc-950/95 border border-white/10 rounded-2xl shadow-2xl overflow-hidden">
              {/* Header */}
              <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
                <h2 className="text-base font-medium text-white">About</h2>
                <button
                  onClick={onClose}
                  className="h-8 w-8 rounded-full bg-white/5 hover:bg-white/10 flex items-center justify-center text-white/70 hover:text-white transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto">
                <AnimatedList className="px-5 py-5 space-y-5" delay={0.05}>
                  
                  {/* Hero */}
                  <AnimatedItem className="text-center">
                    <div className="w-20 h-20 mx-auto rounded-full overflow-hidden mb-3 border-2 border-white/15 shadow-lg">
                      <img 
                        src="/kelly/archetypes/storyteller.png" 
                        alt="Kelly" 
                        className="w-full h-full object-cover"
                        style={{ objectPosition: 'center 20%' }}
                      />
                    </div>
                    <h3 className="font-serif text-xl text-white mb-0.5">The Daily Lesson</h3>
                    <p className="text-xs text-zinc-400 tracking-wide">with Curious Kelly</p>
                    <span className="inline-flex items-center gap-1.5 mt-2 px-2 py-0.5 bg-amber-500/10 border border-amber-500/20 rounded-full text-[10px] text-amber-400 font-medium uppercase">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                      Private Alpha
                    </span>
                  </AnimatedItem>

                  {/* Mission */}
                  <AnimatedItem className="text-center py-4 px-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                    <p className="font-serif text-[15px] text-white/90 leading-relaxed mb-1">
                      A new lesson every day, 365 days a year.
                    </p>
                    <p className="text-sm text-zinc-400">
                      Universal knowledge for every age, language, and learning style.
                    </p>
                  </AnimatedItem>

                  {/* Metrics - 2x2 grid */}
                  <AnimatedItem className="grid grid-cols-2 gap-2">
                    {METRICS.map((m, i) => (
                      <div key={i} className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06]">
                        <div className="font-mono text-lg font-semibold text-white">{m.value}</div>
                        <p className="text-[11px] font-medium text-zinc-300 mt-0.5">{m.label}</p>
                        <p className="text-[10px] text-zinc-500">{m.sub}</p>
                      </div>
                    ))}
                  </AnimatedItem>

                  {/* Team */}
                  <AnimatedItem className="space-y-2">
                    {/* Kelly */}
                    <div className="flex items-center gap-3 p-3 rounded-lg bg-white/[0.03] border border-white/[0.06]">
                      <div className="w-10 h-10 rounded-lg overflow-hidden border border-white/10 flex-shrink-0">
                        <img src="/kelly/archetypes/explorer.png" alt="Kelly" className="w-full h-full object-cover object-top" />
                      </div>
                      <div className="min-w-0">
                        <h4 className="font-serif text-sm text-white">Meet Kelly</h4>
                        <p className="text-[11px] text-zinc-400 leading-snug">AI teacher. 12 archetypes. 5-phase lessons.</p>
                      </div>
                    </div>

                    {/* Nicolette - REAL PHOTO */}
                    <div className="flex items-center gap-3 p-3 rounded-lg bg-white/[0.03] border border-white/[0.06]">
                      <div className="w-10 h-10 rounded-lg overflow-hidden border border-white/10 flex-shrink-0">
                        <img src={NICOLETTE_PHOTO || "/placeholder.svg"} alt="Nicolette Rankin" className="w-full h-full object-cover" />
                      </div>
                      <div className="min-w-0">
                        <h4 className="font-serif text-sm text-white">Nicolette Rankin</h4>
                        <p className="text-[11px] text-zinc-400 leading-snug">Founder & CEO. OpenEnglish (2M+ students).</p>
                      </div>
                    </div>
                  </AnimatedItem>

                  {/* Links */}
                  <AnimatedItem className="space-y-0.5">
                    {[
                      { label: 'Strategic Overview', href: '/strategic' },
                      { label: 'Investor Materials', href: '/ziggurat/investors' },
                    ].map((link) => (
                      <a 
                        key={link.label}
                        href={link.href}
                        className="flex items-center justify-between py-2 px-2 rounded-lg hover:bg-white/[0.04] transition-colors"
                      >
                        <span className="text-sm text-zinc-300">{link.label}</span>
                        <ChevronRight className="h-3.5 w-3.5 text-zinc-600" />
                      </a>
                    ))}
                  </AnimatedItem>
                </AnimatedList>
              </div>

              {/* Footer */}
              <div className="px-5 py-3 text-center border-t border-white/5">
                <p className="text-[10px] text-zinc-600">2026 Lesson of the Day, Inc. | v3.0</p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
