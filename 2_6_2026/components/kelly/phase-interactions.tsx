'use client'

// Phase-Specific Interaction Components
// Makes Kelly the BEST TEACHER on the planet with engaging interactions for each phase

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { 
  Check, X, Lightbulb, BookOpen, Sparkles, 
  MessageSquare, ThumbsUp, RefreshCw, ChevronRight,
  Brain, Rocket, Heart, Star, Zap
} from 'lucide-react'
import { useVote } from '@/hooks/use-vote'
import { LessonPhase } from '@/lib/types'

// ============================================================================
// HOOK PHASE - TRUE/FALSE voting with cinematic teaching flow
// ALL hook facts are TRUE - the teaching explains WHY they're true
// Cinematic Flow:
// 1. hookFact displayed for TRUE/FALSE voting
// 2. Results + Teaching Moment revealed
// 3. hookQuestion (Kelly's curiosity spark) shown after
// ============================================================================
interface HookInteractionProps {
  dayOfYear: number
  language: string
  hookFact?: string // The TRUE fact statement to vote on ("Sunlight bounces off...")
  hookQuestion?: string // Kelly's curiosity spark ("Have you ever wondered...")
  explanation?: string  // Full teaching moment (wonder_script)
  correctAnswer?: 'true' | 'false' // Always 'true' - all hooks are true facts
}

export function HookInteraction({ dayOfYear, language, hookFact, explanation, correctAnswer = 'true' }: HookInteractionProps) {
  const { submitVote, userVote, isLoading, truePercent, totalVotes } = useVote(dayOfYear, 'hook')
  const [showTeaching, setShowTeaching] = useState(false)
  
  const hasVoted = userVote !== null
  const falsePercent = 100 - truePercent
  // All hooks are TRUE - so voting 'true' is correct
  const userWasCorrect = userVote === 'true'
  // GUARD: hook_fact must be a statement, not a question. If it ends with "?" it leaked from hook_script.
  const isQuestion = hookFact ? hookFact.trim().endsWith('?') : false
  
  // Show the teaching moment 1.5 seconds after voting
  React.useEffect(() => {
    if (hasVoted && !showTeaching) {
      const timer = setTimeout(() => setShowTeaching(true), 1500)
      return () => clearTimeout(timer)
    }
  }, [hasVoted, showTeaching])
  
  return (
    <div className="w-full">
      {/* Hook Fact Display */}
      <div className="text-center mb-4">
        <div className="inline-flex items-center gap-1.5 text-zinc-400 text-[10px] uppercase tracking-[0.15em] mb-2 font-medium">
          <span>{isQuestion ? 'Today\'s curiosity' : 'Is this true?'}</span>
        </div>
        <p 
          className="text-white text-[15px] leading-relaxed text-center text-balance font-medium"
          style={{ textShadow: '0 1px 6px rgba(0,0,0,0.7)' }}
        >
          {hookFact || "Loading today's fact..."}
        </p>
      </div>
      
      {/* TRUE / FALSE Buttons */}
      {!hasVoted && (
        <div className="flex gap-2.5">
          <button
            onClick={() => submitVote('true')}
            disabled={isLoading}
            className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm tracking-wide transition-all bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/30 hover:border-emerald-500/50 active:scale-[0.97]"
          >
            <Check className="w-4 h-4" />
            <span>True</span>
          </button>
          
          <button
            onClick={() => submitVote('false')}
            disabled={isLoading}
            className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm tracking-wide transition-all bg-rose-500/20 text-rose-300 border border-rose-500/30 hover:bg-rose-500/30 hover:border-rose-500/50 active:scale-[0.97]"
          >
            <X className="w-4 h-4" />
            <span>False</span>
          </button>
        </div>
      )}
      
      {/* Results Bar - After voting */}
      <AnimatePresence>
        {hasVoted && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="overflow-hidden"
          >
            <div className="flex gap-2 mb-2">
              <div className={cn(
                "flex-1 py-2 px-3 rounded-xl text-center text-xs font-semibold",
                userVote === 'true' 
                  ? "bg-emerald-500/30 text-emerald-300 border border-emerald-500/40" 
                  : "bg-white/[0.03] text-white/40"
              )}>
                True {truePercent}%
              </div>
              <div className={cn(
                "flex-1 py-2 px-3 rounded-xl text-center text-xs font-semibold",
                userVote === 'false' 
                  ? "bg-rose-500/30 text-rose-300 border border-rose-500/40" 
                  : "bg-white/[0.03] text-white/40"
              )}>
                False {falsePercent}%
              </div>
            </div>
            <div className="h-1.5 rounded-full overflow-hidden bg-white/[0.08] flex">
              <div className="h-full bg-emerald-500/70 transition-all duration-500" style={{ width: `${truePercent}%` }} />
              <div className="h-full bg-rose-500/70 transition-all duration-500" style={{ width: `${falsePercent}%` }} />
            </div>
            <p className="text-white/40 text-[10px] text-center mt-1.5">
              {totalVotes.toLocaleString()} votes
            </p>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* THE TEACHING MOMENT - Explains WHY it's true */}
      <AnimatePresence>
        {showTeaching && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
            className={cn(
              "mt-4 p-4 rounded-xl border",
              userWasCorrect
                ? "bg-emerald-950/30 border-emerald-500/20" 
                : "bg-amber-950/30 border-amber-500/20"
            )}
            style={{ backdropFilter: 'blur(12px)' }}
          >
            {userWasCorrect ? (
              <>
                <p className="text-emerald-300 text-sm font-semibold mb-2">
                  Correct! This is TRUE.
                </p>
                {explanation && (
                  <p className="text-white/85 text-sm leading-relaxed">
                    <span className="text-emerald-400 font-medium">Here's why: </span>
                    {explanation}
                  </p>
                )}
              </>
            ) : (
              <>
                <p className="text-amber-300 text-sm font-semibold mb-2">
                  Actually, this is TRUE!
                </p>
                {explanation && (
                  <p className="text-white/85 text-sm leading-relaxed">
                    <span className="text-amber-400 font-medium">Here's why: </span>
                    {explanation}
                  </p>
                )}
                <p className="text-white/50 text-xs mt-2 italic">
                  Many people get this wrong - now you know!
                </p>
              </>
            )}
            
            {/* Nudge to continue */}
            <div className="mt-3 pt-2 border-t border-white/10">
              <p className="text-white/40 text-xs text-center">
                Swipe up for the full story
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ============================================================================
// STORY PHASE - Comprehension check with key takeaways
// ============================================================================
interface StoryInteractionProps {
  dayOfYear: number
  keyTakeaways?: string[]
  onTakeawaySelect?: (takeaway: string) => void
}

export function StoryInteraction({ dayOfYear, keyTakeaways = [], onTakeawaySelect }: StoryInteractionProps) {
  const [selectedTakeaway, setSelectedTakeaway] = useState<string | null>(null)
  
  const defaultTakeaways = [
    'This connects to my life because...',
    'I want to learn more about...',
    'This reminds me of...'
  ]
  
  const takeaways = keyTakeaways.length > 0 ? keyTakeaways : defaultTakeaways
  
  return (
    <div className="w-full p-4 rounded-2xl border border-white/[0.06] bg-blue-500/[0.03]" style={{ backdropFilter: 'blur(16px)' }}>
      <div className="flex items-center gap-2.5 mb-3">
        <div className="w-9 h-9 rounded-xl bg-blue-500/20 flex items-center justify-center">
          <BookOpen className="w-4.5 h-4.5 text-blue-400" />
        </div>
        <div>
          <p className="text-white font-semibold text-base">Story Reflection</p>
          <p className="text-white/50 text-sm">What resonates with you?</p>
        </div>
      </div>
      
      <div className="space-y-2">
        {takeaways.map((takeaway, i) => (
          <button
            key={i}
            onClick={() => { setSelectedTakeaway(takeaway); onTakeawaySelect?.(takeaway) }}
            className={cn(
              "w-full p-3 rounded-xl text-left text-base transition-all flex items-center gap-2",
              selectedTakeaway === takeaway
                ? "bg-blue-500/25 border border-blue-500/40 text-white"
                : "bg-white/[0.04] border border-white/[0.06] text-white/70 hover:bg-white/[0.08]"
            )}
          >
            {selectedTakeaway === takeaway && <Check className="w-4 h-4 text-blue-400 flex-shrink-0" />}
            {takeaway}
          </button>
        ))}
      </div>
    </div>
  )
}

// ============================================================================
// WONDER PHASE - Curiosity sparks and "dig deeper" options
// ============================================================================
interface WonderInteractionProps {
  relatedTopics?: string[]
  onTopicSelect?: (topic: string) => void
}

export function WonderInteraction({ relatedTopics = [], onTopicSelect }: WonderInteractionProps) {
  const defaultTopics = [
    'How does this work in nature?',
    'What do scientists say?',
    'Tell me a fun fact!'
  ]
  
  const topics = relatedTopics.length > 0 ? relatedTopics : defaultTopics
  
  return (
    <div className="w-full p-4 rounded-2xl border border-white/[0.06] bg-violet-500/[0.03]" style={{ backdropFilter: 'blur(16px)' }}>
      <div className="flex items-center gap-2.5 mb-3">
        <div className="w-9 h-9 rounded-xl bg-violet-500/20 flex items-center justify-center">
          <Brain className="w-4.5 h-4.5 text-violet-400" />
        </div>
        <div>
          <p className="text-white font-semibold text-base">Spark Your Curiosity</p>
          <p className="text-white/50 text-sm">Dig deeper into what interests you</p>
        </div>
      </div>
      
      <div className="flex flex-wrap gap-2">
        {topics.map((topic, i) => (
          <button
            key={i}
            onClick={() => onTopicSelect?.(topic)}
            className="px-3.5 py-2.5 rounded-xl bg-violet-500/15 border border-violet-500/25 text-white text-sm hover:bg-violet-500/25 transition-all active:scale-95 flex items-center gap-1.5"
          >
            <Zap className="w-3.5 h-3.5 text-violet-400" />
            {topic}
          </button>
        ))}
      </div>
    </div>
  )
}

// ============================================================================
// ACTION PHASE - Interactive challenge or activity
// ============================================================================
interface ActionInteractionProps {
  challenge?: string
  onComplete?: () => void
}

export function ActionInteraction({ challenge, onComplete }: ActionInteractionProps) {
  const [completed, setCompleted] = useState(false)
  
  const handleComplete = () => {
    setCompleted(true)
    onComplete?.()
  }
  
  return (
    <div className="w-full p-4 rounded-2xl border border-white/[0.06] bg-emerald-500/[0.03]" style={{ backdropFilter: 'blur(16px)' }}>
      <div className="flex items-center gap-2.5 mb-3">
        <div className="w-9 h-9 rounded-xl bg-emerald-500/20 flex items-center justify-center">
          <Rocket className="w-4.5 h-4.5 text-emerald-400" />
        </div>
        <div>
          <p className="text-white font-semibold text-base">Take Action!</p>
          <p className="text-white/50 text-sm">Try this mini-challenge</p>
        </div>
      </div>
      
      <p className="text-white/70 text-base mb-3 leading-relaxed">
        {challenge || 'Share what you learned with someone today. Teaching others helps you remember!'}
      </p>
      
      {!completed ? (
        <button
          onClick={handleComplete}
          className="w-full py-3.5 rounded-xl bg-emerald-500/25 border border-emerald-500/40 text-white font-semibold text-base hover:bg-emerald-500/35 transition-all active:scale-[0.98] flex items-center justify-center gap-2"
        >
          <Check className="w-5 h-5" />
          I did it!
        </button>
      ) : (
        <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="text-center py-2">
          <div className="flex items-center justify-center gap-2 text-emerald-400 font-semibold text-base">
            <Star className="w-5 h-5 fill-emerald-400" /> Amazing work! <Star className="w-5 h-5 fill-emerald-400" />
          </div>
          <p className="text-white/40 text-sm mt-1">+10 XP earned</p>
        </motion.div>
      )}
    </div>
  )
}

// ============================================================================
// WISDOM PHASE - Quote reflection and journaling prompt
// ============================================================================
interface WisdomInteractionProps {
  quote?: string
  quoteAuthor?: string
  reflectionPrompt?: string
}

export function WisdomInteraction({ quote, quoteAuthor, reflectionPrompt }: WisdomInteractionProps) {
  const [reflected, setReflected] = useState(false)
  
  return (
    <div className="w-full p-4 rounded-2xl border border-white/[0.06] bg-amber-500/[0.03]" style={{ backdropFilter: 'blur(16px)' }}>
      <div className="flex items-center gap-2.5 mb-3">
        <div className="w-9 h-9 rounded-xl bg-amber-500/20 flex items-center justify-center">
          <Heart className="w-4.5 h-4.5 text-amber-400" />
        </div>
        <div>
          <p className="text-white font-semibold text-base">Wisdom Moment</p>
          <p className="text-white/50 text-sm">Take a moment to reflect</p>
        </div>
      </div>
      
      {quote && (
        <blockquote className="mb-3 pl-3 border-l-2 border-amber-500/50">
          <p className="text-white text-lg italic leading-relaxed">"{quote}"</p>
          {quoteAuthor && <cite className="text-white/50 text-base mt-1 block not-italic">— {quoteAuthor}</cite>}
        </blockquote>
      )}
      
      <p className="text-white/60 text-base mb-3">{reflectionPrompt || 'How will you apply this wisdom today?'}</p>
      
      {!reflected ? (
        <div className="flex gap-3">
          <button onClick={() => setReflected(true)} className="flex-1 py-3 rounded-xl bg-amber-500/20 border border-amber-500/40 text-white font-semibold text-base hover:bg-amber-500/30 transition-all flex items-center justify-center gap-2">
            <ThumbsUp className="w-5 h-5" /> This resonates
          </button>
          <button onClick={() => setReflected(true)} className="flex-1 py-3 rounded-xl bg-white/[0.05] border border-white/[0.08] text-white/80 font-semibold text-base hover:bg-white/[0.1] transition-all flex items-center justify-center gap-2">
            <MessageSquare className="w-5 h-5" /> Journal it
          </button>
        </div>
      ) : (
        <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="text-center py-2 text-amber-400 font-semibold text-base">
          Wisdom captured. See you tomorrow!
        </motion.div>
      )}
    </div>
  )
}

// ============================================================================
// MAIN PHASE INTERACTION ROUTER
// ============================================================================
interface PhaseInteractionProps {
  phase: LessonPhase
  dayOfYear: number
  language: string
  hookText?: string | null
  explanation?: string | null  // The wonder_script - reveals after voting
  quote?: string
  quoteAuthor?: string
}

export default function PhaseInteraction({
  phase,
  dayOfYear,
  language,
  hookText,
  explanation,
  quote,
  quoteAuthor
}: PhaseInteractionProps) {
  switch (phase) {
    case 'hook':
      return <HookInteraction dayOfYear={dayOfYear} language={language} hookFact={hookText || undefined} explanation={explanation || undefined} />
    case 'story':
      return <StoryInteraction dayOfYear={dayOfYear} />
    case 'wonder':
      return <WonderInteraction />
    case 'action':
      return <ActionInteraction />
    case 'wisdom':
      return <WisdomInteraction quote={quote} quoteAuthor={quoteAuthor} />
    default:
      return null
  }
}
