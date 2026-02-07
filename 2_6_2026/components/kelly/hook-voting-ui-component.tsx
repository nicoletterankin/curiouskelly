'use client'

// Hook Phase TRUE/FALSE Cinematic UI
// ===================================
// CINEMATIC FLOW:
// 1. VOTEABLE FACT displayed: "Is this true? [TRUE FACT STATEMENT]"
// 2. Learner votes TRUE or FALSE
// 3. Results revealed with community percentages
// 4. TEACHING MOMENT: "Correct! Here's why..." or "Actually, this is TRUE!"
// 5. KELLY'S CURIOSITY SPARK: "Have you ever wondered..." (engagement question)
// 6. Nudge to continue: "Swipe up for the full story"

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { Check, X, Lightbulb, Sparkles, ChevronUp } from 'lucide-react'
import { useVote } from '@/hooks/use-vote'

interface HookVotingUIProps {
  dayOfYear: number
  language: string
  hookFact?: string | null // The TRUE fact statement learner votes on
  hookQuestion?: string | null // Kelly's curiosity question (shown AFTER voting)
  correctAnswer?: boolean | null // true or false - mix for real engagement
  teachingMoment?: string | null // Full explanation revealed after voting
}

export default function HookVotingUIComponent({ 
  dayOfYear, 
  language, 
  hookFact,
  hookQuestion,
  correctAnswer = true, // Fallback to true if DB value missing
  teachingMoment
}: HookVotingUIProps) {
  const { submitVote, userVote, isLoading, truePercent, totalVotes } = useVote(dayOfYear, 'hook')
  const [hasVoted, setHasVoted] = useState(userVote !== null)
  const [selectedAnswer, setSelectedAnswer] = useState<'true' | 'false' | null>(
    userVote as 'true' | 'false' | null
  )
  const [showTeachingMoment, setShowTeachingMoment] = useState(false)
  const [showCuriositySpark, setShowCuriositySpark] = useState(false)
  
  const falsePercent = 100 - truePercent
  const isFactTrue = correctAnswer !== false // The actual answer from the DB
  const userWasCorrect = selectedAnswer === (isFactTrue ? 'true' : 'false')
  
  // Cinematic reveal sequence after voting
  useEffect(() => {
    if (hasVoted && !showTeachingMoment) {
      // Show teaching moment after 1.5s
      const teachingTimer = setTimeout(() => setShowTeachingMoment(true), 1500)
      return () => clearTimeout(teachingTimer)
    }
  }, [hasVoted, showTeachingMoment])
  
  useEffect(() => {
    if (showTeachingMoment && !showCuriositySpark && hookQuestion) {
      // Show Kelly's curiosity spark after teaching moment
      const sparkTimer = setTimeout(() => setShowCuriositySpark(true), 2000)
      return () => clearTimeout(sparkTimer)
    }
  }, [showTeachingMoment, showCuriositySpark, hookQuestion])
  
  const handleVote = async (vote: 'true' | 'false') => {
    setSelectedAnswer(vote)
    await submitVote(vote)
    setHasVoted(true)
  }
  
  // GUARD: hook_fact MUST be a declarative statement, NEVER a question.
  // If it ends with "?" it's a hook_script question that leaked through - don't show True/False.
  const isQuestion = hookFact ? hookFact.trim().endsWith('?') : false
  const displayFact = hookFact || 'Loading today\'s fact...'
  
  return (
    <div className="mt-3 space-y-3">
      {/* ===== STAGE 1: VOTEABLE FACT CARD ===== */}
      <div 
        className="py-5 px-6 rounded-[20px]"
        style={{
          background: 'rgba(0, 0, 0, 0.45)',
          backdropFilter: 'blur(48px) saturate(180%)',
          WebkitBackdropFilter: 'blur(48px) saturate(180%)',
          border: '0.5px solid rgba(255,255,255,0.1)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.2), inset 0 0.5px 0 rgba(255,255,255,0.06)',
        }}
      >
        {/* The TRUE fact to vote on */}
        <div className="text-center mb-5">
          <div className="inline-flex items-center gap-1.5 text-white/35 text-[9px] uppercase tracking-[0.2em] mb-3 font-semibold">
            <span>{isQuestion ? 'Today\'s curiosity' : 'Is this true?'}</span>
          </div>
          <p 
            className="text-white text-[16px] font-medium text-balance leading-relaxed"
            style={{ textShadow: '0 2px 8px rgba(0,0,0,0.6)' }}
          >
            {displayFact}
          </p>
        </div>
        
        {/* ===== STAGE 2: TRUE / FALSE VOTING BUTTONS ===== */}
        {!hasVoted && !isQuestion && (
          <div className="flex gap-3">
            <motion.button
              onClick={() => handleVote('true')}
              disabled={isLoading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.96 }}
              className={cn(
                "flex-1 flex items-center justify-center gap-2.5 py-3.5 rounded-2xl font-semibold text-sm tracking-wide transition-colors",
                "bg-emerald-500/15 text-emerald-300 border border-emerald-500/25",
                "hover:bg-emerald-500/25 hover:border-emerald-500/40"
              )}
            >
              <Check className="w-4 h-4" />
              <span>True</span>
            </motion.button>
            
            <motion.button
              onClick={() => handleVote('false')}
              disabled={isLoading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.96 }}
              className={cn(
                "flex-1 flex items-center justify-center gap-2.5 py-3.5 rounded-2xl font-semibold text-sm tracking-wide transition-colors",
                "bg-rose-500/15 text-rose-300 border border-rose-500/25",
                "hover:bg-rose-500/25 hover:border-rose-500/40"
              )}
            >
              <X className="w-4 h-4" />
              <span>False</span>
            </motion.button>
          </div>
        )}
        {/* If it's a question (data error), show it as a curiosity prompt instead */}
        {!hasVoted && isQuestion && (
          <div className="pt-2 text-center">
            <div className="inline-flex items-center gap-1.5 text-white/40 text-xs">
              <ChevronUp className="w-3.5 h-3.5 animate-bounce" />
              <span>Swipe up to discover the answer</span>
              <ChevronUp className="w-3.5 h-3.5 animate-bounce" />
            </div>
          </div>
        )}
        
        {/* ===== STAGE 3: RESULTS REVEAL ===== */}
        <AnimatePresence>
          {hasVoted && totalVotes > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="overflow-hidden"
            >
              <div className="flex gap-2 mb-2">
                <div className={cn(
                  "flex-1 py-2 px-3 rounded-xl text-center text-xs font-semibold transition-all",
                  selectedAnswer === 'true' 
                    ? "bg-emerald-500/30 text-emerald-300 border border-emerald-500/40" 
                    : "bg-white/[0.03] text-white/40"
                )}>
                  True {truePercent}%
                </div>
                <div className={cn(
                  "flex-1 py-2 px-3 rounded-xl text-center text-xs font-semibold transition-all",
                  selectedAnswer === 'false' 
                    ? "bg-rose-500/30 text-rose-300 border border-rose-500/40" 
                    : "bg-white/[0.03] text-white/40"
                )}>
                  False {falsePercent}%
                </div>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden bg-white/[0.08] flex">
                <div 
                  className="h-full bg-emerald-500/70 transition-all duration-500"
                  style={{ width: `${truePercent}%` }}
                />
                <div 
                  className="h-full bg-rose-500/70 transition-all duration-500"
                  style={{ width: `${falsePercent}%` }}
                />
              </div>
              <p className="text-white/40 text-[10px] text-center mt-1.5 tracking-wide">
                {totalVotes.toLocaleString()} votes
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ===== STAGE 4: TEACHING MOMENT - Explains WHY it's true ===== */}
      <AnimatePresence>
        {showTeachingMoment && (
          <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ delay: 0.1, duration: 0.4 }}
            className={cn(
              "p-4 rounded-xl border flex gap-3",
              userWasCorrect
                ? "bg-emerald-950/40 border-emerald-500/30"
                : "bg-amber-950/40 border-amber-500/30"
            )}
            style={{
              backdropFilter: 'blur(12px)',
              WebkitBackdropFilter: 'blur(12px)',
            }}
          >
            <div className="flex-shrink-0 mt-0.5">
              <Lightbulb className={cn(
                "w-5 h-5",
                userWasCorrect ? "text-emerald-400" : "text-amber-400"
              )} />
            </div>
            
            <div className="flex-1 space-y-2">
              {userWasCorrect ? (
                <>
                  <p className="text-emerald-300 text-sm font-semibold">
                    {isFactTrue ? 'Correct! This is TRUE.' : 'Correct! This is FALSE.'}
                  </p>
                  {teachingMoment && (
                    <p className="text-white/90 text-sm leading-relaxed">
                      <span className="text-emerald-400 font-medium">Here's why: </span>
                      {teachingMoment}
                    </p>
                  )}
                </>
              ) : (
                <>
                  <p className="text-amber-300 text-sm font-semibold">
                    {isFactTrue ? 'Actually, this IS true!' : 'Actually, this is FALSE!'}
                  </p>
                  {teachingMoment && (
                    <p className="text-white/90 text-sm leading-relaxed">
                      <span className="text-amber-400 font-medium">Here's why: </span>
                      {teachingMoment}
                    </p>
                  )}
                  <p className="text-white/50 text-xs italic">
                    Many people get this wrong - now you know!
                  </p>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ===== STAGE 5: KELLY'S CURIOSITY SPARK ===== */}
      <AnimatePresence>
        {showCuriositySpark && hookQuestion && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ delay: 0.1, duration: 0.4 }}
            className="p-4 rounded-xl border border-violet-500/20 bg-violet-950/30"
            style={{
              backdropFilter: 'blur(12px)',
              WebkitBackdropFilter: 'blur(12px)',
            }}
          >
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 mt-0.5">
                <Sparkles className="w-5 h-5 text-violet-400" />
              </div>
              <div className="flex-1">
                <p className="text-violet-300 text-xs font-semibold uppercase tracking-wider mb-1.5">
                  Kelly wonders...
                </p>
                <p className="text-white/90 text-sm leading-relaxed italic">
                  "{hookQuestion}"
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ===== STAGE 6: CONTINUE NUDGE ===== */}
      <AnimatePresence>
        {showTeachingMoment && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: showCuriositySpark ? 0.5 : 1.5 }}
            className="pt-2 text-center"
          >
            <div className="inline-flex items-center gap-1.5 text-white/40 text-xs">
              <ChevronUp className="w-3.5 h-3.5 animate-bounce" />
              <span>Swipe up for the full story</span>
              <ChevronUp className="w-3.5 h-3.5 animate-bounce" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Fallback if no teaching moment available */}
      {hasVoted && !teachingMoment && !showTeachingMoment && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="p-3 rounded-xl bg-white/5 border border-white/10 text-center"
        >
          <p className={cn(
            "text-sm font-semibold mb-1",
            userWasCorrect ? "text-emerald-400" : "text-amber-400"
          )}>
            {userWasCorrect 
              ? (isFactTrue ? "Correct! This is true." : "Correct! This is false.") 
              : (isFactTrue ? "Actually, this IS true!" : "Actually, this is false!")}
          </p>
          <p className="text-white/50 text-xs">
            Swipe up to learn why
          </p>
        </motion.div>
      )}
    </div>
  )
}
