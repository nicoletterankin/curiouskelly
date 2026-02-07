'use client'

import { motion } from 'framer-motion'
import { 
  Flame, BookOpen, Clock, Trophy, Star, Calendar,
  TrendingUp, Heart, Target, Sparkles, Zap, Award
} from 'lucide-react'
import { cn } from '@/lib/utils'

// Widget sizes like iPadOS
type WidgetSize = 'small' | 'medium' | 'large'

interface StatsWidgetProps {
  size?: WidgetSize
  className?: string
}

// Streak Widget - Shows learning streak
export function StreakWidget({ 
  streak = 0, 
  bestStreak = 0,
  size = 'small',
  className 
}: StatsWidgetProps & { streak?: number; bestStreak?: number }) {
  const isOnFire = streak >= 7

  return (
    <motion.div
      className={cn(
        "rounded-[20px] p-4 overflow-hidden relative",
        size === 'small' && "aspect-square",
        size === 'medium' && "aspect-[2/1]",
        className
      )}
      style={{
        background: isOnFire 
          ? 'linear-gradient(135deg, rgba(251, 146, 60, 0.2), rgba(239, 68, 68, 0.2))'
          : 'rgba(255, 255, 255, 0.05)',
        backdropFilter: 'blur(20px)',
        border: isOnFire ? '1px solid rgba(251, 146, 60, 0.3)' : '1px solid rgba(255, 255, 255, 0.1)'
      }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      {/* Background glow for fire state */}
      {isOnFire && (
        <div className="absolute inset-0 bg-gradient-to-br from-orange-500/10 to-red-500/10" />
      )}
      
      <div className="relative z-10 h-full flex flex-col justify-between">
        <div className="flex items-start justify-between">
          <div className={cn(
            "w-10 h-10 rounded-xl flex items-center justify-center",
            isOnFire ? "bg-gradient-to-br from-orange-500 to-red-500" : "bg-white/10"
          )}>
            <Flame className={cn("w-5 h-5", isOnFire ? "text-white" : "text-orange-400")} />
          </div>
          {isOnFire && (
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="text-orange-400 text-xs font-bold"
            >
              ON FIRE
            </motion.div>
          )}
        </div>
        
        <div>
          <div className={cn(
            "text-3xl font-bold",
            isOnFire ? "text-orange-400" : "text-white"
          )}>
            {streak}
          </div>
          <div className="text-white/50 text-xs">Day streak</div>
          {bestStreak > 0 && (
            <div className="text-white/30 text-[10px] mt-1">Best: {bestStreak} days</div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// Progress Widget - Shows yearly progress
export function ProgressWidget({ 
  currentDay = 1,
  totalDays = 365,
  size = 'medium',
  className 
}: StatsWidgetProps & { currentDay?: number; totalDays?: number }) {
  const progress = (currentDay / totalDays) * 100
  const circumference = 2 * Math.PI * 40

  return (
    <motion.div
      className={cn(
        "rounded-[20px] p-4 overflow-hidden",
        size === 'medium' && "aspect-[2/1]",
        className
      )}
      style={{
        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15))',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(59, 130, 246, 0.2)'
      }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="h-full flex items-center gap-4">
        {/* Circular progress */}
        <div className="relative w-20 h-20 flex-shrink-0">
          <svg className="w-full h-full -rotate-90">
            <circle
              cx="40"
              cy="40"
              r="36"
              fill="none"
              stroke="rgba(255,255,255,0.1)"
              strokeWidth="6"
            />
            <motion.circle
              cx="40"
              cy="40"
              r="36"
              fill="none"
              stroke="url(#progressGradient)"
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={circumference}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset: circumference * (1 - progress / 100) }}
              transition={{ duration: 1, ease: 'easeOut' }}
            />
            <defs>
              <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#3b82f6" />
                <stop offset="100%" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-white font-bold text-lg">{Math.round(progress)}%</span>
          </div>
        </div>
        
        <div className="flex-1">
          <div className="text-white/50 text-xs uppercase tracking-wider mb-1">Year Progress</div>
          <div className="text-white font-semibold">
            Day {currentDay} of {totalDays}
          </div>
          <div className="text-white/40 text-sm mt-1">
            {totalDays - currentDay} days remaining
          </div>
        </div>
      </div>
    </motion.div>
  )
}

// Lessons Completed Widget
export function LessonsWidget({ 
  completed = 0,
  total = 365,
  favorites = 0,
  size = 'small',
  className 
}: StatsWidgetProps & { completed?: number; total?: number; favorites?: number }) {
  return (
    <motion.div
      className={cn(
        "rounded-[20px] p-4 overflow-hidden",
        size === 'small' && "aspect-square",
        className
      )}
      style={{
        background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.15))',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(34, 197, 94, 0.2)'
      }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="h-full flex flex-col justify-between">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center">
          <BookOpen className="w-5 h-5 text-white" />
        </div>
        
        <div>
          <div className="text-3xl font-bold text-green-400">{completed}</div>
          <div className="text-white/50 text-xs">Lessons completed</div>
          {favorites > 0 && (
            <div className="flex items-center gap-1 text-white/30 text-[10px] mt-1">
              <Heart className="w-3 h-3 fill-current" />
              {favorites} saved
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// Time Spent Widget
export function TimeWidget({ 
  minutes = 0,
  size = 'small',
  className 
}: StatsWidgetProps & { minutes?: number }) {
  const hours = Math.floor(minutes / 60)
  const mins = minutes % 60

  return (
    <motion.div
      className={cn(
        "rounded-[20px] p-4 overflow-hidden",
        size === 'small' && "aspect-square",
        className
      )}
      style={{
        background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.15), rgba(219, 39, 119, 0.15))',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(236, 72, 153, 0.2)'
      }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="h-full flex flex-col justify-between">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-pink-500 to-rose-500 flex items-center justify-center">
          <Clock className="w-5 h-5 text-white" />
        </div>
        
        <div>
          <div className="text-3xl font-bold text-pink-400">
            {hours > 0 ? `${hours}h` : `${mins}m`}
          </div>
          <div className="text-white/50 text-xs">Time learning</div>
          {hours > 0 && mins > 0 && (
            <div className="text-white/30 text-[10px] mt-1">{mins}m today</div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// Achievement Widget
export function AchievementWidget({ 
  unlocked = 0,
  total = 50,
  latest,
  size = 'medium',
  className 
}: StatsWidgetProps & { unlocked?: number; total?: number; latest?: string }) {
  return (
    <motion.div
      className={cn(
        "rounded-[20px] p-4 overflow-hidden",
        size === 'medium' && "aspect-[2/1]",
        className
      )}
      style={{
        background: 'linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.15))',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(251, 191, 36, 0.2)'
      }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="h-full flex items-center gap-4">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-amber-500 to-yellow-500 flex items-center justify-center flex-shrink-0">
          <Trophy className="w-8 h-8 text-white" />
        </div>
        
        <div className="flex-1">
          <div className="text-white/50 text-xs uppercase tracking-wider mb-1">Achievements</div>
          <div className="text-2xl font-bold text-amber-400">
            {unlocked} / {total}
          </div>
          {latest && (
            <div className="flex items-center gap-1.5 mt-2 text-white/60 text-sm">
              <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" />
              <span className="truncate">{latest}</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// Stats Dashboard - Combines multiple widgets
export function StatsDashboard({
  streak = 0,
  bestStreak = 0,
  currentDay = 1,
  lessonsCompleted = 0,
  minutesLearned = 0,
  achievements = 0,
  favorites = 0,
  className,
}: {
  streak?: number
  bestStreak?: number
  currentDay?: number
  lessonsCompleted?: number
  minutesLearned?: number
  achievements?: number
  favorites?: number
  className?: string
}) {
  return (
    <div className={cn("grid grid-cols-2 gap-3", className)}>
      {/* Progress - Large spanning both columns */}
      <ProgressWidget 
        currentDay={currentDay} 
        size="medium"
        className="col-span-2"
      />
      
      {/* Streak and Lessons - Small side by side */}
      <StreakWidget streak={streak} bestStreak={bestStreak} size="small" />
      <LessonsWidget completed={lessonsCompleted} favorites={favorites} size="small" />
      
      {/* Time and Achievement */}
      <TimeWidget minutes={minutesLearned} size="small" />
      <motion.div
        className="rounded-[20px] p-4 aspect-square"
        style={{
          background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(124, 58, 237, 0.15))',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(139, 92, 246, 0.2)'
        }}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="h-full flex flex-col justify-between">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
            <Award className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="text-3xl font-bold text-violet-400">{achievements}</div>
            <div className="text-white/50 text-xs">Achievements</div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

// Compact inline stats bar
export function StatsBar({
  streak = 0,
  currentDay = 1,
  progress = 0,
  className,
}: {
  streak?: number
  currentDay?: number
  progress?: number
  className?: string
}) {
  return (
    <div className={cn(
      "flex items-center gap-4 px-4 py-2 rounded-full",
      className
    )}
    style={{
      background: 'rgba(255, 255, 255, 0.05)',
      backdropFilter: 'blur(20px)',
      border: '1px solid rgba(255, 255, 255, 0.1)'
    }}
    >
      <div className="flex items-center gap-1.5">
        <Flame className={cn("w-4 h-4", streak > 0 ? "text-orange-400" : "text-white/40")} />
        <span className="text-white/80 text-sm font-medium">{streak}</span>
      </div>
      <div className="w-px h-4 bg-white/10" />
      <div className="flex items-center gap-1.5">
        <Calendar className="w-4 h-4 text-blue-400" />
        <span className="text-white/80 text-sm font-medium">Day {currentDay}</span>
      </div>
      <div className="w-px h-4 bg-white/10" />
      <div className="flex items-center gap-1.5">
        <Target className="w-4 h-4 text-green-400" />
        <span className="text-white/80 text-sm font-medium">{progress}%</span>
      </div>
    </div>
  )
}
