"use client"

import { useState, useEffect } from "react"
import Link from "next/link"

// Get day of year
function getDayOfYear() {
  const now = new Date()
  const start = new Date(now.getFullYear(), 0, 0)
  const diff = now.getTime() - start.getTime()
  return Math.floor(diff / (1000 * 60 * 60 * 24))
}

// Get completion data from localStorage
function getCompletionData(): Record<number, { phases: string[], completed: boolean }> {
  if (typeof window === 'undefined') return {}
  const data = localStorage.getItem('kelly_progress')
  return data ? JSON.parse(data) : {}
}

// Get streak from localStorage
function getStreak(): number {
  if (typeof window === 'undefined') return 0
  return parseInt(localStorage.getItem('kelly_streak') || '0', 10)
}

// Calculate total time (estimate 5 min per phase)
function getTotalTime(data: Record<number, { phases: string[], completed: boolean }>): number {
  let totalPhases = 0
  Object.values(data).forEach(day => {
    totalPhases += day.phases?.length || 0
  })
  return totalPhases * 5 // 5 minutes per phase
}

export default function ProgressPage() {
  const [completionData, setCompletionData] = useState<Record<number, { phases: string[], completed: boolean }>>({})
  const [streak, setStreak] = useState(0)
  const [currentDay, setCurrentDay] = useState(1)
  
  useEffect(() => {
    setCompletionData(getCompletionData())
    setStreak(getStreak())
    setCurrentDay(getDayOfYear())
  }, [])
  
  // Calculate stats
  const completedDays = Object.values(completionData).filter(d => d.completed).length
  const partialDays = Object.values(completionData).filter(d => !d.completed && d.phases?.length > 0).length
  const totalTime = getTotalTime(completionData)
  
  // Find next incomplete day
  const nextDay = (() => {
    for (let i = 1; i <= 365; i++) {
      if (!completionData[i]?.completed) return i
    }
    return currentDay
  })()
  
  // Generate 365-day grid
  const months = [
    { name: 'Jan', days: 31 },
    { name: 'Feb', days: 29 },
    { name: 'Mar', days: 31 },
    { name: 'Apr', days: 30 },
    { name: 'May', days: 31 },
    { name: 'Jun', days: 30 },
    { name: 'Jul', days: 31 },
    { name: 'Aug', days: 31 },
    { name: 'Sep', days: 30 },
    { name: 'Oct', days: 31 },
    { name: 'Nov', days: 30 },
    { name: 'Dec', days: 31 },
  ]
  
  let dayCounter = 0
  
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-white/5 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <Link href="/play" className="font-serif text-lg hover:text-zinc-300 transition-colors">
            The Daily Lesson
          </Link>
          <nav className="flex items-center gap-6 text-sm text-zinc-500">
            <Link href="/play" className="hover:text-white transition-colors">Learn</Link>
            <Link href="/calendar" className="hover:text-white transition-colors">Calendar</Link>
            <Link href="/settings" className="hover:text-white transition-colors">Settings</Link>
          </nav>
        </div>
      </header>
      
      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Stats Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
          {/* Streak */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-6">
            <div className="text-4xl font-bold text-amber-400 mb-1">{streak}</div>
            <div className="text-sm text-zinc-500">Day Streak</div>
          </div>
          
          {/* Completed */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-6">
            <div className="text-4xl font-bold text-emerald-400 mb-1">{completedDays}</div>
            <div className="text-sm text-zinc-500">Lessons Complete</div>
          </div>
          
          {/* In Progress */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-6">
            <div className="text-4xl font-bold text-yellow-400 mb-1">{partialDays}</div>
            <div className="text-sm text-zinc-500">In Progress</div>
          </div>
          
          {/* Time */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-6">
            <div className="text-4xl font-bold text-blue-400 mb-1">{totalTime}</div>
            <div className="text-sm text-zinc-500">Minutes Learned</div>
          </div>
        </div>
        
        {/* Continue Learning CTA */}
        <div className="bg-gradient-to-r from-white/5 to-white/10 border border-white/10 rounded-xl p-8 mb-12 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-serif mb-2">Continue Your Journey</h2>
            <p className="text-zinc-400">Day {nextDay} is waiting for you</p>
          </div>
          <Link 
            href={`/play?day=${nextDay}`}
            className="px-8 py-3 bg-white text-black font-medium rounded-lg hover:bg-zinc-200 transition-colors"
          >
            Continue Learning
          </Link>
        </div>
        
        {/* 365-Day Calendar Grid */}
        <h3 className="text-lg font-medium mb-6">Your 365-Day Journey</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {months.map((month, monthIndex) => {
            const monthDays = []
            for (let d = 1; d <= month.days; d++) {
              dayCounter++
              const dayNum = dayCounter
              const dayData = completionData[dayNum]
              const isComplete = dayData?.completed
              const isPartial = !isComplete && dayData?.phases?.length > 0
              const isToday = dayNum === currentDay
              const isFuture = dayNum > currentDay
              
              monthDays.push(
                <Link
                  key={dayNum}
                  href={`/play?day=${dayNum}`}
                  className={`
                    w-6 h-6 rounded text-[10px] flex items-center justify-center transition-all
                    ${isComplete ? 'bg-emerald-500/80 text-white hover:bg-emerald-500' : ''}
                    ${isPartial ? 'bg-yellow-500/80 text-white hover:bg-yellow-500' : ''}
                    ${!isComplete && !isPartial && !isFuture ? 'bg-white/5 text-zinc-600 hover:bg-white/10' : ''}
                    ${isFuture ? 'bg-white/[0.02] text-zinc-700' : ''}
                    ${isToday ? 'ring-2 ring-white ring-offset-1 ring-offset-black' : ''}
                  `}
                  title={`Day ${dayNum}`}
                >
                  {d}
                </Link>
              )
            }
            
            return (
              <div key={month.name} className="bg-white/[0.02] border border-white/5 rounded-lg p-4">
                <div className="text-sm font-medium text-zinc-400 mb-3">{month.name}</div>
                <div className="grid grid-cols-7 gap-1">
                  {monthDays}
                </div>
              </div>
            )
          })}
        </div>
        
        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-8 text-sm text-zinc-500">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-emerald-500/80" />
            <span>Complete</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-yellow-500/80" />
            <span>In Progress</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-white/5" />
            <span>Not Started</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-white/5 ring-2 ring-white ring-offset-1 ring-offset-black" />
            <span>Today</span>
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="border-t border-white/5 px-6 py-4 mt-12">
        <div className="max-w-6xl mx-auto flex items-center justify-between text-xs text-zinc-600">
          <span>Lesson of the Day, PBC</span>
          <span>Private Beta</span>
        </div>
      </footer>
    </div>
  )
}
