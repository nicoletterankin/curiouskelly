"use client"

import { useState, useEffect } from "react"
import Link from "next/link"

// Sample lesson titles by day (in production, fetch from API)
const LESSON_TITLES: Record<number, string> = {
  1: "Why Is the Sky Blue?",
  2: "The Science of Sleep",
  3: "Why We Dream",
  4: "How Memory Forms",
  5: "The Power of Questions",
  6: "Why Music Moves Us",
  7: "How Plants Communicate",
  8: "The Mystery of Time",
  9: "Why We Laugh",
  10: "How Languages Evolve",
  27: "The Art of Patience",
  32: "Why Soap Makes Bubbles",
  // Add more as content is created
}

// Get visual URL for a day (Days 1-90 have hook visuals)
const getVisualUrl = (dayOfYear: number): string | null => {
  if (dayOfYear >= 1 && dayOfYear <= 90) {
    const dayStr = dayOfYear.toString().padStart(3, '0')
    return `/visuals/hook/day-${dayStr}.jpg`
  }
  return null
}

function getDayOfYear() {
  const now = new Date()
  const start = new Date(now.getFullYear(), 0, 0)
  const diff = now.getTime() - start.getTime()
  return Math.floor(diff / (1000 * 60 * 60 * 24))
}

function getCompletionData(): Record<number, { phases: string[], completed: boolean }> {
  if (typeof window === 'undefined') return {}
  const data = localStorage.getItem('kelly_progress')
  return data ? JSON.parse(data) : {}
}

const MONTHS = [
  { name: 'January', short: 'Jan', days: 31, startDay: 1 },
  { name: 'February', short: 'Feb', days: 29, startDay: 32 },
  { name: 'March', short: 'Mar', days: 31, startDay: 61 },
  { name: 'April', short: 'Apr', days: 30, startDay: 92 },
  { name: 'May', short: 'May', days: 31, startDay: 122 },
  { name: 'June', short: 'Jun', days: 30, startDay: 153 },
  { name: 'July', short: 'Jul', days: 31, startDay: 183 },
  { name: 'August', short: 'Aug', days: 31, startDay: 214 },
  { name: 'September', short: 'Sep', days: 30, startDay: 245 },
  { name: 'October', short: 'Oct', days: 31, startDay: 275 },
  { name: 'November', short: 'Nov', days: 30, startDay: 306 },
  { name: 'December', short: 'Dec', days: 31, startDay: 336 },
]

export default function CalendarPage() {
  const [currentMonth, setCurrentMonth] = useState(new Date().getMonth())
  const [completionData, setCompletionData] = useState<Record<number, { phases: string[], completed: boolean }>>({})
  const [hoveredDay, setHoveredDay] = useState<number | null>(null)
  const [currentDay, setCurrentDay] = useState(1)
  
  useEffect(() => {
    setCompletionData(getCompletionData())
    setCurrentDay(getDayOfYear())
  }, [])
  
  const month = MONTHS[currentMonth]
  
  // Calculate what day of week the month starts on (0 = Sunday)
  const firstDayOfMonth = new Date(2024, currentMonth, 1).getDay()
  
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-white/5 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <Link href="/play" className="font-serif text-lg hover:text-zinc-300 transition-colors">
            The Daily Lesson
          </Link>
          <nav className="flex items-center gap-6 text-sm text-zinc-500">
            <Link href="/play" className="hover:text-white transition-colors">Learn</Link>
            <Link href="/progress" className="hover:text-white transition-colors">Progress</Link>
            <Link href="/settings" className="hover:text-white transition-colors">Settings</Link>
          </nav>
        </div>
      </header>
      
      <main className="max-w-4xl mx-auto px-6 py-12">
        {/* Month Navigation */}
        <div className="flex items-center justify-between mb-8">
          <button 
            onClick={() => setCurrentMonth(m => Math.max(0, m - 1))}
            className="p-2 hover:bg-white/5 rounded-lg transition-colors disabled:opacity-30"
            disabled={currentMonth === 0}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.75 19.5L8.25 12l7.5-7.5" />
            </svg>
          </button>
          
          <h1 className="text-3xl font-serif">{month.name}</h1>
          
          <button 
            onClick={() => setCurrentMonth(m => Math.min(11, m + 1))}
            className="p-2 hover:bg-white/5 rounded-lg transition-colors disabled:opacity-30"
            disabled={currentMonth === 11}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.25 4.5l7.5 7.5-7.5 7.5" />
            </svg>
          </button>
        </div>
        
        {/* Weekday Headers */}
        <div className="grid grid-cols-7 gap-2 mb-2">
          {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
            <div key={day} className="text-center text-xs text-zinc-600 py-2">
              {day}
            </div>
          ))}
        </div>
        
        {/* Calendar Grid */}
        <div className="grid grid-cols-7 gap-2">
          {/* Empty cells for days before month starts */}
          {Array.from({ length: firstDayOfMonth }).map((_, i) => (
            <div key={`empty-${i}`} className="aspect-square" />
          ))}
          
          {/* Days of the month */}
          {Array.from({ length: month.days }).map((_, i) => {
            const dayOfMonth = i + 1
            const dayOfYear = month.startDay + i
            const dayData = completionData[dayOfYear]
            const isComplete = dayData?.completed
            const isPartial = !isComplete && dayData?.phases?.length > 0
            const isToday = dayOfYear === currentDay
            const isFuture = dayOfYear > currentDay
            const title = LESSON_TITLES[dayOfYear] || "Today's Lesson"
            const visualUrl = getVisualUrl(dayOfYear)
            
            return (
              <Link
                key={dayOfMonth}
                href={`/play?day=${dayOfYear}`}
                className={`
                  aspect-square rounded-lg border transition-all relative group overflow-hidden
                  ${isComplete ? 'border-emerald-500/40' : ''}
                  ${isPartial ? 'border-yellow-500/40' : ''}
                  ${!isComplete && !isPartial && !isFuture ? 'border-white/5 hover:border-white/10' : ''}
                  ${isFuture ? 'border-white/[0.03] opacity-50' : ''}
                  ${isToday ? 'ring-2 ring-white ring-offset-2 ring-offset-black' : ''}
                `}
                onMouseEnter={() => setHoveredDay(dayOfYear)}
                onMouseLeave={() => setHoveredDay(null)}
              >
                {/* Background visual for days with images */}
                {visualUrl && !isFuture && (
                  <div 
                    className="absolute inset-0 bg-cover bg-center opacity-70 group-hover:opacity-90 transition-opacity"
                    style={{ backgroundImage: `url(${visualUrl})` }}
                  />
                )}
                {/* Gradient overlay for readability */}
                {visualUrl && !isFuture && (
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                )}
                {/* Default background for days without visuals */}
                {(!visualUrl || isFuture) && (
                  <div className={`absolute inset-0 ${isComplete ? 'bg-emerald-500/20' : isPartial ? 'bg-yellow-500/20' : 'bg-white/[0.02]'}`} />
                )}
                
                <div className="relative p-2 h-full flex flex-col z-10">
                  <span className={`text-sm font-medium ${isToday || visualUrl ? 'text-white' : 'text-zinc-400'} ${visualUrl ? 'drop-shadow-md' : ''}`}>
                    {dayOfMonth}
                  </span>
                  {isComplete && (
                    <div className="mt-auto">
                      <svg className="w-4 h-4 text-emerald-400 drop-shadow" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    </div>
                  )}
                </div>
                
                {/* Hover tooltip */}
                {hoveredDay === dayOfYear && !isFuture && (
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-zinc-900 border border-white/10 rounded-lg text-xs whitespace-nowrap z-10 pointer-events-none">
                    <div className="font-medium text-white mb-1">{title}</div>
                    <div className="text-zinc-500">
                      {isComplete ? '5/5 phases complete' : isPartial ? `${dayData?.phases?.length || 0}/5 phases` : 'Not started'}
                    </div>
                  </div>
                )}
              </Link>
            )
          })}
        </div>
        
        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-8 text-sm text-zinc-500">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-emerald-500/20 border border-emerald-500/40" />
            <span>Complete</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-yellow-500/20 border border-yellow-500/40" />
            <span>In Progress</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-white/[0.02] border border-white/5" />
            <span>Not Started</span>
          </div>
        </div>
        
        {/* Quick jump */}
        <div className="mt-12 flex items-center justify-center gap-2">
          <span className="text-sm text-zinc-600 mr-2">Jump to:</span>
          {MONTHS.map((m, i) => (
            <button
              key={m.short}
              onClick={() => setCurrentMonth(i)}
              className={`px-2 py-1 text-xs rounded transition-colors ${currentMonth === i ? 'bg-white/10 text-white' : 'text-zinc-600 hover:text-white'}`}
            >
              {m.short}
            </button>
          ))}
        </div>
      </main>
      
      {/* Footer */}
      <footer className="border-t border-white/5 px-6 py-4 mt-12">
        <div className="max-w-4xl mx-auto flex items-center justify-between text-xs text-zinc-600">
          <span>Lesson of the Day, PBC</span>
          <span>Private Beta</span>
        </div>
      </footer>
    </div>
  )
}
