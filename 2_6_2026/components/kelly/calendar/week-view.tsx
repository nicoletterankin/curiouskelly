'use client'

import { cn } from '@/lib/utils'

interface WeekViewProps {
  viewDate: Date
  currentDay: number
  completedDays: number[]
  onDaySelect: (day: number) => void
}

export function WeekView({ 
  viewDate, 
  currentDay, 
  completedDays, 
  onDaySelect 
}: WeekViewProps) {
  const year = viewDate.getFullYear()
  
  // Get start of week (Sunday)
  const startOfWeek = new Date(viewDate)
  startOfWeek.setDate(viewDate.getDate() - viewDate.getDay())
  
  // Today's day of year
  const today = new Date()
  const todayDayOfYear = today.getFullYear() === year 
    ? Math.floor((today.getTime() - new Date(year, 0, 0).getTime()) / (1000 * 60 * 60 * 24))
    : -1

  const getDayOfYear = (date: Date) => {
    const startOfYear = new Date(date.getFullYear(), 0, 0)
    const diff = date.getTime() - startOfYear.getTime()
    return Math.floor(diff / (1000 * 60 * 60 * 24))
  }

  // Build week days
  const weekDays = []
  for (let i = 0; i < 7; i++) {
    const date = new Date(startOfWeek)
    date.setDate(startOfWeek.getDate() + i)
    const dayOfYear = getDayOfYear(date)
    weekDays.push({
      date,
      dayOfYear,
      dayName: date.toLocaleDateString('en-US', { weekday: 'short' }),
      dayNum: date.getDate(),
      monthName: date.toLocaleDateString('en-US', { month: 'short' }),
    })
  }

  // Lesson themes for the week (placeholder - should come from DB)
  const lessonThemes: Record<number, string> = {
    15: 'The Science of Sleep',
    16: 'Why We Dream', 
    17: 'Memory Formation',
    18: 'Brain Plasticity',
    19: 'Learning & Growth',
    20: 'Focus & Attention',
    21: 'Rest & Recovery',
  }

  return (
    <div className="flex flex-col h-full">
      {/* Week header with day names - DARK GLASS */}
      <div className="grid grid-cols-7 border-b border-white/10">
        {weekDays.map((day) => {
          const isToday = day.dayOfYear === todayDayOfYear
          const isCurrent = day.dayOfYear === currentDay
          
          return (
            <div 
              key={day.dayOfYear}
              className="flex flex-col items-center py-3"
            >
              <span className={cn(
                'text-[13px] font-medium',
                isToday ? 'text-[var(--kelly-blue)]' : 'text-white/40'
              )}>
                {day.dayName}
              </span>
              <span className={cn(
                'w-8 h-8 rounded-full flex items-center justify-center text-[17px] font-semibold mt-1',
                isCurrent && 'bg-[var(--kelly-blue)] text-white',
                isToday && !isCurrent && 'ring-2 ring-[var(--kelly-blue)] text-[var(--kelly-blue)]',
                !isCurrent && !isToday && 'text-white'
              )}>
                {day.dayNum}
              </span>
            </div>
          )
        })}
      </div>
      
      {/* Day columns with lesson cards - DARK GLASS */}
      <div className="flex-1 grid grid-cols-7 divide-x divide-white/[0.06]">
        {weekDays.map((day) => {
          const isToday = day.dayOfYear === todayDayOfYear
          const isCurrent = day.dayOfYear === currentDay
          const isCompleted = completedDays.includes(day.dayOfYear)
          const isFuture = day.dayOfYear > todayDayOfYear && todayDayOfYear > 0
          const theme = lessonThemes[day.dayOfYear] || "Today's Lesson"
          
          return (
            <button
              key={day.dayOfYear}
              onClick={() => onDaySelect(day.dayOfYear)}
              disabled={isFuture}
              className={cn(
                'flex flex-col p-2 text-left transition-colors',
                !isFuture && 'hover:bg-white/5 active:bg-white/10',
                isCurrent && 'bg-[var(--kelly-blue)]/20',
                isFuture && 'opacity-40'
              )}
            >
              {/* Lesson card - DARK GLASS */}
              <div className={cn(
                'p-2 rounded-lg text-[11px] leading-tight',
                isCompleted ? 'bg-[var(--kelly-blue)]/20 text-[var(--kelly-blue)]' : 'bg-white/10 text-white/70'
              )}>
                <div className="font-semibold mb-0.5 text-white/90">
                  {day.monthName} {day.dayNum}
                </div>
                <div className="line-clamp-2">
                  {theme}
                </div>
                {isCompleted && (
                  <div className="mt-1 text-[var(--kelly-blue)] font-medium">
                    Completed
                  </div>
                )}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
