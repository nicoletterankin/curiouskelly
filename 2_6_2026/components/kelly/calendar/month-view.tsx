'use client'

import { cn } from '@/lib/utils'
import { useLessonThumbnails } from '@/hooks/use-lesson-thumbnails'
import Image from 'next/image'

interface MonthViewProps {
  year: number
  month: number
  currentDay: number
  completedDays: number[]
  onDaySelect: (day: number) => void
}

export function MonthView({ 
  year, 
  month, 
  currentDay, 
  completedDays, 
  onDaySelect 
}: MonthViewProps) {
  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  
  // Calculate day range for this month to fetch thumbnails
  const firstDayOfMonth = new Date(year, month, 1)
  const lastDayOfMonth = new Date(year, month + 1, 0)
  const startOfYear = new Date(year, 0, 0)
  const startDayOfYear = Math.floor((firstDayOfMonth.getTime() - startOfYear.getTime()) / (1000 * 60 * 60 * 24))
  const endDayOfYear = Math.floor((lastDayOfMonth.getTime() - startOfYear.getTime()) / (1000 * 60 * 60 * 24))
  
  // Fetch thumbnails for this month's days
  const { thumbnails } = useLessonThumbnails(startDayOfYear, endDayOfYear)
  
  const getDaysInMonth = () => new Date(year, month + 1, 0).getDate()
  const getFirstDayOfMonth = () => new Date(year, month, 1).getDay()
  
  const getDayOfYear = (day: number) => {
    const date = new Date(year, month, day)
    const startOfYear = new Date(year, 0, 0)
    const diff = date.getTime() - startOfYear.getTime()
    return Math.floor(diff / (1000 * 60 * 60 * 24))
  }

  // Today's day of year
  const today = new Date()
  const todayDayOfYear = today.getFullYear() === year 
    ? Math.floor((today.getTime() - new Date(year, 0, 0).getTime()) / (1000 * 60 * 60 * 24))
    : -1

  const daysInMonth = getDaysInMonth()
  const firstDay = getFirstDayOfMonth()
  
  // Get previous month days to fill in
  const prevMonthDays = new Date(year, month, 0).getDate()
  
  // Build calendar grid
  const weeks: Array<Array<{ day: number; dayOfYear: number; isCurrentMonth: boolean }>> = []
  let currentWeek: Array<{ day: number; dayOfYear: number; isCurrentMonth: boolean }> = []
  
  // Previous month days
  for (let i = firstDay - 1; i >= 0; i--) {
    const day = prevMonthDays - i
    const date = new Date(year, month - 1, day)
    const startOfYear = new Date(year, 0, 0)
    const dayOfYear = Math.floor((date.getTime() - startOfYear.getTime()) / (1000 * 60 * 60 * 24))
    currentWeek.push({ day, dayOfYear, isCurrentMonth: false })
  }
  
  // Current month days
  for (let day = 1; day <= daysInMonth; day++) {
    const dayOfYear = getDayOfYear(day)
    currentWeek.push({ day, dayOfYear, isCurrentMonth: true })
    
    if (currentWeek.length === 7) {
      weeks.push(currentWeek)
      currentWeek = []
    }
  }
  
  // Next month days
  if (currentWeek.length > 0) {
    let nextDay = 1
    while (currentWeek.length < 7) {
      const date = new Date(year, month + 1, nextDay)
      const startOfYear = new Date(year, 0, 0)
      const dayOfYear = Math.floor((date.getTime() - startOfYear.getTime()) / (1000 * 60 * 60 * 24))
      currentWeek.push({ day: nextDay, dayOfYear, isCurrentMonth: false })
      nextDay++
    }
    weeks.push(currentWeek)
  }

  return (
    <div className="p-4">
      {/* Day headers - DARK GLASS */}
      <div className="grid grid-cols-7 mb-2">
        {dayNames.map((name) => (
          <div 
            key={name} 
            className="text-center text-[13px] font-semibold text-white/40 py-2"
          >
            {name}
          </div>
        ))}
      </div>
      
      {/* Calendar grid - DARK GLASS */}
      <div className="space-y-1">
        {weeks.map((week, weekIndex) => (
          <div key={weekIndex} className="grid grid-cols-7">
            {week.map((cell, cellIndex) => {
              const isToday = cell.dayOfYear === todayDayOfYear
              const isCurrent = cell.dayOfYear === currentDay
              const isCompleted = completedDays.includes(cell.dayOfYear)
              const isFuture = cell.dayOfYear > todayDayOfYear && todayDayOfYear > 0
              
              // Get thumbnail - try DB first, then fall back to static files
              const thumbnail = thumbnails.get(cell.dayOfYear)
              const staticFallback = cell.dayOfYear <= 90 
                ? `/visuals/hook/day-${cell.dayOfYear.toString().padStart(3, '0')}.jpg`
                : null
              const thumbUrl = thumbnail?.thumbnailUrl || staticFallback
              
              return (
                  <button
                    key={cellIndex}
                    onClick={() => onDaySelect(cell.dayOfYear)}
                    disabled={isFuture || !cell.isCurrentMonth}
                    className={cn(
                      'aspect-square flex flex-col items-center justify-center rounded-xl transition-all relative overflow-hidden',
                      cell.isCurrentMonth ? 'text-white' : 'text-white/20',
                      isCurrent && cell.isCurrentMonth && 'ring-2 ring-[var(--kelly-blue)]',
                      isToday && !isCurrent && cell.isCurrentMonth && 'ring-2 ring-inset ring-white/40',
                      !isCurrent && !isToday && cell.isCurrentMonth && !isFuture && 'hover:ring-2 hover:ring-white/20 active:scale-95',
                      isFuture && cell.isCurrentMonth && 'opacity-40'
                    )}
                  >
                    {/* Thumbnail background - show for ALL days (future just dimmed) */}
                    {thumbUrl && cell.isCurrentMonth && (
                      <Image
                        src={thumbUrl || "/placeholder.svg"}
                        alt=""
                        fill
                        className={cn("object-cover", isFuture && "opacity-50")}
                        sizes="(max-width: 768px) 14vw, 60px"
                        unoptimized={thumbUrl.startsWith('http')}
                      />
                    )}
                    
                    {/* Dark overlay for text readability */}
                    {thumbUrl && cell.isCurrentMonth && (
                      <div className={cn(
                        "absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent",
                        isFuture && "opacity-70"
                      )} />
                    )}
                    
                    {/* Day number */}
                    <span className={cn(
                      'relative z-10 text-[15px] font-medium',
                      thumbUrl && cell.isCurrentMonth && !isFuture ? 'text-white drop-shadow-md' : '',
                      isCurrent && 'font-bold'
                    )}>
                      {cell.day}
                    </span>
                    
                    {/* Completed indicator */}
                    {isCompleted && !isCurrent && cell.isCurrentMonth && (
                      <div className="absolute bottom-1 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-[var(--kelly-blue)] z-10" />
                    )}
                    
                    {/* Current day indicator */}
                    {isCurrent && cell.isCurrentMonth && (
                      <div className="absolute inset-0 bg-[var(--kelly-blue)]/30 z-0" />
                    )}
                  </button>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}
