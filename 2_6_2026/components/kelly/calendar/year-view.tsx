'use client'

import { cn } from '@/lib/utils'

interface YearViewProps {
  year: number
  currentDay: number
  completedDays: number[]
  onDaySelect: (day: number) => void
  onMonthSelect: (month: number) => void
}

export function YearView({ 
  year, 
  currentDay, 
  completedDays, 
  onDaySelect,
  onMonthSelect 
}: YearViewProps) {
  const monthNames = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
  ]

  const getDaysInMonth = (month: number) => {
    return new Date(year, month + 1, 0).getDate()
  }

  const getFirstDayOfMonth = (month: number) => {
    return new Date(year, month, 1).getDay()
  }

  const getDayOfYear = (month: number, day: number) => {
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

  return (
    <div className="p-4">
      <h1 className="text-[34px] font-bold text-white mb-6 px-2">{year}</h1>
      
      <div className="grid grid-cols-3 gap-6">
        {monthNames.map((monthName, monthIndex) => {
          const daysInMonth = getDaysInMonth(monthIndex)
          const firstDay = getFirstDayOfMonth(monthIndex)
          const days = []
          
          // Empty cells for days before month starts
          for (let i = 0; i < firstDay; i++) {
            days.push(<div key={`empty-${i}`} className="w-4 h-4" />)
          }
          
          // Days of month - DARK GLASS
          for (let day = 1; day <= daysInMonth; day++) {
            const dayOfYear = getDayOfYear(monthIndex, day)
            const isToday = dayOfYear === todayDayOfYear
            const isCurrent = dayOfYear === currentDay
            const isCompleted = completedDays.includes(dayOfYear)
            const isFuture = dayOfYear > todayDayOfYear && todayDayOfYear > 0
            
            days.push(
              <button
                key={day}
                onClick={() => onDaySelect(dayOfYear)}
                disabled={isFuture}
                className={cn(
                  'w-4 h-4 rounded-full text-[8px] font-medium flex items-center justify-center transition-all',
                  isCurrent && 'bg-[var(--kelly-blue)] text-white',
                  isToday && !isCurrent && 'ring-1 ring-[var(--kelly-blue)] text-[var(--kelly-blue)]',
                  isCompleted && !isCurrent && !isToday && 'bg-[var(--kelly-blue)]/80 text-white',
                  !isCurrent && !isToday && !isCompleted && !isFuture && 'text-white/70 hover:bg-white/10',
                  isFuture && 'text-white/20'
                )}
              >
                {day}
              </button>
            )
          }

          return (
            <button
              key={monthIndex}
              onClick={() => onMonthSelect(monthIndex)}
              className="text-left p-3 rounded-xl hover:bg-white/5 transition-colors"
            >
              <h3 className="text-[15px] font-semibold text-white mb-2">{monthName}</h3>
              <div className="grid grid-cols-7 gap-0.5">
                {days}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
