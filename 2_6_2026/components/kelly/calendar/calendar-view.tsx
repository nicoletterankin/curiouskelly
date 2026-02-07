'use client'

import React from "react"

import { useState, useRef, useEffect } from 'react'
import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { YearView } from './year-view'
import { MonthView } from './month-view'
import { WeekView } from './week-view'

type ViewMode = 'year' | 'month' | 'week'

interface CalendarViewProps {
  currentDay: number
  onDaySelect: (day: number) => void
  completedDays?: number[]
  onClose: () => void
}

export function CalendarView({ 
  currentDay, 
  onDaySelect, 
  completedDays = [],
  onClose 
}: CalendarViewProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('month')
  const [viewDate, setViewDate] = useState(() => {
    const date = new Date(2026, 0, 1)
    date.setDate(currentDay)
    return date
  })
  const [isExpanded, setIsExpanded] = useState(false)
  const sheetRef = useRef<HTMLDivElement>(null)
  const dragStartY = useRef<number | null>(null)
  const currentHeight = useRef<number>(60) // Start at 60% height

  const year = viewDate.getFullYear()
  const month = viewDate.getMonth()

  const navigateMonth = (delta: number) => {
    const newDate = new Date(viewDate)
    newDate.setMonth(newDate.getMonth() + delta)
    setViewDate(newDate)
  }

  const navigateWeek = (delta: number) => {
    const newDate = new Date(viewDate)
    newDate.setDate(newDate.getDate() + (delta * 7))
    setViewDate(newDate)
  }

  const goToToday = () => {
    const today = new Date()
    const startOfYear = new Date(today.getFullYear(), 0, 0)
    const diff = today.getTime() - startOfYear.getTime()
    const dayOfYear = Math.floor(diff / (1000 * 60 * 60 * 24))
    onDaySelect(dayOfYear)
    onClose()
  }

  const handleDaySelect = (day: number) => {
    onDaySelect(day)
    onClose()
  }

  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ]

  // Handle drag to expand/collapse
  const handleTouchStart = (e: React.TouchEvent) => {
    dragStartY.current = e.touches[0].clientY
  }

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (dragStartY.current === null) return
    const deltaY = dragStartY.current - e.changedTouches[0].clientY
    dragStartY.current = null
    
    // Drag up = expand, drag down = collapse or close
    if (deltaY > 50) {
      setIsExpanded(true)
    } else if (deltaY < -50) {
      if (isExpanded) {
        setIsExpanded(false)
      } else {
        onClose()
      }
    }
  }

  // Click backdrop to close
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <div 
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm" 
      onClick={handleBackdropClick}
    >
      {/* Bottom sheet - OLD MONEY styling: black, white, minimal */}
      <div 
        ref={sheetRef}
        className={cn(
          "absolute left-0 right-0 bottom-0 rounded-t-3xl flex flex-col transition-all duration-300 ease-out",
          "bg-black border-t border-white/20",
          "shadow-[0_-8px_40px_-4px_rgba(0,0,0,0.8)]",
          isExpanded ? "h-[92%]" : "h-[60%]"
        )}
        style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
      >
        {/* Drag handle - refined */}
        <div className="flex justify-center pt-3 pb-2">
          <div className="w-10 h-1 rounded-full bg-white/20" />
        </div>
        
        {/* Header - OLD MONEY: serif typography, minimal chrome */}
        <header className="flex items-center justify-between px-5 h-14 border-b border-white/10">
          <button
            onClick={onClose}
            className="text-sm font-medium text-white/60 hover:text-white tracking-wide transition-colors"
          >
            Close
          </button>
          
          <div className="flex items-center gap-1">
            {/* View mode switcher - understated elegance */}
            <div className="flex items-center bg-white/5 border border-white/10 rounded-lg p-0.5">
              <button
                onClick={() => setViewMode('year')}
                className={cn(
                  'px-4 py-1.5 rounded-md text-xs font-medium tracking-wider uppercase transition-all',
                  viewMode === 'year' 
                    ? 'bg-white text-black' 
                    : 'text-white/50 hover:text-white/80'
                )}
              >
                Year
              </button>
              <button
                onClick={() => setViewMode('month')}
                className={cn(
                  'px-4 py-1.5 rounded-md text-xs font-medium tracking-wider uppercase transition-all',
                  viewMode === 'month' 
                    ? 'bg-white text-black' 
                    : 'text-white/50 hover:text-white/80'
                )}
              >
                Month
              </button>
              <button
                onClick={() => setViewMode('week')}
                className={cn(
                  'px-4 py-1.5 rounded-md text-xs font-medium tracking-wider uppercase transition-all',
                  viewMode === 'week' 
                    ? 'bg-white text-black' 
                    : 'text-white/50 hover:text-white/80'
                )}
              >
                Week
              </button>
            </div>
          </div>

          <button
            onClick={goToToday}
            className="text-sm font-medium text-white/60 hover:text-white tracking-wide transition-colors"
          >
            Today
          </button>
        </header>

        {/* Navigation for Month/Week views - refined typography */}
        {viewMode !== 'year' && (
          <div className="flex items-center justify-between px-5 h-14 border-b border-white/5">
            <button
              onClick={() => viewMode === 'month' ? navigateMonth(-1) : navigateWeek(-1)}
              className="p-2 -ml-2 text-white/40 hover:text-white transition-colors"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
            
            <h2 className="text-lg font-serif font-medium text-white tracking-wide">
              {viewMode === 'month' 
                ? `${monthNames[month]} ${year}`
                : `Week of ${viewDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`
              }
            </h2>
            
            <button
              onClick={() => viewMode === 'month' ? navigateMonth(1) : navigateWeek(1)}
              className="p-2 -mr-2 text-white/40 hover:text-white transition-colors"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>
        )}

        {/* Calendar Content */}
        <div className="flex-1 overflow-auto">
          {viewMode === 'year' && (
            <YearView
              year={year}
              currentDay={currentDay}
              completedDays={completedDays}
              onDaySelect={handleDaySelect}
              onMonthSelect={(m) => {
                setViewDate(new Date(year, m, 1))
                setViewMode('month')
              }}
            />
          )}
          
          {viewMode === 'month' && (
            <MonthView
              year={year}
              month={month}
              currentDay={currentDay}
              completedDays={completedDays}
              onDaySelect={handleDaySelect}
            />
          )}
          
          {viewMode === 'week' && (
            <WeekView
              viewDate={viewDate}
              currentDay={currentDay}
              completedDays={completedDays}
              onDaySelect={handleDaySelect}
            />
          )}
        </div>
      </div>
    </div>
  )
}
