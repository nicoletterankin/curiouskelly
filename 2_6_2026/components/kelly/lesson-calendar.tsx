'use client'

import { cn } from "@/lib/utils"
import { ChevronLeft, ChevronRight } from "lucide-react"

import { SheetTitle } from "@/components/ui/sheet"
import { SheetContent } from "@/components/ui/sheet"
import { Sheet } from "@/components/ui/sheet"

import * as React from 'react'
import { CalendarView } from './calendar/calendar-view'

interface LessonCalendarProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  selectedDay: number
  onDaySelect: (day: number) => void
  completedDays?: number[]
}

export function LessonCalendar({
  open,
  onOpenChange,
  selectedDay,
  onDaySelect,
  completedDays = [],
}: LessonCalendarProps) {
  if (!open) return null
  
  return (
    <CalendarView
      currentDay={selectedDay}
      onDaySelect={onDaySelect}
      completedDays={completedDays}
      onClose={() => onOpenChange(false)}
    />
  )
}
