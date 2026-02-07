"use client"

import React from "react"

import { useState, useMemo } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { 
  Calendar, 
  Clock, 
  Target, 
  AlertTriangle, 
  CheckCircle2, 
  ChevronLeft, 
  ChevronRight,
  Building2,
  Rocket,
  DollarSign,
  Users,
  Globe
} from "lucide-react"
import { DAILY_PRIORITIES, DEADLINE, PHASE_1_TARGET, formatMoney, TIERS } from "@/data/coalition-data"

// ═══════════════════════════════════════════════════════════════════════════
// CALENDAR DATA
// ═══════════════════════════════════════════════════════════════════════════

interface CalendarEvent {
  date: string
  title: string
  description: string
  type: 'critical' | 'milestone' | 'deadline' | 'target' | 'launch'
  icon: React.ReactNode
  amount?: number
}

const CALENDAR_EVENTS: CalendarEvent[] = [
  {
    date: '2026-02-01',
    title: 'GSA Third Auction Opens',
    description: 'Federal auction portal opens for final bidding round',
    type: 'critical',
    icon: <Building2 className="w-4 h-4" />,
  },
  {
    date: '2026-02-04',
    title: 'Offer Submission Deadline',
    description: 'Submit coalition offer to GSA by 5pm PT',
    type: 'deadline',
    icon: <AlertTriangle className="w-4 h-4" />,
    amount: 200_000_000,
  },
  {
    date: '2026-02-07',
    title: 'Target: Offer Acceptance',
    description: 'Expected GSA response to coalition bid',
    type: 'target',
    icon: <Target className="w-4 h-4" />,
  },
  {
    date: '2026-02-28',
    title: 'Founding Partner Close',
    description: 'Target deadline for Founding tier commitments',
    type: 'milestone',
    icon: <DollarSign className="w-4 h-4" />,
    amount: 1_500_000_000,
  },
  {
    date: '2026-03-15',
    title: 'Strategic Partner Demo',
    description: 'First Kelly demonstration at Ziggurat',
    type: 'milestone',
    icon: <Users className="w-4 h-4" />,
  },
  {
    date: '2026-06-01',
    title: 'Escrow Closes',
    description: 'Full Ziggurat ownership begins',
    type: 'critical',
    icon: <Building2 className="w-4 h-4" />,
    amount: 309_000_000,
  },
  {
    date: '2026-12-01',
    title: 'Kelly Launch',
    description: 'First lesson broadcasts from the Ziggurat',
    type: 'launch',
    icon: <Rocket className="w-4 h-4" />,
  },
  {
    date: '2028-07-01',
    title: 'LA Olympics',
    description: 'Ziggurat as global education showcase',
    type: 'milestone',
    icon: <Globe className="w-4 h-4" />,
  },
]

// ═══════════════════════════════════════════════════════════════════════════
// COUNTDOWN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

function CountdownTimer() {
  const [now, setNow] = useState(new Date())
  
  // Update every second
  useState(() => {
    const interval = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(interval)
  })
  
  const timeLeft = useMemo(() => {
    const diff = DEADLINE.getTime() - now.getTime()
    if (diff <= 0) return { days: 0, hours: 0, minutes: 0, seconds: 0, passed: true }
    
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60))
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60))
    const seconds = Math.floor((diff % (1000 * 60)) / 1000)
    
    return { days, hours, minutes, seconds, passed: false }
  }, [now])
  
  if (timeLeft.passed) {
    return (
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-6 text-center">
        <p className="text-amber-400 font-semibold">Deadline Passed</p>
        <p className="text-amber-300/70 text-sm mt-1">GSA offer window has closed</p>
      </div>
    )
  }
  
  return (
    <div className="bg-gradient-to-br from-red-900/20 to-orange-900/20 border border-red-500/30 rounded-xl p-6">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-red-400" />
        <p className="text-red-400 font-semibold text-sm uppercase tracking-wider">Time Until Deadline</p>
      </div>
      
      <div className="grid grid-cols-4 gap-3">
        {[
          { value: timeLeft.days, label: 'Days' },
          { value: timeLeft.hours, label: 'Hours' },
          { value: timeLeft.minutes, label: 'Min' },
          { value: timeLeft.seconds, label: 'Sec' },
        ].map((item) => (
          <div key={item.label} className="text-center">
            <div className="bg-black/40 rounded-lg py-3 px-2">
              <span className="text-3xl font-mono font-bold text-white tabular-nums">
                {String(item.value).padStart(2, '0')}
              </span>
            </div>
            <p className="text-xs text-zinc-500 mt-1">{item.label}</p>
          </div>
        ))}
      </div>
      
      <p className="text-center text-sm text-zinc-400 mt-4">
        Feb 4, 2026 at 5:00 PM PT
      </p>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// CALENDAR GRID
// ═══════════════════════════════════════════════════════════════════════════

function CalendarGrid() {
  const [selectedMonth, setSelectedMonth] = useState(1) // February 2026
  
  const months = [
    { month: 1, year: 2026, name: 'February 2026' },
    { month: 2, year: 2026, name: 'March 2026' },
    { month: 5, year: 2026, name: 'June 2026' },
    { month: 11, year: 2026, name: 'December 2026' },
  ]
  
  const currentMonthData = months.find(m => m.month === selectedMonth) || months[0]
  
  // Get events for the selected month
  const monthEvents = CALENDAR_EVENTS.filter(event => {
    const eventDate = new Date(event.date)
    return eventDate.getMonth() === selectedMonth && eventDate.getFullYear() === currentMonthData.year
  })
  
  // Generate calendar days
  const firstDay = new Date(currentMonthData.year, selectedMonth, 1).getDay()
  const daysInMonth = new Date(currentMonthData.year, selectedMonth + 1, 0).getDate()
  const days = Array.from({ length: daysInMonth }, (_, i) => i + 1)
  const emptyDays = Array.from({ length: firstDay }, (_, i) => null)
  
  const getEventForDay = (day: number) => {
    const dateStr = `${currentMonthData.year}-${String(selectedMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
    return CALENDAR_EVENTS.find(e => e.date === dateStr)
  }
  
  const getEventColor = (type: CalendarEvent['type']) => {
    switch (type) {
      case 'critical': return 'bg-red-500'
      case 'deadline': return 'bg-orange-500'
      case 'target': return 'bg-blue-500'
      case 'milestone': return 'bg-emerald-500'
      case 'launch': return 'bg-purple-500'
      default: return 'bg-zinc-500'
    }
  }
  
  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
      {/* Month Navigation */}
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={() => {
            const currentIndex = months.findIndex(m => m.month === selectedMonth)
            if (currentIndex > 0) setSelectedMonth(months[currentIndex - 1].month)
          }}
          className="p-2 hover:bg-zinc-800 rounded-lg transition"
          disabled={months.findIndex(m => m.month === selectedMonth) === 0}
        >
          <ChevronLeft className="w-5 h-5 text-zinc-400" />
        </button>
        
        <h3 className="text-lg font-semibold text-white">{currentMonthData.name}</h3>
        
        <button
          onClick={() => {
            const currentIndex = months.findIndex(m => m.month === selectedMonth)
            if (currentIndex < months.length - 1) setSelectedMonth(months[currentIndex + 1].month)
          }}
          className="p-2 hover:bg-zinc-800 rounded-lg transition"
          disabled={months.findIndex(m => m.month === selectedMonth) === months.length - 1}
        >
          <ChevronRight className="w-5 h-5 text-zinc-400" />
        </button>
      </div>
      
      {/* Calendar Header */}
      <div className="grid grid-cols-7 gap-1 mb-2">
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
          <div key={day} className="text-center text-xs text-zinc-500 py-2">
            {day}
          </div>
        ))}
      </div>
      
      {/* Calendar Days */}
      <div className="grid grid-cols-7 gap-1">
        {emptyDays.map((_, i) => (
          <div key={`empty-${i}`} className="aspect-square" />
        ))}
        {days.map(day => {
          const event = getEventForDay(day)
          const isToday = new Date().getDate() === day && 
                          new Date().getMonth() === selectedMonth &&
                          new Date().getFullYear() === currentMonthData.year
          
          return (
            <motion.div
              key={day}
              className={`
                aspect-square rounded-lg flex flex-col items-center justify-center relative cursor-pointer
                ${isToday ? 'bg-blue-600 text-white' : 'hover:bg-zinc-800'}
                ${event ? 'ring-2 ring-offset-2 ring-offset-zinc-900' : ''}
              `}
              style={event ? { '--tw-ring-color': getEventColor(event.type).replace('bg-', '#') } as React.CSSProperties : undefined}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className={`text-sm ${isToday ? 'font-bold' : 'text-zinc-400'}`}>{day}</span>
              {event && (
                <div className={`absolute bottom-1 w-1.5 h-1.5 rounded-full ${getEventColor(event.type)}`} />
              )}
            </motion.div>
          )
        })}
      </div>
      
      {/* Events List */}
      <div className="mt-6 space-y-3">
        <p className="text-xs text-zinc-500 uppercase tracking-wider">Events This Month</p>
        {monthEvents.length > 0 ? (
          monthEvents.map((event, i) => (
            <motion.div
              key={event.date}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex items-start gap-3 p-3 bg-zinc-800/50 rounded-lg"
            >
              <div className={`p-2 rounded-lg ${getEventColor(event.type)}/20`}>
                {event.icon}
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-white">{event.title}</p>
                <p className="text-xs text-zinc-500">{event.description}</p>
                {event.amount && (
                  <p className="text-xs text-emerald-400 mt-1 font-mono">
                    {formatMoney(event.amount)}
                  </p>
                )}
              </div>
              <span className="text-xs text-zinc-600">
                {new Date(event.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              </span>
            </motion.div>
          ))
        ) : (
          <p className="text-sm text-zinc-600 text-center py-4">No events this month</p>
        )}
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// PRIORITY TASKS
// ═══════════════════════════════════════════════════════════════════════════

function PriorityTasks() {
  const [expandedDay, setExpandedDay] = useState<string | null>(DAILY_PRIORITIES[0]?.day || null)
  
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'CRITICAL': return 'border-red-500/50 bg-red-500/5'
      case 'HIGH': return 'border-amber-500/50 bg-amber-500/5'
      default: return 'border-zinc-700 bg-zinc-900/50'
    }
  }
  
  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case 'CRITICAL': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'HIGH': return 'bg-amber-500/20 text-amber-400 border-amber-500/30'
      default: return 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30'
    }
  }
  
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-4">
        <CheckCircle2 className="w-5 h-5 text-emerald-400" />
        <p className="text-sm font-semibold text-white uppercase tracking-wider">Priority Checklist</p>
      </div>
      
      {DAILY_PRIORITIES.map((item, i) => (
        <motion.div
          key={item.day}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.05 }}
          className={`border rounded-xl overflow-hidden ${getPriorityColor(item.priority)}`}
        >
          <button
            onClick={() => setExpandedDay(expandedDay === item.day ? null : item.day)}
            className="w-full flex items-center justify-between p-4 hover:bg-white/5 transition"
          >
            <div className="flex items-center gap-3">
              <span className={`px-2 py-0.5 text-xs font-medium rounded border ${getPriorityBadge(item.priority)}`}>
                {item.priority}
              </span>
              <span className="text-white font-medium">{item.day}</span>
            </div>
            <ChevronRight 
              className={`w-4 h-4 text-zinc-500 transition-transform ${expandedDay === item.day ? 'rotate-90' : ''}`} 
            />
          </button>
          
          <AnimatePresence>
            {expandedDay === item.day && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="border-t border-zinc-800"
              >
                <ul className="p-4 space-y-2">
                  {item.tasks.map((task, j) => (
                    <li key={j} className="flex items-start gap-2 text-sm">
                      <span className="text-zinc-600 mt-0.5">
                        {task.startsWith('✓') ? '✓' : task.startsWith('⏰') ? '⏰' : '→'}
                      </span>
                      <span className={`${task.startsWith('✓') ? 'text-emerald-400' : 'text-zinc-400'}`}>
                        {task.replace(/^[✓⏰→]\s*/, '')}
                      </span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      ))}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// TIER PROGRESS
// ═══════════════════════════════════════════════════════════════════════════

function TierProgress() {
  return (
    <div className="space-y-4">
      <p className="text-xs text-zinc-500 uppercase tracking-wider">Investment Tiers</p>
      
      {TIERS.map((tier) => {
        // Calculate hypothetical progress (currently 0 committed)
        const committed = 0
        const progress = (committed / tier.targetAmount) * 100
        
        return (
          <div key={tier.id} className="bg-zinc-800/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-lg">{tier.emoji}</span>
                <span className="text-sm font-medium text-white">{tier.name}</span>
              </div>
              <span className="text-xs text-zinc-500">{formatMoney(tier.targetAmount)}</span>
            </div>
            
            <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-blue-500 to-emerald-500"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 1, ease: 'easeOut' }}
              />
            </div>
            
            <div className="flex justify-between mt-2 text-xs">
              <span className="text-zinc-500">{formatMoney(committed)} committed</span>
              <span className="text-zinc-600">{progress.toFixed(0)}%</span>
            </div>
          </div>
        )
      })}
      
      {/* Total */}
      <div className="border-t border-zinc-800 pt-4 mt-4">
        <div className="flex justify-between items-center">
          <span className="text-sm text-zinc-400">Phase 1 Target</span>
          <span className="text-lg font-bold text-white font-mono">{formatMoney(PHASE_1_TARGET)}</span>
        </div>
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════

export function InvestmentCalendar() {
  return (
    <section className="py-16 px-6 bg-black/50">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-2 mb-4">
            <Calendar className="w-6 h-6 text-blue-400" />
            <h2 className="text-3xl font-bold text-white">Investment Calendar</h2>
          </div>
          <p className="text-zinc-400 max-w-2xl mx-auto">
            Fast financing in February. Critical deadlines for the coalition to save this building 
            from demolition and transform it into the world's largest learning center.
          </p>
        </motion.div>
        
        {/* Main Grid */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Column - Countdown + Calendar */}
          <div className="lg:col-span-2 space-y-6">
            <CountdownTimer />
            <CalendarGrid />
          </div>
          
          {/* Right Column - Priority Tasks + Tier Progress */}
          <div className="space-y-6">
            <PriorityTasks />
            <TierProgress />
          </div>
        </div>
      </div>
    </section>
  )
}

export default InvestmentCalendar
