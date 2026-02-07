'use client'

import React from "react"

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Search, X, Calendar, Clock, ChevronRight, Sparkles, 
  BookOpen, Beaker, Globe, Brain, Cpu, Mountain, Plane,
  Leaf, Dna, Magnet, Command, ArrowUp, ArrowDown
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface SearchOverlayProps {
  isOpen: boolean
  onClose: () => void
  onSelectLesson?: (day: number) => void
}

// Categories with icons
const CATEGORIES = [
  { id: 'all', label: 'All', icon: Sparkles, color: 'from-purple-500 to-pink-500' },
  { id: 'physics', label: 'Physics', icon: Magnet, color: 'from-blue-500 to-cyan-500' },
  { id: 'biology', label: 'Biology', icon: Dna, color: 'from-green-500 to-emerald-500' },
  { id: 'astronomy', label: 'Space', icon: Globe, color: 'from-indigo-500 to-purple-500' },
  { id: 'psychology', label: 'Mind', icon: Brain, color: 'from-pink-500 to-rose-500' },
  { id: 'technology', label: 'Tech', icon: Cpu, color: 'from-orange-500 to-amber-500' },
  { id: 'geology', label: 'Earth', icon: Mountain, color: 'from-amber-500 to-yellow-500' },
]

// Sample lesson topics for search - expanded
const SAMPLE_LESSONS = [
  { day: 1, title: "Why is the sky blue?", category: "physics", keywords: ['light', 'scattering', 'atmosphere'] },
  { day: 2, title: "How do plants make food?", category: "biology", keywords: ['photosynthesis', 'leaves', 'sunlight'] },
  { day: 3, title: "What are stars made of?", category: "astronomy", keywords: ['nuclear', 'fusion', 'hydrogen'] },
  { day: 4, title: "Why do we dream?", category: "psychology", keywords: ['sleep', 'brain', 'REM'] },
  { day: 5, title: "How does the internet work?", category: "technology", keywords: ['network', 'data', 'packets'] },
  { day: 6, title: "What causes earthquakes?", category: "geology", keywords: ['tectonic', 'plates', 'faults'] },
  { day: 7, title: "How do airplanes fly?", category: "physics", keywords: ['lift', 'thrust', 'aerodynamics'] },
  { day: 8, title: "Why do leaves change color?", category: "biology", keywords: ['chlorophyll', 'autumn', 'pigments'] },
  { day: 9, title: "What is DNA?", category: "biology", keywords: ['genes', 'heredity', 'double helix'] },
  { day: 10, title: "How do magnets work?", category: "physics", keywords: ['electrons', 'poles', 'attraction'] },
  { day: 11, title: "What causes thunder?", category: "physics", keywords: ['lightning', 'sound', 'electricity'] },
  { day: 12, title: "How do bees make honey?", category: "biology", keywords: ['nectar', 'hive', 'enzymes'] },
  { day: 13, title: "Why is the ocean salty?", category: "geology", keywords: ['minerals', 'rocks', 'evaporation'] },
  { day: 14, title: "How do vaccines work?", category: "biology", keywords: ['immunity', 'antibodies', 'protection'] },
  { day: 15, title: "What is a black hole?", category: "astronomy", keywords: ['gravity', 'singularity', 'event horizon'] },
]

// Quick actions for Spotlight
const QUICK_ACTIONS = [
  { id: 'today', label: "Today's Lesson", icon: Calendar, action: 'today' },
  { id: 'random', label: 'Random Lesson', icon: Sparkles, action: 'random' },
  { id: 'favorites', label: 'My Favorites', icon: BookOpen, action: 'favorites' },
]

export function SearchOverlay({ isOpen, onClose, onSelectLesson }: SearchOverlayProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(SAMPLE_LESSONS.slice(0, 5))
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  // Focus input on open
  useEffect(() => {
    if (isOpen) {
      setQuery('')
      setSelectedCategory('all')
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isOpen])

  // Filter results
  useEffect(() => {
    let filtered = SAMPLE_LESSONS
    
    // Filter by category
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(l => l.category === selectedCategory)
    }
    
    // Filter by search query
    if (query.trim()) {
      const q = query.toLowerCase()
      filtered = filtered.filter(lesson => 
        lesson.title.toLowerCase().includes(q) ||
        lesson.category.toLowerCase().includes(q) ||
        lesson.keywords.some(k => k.toLowerCase().includes(q))
      )
    }
    
    setResults(filtered.slice(0, 8))
    setSelectedIndex(0)
  }, [query, selectedCategory])

  // Keyboard navigation
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex(i => Math.min(i + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex(i => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && results[selectedIndex]) {
      e.preventDefault()
      handleSelect(results[selectedIndex].day)
    } else if (e.key === 'Escape') {
      onClose()
    }
  }, [results, selectedIndex, onClose])

  const handleSelect = (day: number) => {
    onSelectLesson?.(day)
    onClose()
  }

  const handleQuickAction = (action: string) => {
    if (action === 'today') {
      const today = Math.floor((Date.now() - new Date('2026-01-01').getTime()) / (1000 * 60 * 60 * 24)) + 1
      handleSelect(Math.max(1, Math.min(365, today)))
    } else if (action === 'random') {
      handleSelect(Math.floor(Math.random() * 365) + 1)
    }
  }

  const getCategoryIcon = (category: string) => {
    const cat = CATEGORIES.find(c => c.id === category)
    return cat ? cat.icon : BookOpen
  }

  const getCategoryColor = (category: string) => {
    const cat = CATEGORIES.find(c => c.id === category)
    return cat ? cat.color : 'from-gray-500 to-gray-600'
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop - iPadOS style blur */}
          <motion.div
            className="fixed inset-0 z-[300]"
            style={{ 
              background: 'rgba(0, 0, 0, 0.5)',
              backdropFilter: 'blur(40px) saturate(180%)',
              WebkitBackdropFilter: 'blur(40px) saturate(180%)'
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Spotlight Panel - centered like macOS/iPadOS */}
          <motion.div
            className="fixed inset-x-4 top-[12vh] max-w-[600px] mx-auto z-[301]"
            initial={{ opacity: 0, y: -30, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -30, scale: 0.9 }}
            transition={{ type: 'spring', damping: 30, stiffness: 400 }}
            onKeyDown={handleKeyDown}
          >
            <div 
              className="rounded-[24px] overflow-hidden"
              style={{
                background: 'rgba(30, 30, 32, 0.92)',
                backdropFilter: 'blur(60px) saturate(180%)',
                WebkitBackdropFilter: 'blur(60px) saturate(180%)',
                boxShadow: '0 25px 80px -12px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
              }}
            >
              {/* Search Input - Large Spotlight style */}
              <div className="flex items-center gap-4 px-5 py-5 border-b border-white/5">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                  <Search className="w-5 h-5 text-white" />
                </div>
                <input
                  ref={inputRef}
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search lessons, topics, or keywords..."
                  className="flex-1 bg-transparent text-white placeholder:text-white/30 outline-none text-xl font-light tracking-tight"
                />
                {query && (
                  <button
                    onClick={() => setQuery('')}
                    className="p-2 rounded-full hover:bg-white/10 transition-colors"
                  >
                    <X className="w-4 h-4 text-white/40" />
                  </button>
                )}
              </div>

              {/* Category Pills - Horizontal scroll */}
              <div className="px-4 py-3 border-b border-white/5 overflow-x-auto scrollbar-hide">
                <div className="flex gap-2">
                  {CATEGORIES.map((cat) => {
                    const Icon = cat.icon
                    const isActive = selectedCategory === cat.id
                    return (
                      <button
                        key={cat.id}
                        onClick={() => setSelectedCategory(cat.id)}
                        className={cn(
                          "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium whitespace-nowrap transition-all",
                          isActive 
                            ? "bg-white text-black" 
                            : "bg-white/10 text-white/70 hover:bg-white/15"
                        )}
                      >
                        <Icon className="w-3.5 h-3.5" />
                        {cat.label}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Quick Actions - When no query */}
              {!query && (
                <div className="px-4 py-3 border-b border-white/5">
                  <div className="text-xs text-white/30 uppercase tracking-wider font-medium mb-2 px-1">Quick Actions</div>
                  <div className="flex gap-2">
                    {QUICK_ACTIONS.map((action) => {
                      const Icon = action.icon
                      return (
                        <button
                          key={action.id}
                          onClick={() => handleQuickAction(action.action)}
                          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
                        >
                          <Icon className="w-4 h-4 text-blue-400" />
                          <span className="text-white/80 text-sm">{action.label}</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Results */}
              <div className="max-h-[50vh] overflow-y-auto">
                {results.length > 0 ? (
                  <div className="py-2">
                    <div className="px-5 py-2 text-xs text-white/30 uppercase tracking-wider font-medium">
                      {query ? `${results.length} Results` : 'Suggested Lessons'}
                    </div>
                    {results.map((lesson, index) => {
                      const Icon = getCategoryIcon(lesson.category)
                      const isSelected = index === selectedIndex
                      return (
                        <motion.button
                          key={lesson.day}
                          onClick={() => handleSelect(lesson.day)}
                          onMouseEnter={() => setSelectedIndex(index)}
                          className={cn(
                            "w-full flex items-center gap-4 px-5 py-3 transition-colors text-left",
                            isSelected ? "bg-white/10" : "hover:bg-white/5"
                          )}
                          initial={false}
                          animate={{ 
                            backgroundColor: isSelected ? 'rgba(255,255,255,0.1)' : 'transparent' 
                          }}
                        >
                          <div className={cn(
                            "w-11 h-11 rounded-2xl flex items-center justify-center flex-shrink-0 bg-gradient-to-br",
                            getCategoryColor(lesson.category)
                          )}>
                            <Icon className="w-5 h-5 text-white" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-white font-medium truncate">{lesson.title}</div>
                            <div className="text-white/40 text-sm flex items-center gap-2">
                              <span className="capitalize">{lesson.category}</span>
                              <span className="w-1 h-1 rounded-full bg-white/30" />
                              <span>Day {lesson.day}</span>
                            </div>
                          </div>
                          {isSelected && (
                            <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-white/10">
                              <span className="text-white/50 text-xs">Open</span>
                              <ChevronRight className="w-3 h-3 text-white/50" />
                            </div>
                          )}
                        </motion.button>
                      )
                    })}
                  </div>
                ) : (
                  <div className="py-16 text-center">
                    <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-4">
                      <Search className="w-7 h-7 text-white/20" />
                    </div>
                    <p className="text-white/50 font-medium">No lessons found</p>
                    <p className="text-white/30 text-sm mt-1">Try a different search term or category</p>
                  </div>
                )}
              </div>

              {/* Footer - Keyboard hints like Spotlight */}
              <div className="px-5 py-3 border-t border-white/5 flex items-center justify-between text-xs text-white/30">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1">
                    <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-white/50 font-mono">
                      <ArrowUp className="w-3 h-3 inline" />
                    </kbd>
                    <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-white/50 font-mono">
                      <ArrowDown className="w-3 h-3 inline" />
                    </kbd>
                    <span className="ml-1">Navigate</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-white/50 font-mono text-[10px]">Enter</kbd>
                    <span className="ml-1">Open</span>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-white/50 font-mono">
                    <Command className="w-3 h-3 inline" />
                  </kbd>
                  <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-white/50 font-mono text-[10px]">K</kbd>
                  <span className="ml-1">to open</span>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
