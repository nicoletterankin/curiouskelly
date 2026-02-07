"use client"

import { useState, useEffect } from "react"
import Link from "next/link"

const AGES = [
  { id: "child", label: "Child (2-7)", emoji: "👒" },
  { id: "youth", label: "Youth (8-14)", emoji: "🎒" },
  { id: "adult", label: "Adult (15-44)", emoji: "💼" },
  { id: "mature", label: "Mature (45-64)", emoji: "👔" },
  { id: "elder", label: "Elder (65+)", emoji: "📚" },
]

const LANGUAGES = [
  { code: "en", label: "English", flag: "🇺🇸" },
  { code: "es", label: "Español", flag: "🇪🇸" },
  { code: "fr", label: "Français", flag: "🇫🇷" },
  { code: "de", label: "Deutsch", flag: "🇩🇪" },
  { code: "pt", label: "Português", flag: "🇧🇷" },
  { code: "zh", label: "中文", flag: "🇨🇳" },
]

const ARCHETYPES = [
  { id: "explorer", name: "Explorer", emoji: "🧭", desc: "Discovery, adventure - lessons as journeys" },
  { id: "sage", name: "Sage", emoji: "📚", desc: "Wisdom, understanding - lessons as insights" },
  { id: "hero", name: "Hero", emoji: "⚔️", desc: "Achievement, mastery - lessons as challenges" },
  { id: "caregiver", name: "Caregiver", emoji: "💗", desc: "Helping others - lessons as tools to support" },
  { id: "creator", name: "Creator", emoji: "🎨", desc: "Making new things - lessons as building blocks" },
  { id: "rebel", name: "Rebel", emoji: "⚡", desc: "Independence, change - lessons as tools for freedom" },
  { id: "lover", name: "Lover", emoji: "💕", desc: "Connection, intimacy - lessons as ways to bond" },
  { id: "jester", name: "Jester", emoji: "🎭", desc: "Joy, humor - lessons as playful discoveries" },
  { id: "everyman", name: "Everyman", emoji: "🤝", desc: "Belonging, connection - lessons as shared experience" },
  { id: "ruler", name: "Ruler", emoji: "👑", desc: "Control, order - lessons as leadership tools" },
  { id: "magician", name: "Magician", emoji: "✨", desc: "Transformation - lessons as powers to wield" },
  { id: "innocent", name: "Innocent", emoji: "🌸", desc: "Safety, happiness - lessons as paths to peace" },
]

const QUIZ_QUESTIONS = [
  { q: "When learning something new, I prefer to:", a: "Dive in and explore", b: "Seek wisdom from experts", archetypeA: "explorer", archetypeB: "sage" },
  { q: "I feel most fulfilled when I:", a: "Help others feel cared for", b: "Challenge the status quo", archetypeA: "caregiver", archetypeB: "rebel" },
  { q: "In a group project, I naturally:", a: "Lead and organize", b: "Transform and innovate", archetypeA: "ruler", archetypeB: "magician" },
  { q: "My ideal weekend involves:", a: "Creating something new", b: "Having fun and laughing", archetypeA: "creator", archetypeB: "jester" },
  { q: "When facing a problem, I:", a: "Connect with others for support", b: "Rise to the challenge heroically", archetypeA: "lover", archetypeB: "hero" },
  { q: "I find meaning through:", a: "Simple pleasures and peace", b: "Belonging to a community", archetypeA: "innocent", archetypeB: "everyman" },
]

export default function SettingsPage() {
  const [age, setAge] = useState("adult")
  const [language, setLanguage] = useState("en")
  const [archetype, setArchetype] = useState<string | null>(null)
  const [notifications, setNotifications] = useState({ daily: true, weekly: false, achievements: true })
  const [showQuiz, setShowQuiz] = useState(false)
  const [quizStep, setQuizStep] = useState(0)
  const [quizAnswers, setQuizAnswers] = useState<string[]>([])
  const [saved, setSaved] = useState(false)
  
  // Load settings from localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      setAge(localStorage.getItem('kelly_age') || 'adult')
      setLanguage(localStorage.getItem('kelly_language') || 'en')
      setArchetype(localStorage.getItem('kelly_archetype'))
      const notifs = localStorage.getItem('kelly_notifications')
      if (notifs) setNotifications(JSON.parse(notifs))
    }
  }, [])
  
  // Save settings
  const saveSettings = () => {
    localStorage.setItem('kelly_age', age)
    localStorage.setItem('kelly_language', language)
    if (archetype) localStorage.setItem('kelly_archetype', archetype)
    localStorage.setItem('kelly_notifications', JSON.stringify(notifications))
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }
  
  // Handle quiz answer
  const handleQuizAnswer = (answer: 'a' | 'b') => {
    const question = QUIZ_QUESTIONS[quizStep]
    const selectedArchetype = answer === 'a' ? question.archetypeA : question.archetypeB
    const newAnswers = [...quizAnswers, selectedArchetype]
    setQuizAnswers(newAnswers)
    
    if (quizStep < QUIZ_QUESTIONS.length - 1) {
      setQuizStep(quizStep + 1)
    } else {
      // Calculate result - most common archetype
      const counts: Record<string, number> = {}
      newAnswers.forEach(a => { counts[a] = (counts[a] || 0) + 1 })
      const result = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]
      setArchetype(result)
      setShowQuiz(false)
      setQuizStep(0)
      setQuizAnswers([])
    }
  }
  
  // Export data
  const exportData = () => {
    const data = {
      age,
      language,
      archetype,
      notifications,
      progress: localStorage.getItem('kelly_progress'),
      streak: localStorage.getItem('kelly_streak'),
      exportedAt: new Date().toISOString()
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'kelly-data-export.json'
    a.click()
    URL.revokeObjectURL(url)
  }
  
  const currentArchetype = ARCHETYPES.find(a => a.id === archetype)
  
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-white/5 px-6 py-4">
        <div className="max-w-2xl mx-auto flex items-center justify-between">
          <Link href="/play" className="font-serif text-lg hover:text-zinc-300 transition-colors">
            The Daily Lesson
          </Link>
          <nav className="flex items-center gap-6 text-sm text-zinc-500">
            <Link href="/play" className="hover:text-white transition-colors">Learn</Link>
            <Link href="/progress" className="hover:text-white transition-colors">Progress</Link>
            <Link href="/calendar" className="hover:text-white transition-colors">Calendar</Link>
          </nav>
        </div>
      </header>
      
      <main className="max-w-2xl mx-auto px-6 py-12">
        <h1 className="text-3xl font-serif mb-8">Settings</h1>
        
        {/* Age Selection */}
        <section className="mb-10">
          <h2 className="text-lg font-medium mb-4">Age Group</h2>
          <p className="text-sm text-zinc-500 mb-4">Kelly adjusts her teaching style based on your age</p>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            {AGES.map(a => (
              <button
                key={a.id}
                onClick={() => setAge(a.id)}
                className={`p-4 rounded-lg border text-center transition-all ${age === a.id ? 'bg-white/10 border-white/30' : 'bg-white/[0.02] border-white/5 hover:bg-white/5'}`}
              >
                <div className="text-2xl mb-2">{a.emoji}</div>
                <div className="text-xs text-zinc-400">{a.label}</div>
              </button>
            ))}
          </div>
        </section>
        
        {/* Language Selection */}
        <section className="mb-10">
          <h2 className="text-lg font-medium mb-4">Language</h2>
          <p className="text-sm text-zinc-500 mb-4">Choose your preferred language for lessons</p>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            {LANGUAGES.map(l => (
              <button
                key={l.code}
                onClick={() => setLanguage(l.code)}
                className={`p-4 rounded-lg border text-center transition-all ${language === l.code ? 'bg-white/10 border-white/30' : 'bg-white/[0.02] border-white/5 hover:bg-white/5'}`}
              >
                <div className="text-2xl mb-2">{l.flag}</div>
                <div className="text-xs text-zinc-400">{l.label}</div>
              </button>
            ))}
          </div>
        </section>
        
        {/* Archetype */}
        <section className="mb-10">
          <h2 className="text-lg font-medium mb-4">Learning Archetype</h2>
          <p className="text-sm text-zinc-500 mb-4">Discover how you learn best</p>
          
          {currentArchetype ? (
            <div className="bg-white/5 border border-white/10 rounded-xl p-6 mb-4">
              <div className="flex items-center gap-4">
                <div className="text-4xl">{currentArchetype.emoji}</div>
                <div>
                  <div className="text-xl font-medium">{currentArchetype.name}</div>
                  <div className="text-sm text-zinc-400">{currentArchetype.desc}</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white/[0.02] border border-white/5 rounded-xl p-6 mb-4 text-center text-zinc-500">
              Take the quiz to discover your archetype
            </div>
          )}
          
          <button
            onClick={() => { setShowQuiz(true); setQuizStep(0); setQuizAnswers([]) }}
            className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 transition-colors text-sm"
          >
            {archetype ? 'Retake Quiz' : 'Take Quiz'}
          </button>
        </section>
        
        {/* Quiz Modal */}
        {showQuiz && (
          <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
            <div className="bg-zinc-900 border border-white/10 rounded-2xl p-8 max-w-md w-full">
              <div className="text-sm text-zinc-500 mb-2">Question {quizStep + 1} of {QUIZ_QUESTIONS.length}</div>
              <h3 className="text-xl font-medium mb-6">{QUIZ_QUESTIONS[quizStep].q}</h3>
              <div className="space-y-3">
                <button
                  onClick={() => handleQuizAnswer('a')}
                  className="w-full p-4 bg-white/5 border border-white/10 rounded-lg text-left hover:bg-white/10 transition-colors"
                >
                  {QUIZ_QUESTIONS[quizStep].a}
                </button>
                <button
                  onClick={() => handleQuizAnswer('b')}
                  className="w-full p-4 bg-white/5 border border-white/10 rounded-lg text-left hover:bg-white/10 transition-colors"
                >
                  {QUIZ_QUESTIONS[quizStep].b}
                </button>
              </div>
              <button
                onClick={() => setShowQuiz(false)}
                className="mt-6 text-sm text-zinc-500 hover:text-white transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
        
        {/* Notifications */}
        <section className="mb-10">
          <h2 className="text-lg font-medium mb-4">Notifications</h2>
          <div className="space-y-3">
            <label className="flex items-center justify-between p-4 bg-white/[0.02] border border-white/5 rounded-lg cursor-pointer hover:bg-white/5 transition-colors">
              <span className="text-sm">Daily lesson reminders</span>
              <input
                type="checkbox"
                checked={notifications.daily}
                onChange={e => setNotifications({ ...notifications, daily: e.target.checked })}
                className="w-5 h-5 rounded bg-white/10 border-white/20"
              />
            </label>
            <label className="flex items-center justify-between p-4 bg-white/[0.02] border border-white/5 rounded-lg cursor-pointer hover:bg-white/5 transition-colors">
              <span className="text-sm">Weekly progress summary</span>
              <input
                type="checkbox"
                checked={notifications.weekly}
                onChange={e => setNotifications({ ...notifications, weekly: e.target.checked })}
                className="w-5 h-5 rounded bg-white/10 border-white/20"
              />
            </label>
            <label className="flex items-center justify-between p-4 bg-white/[0.02] border border-white/5 rounded-lg cursor-pointer hover:bg-white/5 transition-colors">
              <span className="text-sm">Achievement notifications</span>
              <input
                type="checkbox"
                checked={notifications.achievements}
                onChange={e => setNotifications({ ...notifications, achievements: e.target.checked })}
                className="w-5 h-5 rounded bg-white/10 border-white/20"
              />
            </label>
          </div>
        </section>
        
        {/* Data */}
        <section className="mb-10">
          <h2 className="text-lg font-medium mb-4">Your Data</h2>
          <button
            onClick={exportData}
            className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 transition-colors text-sm"
          >
            Export My Data
          </button>
        </section>
        
        {/* Save Button */}
        <div className="flex items-center gap-4">
          <button
            onClick={saveSettings}
            className="px-8 py-3 bg-white text-black font-medium rounded-lg hover:bg-zinc-200 transition-colors"
          >
            Save Settings
          </button>
          {saved && <span className="text-emerald-400 text-sm">Settings saved!</span>}
        </div>
      </main>
      
      {/* Footer */}
      <footer className="border-t border-white/5 px-6 py-4 mt-12">
        <div className="max-w-2xl mx-auto flex items-center justify-between text-xs text-zinc-600">
          <span>Lesson of the Day, PBC</span>
          <span>Private Beta</span>
        </div>
      </footer>
    </div>
  )
}
