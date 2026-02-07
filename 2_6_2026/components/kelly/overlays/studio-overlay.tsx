'use client'

import React, { useState, useRef } from 'react'
import useSWR, { mutate } from 'swr'
import { cn } from '@/lib/utils'
import { Sparkles, Pause, Play, Send, Target, Shield, BookOpen, X, CheckCircle2, Circle, Eye, Undo2, Grid3X3, Check, Rocket } from 'lucide-react'

const fetcher = (url: string) => fetch(url).then(r => r.json())

interface StudioOverlayProps {
  isOpen: boolean
  onClose: () => void
}

interface KellyRule {
  id: string
  text: string
}

interface ProgressItem {
  id: string
  label: string
  completed: boolean
}

interface Message {
  id: string
  role: 'founder' | 'system'
  content: string
  timestamp: Date
}

const QUICK_SUGGESTIONS = [
  "Her eyes feel dead",
  "Smile feels fake", 
  "Timing is off",
  "She's not looking at me",
  "Voice sounds robotic",
  "Gestures feel unnatural"
]

const DEFAULT_PROGRESS: ProgressItem[] = [
  { id: '1', label: "Voice feels human", completed: true },
  { id: '2', label: "Lips sync naturally", completed: true },
  { id: '3', label: "Eyes feel alive", completed: false },
  { id: '4', label: "Micro-expressions present", completed: false },
  { id: '5', label: "Student would lean in, not pull back", completed: false },
]

const DEFAULT_CONSTRAINTS = [
  "We own our infrastructure",
  "No dependency on any single vendor", 
  "Kelly must feel present, not just working"
]

export function StudioOverlay({ isOpen, onClose }: StudioOverlayProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentGoal, setCurrentGoal] = useState("Break through the uncanny valley")
  const [sessionName, setSessionName] = useState("Uncanny Valley Session")
  const { data: rulesData } = useSWR(isOpen ? '/api/studio/rules' : null, fetcher)
  const dbRules = rulesData?.rules || []
  const [localRules, setLocalRules] = useState<KellyRule[]>([])
  const rules = [...dbRules.map((r: { id: string; rule_text: string }) => ({ id: r.id, text: r.rule_text })), ...localRules]
  const [progress, setProgress] = useState<ProgressItem[]>(DEFAULT_PROGRESS)
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', role: 'system', content: "Good morning! Kelly is ready. What would you like to work on today?", timestamp: new Date() },
  ])
  const [inputValue, setInputValue] = useState('')
  const [isWorking, setIsWorking] = useState(false)
  const [showConfetti, setShowConfetti] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const removeRule = async (id: string) => {
    setLocalRules(localRules.filter(r => r.id !== id))
    if (!id.startsWith('local-')) {
      try {
        await fetch('/api/studio/rules', {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id })
        })
        mutate('/api/studio/rules')
      } catch (e) {
        console.error('Failed to delete rule:', e)
      }
    }
  }

  const toggleProgress = (id: string) => {
    setProgress(progress.map(p => {
      if (p.id === id) {
        const newCompleted = !p.completed
        if (newCompleted) {
          setShowConfetti(id)
          setTimeout(() => setShowConfetti(null), 2000)
        }
        return { ...p, completed: newCompleted }
      }
      return p
    }))
  }

  const sendMessage = () => {
    if (!inputValue.trim()) return
    const newMessage: Message = {
      id: Date.now().toString(),
      role: 'founder',
      content: inputValue,
      timestamp: new Date()
    }
    setMessages([...messages, newMessage])
    setInputValue('')
    setIsWorking(true)
    setTimeout(async () => {
      const isDirective = inputValue.toLowerCase().includes('should') || 
                          inputValue.toLowerCase().includes('always') ||
                          inputValue.toLowerCase().includes('never')
      let responseContent = "I hear you. Let me adjust that for Kelly..."
      if (isDirective) {
        try {
          const ruleText = inputValue.replace(/^(kelly\s+)?/i, '').trim()
          await fetch('/api/studio/rules', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rule_text: ruleText, source_session: sessionName })
          })
          mutate('/api/studio/rules')
          responseContent = `Got it. I've saved this as a rule: "${ruleText}"`
        } catch (e) {
          console.error('Failed to create rule:', e)
        }
      }
      const response: Message = {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: responseContent,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, response])
      setIsWorking(false)
    }, 1500)
  }

  const completedCount = progress.filter(p => p.completed).length

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-[100] bg-[#0f1419]">
      {/* Header */}
      <header className="h-14 border-b border-[#2f3842] flex items-center justify-between px-6">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-[#1d9bf0]" />
          <span className="text-lg font-semibold text-white">Kelly Studio</span>
          <span className="text-xs text-[#8b98a5] ml-2">Founder Only</span>
        </div>
        <input
          type="text"
          value={sessionName}
          onChange={(e) => setSessionName(e.target.value)}
          className="bg-transparent text-center text-sm text-[#8b98a5] hover:text-white focus:text-white focus:outline-none focus:ring-1 focus:ring-[#1d9bf0] rounded px-3 py-1"
        />
        <button onClick={onClose} className="w-10 h-10 rounded-full hover:bg-white/10 flex items-center justify-center">
          <X className="w-5 h-5 text-white" />
        </button>
      </header>

      {/* Main Content */}
      <div className="flex h-[calc(100vh-3.5rem)] overflow-hidden">
        {/* Left - Preview */}
        <div className="w-[40%] border-r border-[#2f3842] flex flex-col p-4 gap-4">
          <div className="flex-1 relative rounded-xl overflow-hidden bg-[#1a1f26] border border-[#2f3842]">
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="w-32 h-32 rounded-full bg-gradient-to-br from-[#1d9bf0]/20 to-[#00ba7c]/20 border-2 border-[#2f3842] flex items-center justify-center mb-4">
                <span className="text-5xl">K</span>
              </div>
              <p className="text-[#8b98a5] text-sm">Kelly Preview</p>
            </div>
            {isWorking && (
              <div className="absolute inset-0 bg-[#0f1419]/80 flex items-center justify-center">
                <div className="w-8 h-8 border-2 border-[#1d9bf0] border-t-transparent rounded-full animate-spin" />
              </div>
            )}
          </div>
          <div className="flex items-center gap-3 px-2">
            <button onClick={() => setIsPlaying(!isPlaying)} className="w-10 h-10 rounded-full bg-[#1d9bf0] flex items-center justify-center">
              {isPlaying ? <Pause className="w-5 h-5 text-white" /> : <Play className="w-5 h-5 text-white ml-0.5" />}
            </button>
            <div className="flex-1 h-1 bg-[#2f3842] rounded-full"><div className="h-full w-1/3 bg-[#1d9bf0] rounded-full" /></div>
            <span className="text-xs text-[#8b98a5]">0:12 / 0:45</span>
          </div>
        </div>

        {/* Center - Conversation */}
        <div className="w-[40%] border-r border-[#2f3842] flex flex-col">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg) => (
              <div key={msg.id} className={cn(
                "max-w-[85%] p-3 rounded-2xl text-sm",
                msg.role === 'founder' ? "ml-auto bg-[#1d9bf0] text-white rounded-br-sm" : "mr-auto bg-[#1a1f26] border border-[#2f3842] text-white rounded-bl-sm"
              )}>
                {msg.content}
              </div>
            ))}
            {isWorking && (
              <div className="mr-auto bg-[#1a1f26] border border-[#2f3842] rounded-2xl rounded-bl-sm p-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 rounded-full bg-[#8b98a5] animate-bounce" />
                  <div className="w-2 h-2 rounded-full bg-[#8b98a5] animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 rounded-full bg-[#8b98a5] animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            )}
          </div>
          <div className="px-4 pb-2 flex flex-wrap gap-2">
            {QUICK_SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => setInputValue(s)} className="text-xs px-3 py-1.5 rounded-full bg-[#1a1f26] border border-[#2f3842] text-[#8b98a5] hover:text-white hover:border-[#1d9bf0]/50">
                {s}
              </button>
            ))}
          </div>
          <div className="p-4 border-t border-[#2f3842]">
            <div className="flex items-center gap-2 bg-[#1a1f26] border border-[#2f3842] rounded-xl px-4 py-2 focus-within:border-[#1d9bf0]/50">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="Tell Kelly what she needs..."
                className="flex-1 bg-transparent text-sm text-white placeholder:text-[#8b98a5]/50 focus:outline-none"
              />
              <button onClick={sendMessage} disabled={!inputValue.trim()} className="w-8 h-8 rounded-lg bg-[#1d9bf0] disabled:opacity-30 flex items-center justify-center">
                <Send className="w-4 h-4 text-white" />
              </button>
            </div>
          </div>
        </div>

        {/* Right - Mission Control */}
        <div className="w-[20%] flex flex-col overflow-y-auto">
          <div className="p-4 border-b border-[#2f3842]">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-[#1d9bf0]" />
              <span className="text-xs uppercase tracking-wider text-[#8b98a5]">Current Goal</span>
            </div>
            <input type="text" value={currentGoal} onChange={(e) => setCurrentGoal(e.target.value)} className="w-full bg-[#1a1f26] border border-[#2f3842] rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-[#1d9bf0]/50" />
          </div>
          <div className="p-4 border-b border-[#2f3842]">
            <div className="flex items-center gap-2 mb-3">
              <Shield className="w-4 h-4 text-[#8b98a5]" />
              <span className="text-xs uppercase tracking-wider text-[#8b98a5]">Constraints</span>
            </div>
            <ul className="space-y-2">
              {DEFAULT_CONSTRAINTS.map((c, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-[#8b98a5]"><span className="text-[#2f3842] mt-0.5">*</span>{c}</li>
              ))}
            </ul>
          </div>
          <div className="p-4 border-b border-[#2f3842]">
            <div className="flex items-center gap-2 mb-3">
              <BookOpen className="w-4 h-4 text-[#00ba7c]" />
              <span className="text-xs uppercase tracking-wider text-[#8b98a5]">Kelly Rules</span>
              <span className="ml-auto text-[10px] text-[#8b98a5]/60">{rules.length}</span>
            </div>
            <ul className="space-y-2 max-h-40 overflow-y-auto">
              {rules.map((rule) => (
                <li key={rule.id} className="flex items-start gap-2 text-xs text-white group">
                  <span className="text-[#00ba7c] mt-0.5">*</span>
                  <span className="flex-1">{rule.text}</span>
                  <button onClick={() => removeRule(rule.id)} className="opacity-0 group-hover:opacity-100 text-[#8b98a5] hover:text-red-400"><X className="w-3 h-3" /></button>
                </li>
              ))}
            </ul>
          </div>
          <div className="p-4 flex-1">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 className="w-4 h-4 text-[#1d9bf0]" />
              <span className="text-xs uppercase tracking-wider text-[#8b98a5]">Progress</span>
              <span className="ml-auto text-[10px] text-[#00ba7c]">{completedCount}/{progress.length}</span>
            </div>
            <ul className="space-y-2">
              {progress.map((item) => (
                <li key={item.id} className="relative">
                  <button onClick={() => toggleProgress(item.id)} className={cn("flex items-center gap-2 text-xs w-full text-left", item.completed ? "text-[#00ba7c]" : "text-[#8b98a5] hover:text-white")}>
                    {item.completed ? <CheckCircle2 className="w-4 h-4" /> : <Circle className="w-4 h-4" />}
                    <span className={item.completed ? "line-through opacity-70" : ""}>{item.label}</span>
                  </button>
                  {showConfetti === item.id && (
                    <div className="absolute inset-0 pointer-events-none overflow-hidden">
                      <div className="absolute top-1/2 left-4 w-1 h-1 bg-[#00ba7c] rounded-full animate-ping" />
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Bottom Actions */}
      <div className="absolute bottom-0 left-0 right-0 h-16 border-t border-[#2f3842] bg-[#0f1419] flex items-center justify-center gap-4 px-6">
        <ActionButton icon={<Eye className="w-4 h-4" />} label="Show changes" />
        <ActionButton icon={<Undo2 className="w-4 h-4" />} label="Go back" />
        <ActionButton icon={<Grid3X3 className="w-4 h-4" />} label="3 variations" />
        <ActionButton icon={<Check className="w-4 h-4" />} label="Save standard" primary />
        <ActionButton icon={<Rocket className="w-4 h-4" />} label="Ship to production" primary />
      </div>
    </div>
  )
}

function ActionButton({ icon, label, primary }: { icon: React.ReactNode; label: string; primary?: boolean }) {
  return (
    <button className={cn(
      "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
      primary ? "bg-[#1d9bf0] text-white hover:bg-[#1a8cd8]" : "bg-[#1a1f26] text-[#8b98a5] hover:text-white border border-[#2f3842] hover:border-[#1d9bf0]/50"
    )}>
      {icon}
      {label}
    </button>
  )
}
