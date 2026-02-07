'use client'

/**
 * KellyOS Chat Overlay
 * Apple-quality AI chat panel - Dark glass aesthetic
 * Connected to real AI API via Vercel AI Gateway
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel } from './index'
import { Send, Sparkles, User, Loader2, AlertCircle, RefreshCw } from 'lucide-react'

interface ChatOverlayProps {
  isOpen: boolean
  onClose: () => void
  lessonContext?: string
  lessonTitle?: string
  lessonTopic?: string
  dayOfYear?: number
  kellyAge?: number
  archetype?: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

export function ChatOverlay({ 
  isOpen, 
  onClose, 
  lessonContext,
  lessonTitle,
  lessonTopic,
  dayOfYear = 1,
  kellyAge = 30,
  archetype = 'storyteller'
}: ChatOverlayProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hi! I'm Kelly, your curious learning companion. Ask me anything about today's lesson or any topic you'd like to explore!"
    }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [questionsToday, setQuestionsToday] = useState(0)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Derive age variant from kellyAge
  const ageVariant = kellyAge < 18 ? 'kid' : kellyAge >= 60 ? 'elder' : 'adult'

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isOpen])

  // Clean up abort controller on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

  const handleSend = useCallback(async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setError(null)

    // Create abort controller for this request
    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages.filter(m => m.id !== '1'), userMessage].map(m => ({
            role: m.role,
            content: m.content
          })),
          day: dayOfYear,
          ageVariant,
          archetype,
          lessonContext,
          lessonTitle,
          lessonTopic
        }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) {
        throw new Error('Failed to get response from Kelly')
      }

      // Handle streaming response
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let assistantContent = ''
      const assistantMessageId = (Date.now() + 1).toString()

      // Add placeholder message for streaming
      setMessages(prev => [...prev, {
        id: assistantMessageId,
        role: 'assistant',
        content: ''
      }])

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          const chunk = decoder.decode(value, { stream: true })
          assistantContent += chunk
          
          // Update message with streamed content
          setMessages(prev => prev.map(m => 
            m.id === assistantMessageId 
              ? { ...m, content: assistantContent }
              : m
          ))
        }
      }

      setQuestionsToday(prev => prev + 1)
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        // Request was cancelled, ignore
        return
      }
      
      console.error('Chat error:', err)
      setError('Kelly is having trouble responding. Please try again.')
      
      // Remove the empty assistant message if streaming failed
      setMessages(prev => prev.filter(m => m.content !== ''))
    } finally {
      setIsLoading(false)
    }
  }, [input, isLoading, messages, dayOfYear, ageVariant, archetype, lessonContext, lessonTitle, lessonTopic])

  if (!isOpen) return null

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose} side="right">
      <OverlayPanel
        title="Ask Kelly"
        subtitle="Your AI learning companion"
        onClose={onClose}
      >
        <div className="flex flex-col h-[60vh]">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex gap-3",
                  message.role === 'user' && "flex-row-reverse"
                )}
              >
                <div className={cn(
                  "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
                  message.role === 'assistant' 
                    ? "bg-gradient-to-br from-emerald-500/30 to-teal-500/30 border border-emerald-500/20"
                    : "bg-white/10 border border-white/10"
                )}>
                  {message.role === 'assistant' ? (
                    <Sparkles className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <User className="w-4 h-4 text-white/70" />
                  )}
                </div>
                <div className={cn(
                  "max-w-[80%] rounded-2xl px-4 py-2.5",
                  message.role === 'assistant'
                    ? "bg-white/5 border border-white/10 text-white/90 rounded-tl-sm"
                    : "bg-emerald-500/20 border border-emerald-500/30 text-white rounded-tr-sm"
                )}>
                  <p className="text-sm leading-relaxed">{message.content}</p>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500/30 to-teal-500/30 border border-emerald-500/20 flex items-center justify-center">
                  <Sparkles className="w-4 h-4 text-emerald-400" />
                </div>
                <div className="bg-white/5 border border-white/10 rounded-2xl rounded-tl-sm px-4 py-3">
                  <Loader2 className="w-5 h-5 text-white/40 animate-spin" />
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-sm">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span className="flex-1">{error}</span>
                <button 
                  onClick={() => setError(null)}
                  className="p-1 hover:bg-red-500/20 rounded-full transition-colors"
                >
                  <RefreshCw className="w-3 h-3" />
                </button>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-white/10">
            <div className="flex items-center gap-2 bg-white/5 rounded-full border border-white/10 px-4 py-2">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask Kelly anything..."
                className="flex-1 bg-transparent outline-none text-white placeholder:text-white/40 text-sm"
                disabled={isLoading}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className={cn(
                  "p-2 rounded-full transition-all",
                  input.trim() && !isLoading
                    ? "bg-emerald-500/30 text-emerald-300 hover:bg-emerald-500/40"
                    : "bg-white/5 text-white/30"
                )}
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            <p className="text-[10px] text-white/30 text-center mt-2">
              Kelly learns from today's lesson context
            </p>
          </div>
        </div>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
