'use client'

import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { 
  X, 
  Minus, 
  Maximize2,
  ChevronUp,
  ChevronDown,
  Languages,
  BookOpen,
  Volume2,
  Settings,
  Search,
  Copy,
  Check
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from '@/components/ui/dropdown-menu'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'

interface InteractiveCaptionProps {
  script: string
  currentTime: number
  isPlaying: boolean
  language: string
  onLanguageChange?: (lang: string) => void
  className?: string
}

type CaptionState = 'full' | 'compact' | 'minimized' | 'hidden'

interface WordDefinition {
  word: string
  definition: string
  pronunciation?: string
  partOfSpeech?: string
}

// Available languages
const LANGUAGES = [
  { code: 'en', name: 'English', native: 'English' },
  { code: 'es', name: 'Spanish', native: 'Español' },
  { code: 'fr', name: 'French', native: 'Français' },
  { code: 'de', name: 'German', native: 'Deutsch' },
  { code: 'zh', name: 'Chinese', native: '中文' },
  { code: 'ja', name: 'Japanese', native: '日本語' },
  { code: 'ko', name: 'Korean', native: '한국어' },
  { code: 'pt', name: 'Portuguese', native: 'Português' },
  { code: 'ar', name: 'Arabic', native: 'العربية' },
  { code: 'hi', name: 'Hindi', native: 'हिन्दी' },
]

// Font size options
const FONT_SIZES = [
  { label: 'Small', value: 'text-sm' },
  { label: 'Medium', value: 'text-base' },
  { label: 'Large', value: 'text-lg' },
  { label: 'Extra Large', value: 'text-xl' },
]

export function InteractiveCaption({
  script,
  currentTime,
  isPlaying,
  language,
  onLanguageChange,
  className,
}: InteractiveCaptionProps) {
  const [state, setState] = useState<CaptionState>('full')
  const [fontSize, setFontSize] = useState('text-base')
  const [selectedWord, setSelectedWord] = useState<string | null>(null)
  const [wordDefinition, setWordDefinition] = useState<WordDefinition | null>(null)
  const [isLoadingDefinition, setIsLoadingDefinition] = useState(false)
  const [copied, setCopied] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Parse script into sentences with timing
  const sentences = useMemo(() => {
    if (!script) return []
    const sentenceMatches = script.match(/[^.!?]+[.!?]+/g) || [script]
    const avgDuration = 3 // seconds per sentence estimate
    
    return sentenceMatches.map((sentence, index) => ({
      text: sentence.trim(),
      startTime: index * avgDuration,
      endTime: (index + 1) * avgDuration,
    }))
  }, [script])

  // Find current sentence
  const currentSentence = useMemo(() => {
    return sentences.find(
      (s) => currentTime >= s.startTime && currentTime < s.endTime
    )
  }, [sentences, currentTime])

  // Parse current sentence into words with timing
  const words = useMemo(() => {
    if (!currentSentence) return []
    const wordList = currentSentence.text.split(/\s+/)
    const sentenceDuration = currentSentence.endTime - currentSentence.startTime
    const wordDuration = sentenceDuration / wordList.length
    
    return wordList.map((word, index) => ({
      word,
      startTime: currentSentence.startTime + (index * wordDuration),
      endTime: currentSentence.startTime + ((index + 1) * wordDuration),
    }))
  }, [currentSentence])

  // Look up word definition
  const lookupWord = useCallback(async (word: string) => {
    setSelectedWord(word)
    setIsLoadingDefinition(true)
    
    // Clean the word (remove punctuation)
    const cleanWord = word.replace(/[^a-zA-Z]/g, '').toLowerCase()
    
    try {
      // Use free dictionary API
      const res = await fetch(`https://api.dictionaryapi.dev/api/v2/entries/en/${cleanWord}`)
      if (res.ok) {
        const data = await res.json()
        if (data[0]) {
          const entry = data[0]
          const meaning = entry.meanings?.[0]
          setWordDefinition({
            word: entry.word,
            definition: meaning?.definitions?.[0]?.definition || 'No definition found',
            pronunciation: entry.phonetic || entry.phonetics?.[0]?.text,
            partOfSpeech: meaning?.partOfSpeech,
          })
        }
      } else {
        setWordDefinition({
          word: cleanWord,
          definition: 'Definition not found. Try looking up a different word.',
        })
      }
    } catch {
      setWordDefinition({
        word: cleanWord,
        definition: 'Unable to look up definition. Please try again.',
      })
    }
    
    setIsLoadingDefinition(false)
  }, [])

  // Copy text to clipboard
  const copyText = useCallback(() => {
    navigator.clipboard.writeText(script)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [script])

  // Speak selected word
  const speakWord = useCallback((word: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(word)
      utterance.lang = language
      speechSynthesis.speak(utterance)
    }
  }, [language])

  if (state === 'hidden') {
    return (
      <Button
        variant="outline"
        size="sm"
        className={cn('fixed bottom-4 right-4 z-30', className)}
        onClick={() => setState('full')}
      >
        <BookOpen className="h-4 w-4 mr-2" />
        Show Captions
      </Button>
    )
  }

  if (state === 'minimized') {
    return (
      <div 
        className={cn(
          'bg-background/95 backdrop-blur-md border rounded-full',
          'px-4 py-2 flex items-center gap-3 shadow-lg',
          className
        )}
      >
        <BookOpen className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium truncate max-w-[200px]">
          {currentSentence?.text || 'Captions'}
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setState('full')}
        >
          <ChevronUp className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        'bg-background/95 backdrop-blur-md border rounded-2xl',
        'shadow-lg overflow-hidden',
        'transition-all duration-300',
        state === 'compact' ? 'max-h-24' : 'max-h-96',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b bg-muted/50">
        <div className="flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Captions</span>
        </div>
        
        <div className="flex items-center gap-1">
          {/* Language selector */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-7 w-7">
                <Languages className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Language</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {LANGUAGES.map((lang) => (
                <DropdownMenuItem
                  key={lang.code}
                  onClick={() => onLanguageChange?.(lang.code)}
                  className={cn(lang.code === language && 'bg-muted')}
                >
                  <span>{lang.native}</span>
                  <span className="ml-auto text-muted-foreground text-xs">
                    {lang.name}
                  </span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Settings */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-7 w-7">
                <Settings className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Font Size</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {FONT_SIZES.map((size) => (
                <DropdownMenuItem
                  key={size.value}
                  onClick={() => setFontSize(size.value)}
                  className={cn(size.value === fontSize && 'bg-muted')}
                >
                  {size.label}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Copy */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={copyText}
          >
            {copied ? <Check className="h-4 w-4 text-white" /> : <Copy className="h-4 w-4" />}
          </Button>

          <div className="w-px h-4 bg-border mx-1" />

          {/* Toggle compact */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setState(state === 'compact' ? 'full' : 'compact')}
          >
            {state === 'compact' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </Button>

          {/* Minimize */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setState('minimized')}
          >
            <Minus className="h-4 w-4" />
          </Button>

          {/* Close */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setState('hidden')}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Caption Content */}
      <div className={cn('px-4 py-3', state === 'compact' ? 'py-2' : 'py-3')}>
        {currentSentence ? (
          <p className={cn('leading-relaxed', fontSize)}>
            {words.map((w, i) => {
              const isActive = currentTime >= w.startTime && currentTime < w.endTime
              const isPast = currentTime >= w.endTime

              return (
                <Popover key={i}>
                  <PopoverTrigger asChild>
                    <span
                      className={cn(
                        'cursor-pointer transition-all duration-150 hover:bg-primary/10 rounded px-0.5',
                        isActive
                          ? 'text-primary font-semibold bg-primary/5'
                          : isPast
                          ? 'text-foreground'
                          : 'text-muted-foreground'
                      )}
                      onClick={() => lookupWord(w.word)}
                    >
                      {w.word}{' '}
                    </span>
                  </PopoverTrigger>
                  <PopoverContent className="w-72 p-3" align="center">
                    {isLoadingDefinition ? (
                      <div className="flex items-center justify-center py-4">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary" />
                      </div>
                    ) : wordDefinition ? (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-semibold text-lg">{wordDefinition.word}</h4>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7"
                            onClick={() => speakWord(wordDefinition.word)}
                          >
                            <Volume2 className="h-4 w-4" />
                          </Button>
                        </div>
                        {wordDefinition.pronunciation && (
                          <p className="text-sm text-muted-foreground">
                            {wordDefinition.pronunciation}
                          </p>
                        )}
                        {wordDefinition.partOfSpeech && (
                          <span className="inline-block text-xs bg-muted px-2 py-0.5 rounded">
                            {wordDefinition.partOfSpeech}
                          </span>
                        )}
                        <p className="text-sm">{wordDefinition.definition}</p>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Search className="h-4 w-4" />
                        <span className="text-sm">Click a word to look it up</span>
                      </div>
                    )}
                  </PopoverContent>
                </Popover>
              )
            })}
          </p>
        ) : (
          <p className={cn('text-center text-muted-foreground', fontSize)}>
            {isPlaying ? 'Listening...' : 'Tap play to start'}
          </p>
        )}
      </div>

      {/* Full transcript toggle */}
      {state === 'full' && script && (
        <div className="border-t px-4 py-2 max-h-32 overflow-y-auto">
          <p className="text-xs text-muted-foreground leading-relaxed">
            <span className="font-medium">Full transcript: </span>
            {script}
          </p>
        </div>
      )}
    </div>
  )
}
