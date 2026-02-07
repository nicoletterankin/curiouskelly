'use client'

import React from "react"

/**
 * KellyOS Dictionary Overlay
 * Apple-quality word lookup and learning tool
 */

import { useState, useCallback } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel, AnimatedList, AnimatedItem } from './index'
import { Search, Volume2, BookOpen, Star, History, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface DictionaryOverlayProps {
  isOpen: boolean
  onClose: () => void
  initialWord?: string
  language?: string
}

interface Definition {
  word: string
  phonetic?: string
  partOfSpeech: string
  definition: string
  example?: string
  synonyms?: string[]
}

export function DictionaryOverlay({
  isOpen,
  onClose,
  initialWord = '',
  language = 'en',
}: DictionaryOverlayProps) {
  const [searchTerm, setSearchTerm] = useState(initialWord)
  const [definitions, setDefinitions] = useState<Definition[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [savedWords, setSavedWords] = useState<string[]>([])
  const [recentSearches, setRecentSearches] = useState<string[]>([])

  const lookupWord = useCallback(async (word: string) => {
    if (!word.trim()) return
    
    setIsLoading(true)
    setError(null)
    
    const cleanWord = word.replace(/[^a-zA-Z\s-]/g, '').toLowerCase().trim()
    
    try {
      const res = await fetch(`https://api.dictionaryapi.dev/api/v2/entries/en/${cleanWord}`)
      
      if (res.ok) {
        const data = await res.json()
        if (data[0]) {
          const entry = data[0]
          const defs: Definition[] = []
          
          for (const meaning of entry.meanings || []) {
            for (const def of meaning.definitions?.slice(0, 2) || []) {
              defs.push({
                word: entry.word,
                phonetic: entry.phonetic || entry.phonetics?.[0]?.text,
                partOfSpeech: meaning.partOfSpeech,
                definition: def.definition,
                example: def.example,
                synonyms: meaning.synonyms?.slice(0, 5),
              })
            }
          }
          
          setDefinitions(defs)
          setRecentSearches(prev => [cleanWord, ...prev.filter(w => w !== cleanWord)].slice(0, 10))
        }
      } else {
        setError('Word not found. Try a different spelling.')
        setDefinitions([])
      }
    } catch {
      setError('Unable to look up word. Please try again.')
      setDefinitions([])
    }
    
    setIsLoading(false)
  }, [])

  const speakWord = useCallback((word: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(word)
      utterance.lang = language
      speechSynthesis.speak(utterance)
    }
  }, [language])

  const toggleSaveWord = useCallback((word: string) => {
    setSavedWords(prev => 
      prev.includes(word) 
        ? prev.filter(w => w !== word)
        : [...prev, word]
    )
  }, [])

  const handleSearch = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    lookupWord(searchTerm)
  }, [searchTerm, lookupWord])

  if (!isOpen) return null

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose}>
      <OverlayPanel
        title="Dictionary"
        subtitle="Look up any word"
        onClose={onClose}
        size="md"
      >
        <AnimatedList className="p-4 space-y-4" delay={0.05}>
          {/* Search */}
          <AnimatedItem>
            <form onSubmit={handleSearch} className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Type a word..."
                  className="pl-10 bg-muted/30 border-border/50"
                  autoFocus
                />
              </div>
              <Button type="submit" disabled={isLoading}>
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Look up'}
              </Button>
            </form>
          </AnimatedItem>

          {/* Recent searches */}
          {recentSearches.length > 0 && !definitions.length && !isLoading && (
            <AnimatedItem>
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <History className="h-4 w-4" />
                  <span>Recent searches</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {recentSearches.map((word) => (
                    <button
                      key={word}
                      onClick={() => {
                        setSearchTerm(word)
                        lookupWord(word)
                      }}
                      className="px-3 py-1.5 rounded-full text-sm bg-muted/50 hover:bg-muted transition-colors"
                    >
                      {word}
                    </button>
                  ))}
                </div>
              </div>
            </AnimatedItem>
          )}

          {/* Error */}
          {error && (
            <AnimatedItem>
              <div className="p-4 rounded-2xl bg-destructive/10 text-destructive text-center">
                {error}
              </div>
            </AnimatedItem>
          )}

          {/* Definitions */}
          {definitions.length > 0 && (
            <AnimatedItem>
              <div className="space-y-4">
                {/* Word header */}
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-2xl font-bold">{definitions[0].word}</h3>
                    {definitions[0].phonetic && (
                      <p className="text-muted-foreground">{definitions[0].phonetic}</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => speakWord(definitions[0].word)}
                      className="rounded-full"
                    >
                      <Volume2 className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => toggleSaveWord(definitions[0].word)}
                      className={cn(
                        'rounded-full',
                        savedWords.includes(definitions[0].word) && 'text-[var(--kelly-blue)]'
                      )}
                    >
                      <Star className={cn(
                        'h-4 w-4',
                        savedWords.includes(definitions[0].word) && 'fill-[var(--kelly-blue)]'
                      )} />
                    </Button>
                  </div>
                </div>

                {/* Definition cards */}
                <div className="space-y-3">
                  {definitions.map((def, i) => (
                    <div key={i} className="p-4 rounded-2xl bg-muted/30 space-y-2">
                      <span className="inline-block px-2 py-0.5 rounded text-xs font-medium bg-primary/10 text-primary">
                        {def.partOfSpeech}
                      </span>
                      <p className="text-foreground">{def.definition}</p>
                      {def.example && (
                        <p className="text-sm text-muted-foreground italic">
                          "{def.example}"
                        </p>
                      )}
                      {def.synonyms && def.synonyms.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 pt-1">
                          <span className="text-xs text-muted-foreground">Synonyms:</span>
                          {def.synonyms.map((syn) => (
                            <button
                              key={syn}
                              onClick={() => {
                                setSearchTerm(syn)
                                lookupWord(syn)
                              }}
                              className="text-xs text-primary hover:underline"
                            >
                              {syn}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </AnimatedItem>
          )}

          {/* Saved words */}
          {savedWords.length > 0 && !definitions.length && (
            <AnimatedItem>
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Star className="h-4 w-4" />
                  <span>Saved words</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {savedWords.map((word) => (
                    <button
                      key={word}
                      onClick={() => {
                        setSearchTerm(word)
                        lookupWord(word)
                      }}
                      className="px-3 py-1.5 rounded-full text-sm bg-[var(--kelly-blue)]/10 text-[var(--kelly-blue)] hover:bg-[var(--kelly-blue)]/20 transition-colors"
                    >
                      {word}
                    </button>
                  ))}
                </div>
              </div>
            </AnimatedItem>
          )}

          {/* Empty state */}
          {!definitions.length && !error && !isLoading && !recentSearches.length && (
            <AnimatedItem>
              <div className="py-8 text-center">
                <BookOpen className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
                <h4 className="font-medium mb-1">Look up any word</h4>
                <p className="text-sm text-muted-foreground">
                  Type a word above to see its definition, pronunciation, and more.
                </p>
              </div>
            </AnimatedItem>
          )}
        </AnimatedList>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
