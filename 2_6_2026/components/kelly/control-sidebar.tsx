'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { getKellyAge, getAgeVariant } from '@/lib/date-utils'
import type { KellyArchetype, LessonPhase } from '@/lib/types'
import { PhaseNavigationVertical } from './phase-navigation'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  User,
  Globe,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  MessageSquare,
  Users,
} from 'lucide-react'

// Languages - comprehensive global coverage with native names (24 languages)
const LANGUAGES = [
  // Tier 1: Core languages with full support (2B+ speakers)
  { code: 'en', name: 'English', native: 'English', flag: '🇺🇸', tier: 1, speakers: '1.5B' },
  { code: 'zh', name: 'Mandarin', native: '中文', flag: '🇨🇳', tier: 1, speakers: '1.1B' },
  { code: 'hi', name: 'Hindi', native: 'हिन्दी', flag: '🇮🇳', tier: 1, speakers: '600M' },
  { code: 'es', name: 'Spanish', native: 'Español', flag: '🇪🇸', tier: 1, speakers: '550M' },
  { code: 'ar', name: 'Arabic', native: 'العربية', flag: '🇸🇦', tier: 1, speakers: '420M' },
  { code: 'pt', name: 'Portuguese', native: 'Português', flag: '🇧🇷', tier: 1, speakers: '260M' },
  // Tier 2: European languages (100M+ speakers)
  { code: 'fr', name: 'French', native: 'Français', flag: '🇫🇷', tier: 2, speakers: '280M' },
  { code: 'de', name: 'German', native: 'Deutsch', flag: '🇩🇪', tier: 2, speakers: '130M' },
  { code: 'ru', name: 'Russian', native: 'Русский', flag: '🇷🇺', tier: 2, speakers: '250M' },
  { code: 'it', name: 'Italian', native: 'Italiano', flag: '🇮🇹', tier: 2, speakers: '85M' },
  { code: 'nl', name: 'Dutch', native: 'Nederlands', flag: '🇳🇱', tier: 2, speakers: '25M' },
  { code: 'pl', name: 'Polish', native: 'Polski', flag: '🇵🇱', tier: 2, speakers: '45M' },
  // Tier 3: Asian languages (high growth markets)
  { code: 'ja', name: 'Japanese', native: '日本語', flag: '🇯🇵', tier: 3, speakers: '125M' },
  { code: 'ko', name: 'Korean', native: '한국어', flag: '🇰🇷', tier: 3, speakers: '80M' },
  { code: 'vi', name: 'Vietnamese', native: 'Tiếng Việt', flag: '🇻🇳', tier: 3, speakers: '85M' },
  { code: 'th', name: 'Thai', native: 'ภาษาไทย', flag: '🇹🇭', tier: 3, speakers: '60M' },
  { code: 'id', name: 'Indonesian', native: 'Bahasa', flag: '🇮🇩', tier: 3, speakers: '200M' },
  { code: 'tl', name: 'Filipino', native: 'Tagalog', flag: '🇵🇭', tier: 3, speakers: '80M' },
  // Tier 4: Additional global languages (strategic markets)
  { code: 'tr', name: 'Turkish', native: 'Türkçe', flag: '🇹🇷', tier: 4, speakers: '80M' },
  { code: 'uk', name: 'Ukrainian', native: 'Українська', flag: '🇺🇦', tier: 4, speakers: '40M' },
  { code: 'he', name: 'Hebrew', native: 'עברית', flag: '🇮🇱', tier: 4, speakers: '9M' },
  { code: 'sw', name: 'Swahili', native: 'Kiswahili', flag: '🇰🇪', tier: 4, speakers: '100M' },
  { code: 'bn', name: 'Bengali', native: 'বাংলা', flag: '🇧🇩', tier: 4, speakers: '270M' },
  { code: 'ur', name: 'Urdu', native: 'اردو', flag: '🇵🇰', tier: 4, speakers: '230M' },
] as const

// Kelly archetypes - REMOVED storyteller/survivor per brand guidelines
// Only showing archetypes that are actively supported
const ARCHETYPES: { value: KellyArchetype; label: string; description: string }[] = [
  { value: 'explorer', label: 'Explorer', description: 'Adventurous, curious' },
  { value: 'architect', label: 'Architect', description: 'Analytical, methodical' },
  { value: 'empath', label: 'Empath', description: 'Nurturing, understanding' },
  { value: 'rebel', label: 'Rebel', description: 'Bold, independent' },
  { value: 'mystic', label: 'Mystic', description: 'Reflective, intuitive' },
  { value: 'provider', label: 'Provider', description: 'Caring, generous' },
  { value: 'diplomat', label: 'Diplomat', description: 'Balanced, fair' },
  { value: 'macgyver', label: 'MacGyver', description: 'Resourceful, inventive' },
  { value: 'scientist', label: 'Scientist', description: 'Curious, precise' },
  { value: 'strategist', label: 'Strategist', description: 'Tactical, deliberate' },
]

// Age options - 3 core variants (kid, adult, senior)
const AGE_OPTIONS = [
  { id: 'kid', label: 'Young Learner', age: 10, range: '5-12', image: '/kelly/kid/explorer.png' },
  { id: 'adult', label: 'Adult', age: 30, range: '13-64', image: '/kelly/archetypes/explorer.png' },
  { id: 'senior', label: 'Senior', age: 68, range: '65+', image: '/kelly/senior/explorer.png' },
] as const

const AGE_VARIANTS = {
  kid: { label: 'Kid', range: '5-12' },
  adult: { label: 'Adult', range: '13-64' },
  senior: { label: 'Senior', range: '65+' },
}

function getAgeCategory(age: number): 'kid' | 'adult' | 'senior' {
  if (age <= 12) return 'kid'
  if (age >= 65) return 'senior'
  return 'adult'
}

interface LiveComment {
  id: string
  username: string
  text: string
  timestamp: Date
  highlight?: boolean
}

interface ControlSidebarProps {
  // Preferences
  kellyAge: number
  language: string
  archetype: KellyArchetype
  
  // Handlers
  onAgeChange: (age: number) => void
  onLanguageChange: (language: string) => void
  onArchetypeChange: (archetype: KellyArchetype) => void
  
  // Phase navigation
  currentPhase: LessonPhase
  onPhaseChange: (phase: LessonPhase) => void
  completedPhases?: LessonPhase[]
  
  // Comments
  comments?: LiveComment[]
  
  // Layout
  collapsed?: boolean
  onCollapsedChange?: (collapsed: boolean) => void
  className?: string
}

export function ControlSidebar({
  kellyAge,
  language,
  archetype,
  onAgeChange,
  onLanguageChange,
  onArchetypeChange,
  currentPhase,
  onPhaseChange,
  completedPhases = [],
  comments = [],
  collapsed = false,
  onCollapsedChange,
  className,
}: ControlSidebarProps) {
  const ageVariant = getAgeVariant(kellyAge)
  const displayKellyAge = getKellyAge(kellyAge)
  const variantInfo = AGE_VARIANTS[ageVariant]
  const ageCategory = getAgeCategory(kellyAge) // Declare ageCategory variable

  // Collapsed state - minimal rail with icons
  if (collapsed) {
    return (
      <div
        className={cn(
          'w-14 bg-black/95 backdrop-blur-sm border-r border-white/10 flex flex-col items-center py-4 gap-2',
          className
        )}
      >
        <Button
          variant="ghost"
          size="icon"
          onClick={() => onCollapsedChange?.(false)}
          className="text-white/60 hover:text-white hover:bg-white/10 mb-4"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        
        <div className="flex flex-col items-center gap-1 flex-1">
          <Button variant="ghost" size="icon" className="text-white/40 hover:text-white hover:bg-white/10" title="Learner Age">
            <User className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" className="text-white/40 hover:text-white hover:bg-white/10" title="Language">
            <Globe className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" className="text-white/40 hover:text-white hover:bg-white/10" title="Teaching Style">
            <Sparkles className="h-4 w-4" />
          </Button>
        </div>
        
        {/* Phase indicator dots */}
        <div className="flex flex-col items-center gap-1.5 py-4 border-t border-white/10 w-full">
          {['hook', 'story', 'wonder', 'action', 'wisdom'].map((phase) => (
            <div 
              key={phase}
              className={cn(
                'w-2 h-2 rounded-full transition-all',
                currentPhase === phase ? 'bg-white w-3' : 
                completedPhases.includes(phase as LessonPhase) ? 'bg-white/40' : 'bg-white/15'
              )}
            />
          ))}
        </div>
      </div>
    )
  }

  return (
    <aside
      className={cn(
        'w-80 bg-black/95 backdrop-blur-sm border-r border-white/10 flex flex-col',
        className
      )}
    >
      {/* Header - Old Money Aesthetic */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
        <div>
          <h2 className="text-xs font-semibold tracking-[0.2em] text-white/60 uppercase">Preferences</h2>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => onCollapsedChange?.(true)}
          className="text-white/60 hover:text-white hover:bg-white/5 h-8 w-8"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-5 space-y-8">
          {/* Age Selector - Refined 3-tier */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-medium tracking-[0.15em] text-white/50 uppercase">Learner Age</span>
            </div>
            
            <div className="grid grid-cols-3 gap-3">
              {AGE_OPTIONS.map((option) => {
                const isSelected = ageCategory === option.id
                return (
                  <button
                    key={option.id}
                    onClick={() => onAgeChange(option.age)}
                    className={cn(
                      'flex flex-col items-center p-3 rounded-lg transition-all border',
                      isSelected
                        ? 'border-white/40 bg-white/10'
                        : 'border-white/5 bg-white/5 hover:bg-white/10 hover:border-white/20'
                    )}
                  >
                    <div className={cn(
                      'w-14 h-14 rounded-full overflow-hidden mb-2 ring-2 ring-offset-2 ring-offset-black',
                      isSelected ? 'ring-white/60' : 'ring-transparent'
                    )}>
                      <img 
                        src={option.image || "/placeholder.svg"} 
                        alt={`Kelly as ${option.label}`}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <span className={cn(
                      'text-xs font-medium',
                      isSelected ? 'text-white' : 'text-white/60'
                    )}>
                      {option.label}
                    </span>
                    <span className={cn(
                      'text-[10px] mt-0.5',
                      isSelected ? 'text-white/50' : 'text-white/30'
                    )}>
                      {option.range}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Language Selector - Enhanced with native names */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium tracking-[0.15em] text-white/50 uppercase">Language</span>
              <span className="text-[10px] text-white/30">{LANGUAGES.length} available</span>
            </div>
            
            <Select value={language} onValueChange={onLanguageChange}>
              <SelectTrigger className="w-full bg-white/5 border-white/10 text-white hover:bg-white/10 hover:border-white/20 transition-all">
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent className="bg-black/95 border-white/10 backdrop-blur-lg max-h-80">
                {/* Tier 1 - Primary */}
                <div className="px-2 py-1.5 text-[10px] font-medium tracking-widest text-white/40 uppercase">Primary</div>
                {LANGUAGES.filter(l => l.tier === 1).map((lang) => (
                  <SelectItem key={lang.code} value={lang.code} className="text-white hover:bg-white/10 focus:bg-white/10">
                    <span className="flex items-center gap-3">
                      <span className="text-base">{lang.flag}</span>
                      <span className="flex-1">{lang.name}</span>
                      <span className="text-white/40 text-xs">{lang.native}</span>
                    </span>
                  </SelectItem>
                ))}
                {/* Tier 2 - European */}
                <div className="px-2 py-1.5 text-[10px] font-medium tracking-widest text-white/40 uppercase mt-2">European</div>
                {LANGUAGES.filter(l => l.tier === 2).map((lang) => (
                  <SelectItem key={lang.code} value={lang.code} className="text-white hover:bg-white/10 focus:bg-white/10">
                    <span className="flex items-center gap-3">
                      <span className="text-base">{lang.flag}</span>
                      <span className="flex-1">{lang.name}</span>
                      <span className="text-white/40 text-xs">{lang.native}</span>
                    </span>
                  </SelectItem>
                ))}
                {/* Tier 3 - Asian */}
                <div className="px-2 py-1.5 text-[10px] font-medium tracking-widest text-white/40 uppercase mt-2">Asian</div>
                {LANGUAGES.filter(l => l.tier === 3).map((lang) => (
                  <SelectItem key={lang.code} value={lang.code} className="text-white hover:bg-white/10 focus:bg-white/10">
                    <span className="flex items-center gap-3">
                      <span className="text-base">{lang.flag}</span>
                      <span className="flex-1">{lang.name}</span>
                      <span className="text-white/40 text-xs">{lang.native}</span>
                    </span>
                  </SelectItem>
                ))}
                {/* Tier 4 - Global */}
                <div className="px-2 py-1.5 text-[10px] font-medium tracking-widest text-white/40 uppercase mt-2">Global</div>
                {LANGUAGES.filter(l => l.tier === 4).map((lang) => (
                  <SelectItem key={lang.code} value={lang.code} className="text-white hover:bg-white/10 focus:bg-white/10">
                    <span className="flex items-center gap-3">
                      <span className="text-base">{lang.flag}</span>
                      <span className="flex-1">{lang.name}</span>
                      <span className="text-white/40 text-xs">{lang.native}</span>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Archetype Selector - Refined 5x2 Grid */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium tracking-[0.15em] text-white/50 uppercase">Teaching Style</span>
              <span className="text-[10px] text-white/30">{ARCHETYPES.length} styles</span>
            </div>
            
            <div className="grid grid-cols-5 gap-2">
              {ARCHETYPES.map((arch) => {
                const isSelected = archetype === arch.value
                const imagePath = `/kelly/archetypes/${arch.value}.png`
                
                return (
                  <button
                    key={arch.value}
                    onClick={() => onArchetypeChange(arch.value)}
                    className={cn(
                      'flex flex-col items-center p-2 rounded-lg transition-all border group',
                      isSelected
                        ? 'border-white/40 bg-white/10'
                        : 'border-transparent hover:bg-white/5 hover:border-white/10'
                    )}
                    title={arch.description}
                  >
                    <div className={cn(
                      'w-10 h-10 rounded-full overflow-hidden mb-1.5 ring-2 ring-offset-1 ring-offset-black transition-all',
                      isSelected ? 'ring-white/60' : 'ring-transparent group-hover:ring-white/20'
                    )}>
                      <img 
                        src={imagePath || "/placeholder.svg"} 
                        alt={arch.label}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <span className={cn(
                      'text-[9px] font-medium leading-tight text-center transition-colors',
                      isSelected ? 'text-white' : 'text-white/40 group-hover:text-white/60'
                    )}>
                      {arch.label}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Phase Navigation */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium tracking-[0.15em] text-white/50 uppercase">Lesson Phases</span>
              <span className="text-[10px] text-white/30">{completedPhases.length}/5</span>
            </div>
            
            <PhaseNavigationVertical
              currentPhase={currentPhase}
              onPhaseChange={onPhaseChange}
              completedPhases={completedPhases}
            />
          </div>

          {/* Live Comments - Refined */}
          {comments.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium tracking-[0.15em] text-white/50 uppercase">Live Comments</span>
                <span className="text-[10px] text-white bg-white/10 px-2 py-0.5 rounded-full">
                  {comments.length}
                </span>
              </div>
              
              <div className="space-y-2 max-h-48 overflow-y-auto scrollbar-thin">
                {comments.slice(0, 10).map((comment) => (
                  <div
                    key={comment.id}
                    className={cn(
                      'rounded-lg p-3 text-sm border',
                      comment.highlight
                        ? 'bg-white/10 border-white/20'
                        : 'bg-white/5 border-white/5'
                    )}
                  >
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className={cn(
                        'text-[10px] font-medium px-1.5 py-0.5 rounded',
                        comment.highlight ? 'bg-white/20 text-white' : 'bg-white/10 text-white/60'
                      )}>
                        {comment.username}
                      </span>
                    </div>
                    <p className="text-white/80 text-sm leading-relaxed">{comment.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Footer - Refined */}
      <div className="p-4 border-t border-white/10">
        <Button 
          variant="outline" 
          className="w-full gap-2 bg-white/5 border-white/10 text-white/70 hover:bg-white/10 hover:text-white hover:border-white/20 transition-all" 
          size="sm"
        >
          <Users className="h-4 w-4" />
          <span className="text-xs tracking-wide">Join Discussion</span>
        </Button>
      </div>
    </aside>
  )
}

// Mobile-friendly drawer version
export function ControlDrawer({
  kellyAge,
  language,
  archetype,
  onAgeChange,
  onLanguageChange,
  onArchetypeChange,
  currentPhase,
  onPhaseChange,
  completedPhases = [],
  isOpen,
  onClose,
}: ControlSidebarProps & { isOpen: boolean; onClose: () => void }) {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 lg:hidden">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      />
      
      {/* Drawer */}
      <div className="absolute bottom-0 left-0 right-0 bg-sidebar rounded-t-2xl max-h-[80vh] overflow-hidden">
        {/* Handle */}
        <div className="flex justify-center py-2">
          <div className="w-12 h-1 bg-muted rounded-full" />
        </div>
        
        <ScrollArea className="h-full max-h-[calc(80vh-2rem)]">
          <div className="p-4 space-y-6 pb-8">
            {/* Age Selector - 3 Icon Picker */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <User className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Kelly&apos;s Age</span>
              </div>
              
              <div className="grid grid-cols-3 gap-3">
                {AGE_OPTIONS.map((option) => {
                  const isSelected = getAgeCategory(kellyAge) === option.id
                  return (
                    <button
                      key={option.id}
                      onClick={() => onAgeChange(option.age)}
                      className={cn(
                        'flex flex-col items-center p-3 rounded-xl transition-all border-2',
                        isSelected
                          ? 'border-primary bg-primary/10'
                          : 'border-transparent bg-sidebar-accent'
                      )}
                    >
                      <div className={cn(
                        'w-14 h-14 rounded-full overflow-hidden mb-2 ring-2',
                        isSelected ? 'ring-primary' : 'ring-transparent'
                      )}>
                        <img 
                          src={option.image || "/placeholder.svg"} 
                          alt={`Kelly as ${option.label}`}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <span className={cn(
                        'text-sm font-medium',
                        isSelected ? 'text-primary' : 'text-foreground'
                      )}>
                        {option.label}
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Language Selector */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Globe className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Language</span>
              </div>
              
              <Select value={language} onValueChange={onLanguageChange}>
                <SelectTrigger className="w-full bg-sidebar-accent border-sidebar-border">
                  <SelectValue placeholder="Select language" />
                </SelectTrigger>
                <SelectContent>
                  {LANGUAGES.map((lang) => (
                    <SelectItem key={lang.code} value={lang.code}>
                      <span className="flex items-center gap-2">
                        <span className="text-xs">{lang.flag}</span>
                        <span>{lang.name}</span>
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Archetype Selector - 4x3 Grid */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Teaching Style</span>
              </div>
              
              <div className="grid grid-cols-4 gap-2">
                {ARCHETYPES.map((arch) => {
                  const isSelected = archetype === arch.value
                  const imagePath = `/kelly/archetypes/${arch.value}.png`
                  
                  return (
                    <button
                      key={arch.value}
                      onClick={() => onArchetypeChange(arch.value)}
                      className={cn(
                        'flex flex-col items-center p-2 rounded-lg transition-all border-2',
                        isSelected
                          ? 'border-primary bg-primary/10'
                          : 'border-transparent bg-sidebar-accent'
                      )}
                    >
                      <div className={cn(
                        'w-12 h-12 rounded-full overflow-hidden mb-1 ring-2',
                        isSelected ? 'ring-primary' : 'ring-transparent'
                      )}>
                        <img 
                          src={imagePath || "/placeholder.svg"} 
                          alt={arch.label}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <span className={cn(
                        'text-[10px] font-medium',
                        isSelected ? 'text-primary' : 'text-muted-foreground'
                      )}>
                        {arch.label}
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Phase Navigation - Horizontal for mobile */}
            <div className="space-y-3">
              <span className="text-sm font-medium">Lesson Phases</span>
              <PhaseNavigationVertical
                currentPhase={currentPhase}
                onPhaseChange={(phase) => {
                  onPhaseChange(phase)
                  onClose()
                }}
                completedPhases={completedPhases}
              />
            </div>
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}
