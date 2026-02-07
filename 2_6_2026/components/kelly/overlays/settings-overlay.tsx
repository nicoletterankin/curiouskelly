'use client'

/**
 * KellyOS Settings Overlay
 * Apple-quality settings panel - TWO levels max (main + selector)
 * No third-level navigation - Help/About/Privacy are inline or external links
 */

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel, ListItem, SectionHeader, PillSelector } from './index'
import {
  User,
  Globe,
  Sparkles,
  Bell,
  Moon,
  Sun,
  Volume2,
  CreditCard,
  Info,
  Shield,
  LogOut,
  Check,
  ChevronDown,
  ExternalLink,
  Mail,
  HelpCircle,
} from 'lucide-react'
import { Switch } from '@/components/ui/switch'

// ============================================================================
// TYPES
// ============================================================================

type KellyArchetype = 'storyteller' | 'explorer' | 'architect' | 'empath' | 'rebel' | 'mystic' | 'provider' | 'diplomat' | 'macgyver' | 'scientist' | 'strategist' | 'survivor'

interface SettingsOverlayProps {
  isOpen: boolean
  onClose: () => void
  // User settings
  kellyAge: number
  language: string
  archetype: KellyArchetype
  autoplay: boolean
  notifications: boolean
  theme: 'light' | 'dark' | 'system'
  // Handlers
  onAgeChange: (age: number) => void
  onLanguageChange: (lang: string) => void
  onArchetypeChange: (arch: KellyArchetype) => void
  onAutoplayToggle: (enabled: boolean) => void
  onNotificationsToggle: (enabled: boolean) => void
  onThemeChange: (theme: 'light' | 'dark' | 'system') => void
  onSignOut?: () => void
  onOpenSubscription?: () => void
}

// ============================================================================
// DATA
// ============================================================================

const LANGUAGES = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'es', name: 'Español', flag: '🇪🇸' },
  { code: 'fr', name: 'Français', flag: '🇫🇷' },
  { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
  { code: 'pt', name: 'Português', flag: '🇧🇷' },
  { code: 'zh', name: '中文', flag: '🇨🇳' },
  { code: 'ja', name: '日本語', flag: '🇯🇵' },
  { code: 'ko', name: '한국어', flag: '🇰🇷' },
  { code: 'hi', name: 'हिन्दी', flag: '🇮🇳' },
  { code: 'ar', name: 'العربية', flag: '🇸🇦' },
]

const ARCHETYPES: { value: KellyArchetype; label: string; description: string }[] = [
  { value: 'storyteller', label: 'Storyteller', description: 'Warm, narrative-driven' },
  { value: 'explorer', label: 'Explorer', description: 'Adventurous, curious' },
  { value: 'architect', label: 'Architect', description: 'Analytical, structured' },
  { value: 'empath', label: 'Empath', description: 'Nurturing, understanding' },
  { value: 'rebel', label: 'Rebel', description: 'Bold, unconventional' },
  { value: 'mystic', label: 'Mystic', description: 'Spiritual, reflective' },
  { value: 'provider', label: 'Provider', description: 'Caring, supportive' },
  { value: 'diplomat', label: 'Diplomat', description: 'Balanced, thoughtful' },
  { value: 'macgyver', label: 'MacGyver', description: 'Innovative, hands-on' },
  { value: 'scientist', label: 'Scientist', description: 'Curious, methodical' },
  { value: 'strategist', label: 'Strategist', description: 'Tactical, chess-minded' },
  { value: 'survivor', label: 'Survivor', description: 'Resilient, resourceful' },
]

// ============================================================================
// SUBVIEWS
// ============================================================================

type SettingsView = 'main' | 'age' | 'language' | 'archetype'

// Age categories - only 3 that match database (kid, adult, elder/senior)
const AGE_OPTIONS = [
  { id: 'kid', label: 'Kid', age: 10, description: '5-12 years', image: '/kelly/kid/storyteller.png' },
  { id: 'adult', label: 'Adult', age: 30, description: '13-64 years', image: '/kelly/adult/storyteller.png' },
  { id: 'senior', label: 'Senior', age: 68, description: '65+ years', image: '/kelly/senior/storyteller.png' },
] as const

function getAgeCategory(age: number): 'kid' | 'adult' | 'senior' {
  if (age <= 12) return 'kid'
  if (age >= 65) return 'senior'
  return 'adult'
}

function AgeSelector({ 
  kellyAge, 
  onAgeChange, 
  onBack 
}: { 
  kellyAge: number
  onAgeChange: (age: number) => void
  onBack: () => void 
}) {
  const currentCategory = getAgeCategory(kellyAge)

  return (
    <OverlayPanel
      title="Kelly's Age"
      subtitle="Choose how Kelly appears to you"
      showBack
      showClose={false}
      onBack={onBack}
    >
      <div className="p-6">
        <div className="grid grid-cols-3 gap-4">
          {AGE_OPTIONS.map((option) => {
            const isSelected = currentCategory === option.id
            return (
              <button
                key={option.id}
                onClick={() => {
                  onAgeChange(option.age)
                  onBack()
                }}
                className={cn(
                  'flex flex-col items-center p-4 rounded-2xl transition-all border-2',
                  isSelected
                    ? 'border-white/40 bg-white/10'
                    : 'border-transparent bg-white/5 hover:bg-white/10'
                )}
              >
                {/* Kelly Image */}
                <div className={cn(
                  'w-20 h-20 rounded-full overflow-hidden mb-3 ring-2',
                  isSelected ? 'ring-white/60' : 'ring-transparent'
                )}>
                  <img 
                    src={option.image || "/placeholder.svg"} 
                    alt={`Kelly as ${option.label}`}
                    className="w-full h-full object-cover"
                  />
                </div>
                
                {/* Label */}
                <span className={cn(
                  'font-semibold text-base',
                  isSelected ? 'text-white' : 'text-white/70'
                )}>
                  {option.label}
                </span>
                
                {/* Age range */}
                <span className="text-xs text-white/50 mt-1">
                  {option.description}
                </span>
                
                {/* Check indicator */}
                {isSelected && (
                  <div className="mt-2">
                    <Check className="h-5 w-5 text-white" />
                  </div>
                )}
              </button>
            )
          })}
        </div>
      </div>
    </OverlayPanel>
  )
}

function LanguageSelector({
  language,
  onLanguageChange,
  onBack,
}: {
  language: string
  onLanguageChange: (lang: string) => void
  onBack: () => void
}) {
  return (
    <OverlayPanel
      title="Language"
      subtitle="Choose your lesson language"
      showBack
      showClose={false}
      onBack={onBack}
    >
      <div className="divide-y divide-border/30">
        {LANGUAGES.map((lang) => (
          <button
            key={lang.code}
            onClick={() => {
              onLanguageChange(lang.code)
              onBack()
            }}
            className="w-full flex items-center gap-4 px-6 py-4 hover:bg-white/10 transition-colors"
          >
            <span className="text-2xl">{lang.flag}</span>
            <span className="flex-1 text-left font-medium text-white">{lang.name}</span>
            {language === lang.code && (
              <Check className="h-5 w-5 text-white" />
            )}
          </button>
        ))}
      </div>
    </OverlayPanel>
  )
}

function ArchetypeSelector({
  archetype,
  onArchetypeChange,
  onBack,
}: {
  archetype: KellyArchetype
  onArchetypeChange: (arch: KellyArchetype) => void
  onBack: () => void
}) {
  return (
    <OverlayPanel
      title="Teaching Style"
      subtitle="Choose how Kelly teaches you"
      showBack
      showClose={false}
      onBack={onBack}
    >
      <div className="p-4">
        {/* 4x3 grid of archetype images */}
        <div className="grid grid-cols-4 gap-3">
          {ARCHETYPES.map((arch) => {
            const isSelected = archetype === arch.value
            const imagePath = `/kelly/adult/${arch.value}.png`
            
            return (
              <button
                key={arch.value}
                onClick={() => {
                  onArchetypeChange(arch.value)
                  onBack()
                }}
                className={cn(
                  'flex flex-col items-center p-2 rounded-xl transition-all border-2',
                  isSelected
                    ? 'border-white/40 bg-white/10'
                    : 'border-transparent hover:bg-white/10'
                )}
              >
                {/* Kelly archetype image */}
                <div className={cn(
                  'w-14 h-14 md:w-16 md:h-16 rounded-full overflow-hidden mb-2 ring-2',
                  isSelected ? 'ring-white/60' : 'ring-transparent'
                )}>
                  <img 
                    src={imagePath || "/placeholder.svg"} 
                    alt={`Kelly as ${arch.label}`}
                    className="w-full h-full object-cover"
                  />
                </div>
                
                {/* Label */}
                <span className={cn(
                  'font-medium text-xs text-center leading-tight',
                  isSelected ? 'text-white' : 'text-white/70'
                )}>
                  {arch.label}
                </span>
                
                {/* Check indicator */}
                {isSelected && (
                  <Check className="h-4 w-4 text-white mt-1" />
                )}
              </button>
            )
          })}
        </div>
        
        {/* Selected description */}
        <div className="mt-4 p-3 bg-white/5 rounded-xl text-center">
          <span className="text-sm text-white/60">
            {ARCHETYPES.find(a => a.value === archetype)?.description}
          </span>
        </div>
      </div>
    </OverlayPanel>
  )
}

// AboutView removed - now inline expandable section

// ============================================================================
// MAIN SETTINGS OVERLAY
// ============================================================================

export function SettingsOverlay({
  isOpen,
  onClose,
  kellyAge,
  language,
  archetype,
  autoplay,
  notifications,
  theme,
  onAgeChange,
  onLanguageChange,
  onArchetypeChange,
  onAutoplayToggle,
  onNotificationsToggle,
  onThemeChange,
  onSignOut,
  onOpenSubscription,
}: SettingsOverlayProps) {
  const [view, setView] = useState<SettingsView>('main')
  const [aboutExpanded, setAboutExpanded] = useState(false)
  const [helpExpanded, setHelpExpanded] = useState(false)
  
  const selectedLanguage = LANGUAGES.find(l => l.code === language) || LANGUAGES[0]
  const selectedArchetype = ARCHETYPES.find(a => a.value === archetype) || ARCHETYPES[0]

const handleClose = () => {
  setView('main')
  setAboutExpanded(false)
  setHelpExpanded(false)
  onClose()
  }

  if (!isOpen) return null

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={handleClose}>
      {view === 'age' && (
        <AgeSelector
          kellyAge={kellyAge}
          onAgeChange={onAgeChange}
          onBack={() => setView('main')}
        />
      )}
      
      {view === 'language' && (
        <LanguageSelector
          language={language}
          onLanguageChange={onLanguageChange}
          onBack={() => setView('main')}
        />
      )}
      
      {view === 'archetype' && (
        <ArchetypeSelector
          archetype={archetype}
          onArchetypeChange={onArchetypeChange}
          onBack={() => setView('main')}
        />
      )}
      
      {view === 'main' && (
        <OverlayPanel
          title="Settings"
          onClose={handleClose}
          size="md"
        >
          {/* Personalization */}
          <SectionHeader title="Personalization" />
          <div>
            <ListItem
              icon={<User className="h-4 w-4" />}
              title="Your Age"
              value={`${kellyAge} years old`}
              onClick={() => setView('age')}
            />
            <ListItem
              icon={<Globe className="h-4 w-4" />}
              title="Language"
              value={`${selectedLanguage.flag} ${selectedLanguage.name}`}
              onClick={() => setView('language')}
            />
            <ListItem
              icon={<Sparkles className="h-4 w-4" />}
              title="Teaching Style"
              value={selectedArchetype.label}
              onClick={() => setView('archetype')}
            />
          </div>

          {/* Playback */}
          <SectionHeader title="Playback" />
          <div>
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center text-white/70">
                  <Volume2 className="h-4 w-4" />
                </div>
                <span className="font-medium">Autoplay</span>
              </div>
              <Switch checked={autoplay} onCheckedChange={onAutoplayToggle} />
            </div>
          </div>

          {/* Notifications */}
          <SectionHeader title="Notifications" />
          <div>
            <div className="flex items-center justify-between px-4 py-3 border-b border-border/30">
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center text-white/70">
                  <Bell className="h-4 w-4" />
                </div>
                <div>
                  <div className="font-medium">Daily Reminders</div>
                  <div className="text-sm text-muted-foreground">Get notified for new lessons</div>
                </div>
              </div>
              <Switch checked={notifications} onCheckedChange={onNotificationsToggle} />
            </div>
          </div>

          {/* Appearance */}
          <SectionHeader title="Appearance" />
          <div className="px-4 py-4">
            <PillSelector
              options={[
                { value: 'light', label: 'Light', icon: <Sun className="h-4 w-4" /> },
                { value: 'dark', label: 'Dark', icon: <Moon className="h-4 w-4" /> },
                { value: 'system', label: 'System' },
              ]}
              value={theme}
              onChange={onThemeChange}
              className="w-full justify-center"
            />
          </div>

          {/* Support - INLINE expandable sections, no third level */}
          <SectionHeader title="Support" />
          <div>
            {/* Help - Expandable inline */}
            <button
              onClick={() => setHelpExpanded(!helpExpanded)}
              className="w-full flex items-center justify-between px-4 py-3 border-b border-border/30 hover:bg-white/5 transition-colors"
            >
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center text-white/70">
                  <HelpCircle className="h-4 w-4" />
                </div>
                <span className="font-medium">Help</span>
              </div>
              <ChevronDown className={cn(
                "h-4 w-4 text-white/40 transition-transform",
                helpExpanded && "rotate-180"
              )} />
            </button>
            {helpExpanded && (
              <div className="px-4 py-4 bg-white/[0.02] border-b border-border/30 space-y-3">
                <p className="text-sm text-white/60 leading-relaxed">
                  Need help with iLearn? Here are some ways to get support:
                </p>
                <div className="space-y-2">
                  <a
                    href="mailto:support@lessonoftheday.com"
                    className="flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                  >
                    <Mail className="h-4 w-4 text-white/50" />
                    <div className="flex-1">
                      <div className="text-sm font-medium">Email Support</div>
                      <div className="text-xs text-white/40">support@lessonoftheday.com</div>
                    </div>
                    <ExternalLink className="h-4 w-4 text-white/30" />
                  </a>
                  <a
                    href="https://curiouskelly.com/faq"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                  >
                    <HelpCircle className="h-4 w-4 text-white/50" />
                    <div className="flex-1">
                      <div className="text-sm font-medium">FAQs</div>
                      <div className="text-xs text-white/40">Common questions answered</div>
                    </div>
                    <ExternalLink className="h-4 w-4 text-white/30" />
                  </a>
                </div>
              </div>
            )}
            
            {/* About - Expandable inline */}
            <button
              onClick={() => setAboutExpanded(!aboutExpanded)}
              className="w-full flex items-center justify-between px-4 py-3 border-b border-border/30 hover:bg-white/5 transition-colors"
            >
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center text-white/70">
                  <Info className="h-4 w-4" />
                </div>
                <span className="font-medium">About iLearn</span>
              </div>
              <ChevronDown className={cn(
                "h-4 w-4 text-white/40 transition-transform",
                aboutExpanded && "rotate-180"
              )} />
            </button>
            {aboutExpanded && (
              <div className="px-4 py-4 bg-white/[0.02] border-b border-border/30 space-y-3 text-center">
                <div className="w-16 h-16 mx-auto rounded-2xl bg-white/10 border border-white/20 flex items-center justify-center">
                  <span className="text-2xl font-bold text-white">K</span>
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white">iLearn with Kelly</h3>
                  <p className="text-sm text-white/60">The Daily Lesson</p>
                </div>
                <p className="text-xs text-white/50 leading-relaxed max-w-[280px] mx-auto">
                  Kelly is your AI learning companion who adapts to your age, 
                  language, and teaching style. 365 lessons. Infinite curiosity.
                </p>
                <div className="text-xs text-white/40 pt-2">
                  <div>Version 3.0.0</div>
                  <div>© 2026 Lesson of the Day, PBC</div>
                </div>
              </div>
            )}
            
            {/* Privacy - Direct external link */}
            <a
              href="https://curiouskelly.com/privacy"
              target="_blank"
              rel="noopener noreferrer"
              className="w-full flex items-center justify-between px-4 py-3 border-b border-border/30 hover:bg-white/5 transition-colors"
            >
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center text-white/70">
                  <Shield className="h-4 w-4" />
                </div>
                <span className="font-medium">Privacy Policy</span>
              </div>
              <ExternalLink className="h-4 w-4 text-white/40" />
            </a>
          </div>

          {/* Account */}
          <SectionHeader title="Account" />
          <div>
            {onOpenSubscription && (
              <ListItem
                icon={<CreditCard className="h-4 w-4" />}
                title="Subscription"
                onClick={onOpenSubscription}
              />
            )}
            {onSignOut && (
              <button
                onClick={onSignOut}
                className="w-full flex items-center gap-4 px-4 py-3 text-left text-destructive hover:bg-destructive/10 transition-colors"
              >
                <div className="w-8 h-8 rounded-lg bg-destructive/10 flex items-center justify-center">
                  <LogOut className="h-4 w-4" />
                </div>
                <span className="font-medium">Sign Out</span>
              </button>
            )}
          </div>
        </OverlayPanel>
      )}
    </OverlayBackdrop>
  )
}
