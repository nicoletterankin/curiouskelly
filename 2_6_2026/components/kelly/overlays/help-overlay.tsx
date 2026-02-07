'use client'

/**
 * KellyOS Help Overlay
 * Apple-quality help center panel
 */

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel, ListItem, SectionHeader, AnimatedList, AnimatedItem } from './index'
import { 
  Search, 
  MessageCircle, 
  Mail, 
  ChevronRight,
  Play,
  Settings,
  CreditCard,
  User,
  Globe,
  Sparkles,
  BookOpen,
  Video,
  HelpCircle,
  ExternalLink,
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

interface HelpOverlayProps {
  isOpen: boolean
  onClose: () => void
}

const FAQ_CATEGORIES = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    icon: Play,
    questions: [
      { q: 'How do I start my first lesson?', a: 'Tap Play on the main screen. Kelly guides you through today\'s lesson automatically.' },
      { q: 'What is the 5-phase structure?', a: 'Hook (attention), Story (narrative), Wonder (exploration), Action (apply it), Wisdom (takeaways).' },
      { q: 'Do I need an account?', a: 'No! Start learning immediately. Create an account to save progress and unlock premium.' },
    ],
  },
  {
    id: 'personalization',
    title: 'Personalization',
    icon: Sparkles,
    questions: [
      { q: 'How do I change Kelly\'s age?', a: 'Use the age slider in the left panel or Settings. Kelly\'s teaching style adapts to match.' },
      { q: 'What are archetypes?', a: 'Kelly has 12 teaching styles: Storyteller, Explorer, Architect, Empath, Rebel, Mystic, Provider, Diplomat, MacGyver, Scientist, Strategist, Survivor.' },
      { q: 'Can I change the language?', a: 'Yes! We support 12 languages. Change it in Settings or the left panel.' },
    ],
  },
  {
    id: 'subscription',
    title: 'Subscription',
    icon: CreditCard,
    questions: [
      { q: 'What\'s free?', a: 'Today\'s lesson, 1 language, basic captions. Premium unlocks 365 lessons, all 12 languages, and more.' },
      { q: 'Can I cancel anytime?', a: 'Yes. Cancel anytime, no questions asked. Keep access until the billing period ends.' },
      { q: 'Is there a family plan?', a: 'Yes! Family supports up to 6 members with individual profiles and parental controls.' },
    ],
  },
  {
    id: 'technical',
    title: 'Technical',
    icon: Settings,
    questions: [
      { q: 'Video won\'t play?', a: 'Check your internet connection and refresh. Videos need a stable connection to stream.' },
      { q: 'How do I enable captions?', a: 'Tap the CC button in the right panel, or go to Settings > Playback > Captions.' },
      { q: 'Can I download lessons?', a: 'Premium subscribers can download lessons for offline viewing in the mobile app.' },
    ],
  },
]

export function HelpOverlay({
  isOpen,
  onClose,
}: HelpOverlayProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [expandedQuestion, setExpandedQuestion] = useState<string | null>(null)

  if (!isOpen) return null

  const selectedCategoryData = FAQ_CATEGORIES.find(c => c.id === selectedCategory)

  // Filter questions by search
  const filteredCategories = searchTerm
    ? FAQ_CATEGORIES.map(cat => ({
        ...cat,
        questions: cat.questions.filter(
          q => q.q.toLowerCase().includes(searchTerm.toLowerCase()) ||
               q.a.toLowerCase().includes(searchTerm.toLowerCase())
        )
      })).filter(cat => cat.questions.length > 0)
    : FAQ_CATEGORIES

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose}>
      <OverlayPanel
        title={selectedCategory ? selectedCategoryData?.title || 'Help' : 'Help Center'}
        subtitle="Find answers and get support"
        showBack={!!selectedCategory}
        onBack={() => setSelectedCategory(null)}
        onClose={onClose}
        size="md"
      >
        {!selectedCategory ? (
          <AnimatedList className="p-4 space-y-6" delay={0.05}>
            {/* Search */}
            <AnimatedItem>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search help articles..."
                  className="pl-10 bg-muted/30 border-border/50"
                />
              </div>
            </AnimatedItem>

            {/* Quick actions */}
            {!searchTerm && (
              <div className="grid grid-cols-2 gap-3">
                <button
                  className="p-4 rounded-2xl bg-white/5 border border-white/10 text-left hover:bg-white/10 transition-colors"
                  onClick={() => window.open('mailto:support@lessonoftheday.com')}
                >
                  <Mail className="h-6 w-6 text-white/70 mb-2" />
                  <h4 className="font-semibold text-sm text-white">Email Support</h4>
                  <p className="text-xs text-white/50">Get help from our team</p>
                </button>
                <button
                  className="p-4 rounded-2xl bg-white/5 border border-white/10 text-left hover:bg-white/10 transition-colors"
                  onClick={() => {/* Open chat */}}
                >
                  <MessageCircle className="h-6 w-6 text-white/70 mb-2" />
                  <h4 className="font-semibold text-sm text-white">Live Chat</h4>
                  <p className="text-xs text-white/50">Chat with Kelly</p>
                </button>
              </div>
            )}

            {/* Categories or search results */}
            {searchTerm ? (
              <div className="space-y-4">
                {filteredCategories.length > 0 ? (
                  filteredCategories.map((cat) => (
                    <div key={cat.id}>
                      <h4 className="text-sm font-medium text-muted-foreground mb-2">{cat.title}</h4>
                      <div className="space-y-2">
                        {cat.questions.map((q, i) => (
                          <button
                            key={i}
                            onClick={() => setExpandedQuestion(expandedQuestion === `${cat.id}-${i}` ? null : `${cat.id}-${i}`)}
                            className="w-full text-left p-3 rounded-xl bg-muted/30 hover:bg-muted/50 transition-colors"
                          >
                            <p className="font-medium text-sm">{q.q}</p>
                            {expandedQuestion === `${cat.id}-${i}` && (
                              <p className="text-sm text-muted-foreground mt-2">{q.a}</p>
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8">
                    <HelpCircle className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
                    <p className="font-medium">No results found</p>
                    <p className="text-sm text-muted-foreground">Try a different search term</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-muted-foreground px-1">Browse Topics</h4>
                {FAQ_CATEGORIES.map((cat) => (
                  <button
                    key={cat.id}
                    onClick={() => setSelectedCategory(cat.id)}
                    className="w-full flex items-center gap-4 p-4 rounded-xl hover:bg-white/10 transition-colors"
                  >
                    <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center">
                      <cat.icon className="h-5 w-5 text-white/70" />
                    </div>
                    <div className="flex-1 text-left">
                      <p className="font-medium text-white">{cat.title}</p>
                      <p className="text-xs text-white/50">{cat.questions.length} articles</p>
                    </div>
                    <ChevronRight className="h-5 w-5 text-white/50" />
                  </button>
                ))}
              </div>
            )}

            {/* Footer links */}
            {!searchTerm && (
              <div className="pt-4 border-t">
                <a 
                  href="/help"
                  className="flex items-center justify-between p-3 rounded-xl hover:bg-muted/50 transition-colors"
                >
                  <span className="text-sm">View full Help Center</span>
                  <ExternalLink className="h-4 w-4 text-muted-foreground" />
                </a>
              </div>
            )}
          </AnimatedList>
        ) : (
          <div className="p-4 space-y-2">
            {selectedCategoryData?.questions.map((q, i) => (
              <button
                key={i}
                onClick={() => setExpandedQuestion(expandedQuestion === `q-${i}` ? null : `q-${i}`)}
                className={cn(
                  'w-full text-left p-4 rounded-xl transition-colors',
                  expandedQuestion === `q-${i}` ? 'bg-primary/5' : 'bg-muted/30 hover:bg-muted/50'
                )}
              >
                <p className="font-medium">{q.q}</p>
                {expandedQuestion === `q-${i}` && (
                  <p className="text-sm text-muted-foreground mt-2 leading-relaxed">{q.a}</p>
                )}
              </button>
            ))}
          </div>
        )}
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
