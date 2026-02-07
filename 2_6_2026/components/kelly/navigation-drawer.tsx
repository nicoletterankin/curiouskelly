'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Menu, X, User, Settings, CreditCard, Heart, HelpCircle, 
  Info, FileText, Users, Shield, Sparkles, BookOpen, 
  Search, MessageCircle, Calendar, Volume2, Palette,
  Home, Star, Gift, Zap, Pyramid, Target, TrendingUp
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface NavigationDrawerProps {
  isOpen: boolean
  onClose: () => void
  onNavigate: (view: string) => void
  currentDay?: number
  userName?: string
  isPremium?: boolean
}

const MENU_SECTIONS = [
  {
    title: 'Kelly',
    items: [
      { id: 'home', label: 'Today\'s Lesson', icon: Home, description: 'Continue learning' },
      { id: 'calendar', label: 'Lesson Calendar', icon: Calendar, description: 'Browse all 365 days' },
      { id: 'dictionary', label: 'Word Lookup', icon: BookOpen, description: 'Explore vocabulary' },
      { id: 'search', label: 'Search Lessons', icon: Search, description: 'Find any topic' },
    ]
  },
  {
    title: 'Account',
    items: [
      { id: 'profile', label: 'My Profile', icon: User, description: 'View your progress' },
      { id: 'settings', label: 'Settings', icon: Settings, description: 'Customize Kelly' },
      { id: 'subscription', label: 'Subscription', icon: CreditCard, description: 'Manage billing' },
      { id: 'pricing', label: 'Upgrade', icon: Star, description: 'Unlock premium features', highlight: true },
    ]
  },
  {
    title: 'Community',
    items: [
      { id: 'affiliate', label: 'Earn Rewards', icon: Gift, description: 'Share & earn 20-50%' },
      { id: 'chat', label: 'Ask Kelly', icon: MessageCircle, description: 'Chat with AI tutor' },
    ]
  },
  {
    title: 'More',
    items: [
      { id: 'about', label: 'About Kelly', icon: Info, description: 'Our mission' },
      { id: 'help', label: 'Help Center', icon: HelpCircle, description: 'Get support' },
      { id: 'terms', label: 'Terms & Privacy', icon: FileText, description: 'Legal stuff' },
    ]
  },
  {
    title: 'Strategic',
    items: [
      { id: 'strategic', label: 'Strategic Overview', icon: Target, description: 'Vision & coalition' },
      { id: 'investors', label: 'Financial Model', icon: TrendingUp, description: 'Full investor workspace', highlight: true },
    ]
  },
  {
    title: 'Studio',
    items: [
      { id: 'studio', label: 'Content Studio', icon: Sparkles, description: 'Create with Kelly' },
      { id: 'admin', label: 'Admin Panel', icon: Shield, description: 'Manage content', admin: true },
    ]
  }
]

export function NavigationDrawer({ 
  isOpen, 
  onClose, 
  onNavigate, 
  currentDay = 1,
  userName = 'Explorer',
  isPremium = false
}: NavigationDrawerProps) {
  const [hoveredItem, setHoveredItem] = useState<string | null>(null)

  const handleNavigate = (id: string) => {
    onNavigate(id)
    onClose()
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[200]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Drawer */}
          <motion.div
            className="fixed top-0 left-0 bottom-0 w-[85vw] max-w-[320px] z-[201] bg-gradient-to-b from-gray-900 to-black border-r border-white/10 overflow-hidden"
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          >
            {/* Header */}
            <div className="relative px-5 pt-12 pb-6 border-b border-white/10">
              {/* Close button */}
              <button
                onClick={onClose}
                className="absolute top-4 right-4 p-2 rounded-full hover:bg-white/10 transition-colors"
              >
                <X className="w-5 h-5 text-white/70" />
              </button>

              {/* User info */}
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-semibold text-lg">
                  {userName.charAt(0).toUpperCase()}
                </div>
                <div>
                  <h2 className="text-white font-semibold">{userName}</h2>
                  <p className="text-white/50 text-sm">
                    Day {currentDay} of 365 {isPremium && '• Premium'}
                  </p>
                </div>
              </div>

              {/* Quick stats - consistent glass style */}
              <div className="flex gap-3 mt-4">
                <div className="flex-1 bg-white/15 backdrop-blur-sm rounded-xl px-3 py-2.5 text-center border border-white/10">
                  <div className="text-white font-semibold">{currentDay}</div>
                  <div className="text-white/60 text-xs">Days</div>
                </div>
                <div className="flex-1 bg-white/15 backdrop-blur-sm rounded-xl px-3 py-2.5 text-center border border-white/10">
                  <div className="text-white font-semibold">0</div>
                  <div className="text-white/60 text-xs">Streak</div>
                </div>
                <div className="flex-1 bg-white/15 backdrop-blur-sm rounded-xl px-3 py-2.5 text-center border border-white/10">
                  <div className="text-white font-semibold">0</div>
                  <div className="text-white/60 text-xs">Saved</div>
                </div>
              </div>
            </div>

            {/* Menu sections */}
            <div className="overflow-y-auto h-[calc(100vh-200px)] py-4">
              {MENU_SECTIONS.map((section, sectionIndex) => (
                <div key={section.title} className="mb-4">
                  <h3 className="px-5 text-xs font-semibold text-white/40 uppercase tracking-wider mb-2">
                    {section.title}
                  </h3>
                  <div className="space-y-0.5">
                    {section.items.map((item) => {
                      const Icon = item.icon
                      const isHovered = hoveredItem === item.id
                      
                      // Show all items - admin check can be added later
                      
                      return (
                        <motion.button
                          key={item.id}
                          onClick={() => handleNavigate(item.id)}
                          onHoverStart={() => setHoveredItem(item.id)}
                          onHoverEnd={() => setHoveredItem(null)}
                          className={cn(
                            "w-full flex items-center gap-3 px-5 py-3 text-left transition-all",
                            isHovered ? "bg-white/10" : "bg-transparent",
                            item.highlight && "bg-gradient-to-r from-amber-500/20 to-orange-500/20"
                          )}
                          whileTap={{ scale: 0.98 }}
                        >
                          <div className={cn(
                            "w-9 h-9 rounded-xl flex items-center justify-center transition-colors",
                            item.highlight 
                              ? "bg-gradient-to-br from-amber-500 to-orange-500 text-white" 
                              : "bg-white/15 backdrop-blur-sm text-white"
                          )}>
                            <Icon className="w-5 h-5" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className={cn(
                              "font-medium text-sm",
                              item.highlight ? "text-amber-300" : "text-white"
                            )}>
                              {item.label}
                            </div>
                            <div className="text-white/60 text-xs truncate">
                              {item.description}
                            </div>
                          </div>
                          {item.highlight && (
                            <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
                          )}
                        </motion.button>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>

            {/* Footer */}
            <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-white/10 bg-black/50 backdrop-blur-sm">
              <div className="text-center text-white/30 text-xs">
                Curious Kelly v1.0 • Made with love
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

// Floating menu button to open the drawer
export function MenuButton({ onClick }: { onClick: () => void }) {
  return (
    <motion.button
      onClick={onClick}
      className="fixed top-4 left-4 z-[100] w-11 h-11 rounded-full bg-black/40 backdrop-blur-md border border-white/20 flex items-center justify-center text-white hover:bg-black/50 transition-all shadow-lg"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <Menu className="w-6 h-6 drop-shadow-[0_2px_4px_rgba(0,0,0,0.5)]" />
    </motion.button>
  )
}
