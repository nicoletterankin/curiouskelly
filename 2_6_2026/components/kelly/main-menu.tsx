'use client'

import { useState } from 'react'
import { 
  Menu, X, User, Settings, CreditCard, Share2, MessageCircle, 
  Book, Info, HelpCircle, Users, Shield, FileText, Palette, 
  BarChart3, Calendar, Search, MessageSquare, Sparkles
} from 'lucide-react'

interface MainMenuProps {
  isOpen: boolean
  onClose: () => void
  onOpenProfile: () => void
  onOpenSettings: () => void
  onOpenPricing: () => void
  onOpenShare: () => void
  onOpenComments: () => void
  onOpenDictionary: () => void
  onOpenAbout: () => void
  onOpenHelp: () => void
  onOpenSubscription: () => void
  onOpenAffiliate: () => void
  onOpenAdmin: () => void
  onOpenTerms: () => void
  onOpenStudio: () => void
  onOpenZig: () => void
  onOpenCalendar: () => void
  onOpenSearch: () => void
  onOpenChat: () => void
}

const menuSections = [
  {
    title: 'Learning',
    items: [
      { id: 'calendar', label: 'Lesson Calendar', icon: Calendar, action: 'onOpenCalendar' },
      { id: 'dictionary', label: 'Dictionary', icon: Book, action: 'onOpenDictionary' },
      { id: 'search', label: 'Search Lessons', icon: Search, action: 'onOpenSearch' },
      { id: 'chat', label: 'Ask Kelly', icon: MessageSquare, action: 'onOpenChat' },
    ]
  },
  {
    title: 'Account',
    items: [
      { id: 'profile', label: 'My Profile', icon: User, action: 'onOpenProfile' },
      { id: 'settings', label: 'Settings', icon: Settings, action: 'onOpenSettings' },
      { id: 'subscription', label: 'Subscription', icon: CreditCard, action: 'onOpenSubscription' },
      { id: 'pricing', label: 'Upgrade Plans', icon: Sparkles, action: 'onOpenPricing' },
    ]
  },
  {
    title: 'Community',
    items: [
      { id: 'share', label: 'Share Lesson', icon: Share2, action: 'onOpenShare' },
      { id: 'comments', label: 'Comments', icon: MessageCircle, action: 'onOpenComments' },
      { id: 'affiliate', label: 'Become an Affiliate', icon: Users, action: 'onOpenAffiliate' },
    ]
  },
  {
    title: 'More',
    items: [
      { id: 'about', label: 'About Curious Kelly', icon: Info, action: 'onOpenAbout' },
      { id: 'help', label: 'Help & Support', icon: HelpCircle, action: 'onOpenHelp' },
      { id: 'studio', label: 'Creator Studio', icon: Palette, action: 'onOpenStudio' },
      { id: 'zig', label: 'ZIG Model', icon: BarChart3, action: 'onOpenZig' },
      { id: 'terms', label: 'Terms & Privacy', icon: FileText, action: 'onOpenTerms' },
      { id: 'admin', label: 'Admin Dashboard', icon: Shield, action: 'onOpenAdmin' },
    ]
  }
]

export function MainMenu(props: MainMenuProps) {
  const { isOpen, onClose } = props
  
  if (!isOpen) return null
  
  const handleAction = (action: string) => {
    const fn = props[action as keyof MainMenuProps]
    if (typeof fn === 'function') {
      fn()
      onClose()
    }
  }
  
  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[200] animate-in fade-in duration-200"
        onClick={onClose}
      />
      
      {/* Menu Panel */}
      <div className="fixed inset-y-0 left-0 w-[300px] max-w-[85vw] bg-white z-[201] shadow-2xl animate-in slide-in-from-left duration-300">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#00D4FF] to-[#00A3CC] flex items-center justify-center">
              <span className="text-white font-bold text-lg">K</span>
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">Curious Kelly</h2>
              <p className="text-xs text-gray-500">Daily Learning Adventures</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-100 transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>
        
        {/* Menu Content */}
        <div className="overflow-y-auto h-[calc(100%-80px)] pb-safe">
          {menuSections.map((section) => (
            <div key={section.title} className="py-3">
              <h3 className="px-5 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                {section.title}
              </h3>
              <div className="space-y-0.5">
                {section.items.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => handleAction(item.action)}
                    className="w-full flex items-center gap-3 px-5 py-3 hover:bg-gray-50 active:bg-gray-100 transition-colors text-left"
                  >
                    <item.icon className="w-5 h-5 text-gray-400" />
                    <span className="text-gray-700 font-medium">{item.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}

// Floating menu button to open the main menu
export function MenuButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="fixed top-4 left-4 z-[100] w-11 h-11 rounded-full bg-white/90 backdrop-blur-md shadow-lg flex items-center justify-center hover:bg-white hover:scale-105 active:scale-95 transition-all border border-gray-200/50"
      aria-label="Open menu"
    >
      <Menu className="w-5 h-5 text-gray-700" />
    </button>
  )
}
