'use client'

import React from "react"

/**
 * KellyOS Share Overlay
 * Apple-quality sharing panel with AFFILIATE REFERRAL LINKS
 * 
 * Learn, Share, Earn: Every share includes your affiliate code
 * Lifetime cookies - you earn forever from your referrals
 */

import { useState, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel, AnimatedList, AnimatedItem } from './index'
import { 
  Copy, 
  Check, 
  Twitter, 
  Facebook, 
  Linkedin, 
  MessageCircle,
  Mail,
  Link2,
  QrCode,
  DollarSign,
  TrendingUp,
  Users,
  Gift,
} from 'lucide-react'
import { getFingerprint } from '@/lib/fingerprint'

interface ShareOverlayProps {
  isOpen: boolean
  onClose: () => void
  title: string
  url: string
  description?: string
}

interface AffiliateData {
  registered: boolean
  affiliateCode?: string
  tier?: string
  earnings?: number
  referralCount?: number
}

interface WaitlistState {
  email: string
  submitted: boolean
  error: string | null
}

// APPLE MONOCHROME: All share icons use white/glass styling - NO colored icons
const SHARE_OPTIONS = [
  { id: 'copy', label: 'Copy Link', icon: Link2 },
  { id: 'twitter', label: 'Twitter', icon: Twitter },
  { id: 'facebook', label: 'Facebook', icon: Facebook },
  { id: 'linkedin', label: 'LinkedIn', icon: Linkedin },
  { id: 'whatsapp', label: 'WhatsApp', icon: MessageCircle },
  { id: 'email', label: 'Email', icon: Mail },
]

// Tier badges with commission rates
const TIER_INFO: Record<string, { label: string, commission: number, color: string }> = {
  'starter': { label: 'Starter', commission: 20, color: 'bg-white/20' },
  'advocate': { label: 'Advocate', commission: 25, color: 'bg-blue-500/30' },
  'ambassador': { label: 'Ambassador', commission: 30, color: 'bg-purple-500/30' },
  'champion': { label: 'Champion', commission: 35, color: 'bg-amber-500/30' },
  'legend': { label: 'Legend', commission: 40, color: 'bg-gradient-to-r from-amber-500/30 to-orange-500/30' },
}

export function ShareOverlay({
  isOpen,
  onClose,
  title,
  url,
  description,
}: ShareOverlayProps) {
  const [copied, setCopied] = useState(false)
  const [showQR, setShowQR] = useState(false)
  const [affiliate, setAffiliate] = useState<AffiliateData>({ registered: false })
  const [isLoading, setIsLoading] = useState(true)
  const [waitlist, setWaitlist] = useState<WaitlistState>({ email: '', submitted: false, error: null })

  const handleWaitlistSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!waitlist.email) return
    
    try {
      const res = await fetch('/api/waitlist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          email: waitlist.email,
          source: 'share_overlay'
        })
      })
      
      if (res.ok) {
        setWaitlist(prev => ({ ...prev, submitted: true, error: null }))
      } else {
        const data = await res.json()
        setWaitlist(prev => ({ ...prev, error: data.error || 'Failed to join' }))
      }
    } catch {
      setWaitlist(prev => ({ ...prev, error: 'Network error' }))
    }
  }

  // Fetch or create affiliate on mount
  useEffect(() => {
    if (isOpen) {
      fetchOrCreateAffiliate()
    }
  }, [isOpen])

  const fetchOrCreateAffiliate = async () => {
    setIsLoading(true)
    try {
      const userId = getFingerprint()
      
      // Try to get existing affiliate
      const res = await fetch(`/api/affiliate/register?userId=${userId}`)
      const data = await res.json()
      
      if (data.registered) {
        setAffiliate(data)
      } else {
        // Auto-register as affiliate (everyone can earn!)
        const registerRes = await fetch('/api/affiliate/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ userId }),
        })
        const registerData = await registerRes.json()
        setAffiliate({
          registered: true,
          affiliateCode: registerData.affiliateCode,
          tier: registerData.tier || 'starter',
          earnings: registerData.earnings || 0,
          referralCount: registerData.referralCount || 0,
        })
      }
    } catch (e) {
      console.error('[ShareOverlay] Failed to fetch affiliate:', e)
    } finally {
      setIsLoading(false)
    }
  }

  // Build referral URL with affiliate code
  const baseUrl = url || 'https://thedailylesson.com'
  const referralUrl = affiliate.affiliateCode 
    ? `${baseUrl}${baseUrl.includes('?') ? '&' : '?'}ref=${affiliate.affiliateCode}`
    : baseUrl

  if (!isOpen) return null

  const handleShare = async (optionId: string) => {
    const shareText = `${title}${description ? ` - ${description}` : ''}`
    // Always use referral URL for sharing (includes affiliate code)
    const shareUrl = referralUrl
    
    switch (optionId) {
      case 'copy':
        await navigator.clipboard.writeText(shareUrl)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
        break
      case 'twitter':
        window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}`, '_blank')
        break
      case 'facebook':
        window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`, '_blank')
        break
      case 'linkedin':
        window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`, '_blank')
        break
      case 'whatsapp':
        window.open(`https://wa.me/?text=${encodeURIComponent(`${shareText} ${shareUrl}`)}`, '_blank')
        break
      case 'email':
        window.open(`mailto:?subject=${encodeURIComponent(title)}&body=${encodeURIComponent(`${shareText}\n\n${shareUrl}`)}`)
        break
    }
  }

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose} side="right">
      <OverlayPanel
        title="Share"
        onClose={onClose}
        size="md"
      >
        <AnimatedList className="p-6 space-y-5" delay={0.05}>
          {/* EARN CARD - Show affiliate stats */}
          {!isLoading && affiliate.registered && (
            <AnimatedItem>
              <div className={cn(
                'rounded-2xl p-4 border border-white/10',
                TIER_INFO[affiliate.tier || 'starter']?.color || 'bg-white/5'
              )}>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Gift className="w-4 h-4 text-white/70" />
                    <span className="text-xs font-semibold text-white/90 uppercase tracking-wider">
                      Your Referral Link
                    </span>
                  </div>
                  <span className="text-[10px] font-bold text-white/60 px-2 py-0.5 rounded-full bg-white/10">
                    {TIER_INFO[affiliate.tier || 'starter']?.commission || 20}% Commission
                  </span>
                </div>
                
                {/* Referral URL display */}
                <div className="bg-black/30 rounded-lg px-3 py-2 mb-3">
                  <p className="text-[11px] text-white/50 font-mono truncate">
                    {referralUrl}
                  </p>
                </div>
                
                {/* Stats row */}
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="bg-white/5 rounded-lg py-2">
                    <div className="text-sm font-bold text-white">${(affiliate.earnings || 0).toFixed(2)}</div>
                    <div className="text-[9px] text-white/50 uppercase">Earned</div>
                  </div>
                  <div className="bg-white/5 rounded-lg py-2">
                    <div className="text-sm font-bold text-white">{affiliate.referralCount || 0}</div>
                    <div className="text-[9px] text-white/50 uppercase">Referrals</div>
                  </div>
                  <div className="bg-white/5 rounded-lg py-2">
                    <div className="text-sm font-bold text-white capitalize">{affiliate.tier || 'Starter'}</div>
                    <div className="text-[9px] text-white/50 uppercase">Tier</div>
                  </div>
                </div>
              </div>
            </AnimatedItem>
          )}

          {/* Preview Card - Dark glass */}
          <AnimatedItem>
            <div className="bg-white/5 rounded-2xl p-4 flex items-start gap-3 border border-white/10">
              <img 
                src="/kelly/archetypes/storyteller.png" 
                alt="Kelly" 
                className="w-14 h-14 rounded-xl object-cover flex-shrink-0"
              />
              <div className="flex-1 min-w-0">
                <h4 className="font-semibold text-white text-sm truncate">{title}</h4>
                {description && (
                  <p className="text-xs text-white/60 line-clamp-2 mt-1">
                    {description}
                  </p>
                )}
                <p className="text-[10px] text-white/40 mt-1.5 truncate">{referralUrl}</p>
              </div>
            </div>
          </AnimatedItem>

          {/* Share Options Grid */}
          <AnimatedItem>
            <div className="grid grid-cols-3 gap-4">
              {SHARE_OPTIONS.map((option) => {
                const Icon = option.icon
                const isCopy = option.id === 'copy'
                
                return (
                  <button
                    key={option.id}
                    onClick={() => handleShare(option.id)}
                    className="flex flex-col items-center gap-2 p-4 rounded-2xl hover:bg-white/10 transition-all active:scale-95"
                  >
                    <div className={cn(
                      'w-12 h-12 rounded-full flex items-center justify-center',
                      isCopy && copied ? 'bg-white/30' : 'bg-white/10',
                      'text-white/80 border border-white/10'
                    )}>
                      {isCopy && copied ? (
                        <Check className="h-5 w-5 text-white/90" />
                      ) : (
                        <Icon className="h-5 w-5" />
                      )}
                    </div>
                    <span className="text-xs font-medium text-white/70">
                      {isCopy && copied ? 'Copied!' : option.label}
                    </span>
                  </button>
                )
              })}
            </div>
          </AnimatedItem>

          {/* QR Code Toggle */}
          <AnimatedItem>
            <button
              onClick={() => setShowQR(!showQR)}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors text-white"
            >
              <QrCode className="h-4 w-4" />
              <span className="text-sm font-medium">
                {showQR ? 'Hide QR Code' : 'Show QR Code'}
              </span>
            </button>
          </AnimatedItem>

          {/* QR Code Display - Dark glass */}
          {showQR && (
            <AnimatedItem>
              <div className="flex justify-center p-4 bg-white/10 backdrop-blur-sm rounded-2xl border border-white/10">
                <div className="w-40 h-40 bg-white rounded-lg flex items-center justify-center">
                  <QrCode className="h-16 w-16 text-gray-800" />
                </div>
              </div>
            </AnimatedItem>
          )}

          {/* Waitlist Signup - Capture Emails */}
          {!affiliate.registered && (
            <AnimatedItem>
              <div className="bg-gradient-to-br from-white/10 to-white/5 rounded-2xl p-4 border border-white/10">
                <div className="flex items-center gap-2 mb-3">
                  <Mail className="w-4 h-4 text-white/70" />
                  <span className="text-xs font-semibold text-white/90 uppercase tracking-wider">
                    Get Notified
                  </span>
                </div>
                
                {waitlist.submitted ? (
                  <div className="flex items-center gap-2 text-green-400 py-2">
                    <Check className="w-4 h-4" />
                    <span className="text-sm">You are on the list!</span>
                  </div>
                ) : (
                  <form onSubmit={handleWaitlistSubmit} className="flex gap-2">
                    <input
                      type="email"
                      placeholder="your@email.com"
                      value={waitlist.email}
                      onChange={(e) => setWaitlist(prev => ({ ...prev, email: e.target.value }))}
                      className="flex-1 px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm placeholder:text-white/30 focus:outline-none focus:border-white/30"
                      required
                    />
                    <button
                      type="submit"
                      className="px-4 py-2 rounded-lg bg-white text-black font-medium text-sm hover:bg-white/90 transition-colors"
                    >
                      Join
                    </button>
                  </form>
                )}
                {waitlist.error && (
                  <p className="text-xs text-red-400 mt-2">{waitlist.error}</p>
                )}
              </div>
            </AnimatedItem>
          )}
        </AnimatedList>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
