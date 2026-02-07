'use client'

/**
 * KellyOS Affiliate Overlay
 * Unified affiliate dashboard within the 2-layer Kelly architecture
 */

import { useState, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel } from './index'
import { 
  Gift, Users, DollarSign, TrendingUp, Copy, Check, Crown, Trophy, Star, Flame, Sparkles,
  Globe, Clock, Medal, Share2, Twitter, MessageCircle, Mail,
} from 'lucide-react'
import { getFingerprint } from '@/lib/fingerprint'

interface AffiliateOverlayProps {
  isOpen: boolean
  onClose: () => void
}

interface AffiliateData {
  registered: boolean
  affiliateId?: number
  affiliateCode?: string
  tier?: string
  earnings?: number
  referralCount?: number
  pendingPayout?: number
  isLanguageGuardian?: boolean
}

interface Referral {
  id: number
  subscriptionType: string
  commissionAmount: number
  status: string
  createdAt: string
}

interface LeaderboardEntry {
  rank: number
  displayName: string
  referralCount: number
  tier: string
  isLanguageGuardian: boolean
}

const TIERS = [
  { name: 'Starter', minReferrals: 0, commission: 20, icon: Sparkles, color: 'from-slate-500 to-slate-600' },
  { name: 'Advocate', minReferrals: 3, commission: 25, icon: Flame, color: 'from-amber-500 to-amber-600' },
  { name: 'Ambassador', minReferrals: 10, commission: 30, icon: Star, color: 'from-orange-500 to-orange-600' },
  { name: 'Champion', minReferrals: 25, commission: 35, icon: Trophy, color: 'from-rose-500 to-rose-600' },
  { name: 'Legend', minReferrals: 50, commission: 40, icon: Crown, color: 'from-violet-500 to-violet-600', recurring: 10 },
]

export function AffiliateOverlay({ isOpen, onClose }: AffiliateOverlayProps) {
  const [affiliate, setAffiliate] = useState<AffiliateData>({ registered: false })
  const [referrals, setReferrals] = useState<Referral[]>([])
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [copied, setCopied] = useState(false)
  const [email, setEmail] = useState('')
  const [registering, setRegistering] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'referrals' | 'leaderboard'>('overview')

  useEffect(() => {
    if (isOpen) {
      fetchAffiliate()
      fetchLeaderboard()
    }
  }, [isOpen])

  useEffect(() => {
    if (affiliate.affiliateId && activeTab === 'referrals') {
      fetchReferrals()
    }
  }, [affiliate.affiliateId, activeTab])

  const fetchAffiliate = async () => {
    setIsLoading(true)
    try {
      const userId = getFingerprint()
      const res = await fetch(`/api/affiliate/register?userId=${userId}`)
      const data = await res.json()
      setAffiliate(data)
    } catch (e) {
      console.error('Failed to fetch affiliate:', e)
    } finally {
      setIsLoading(false)
    }
  }

  const fetchReferrals = async () => {
    try {
      const res = await fetch(`/api/affiliate/referrals?affiliateId=${affiliate.affiliateId}`)
      const data = await res.json()
      if (data.referrals) setReferrals(data.referrals)
    } catch (e) {
      console.error('Failed to fetch referrals:', e)
    }
  }

  const fetchLeaderboard = async () => {
    try {
      const res = await fetch('/api/affiliate/leaderboard')
      const data = await res.json()
      if (data.leaderboard) setLeaderboard(data.leaderboard)
    } catch {
      setLeaderboard([
        { rank: 1, displayName: 'Sarah M.', referralCount: 127, tier: 'Legend', isLanguageGuardian: true },
        { rank: 2, displayName: 'Marcus T.', referralCount: 89, tier: 'Legend', isLanguageGuardian: false },
        { rank: 3, displayName: 'Elena K.', referralCount: 64, tier: 'Legend', isLanguageGuardian: true },
      ])
    }
  }

  const registerAffiliate = async () => {
    if (!email) return
    setRegistering(true)
    try {
      const userId = getFingerprint()
      const res = await fetch('/api/affiliate/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, email }),
      })
      const data = await res.json()
      if (data.registered) setAffiliate(data)
    } catch (e) {
      console.error('Failed to register:', e)
    } finally {
      setRegistering(false)
    }
  }

  const copyLink = async () => {
    const url = `https://curiouskelly.com?ref=${affiliate.affiliateCode}`
    await navigator.clipboard.writeText(url)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const shareVia = (platform: string) => {
    const url = `https://curiouskelly.com?ref=${affiliate.affiliateCode}`
    const text = "My kids love Curious Kelly! Join us:"
    const shareUrls: Record<string, string> = {
      twitter: `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`,
      whatsapp: `https://wa.me/?text=${encodeURIComponent(text + ' ' + url)}`,
      email: `mailto:?subject=${encodeURIComponent('Check out Curious Kelly!')}&body=${encodeURIComponent(text + '\n\n' + url)}`,
    }
    window.open(shareUrls[platform], '_blank')
  }

  const currentTierIndex = TIERS.findIndex(t => t.name.toLowerCase() === affiliate.tier?.toLowerCase()) || 0
  const currentTier = TIERS[currentTierIndex]
  const nextTier = TIERS[currentTierIndex + 1]
  const referralsToNext = nextTier ? nextTier.minReferrals - (affiliate.referralCount || 0) : 0
  const progressPercent = nextTier ? Math.min(100, ((affiliate.referralCount || 0) / nextTier.minReferrals) * 100) : 100

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose} side="right">
      <OverlayPanel
        title="Affiliate Program"
        subtitle="Earn up to 40% + 10% recurring"
        onClose={onClose}
      >
        <div className="p-4 space-y-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin w-6 h-6 border-2 border-white/20 border-t-white rounded-full" />
            </div>
          ) : !affiliate.registered ? (
            /* Registration */
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                  <Gift className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-semibold">Become an Affiliate</h3>
                  <p className="text-white/50 text-sm">Start earning today</p>
                </div>
              </div>
              <input
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-white/30 focus:outline-none focus:border-white/30"
              />
              <button
                onClick={registerAffiliate}
                disabled={registering || !email}
                className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 font-semibold disabled:opacity-50"
              >
                {registering ? 'Creating...' : 'Get My Link'}
              </button>
            </div>
          ) : (
            <>
              {/* Stats */}
              <div className="grid grid-cols-3 gap-2">
                <div className="bg-white/5 rounded-xl p-3 text-center">
                  <DollarSign className="w-4 h-4 text-green-400 mx-auto mb-1" />
                  <p className="text-lg font-bold">${(affiliate.earnings || 0).toFixed(0)}</p>
                  <p className="text-white/40 text-xs">Earned</p>
                </div>
                <div className="bg-white/5 rounded-xl p-3 text-center">
                  <Users className="w-4 h-4 text-blue-400 mx-auto mb-1" />
                  <p className="text-lg font-bold">{affiliate.referralCount || 0}</p>
                  <p className="text-white/40 text-xs">Referrals</p>
                </div>
                <div className="bg-white/5 rounded-xl p-3 text-center">
                  <Clock className="w-4 h-4 text-amber-400 mx-auto mb-1" />
                  <p className="text-lg font-bold">${(affiliate.pendingPayout || 0).toFixed(0)}</p>
                  <p className="text-white/40 text-xs">Pending</p>
                </div>
              </div>

              {/* Tabs */}
              <div className="flex gap-1 p-1 bg-white/5 rounded-xl">
                {['overview', 'referrals', 'leaderboard'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab as typeof activeTab)}
                    className={cn(
                      'flex-1 py-2 rounded-lg text-xs font-medium transition-all capitalize',
                      activeTab === tab ? 'bg-white/10 text-white' : 'text-white/50'
                    )}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              {activeTab === 'overview' && (
                <>
                  {/* Current Tier */}
                  <div className={cn('rounded-2xl p-4 bg-gradient-to-br', currentTier?.color || 'from-slate-500 to-slate-600')}>
                    <div className="flex items-center gap-3 mb-3">
                      {currentTier && <currentTier.icon className="w-8 h-8" />}
                      <div>
                        <h3 className="font-bold text-xl">{currentTier?.name || 'Starter'}</h3>
                        <p className="text-white/80 text-sm">{currentTier?.commission}% commission</p>
                      </div>
                    </div>
                    {nextTier && (
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-white/80">To {nextTier.name}</span>
                          <span>{referralsToNext} more</span>
                        </div>
                        <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                          <div className="h-full bg-white/90 rounded-full" style={{ width: `${progressPercent}%` }} />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Share */}
                  <div className="space-y-3">
                    <div className="flex gap-2">
                      <div className="flex-1 px-3 py-2 bg-black/30 rounded-lg text-xs text-white/70 truncate font-mono">
                        curiouskelly.com?ref={affiliate.affiliateCode}
                      </div>
                      <button
                        onClick={copyLink}
                        className={cn('px-3 py-2 rounded-lg transition-all', copied ? 'bg-green-500' : 'bg-white/10')}
                      >
                        {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                      </button>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={() => shareVia('twitter')} className="flex-1 py-2 rounded-lg bg-[#1DA1F2]/20 flex items-center justify-center gap-1">
                        <Twitter className="w-3 h-3 text-[#1DA1F2]" /><span className="text-xs">Twitter</span>
                      </button>
                      <button onClick={() => shareVia('whatsapp')} className="flex-1 py-2 rounded-lg bg-[#25D366]/20 flex items-center justify-center gap-1">
                        <MessageCircle className="w-3 h-3 text-[#25D366]" /><span className="text-xs">WhatsApp</span>
                      </button>
                      <button onClick={() => shareVia('email')} className="flex-1 py-2 rounded-lg bg-white/10 flex items-center justify-center gap-1">
                        <Mail className="w-3 h-3" /><span className="text-xs">Email</span>
                      </button>
                    </div>
                  </div>
                </>
              )}

              {activeTab === 'referrals' && (
                <div className="space-y-2">
                  {referrals.length === 0 ? (
                    <div className="text-center py-8">
                      <Users className="w-10 h-10 text-white/20 mx-auto mb-2" />
                      <p className="text-white/50 text-sm">No referrals yet</p>
                    </div>
                  ) : referrals.map((ref) => (
                    <div key={ref.id} className="flex items-center justify-between p-3 bg-white/5 rounded-xl">
                      <div>
                        <p className="font-medium text-sm">{ref.subscriptionType}</p>
                        <p className="text-white/50 text-xs">{new Date(ref.createdAt).toLocaleDateString()}</p>
                      </div>
                      <p className="font-bold text-green-400">+${ref.commissionAmount.toFixed(2)}</p>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'leaderboard' && (
                <div className="space-y-2">
                  {leaderboard.map((entry) => {
                    const tier = TIERS.find(t => t.name.toLowerCase() === entry.tier.toLowerCase())
                    const TierIcon = tier?.icon || Sparkles
                    return (
                      <div key={entry.rank} className="flex items-center gap-3 p-3 bg-white/5 rounded-xl">
                        <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center font-bold text-sm">
                          {entry.rank}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-sm truncate">{entry.displayName}</p>
                          <div className="flex items-center gap-1 text-white/50 text-xs">
                            <TierIcon className="w-3 h-3" />
                            <span>{entry.tier}</span>
                            {entry.isLanguageGuardian && <Globe className="w-3 h-3 text-blue-400" />}
                          </div>
                        </div>
                        <p className="font-bold text-sm">{entry.referralCount}</p>
                      </div>
                    )
                  })}
                </div>
              )}
            </>
          )}
        </div>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
