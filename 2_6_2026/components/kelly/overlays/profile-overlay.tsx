'use client'

/**
 * KellyOS Profile Overlay
 * Apple-quality user profile and progress panel
 */

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel, ListItem, SectionHeader } from './index'
import { 
  User, 
  Mail, 
  Calendar, 
  Award, 
  Flame, 
  Target, 
  BookOpen,
  Clock,
  TrendingUp,
  Edit2,
  Camera,
  Crown,
  ChevronRight,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'

interface ProfileOverlayProps {
  isOpen: boolean
  onClose: () => void
  user?: {
    name: string
    email: string
    avatar?: string
    joinDate: string
    plan: 'free' | 'learner' | 'family'
  }
  stats?: {
    lessonsCompleted: number
    currentStreak: number
    longestStreak: number
    totalMinutes: number
    favoritePhase: string
  }
  earnings?: {
    totalEarnings: number
    pendingEarnings: number
    referralCount: number
    affiliateCode: string | null
  }
  onEditProfile?: () => void
  onUpgrade?: () => void
  onShare?: () => void
}

export function ProfileOverlay({
  isOpen,
  onClose,
  user = {
    name: 'Learner',
    email: 'learner@example.com',
    joinDate: 'January 2026',
    plan: 'free',
  },
  stats = {
    lessonsCompleted: 12,
    currentStreak: 5,
    longestStreak: 14,
    totalMinutes: 180,
    favoritePhase: 'Story',
  },
  earnings = {
    totalEarnings: 0,
    pendingEarnings: 0,
    referralCount: 0,
    affiliateCode: null,
  },
  onEditProfile,
  onUpgrade,
  onShare,
}: ProfileOverlayProps) {
  const [isEditing, setIsEditing] = useState(false)

  if (!isOpen) return null

  const planLabels = {
    free: 'Explorer (Free)',
    learner: 'Learner',
    family: 'Family',
  }

  const nextMilestone = Math.ceil(stats.lessonsCompleted / 10) * 10
  const milestoneProgress = (stats.lessonsCompleted % 10) / 10 * 100

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose}>
      <OverlayPanel
        title="Profile"
        onClose={onClose}
        size="md"
      >
        {/* Profile Header */}
        <div className="p-6 bg-gradient-to-b from-primary/10 to-transparent">
          <div className="flex items-start gap-4">
            {/* Avatar */}
            <div className="relative">
              <div className="w-20 h-20 rounded-full bg-primary/20 flex items-center justify-center overflow-hidden">
                {user.avatar ? (
                  <img src={user.avatar || "/placeholder.svg"} alt="" className="w-full h-full object-cover" />
                ) : (
                  <User className="h-10 w-10 text-primary" />
                )}
              </div>
              <button 
                className="absolute bottom-0 right-0 w-7 h-7 rounded-full bg-background border shadow-sm flex items-center justify-center hover:bg-muted transition-colors"
                onClick={() => setIsEditing(true)}
              >
                <Camera className="h-3.5 w-3.5" />
              </button>
            </div>

            {/* Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="text-xl font-bold">{user.name}</h3>
                {user.plan !== 'free' && (
                  <Crown className="h-4 w-4 text-white/70" />
                )}
              </div>
              <p className="text-sm text-muted-foreground">{user.email}</p>
              <div className="flex items-center gap-2 mt-2">
                <span className={cn(
                  'px-2 py-0.5 rounded-full text-xs font-medium',
                  user.plan === 'free' 
                    ? 'bg-muted text-muted-foreground'
                    : 'bg-primary/10 text-primary'
                )}>
                  {planLabels[user.plan]}
                </span>
                {user.plan === 'free' && (
                  <button 
                    onClick={onUpgrade}
                    className="text-xs text-primary hover:underline"
                  >
                    Upgrade
                  </button>
                )}
              </div>
            </div>

            {/* Edit button */}
            <Button variant="outline" size="sm" onClick={onEditProfile}>
              <Edit2 className="h-3.5 w-3.5 mr-1.5" />
              Edit
            </Button>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="p-4 grid grid-cols-2 gap-3">
          <div className="p-4 rounded-2xl bg-white/5 text-center">
            <Flame className="h-6 w-6 text-white/70 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{stats.currentStreak}</div>
            <div className="text-xs text-white/50">Day Streak</div>
          </div>
          <div className="p-4 rounded-2xl bg-white/5 text-center">
            <BookOpen className="h-6 w-6 text-white/70 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{stats.lessonsCompleted}</div>
            <div className="text-xs text-white/50">Lessons</div>
          </div>
          <div className="p-4 rounded-2xl bg-white/5 text-center">
            <Clock className="h-6 w-6 text-white/70 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{Math.floor(stats.totalMinutes / 60)}h</div>
            <div className="text-xs text-white/50">Learning Time</div>
          </div>
          <div className="p-4 rounded-2xl bg-white/5 text-center">
            <Target className="h-6 w-6 text-white/70 mx-auto mb-2" />
            <div className="text-2xl font-bold">{stats.longestStreak}</div>
            <div className="text-xs text-muted-foreground">Best Streak</div>
          </div>
        </div>

        {/* Learn, Share, Earn - Affiliate Earnings */}
        {(earnings.totalEarnings > 0 || earnings.affiliateCode) && (
          <div className="px-4 pb-4">
            <div className="p-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/20">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-emerald-400" />
                  <span className="font-medium text-white">Your Earnings</span>
                </div>
                <button 
                  onClick={onShare}
                  className="text-xs text-emerald-400 hover:text-emerald-300 font-medium"
                >
                  Share & Earn More
                </button>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <div className="text-xl font-bold text-emerald-400">${(earnings.totalEarnings / 100).toFixed(2)}</div>
                  <div className="text-[10px] text-white/40">Total Earned</div>
                </div>
                <div>
                  <div className="text-xl font-bold text-amber-400">${(earnings.pendingEarnings / 100).toFixed(2)}</div>
                  <div className="text-[10px] text-white/40">Pending</div>
                </div>
                <div>
                  <div className="text-xl font-bold text-white">{earnings.referralCount}</div>
                  <div className="text-[10px] text-white/40">Referrals</div>
                </div>
              </div>
              {earnings.affiliateCode && (
                <div className="mt-3 pt-3 border-t border-white/10 text-center">
                  <p className="text-[10px] text-white/40 mb-1">Your referral code</p>
                  <code className="text-sm font-mono text-white bg-white/10 px-3 py-1 rounded">{earnings.affiliateCode}</code>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Next Milestone */}
        <div className="px-4 pb-4">
          <div className="p-4 rounded-2xl bg-primary/5 border border-primary/10">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Award className="h-5 w-5 text-primary" />
                <span className="font-medium">Next Milestone</span>
              </div>
              <span className="text-sm text-muted-foreground">
                {stats.lessonsCompleted} / {nextMilestone} lessons
              </span>
            </div>
            <Progress value={milestoneProgress} className="h-2" />
            <p className="text-xs text-muted-foreground mt-2">
              Complete {nextMilestone - stats.lessonsCompleted} more lessons to earn your next badge!
            </p>
          </div>
        </div>

        {/* Details */}
        <SectionHeader title="Details" />
        <div>
          <ListItem
            icon={<Calendar className="h-4 w-4" />}
            title="Joined"
            value={user.joinDate}
            chevron={false}
          />
          <ListItem
            icon={<TrendingUp className="h-4 w-4" />}
            title="Favorite Phase"
            value={stats.favoritePhase}
            chevron={false}
          />
        </div>

        {/* Account Actions */}
        <SectionHeader title="Account" />
        <div>
          <ListItem
            icon={<Crown className="h-4 w-4" />}
            title="Subscription"
            value={planLabels[user.plan]}
            onClick={onUpgrade}
          />
          <ListItem
            icon={<Mail className="h-4 w-4" />}
            title="Email Settings"
            onClick={() => { window.location.href = 'mailto:support@lessonoftheday.com?subject=Email%20Settings%20Request'; }}
          />
        </div>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
