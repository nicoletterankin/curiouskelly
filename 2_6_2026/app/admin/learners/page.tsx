"use client"

import React from "react"
import { Suspense } from "react"
import { useSearchParams } from "next/navigation"

import { useState, useMemo, useEffect } from "react"
import {
  Users, Globe, MapPin, Mail, MessageSquare, CreditCard, Clock, Calendar,
  Award, Flame, Eye, Search, Filter, ChevronDown, ChevronRight, Send, Bell,
  Settings, Download, RefreshCw, Shield, UserCheck, UserX, AlertTriangle,
  CheckCircle, XCircle, MoreVertical, ExternalLink, Copy, Star, Activity, X
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Checkbox } from "@/components/ui/checkbox"
import { Textarea } from "@/components/ui/textarea"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

// Types
interface Learner {
  id: string
  name: string | null
  email: string | null
  avatar: string | null
  authProvider: 'google' | 'apple' | 'email'
  createdAt: string
  lastSeenAt: string
  ip: string
  ipFull: string
  city: string
  region: string
  country: string
  countryFlag: string
  timezone: string
  device: 'Desktop' | 'Mobile' | 'Tablet'
  browser: string
  os: string
  subscriptionStatus: 'free' | 'trial' | 'active' | 'canceled' | 'past_due'
  subscriptionPlan: 'monthly' | 'yearly' | 'lifetime' | null
  stripeCustomerId: string | null
  trialEndsAt: string | null
  currentPeriodEnd: string | null
  currentDay: number
  currentPhase: number
  lessonsCompleted: number
  totalTimeSpent: number
  currentStreak: number
  longestStreak: number
  lastLessonTopic: string | null
  kellyAge: 5 | 12 | 18 | 35 | 55 | 77
  preferredLanguage: string
  emailOptIn: boolean
  pushOptIn: boolean
  emailsSent: number
  emailsOpened: number
  lastEmailSent: string | null
  isOnline: boolean
  isVIP: boolean
  needsAttention: boolean
}

interface MessageTemplate {
  id: string
  name: string
  type: 'email' | 'push'
  subject: string
  body: string
}

// Mock data generator
const generateMockLearners = (): Learner[] => {
  const firstNames = ['Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Ethan', 'Sophia', 'Mason', 'Isabella', 'James', 'Mia', 'Lucas', 'Charlotte', 'Benjamin', 'Amelia']
  const lastNames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Wilson', 'Anderson', 'Taylor', 'Thomas']
  
  const locations = [
    { city: 'Irvine', region: 'California', country: 'US', flag: '🇺🇸', tz: 'America/Los_Angeles' },
    { city: 'New York', region: 'New York', country: 'US', flag: '🇺🇸', tz: 'America/New_York' },
    { city: 'Los Angeles', region: 'California', country: 'US', flag: '🇺🇸', tz: 'America/Los_Angeles' },
    { city: 'London', region: 'England', country: 'GB', flag: '🇬🇧', tz: 'Europe/London' },
    { city: 'Toronto', region: 'Ontario', country: 'CA', flag: '🇨🇦', tz: 'America/Toronto' },
    { city: 'Sydney', region: 'NSW', country: 'AU', flag: '🇦🇺', tz: 'Australia/Sydney' },
    { city: 'Berlin', region: 'Berlin', country: 'DE', flag: '🇩🇪', tz: 'Europe/Berlin' },
    { city: 'Mumbai', region: 'Maharashtra', country: 'IN', flag: '🇮🇳', tz: 'Asia/Kolkata' },
    { city: 'Tokyo', region: 'Tokyo', country: 'JP', flag: '🇯🇵', tz: 'Asia/Tokyo' },
    { city: 'Paris', region: 'Île-de-France', country: 'FR', flag: '🇫🇷', tz: 'Europe/Paris' },
    { city: 'Singapore', region: 'Singapore', country: 'SG', flag: '🇸🇬', tz: 'Asia/Singapore' },
    { city: 'Dubai', region: 'Dubai', country: 'AE', flag: '🇦🇪', tz: 'Asia/Dubai' },
  ]

  const topics = ['Photosynthesis', 'Gravity', 'Emotions', 'How the Internet Works', 'Dreams', 'Weather Patterns', 'Ocean Life', 'Space Exploration', 'Human Body', 'Ancient History']

  return Array.from({ length: 50 }, (_, i) => {
    const hasName = Math.random() > 0.25
    const firstName = firstNames[Math.floor(Math.random() * firstNames.length)]
    const lastName = lastNames[Math.floor(Math.random() * lastNames.length)]
    const location = locations[Math.floor(Math.random() * locations.length)]
    const statuses: Learner['subscriptionStatus'][] = ['free', 'trial', 'active', 'active', 'active', 'canceled', 'past_due']
    const status = statuses[Math.floor(Math.random() * statuses.length)]
    const lessonsCompleted = Math.floor(Math.random() * 120)
    const lastSeenMinutes = Math.floor(Math.random() * 10080) // up to 7 days in minutes
    
    return {
      id: `usr_${Math.random().toString(36).substr(2, 9)}`,
      name: hasName ? `${firstName} ${lastName}` : null,
      email: hasName ? `${firstName.toLowerCase()}.${lastName.toLowerCase()}@gmail.com` : null,
      avatar: hasName ? `https://api.dicebear.com/7.x/avataaars/svg?seed=${firstName}${i}` : null,
      authProvider: (['google', 'apple', 'email'] as const)[Math.floor(Math.random() * 3)],
      createdAt: new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString(),
      lastSeenAt: new Date(Date.now() - lastSeenMinutes * 60 * 1000).toISOString(),
      ip: `${Math.floor(Math.random() * 200) + 10}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.xxx`,
      ipFull: `${Math.floor(Math.random() * 200) + 10}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
      city: location.city,
      region: location.region,
      country: location.country,
      countryFlag: location.flag,
      timezone: location.tz,
      device: (['Desktop', 'Mobile', 'Tablet'] as const)[Math.floor(Math.random() * 3)],
      browser: ['Chrome', 'Safari', 'Firefox', 'Edge'][Math.floor(Math.random() * 4)],
      os: ['Windows', 'macOS', 'iOS', 'Android'][Math.floor(Math.random() * 4)],
      subscriptionStatus: status,
      subscriptionPlan: status === 'active' ? (['monthly', 'yearly', 'lifetime'] as const)[Math.floor(Math.random() * 3)] : null,
      stripeCustomerId: status !== 'free' ? `cus_${Math.random().toString(36).substr(2, 14)}` : null,
      trialEndsAt: status === 'trial' ? new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() : null,
      currentPeriodEnd: status === 'active' ? new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString() : null,
      currentDay: Math.min(lessonsCompleted + 1, 365),
      currentPhase: Math.floor(Math.random() * 7) + 1,
      lessonsCompleted,
      totalTimeSpent: lessonsCompleted * 180 + Math.floor(Math.random() * 600),
      currentStreak: lessonsCompleted > 0 ? Math.floor(Math.random() * 30) : 0,
      longestStreak: Math.floor(Math.random() * 50) + 5,
      lastLessonTopic: topics[Math.floor(Math.random() * topics.length)],
      kellyAge: ([5, 12, 18, 35, 55, 77] as const)[Math.floor(Math.random() * 6)],
      preferredLanguage: 'en',
      emailOptIn: Math.random() > 0.2,
      pushOptIn: Math.random() > 0.4,
      emailsSent: Math.floor(Math.random() * 10),
      emailsOpened: Math.floor(Math.random() * 8),
      lastEmailSent: Math.random() > 0.5 ? new Date(Date.now() - Math.random() * 14 * 24 * 60 * 60 * 1000).toISOString() : null,
      isOnline: lastSeenMinutes < 5,
      isVIP: Math.random() > 0.92,
      needsAttention: Math.random() > 0.88,
    }
  }).sort((a, b) => new Date(b.lastSeenAt).getTime() - new Date(a.lastSeenAt).getTime())
}

const MESSAGE_TEMPLATES: MessageTemplate[] = [
  {
    id: '1',
    name: 'Streak Reminder',
    type: 'email',
    subject: "Don't break your streak! 🔥",
    body: "Hi {{name}},\n\nYou've got a {{streak}}-day streak going! Don't let it slip.\n\nYour next lesson is ready: Day {{day}}\n\nSee you soon,\nKelly"
  },
  {
    id: '2',
    name: 'Welcome Back',
    type: 'email',
    subject: "We miss you! 👋",
    body: "Hi {{name}},\n\nIt's been a while since your last lesson. Ready to continue your learning journey?\n\nYou were on Day {{day}} - pick up where you left off!\n\nKelly"
  },
  {
    id: '3',
    name: 'Trial Ending',
    type: 'email',
    subject: "Your trial ends soon ⏰",
    body: "Hi {{name}},\n\nYour free trial is ending soon. You've completed {{lessons}} lessons so far!\n\nUpgrade now to keep learning with Kelly.\n\nBest,\nThe Curious Kelly Team"
  },
]

// Utility functions
const formatRelativeTime = (dateString: string): string => {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)
  
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return date.toLocaleDateString()
}

const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  if (hours > 0) return `${hours}h ${mins}m`
  return `${mins}m`
}

const getStatusBadge = (status: Learner['subscriptionStatus'], plan?: Learner['subscriptionPlan']) => {
  const styles: Record<string, string> = {
    free: 'bg-gray-600 text-gray-100',
    trial: 'bg-amber-500/20 text-amber-400 border border-amber-500/30',
    active: 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30',
    canceled: 'bg-red-500/20 text-red-400 border border-red-500/30',
    past_due: 'bg-orange-500/20 text-orange-400 border border-orange-500/30',
  }
  const labels: Record<string, string> = {
    free: 'Free',
    trial: 'Trial',
    active: plan ? `Active - ${plan}` : 'Active',
    canceled: 'Canceled',
    past_due: 'Past Due',
  }
  return <Badge className={`${styles[status]} text-xs font-medium`}>{labels[status]}</Badge>
}

// Components
function StatsBar({ learners }: { learners: Learner[] }) {
  const stats = useMemo(() => ({
    total: learners.length,
    online: learners.filter(l => l.isOnline).length,
    subscribed: learners.filter(l => l.subscriptionStatus === 'active').length,
    trial: learners.filter(l => l.subscriptionStatus === 'trial').length,
    streaks: learners.filter(l => l.currentStreak > 0).length,
  }), [learners])

  return (
    <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide">
      <StatPill icon={<Users className="w-4 h-4" />} label="Total" value={stats.total} />
      <StatPill icon={<Activity className="w-4 h-4 text-emerald-400" />} label="Online" value={stats.online} pulse />
      <StatPill icon={<CheckCircle className="w-4 h-4 text-blue-400" />} label="Subscribed" value={stats.subscribed} />
      <StatPill icon={<Clock className="w-4 h-4 text-amber-400" />} label="Trial" value={stats.trial} />
      <StatPill icon={<Flame className="w-4 h-4 text-orange-400" />} label="Streaks" value={stats.streaks} />
    </div>
  )
}

function StatPill({ icon, label, value, pulse }: { icon: React.ReactNode; label: string; value: number; pulse?: boolean }) {
  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-gray-800/60 rounded-full border border-gray-700/50 whitespace-nowrap">
      <span className={pulse ? 'animate-pulse' : ''}>{icon}</span>
      <span className="font-bold text-white">{value}</span>
      <span className="text-gray-400 text-sm">{label}</span>
    </div>
  )
}

function LearnerRow({ 
  learner, 
  isSelected, 
  onSelect, 
  onClick 
}: { 
  learner: Learner
  isSelected: boolean
  onSelect: () => void
  onClick: () => void
}) {
  return (
    <tr 
      className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors cursor-pointer group"
      onClick={onClick}
    >
      <td className="p-4" onClick={e => e.stopPropagation()}>
        <Checkbox checked={isSelected} onCheckedChange={onSelect} />
      </td>
      <td className="p-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            {learner.avatar ? (
              <img src={learner.avatar || "/placeholder.svg"} alt="" className="w-10 h-10 rounded-full bg-gray-700" />
            ) : (
              <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center">
                <Users className="w-5 h-5 text-gray-500" />
              </div>
            )}
            {learner.isOnline && (
              <span className="absolute bottom-0 right-0 w-3 h-3 bg-emerald-500 rounded-full border-2 border-gray-900" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={`font-medium ${learner.name ? 'text-white' : 'text-gray-500 italic'}`}>
                {learner.name || 'Anonymous'}
              </span>
              {learner.isVIP && <Star className="w-4 h-4 text-amber-400 fill-amber-400" />}
              {learner.needsAttention && <AlertTriangle className="w-4 h-4 text-orange-400" />}
            </div>
            <div className="text-sm text-gray-500 font-mono">
              {learner.email || learner.id}
            </div>
          </div>
        </div>
      </td>
      <td className="p-4">
        <div className="flex items-center gap-2">
          <span className="text-lg">{learner.countryFlag}</span>
          <div>
            <div className="text-white">{learner.city}</div>
            <div className="text-sm text-gray-500">{learner.country}</div>
          </div>
        </div>
      </td>
      <td className="p-4">
        <code className="text-sm text-gray-400 font-mono bg-gray-800/50 px-2 py-1 rounded">
          {learner.ip}
        </code>
      </td>
      <td className="p-4">
        {getStatusBadge(learner.subscriptionStatus, learner.subscriptionPlan)}
      </td>
      <td className="p-4">
        {learner.currentStreak > 0 ? (
          <span className="flex items-center gap-1 text-orange-400">
            <Flame className="w-4 h-4" />
            <span className="font-bold">{learner.currentStreak}</span>
          </span>
        ) : (
          <span className="text-gray-600">—</span>
        )}
      </td>
      <td className="p-4">
        <div>
          <div className="text-white font-medium">{learner.lessonsCompleted}</div>
          <div className="text-sm text-gray-500">Day {learner.currentDay}</div>
        </div>
      </td>
      <td className="p-4">
        <span className="text-gray-400">{formatRelativeTime(learner.lastSeenAt)}</span>
      </td>
      <td className="p-4">
        <DropdownMenu>
          <DropdownMenuTrigger asChild onClick={e => e.stopPropagation()}>
            <Button variant="ghost" size="icon" className="opacity-0 group-hover:opacity-100">
              <MoreVertical className="w-4 h-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="bg-gray-900 border-gray-700">
            <DropdownMenuItem className="text-white hover:bg-gray-800">
              <Mail className="w-4 h-4 mr-2" /> Send Email
            </DropdownMenuItem>
            <DropdownMenuItem className="text-white hover:bg-gray-800">
              <Bell className="w-4 h-4 mr-2" /> Send Push
            </DropdownMenuItem>
            <DropdownMenuItem className="text-white hover:bg-gray-800">
              <ExternalLink className="w-4 h-4 mr-2" /> View in Stripe
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </td>
    </tr>
  )
}

function LearnerDetailPanel({ 
  learner, 
  onClose,
  onSendMessage 
}: { 
  learner: Learner
  onClose: () => void
  onSendMessage: (type: 'email' | 'push') => void
}) {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <>
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40" onClick={onClose} />
      <div className="fixed right-0 top-0 h-full w-full max-w-lg bg-gray-900 border-l border-gray-700 z-50 overflow-y-auto">
        <div className="sticky top-0 bg-gray-900/95 backdrop-blur border-b border-gray-700 p-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Learner Profile</h2>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-5 h-5" />
          </Button>
        </div>

        <div className="p-6 space-y-6">
          {/* Header */}
          <div className="flex items-start gap-4">
            <div className="relative">
              {learner.avatar ? (
                <img src={learner.avatar || "/placeholder.svg"} alt="" className="w-16 h-16 rounded-full bg-gray-700" />
              ) : (
                <div className="w-16 h-16 rounded-full bg-gray-700 flex items-center justify-center">
                  <Users className="w-8 h-8 text-gray-500" />
                </div>
              )}
              {learner.isOnline && (
                <span className="absolute bottom-0 right-0 w-4 h-4 bg-emerald-500 rounded-full border-2 border-gray-900" />
              )}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <h3 className="text-xl font-bold text-white">{learner.name || 'Anonymous'}</h3>
                {learner.isVIP && <Star className="w-5 h-5 text-amber-400 fill-amber-400" />}
              </div>
              <p className="text-gray-400">{learner.email || learner.id}</p>
              <div className="flex items-center gap-2 mt-2">
                {getStatusBadge(learner.subscriptionStatus, learner.subscriptionPlan)}
                {learner.isOnline && (
                  <Badge className="bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    Online
                  </Badge>
                )}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button 
              className="flex-1 bg-blue-600 hover:bg-blue-700"
              onClick={() => onSendMessage('email')}
              disabled={!learner.emailOptIn}
            >
              <Mail className="w-4 h-4 mr-2" /> Send Email
            </Button>
            <Button 
              variant="outline" 
              className="flex-1 border-gray-600 hover:bg-gray-800 bg-transparent"
              onClick={() => onSendMessage('push')}
              disabled={!learner.pushOptIn}
            >
              <Bell className="w-4 h-4 mr-2" /> Send Push
            </Button>
          </div>

          {/* Location & Device */}
          <Section title="Location & Device" icon={<Globe className="w-4 h-4" />}>
            <InfoRow label="Location" value={`${learner.countryFlag} ${learner.city}, ${learner.region}, ${learner.country}`} />
            <InfoRow label="Timezone" value={learner.timezone} />
            <InfoRow 
              label="IP Address" 
              value={
                <div className="flex items-center gap-2">
                  <code className="font-mono text-sm">{learner.ipFull}</code>
                  <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => copyToClipboard(learner.ipFull)}>
                    <Copy className="w-3 h-3" />
                  </Button>
                </div>
              } 
            />
            <InfoRow label="Device" value={`${learner.device} • ${learner.browser} • ${learner.os}`} />
          </Section>

          {/* Subscription */}
          <Section title="Subscription" icon={<CreditCard className="w-4 h-4" />}>
            <InfoRow label="Status" value={getStatusBadge(learner.subscriptionStatus, learner.subscriptionPlan)} />
            {learner.stripeCustomerId && (
              <InfoRow 
                label="Stripe ID" 
                value={
                  <a 
                    href={`https://dashboard.stripe.com/customers/${learner.stripeCustomerId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:underline flex items-center gap-1"
                  >
                    {learner.stripeCustomerId} <ExternalLink className="w-3 h-3" />
                  </a>
                } 
              />
            )}
            {learner.currentPeriodEnd && (
              <InfoRow label="Renews" value={new Date(learner.currentPeriodEnd).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })} />
            )}
            {learner.trialEndsAt && (
              <InfoRow label="Trial Ends" value={new Date(learner.trialEndsAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })} />
            )}
          </Section>

          {/* Learning Progress */}
          <Section title="Learning Progress" icon={<Award className="w-4 h-4" />}>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{learner.lessonsCompleted}</div>
                <div className="text-xs text-gray-500">Lessons</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-400 flex items-center justify-center gap-1">
                  <Flame className="w-5 h-5" /> {learner.currentStreak}
                </div>
                <div className="text-xs text-gray-500">Current Streak</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{learner.longestStreak}</div>
                <div className="text-xs text-gray-500">Best Streak</div>
              </div>
            </div>
            <InfoRow label="Current Day" value={`Day ${learner.currentDay} / 365`} />
            <InfoRow label="Total Time" value={formatDuration(learner.totalTimeSpent)} />
            <InfoRow label="Kelly Age" value={`${learner.kellyAge} years`} />
            <InfoRow label="Language" value={learner.preferredLanguage.toUpperCase()} />
            {learner.lastLessonTopic && <InfoRow label="Last Lesson" value={learner.lastLessonTopic} />}
          </Section>

          {/* Communication */}
          <Section title="Communication" icon={<MessageSquare className="w-4 h-4" />}>
            <InfoRow label="Email Opt-in" value={learner.emailOptIn ? <CheckCircle className="w-4 h-4 text-emerald-400" /> : <XCircle className="w-4 h-4 text-gray-500" />} />
            <InfoRow label="Push Opt-in" value={learner.pushOptIn ? <CheckCircle className="w-4 h-4 text-emerald-400" /> : <XCircle className="w-4 h-4 text-gray-500" />} />
            <InfoRow label="Emails Sent" value={learner.emailsSent} />
            <InfoRow label="Open Rate" value={learner.emailsSent > 0 ? `${Math.round((learner.emailsOpened / learner.emailsSent) * 100)}%` : '—'} />
            {learner.lastEmailSent && (
              <InfoRow label="Last Email" value={new Date(learner.lastEmailSent).toLocaleString()} />
            )}
          </Section>

          {/* Timeline */}
          <Section title="Timeline" icon={<Clock className="w-4 h-4" />}>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 mt-2 rounded-full bg-gray-500" />
                <div>
                  <div className="text-white">Last seen</div>
                  <div className="text-sm text-gray-500">{new Date(learner.lastSeenAt).toLocaleString()}</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 mt-2 rounded-full bg-blue-500" />
                <div>
                  <div className="text-white">Signed up via {learner.authProvider}</div>
                  <div className="text-sm text-gray-500">{new Date(learner.createdAt).toLocaleString()}</div>
                </div>
              </div>
            </div>
          </Section>
        </div>
      </div>
    </>
  )
}

function Section({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
      <h4 className="flex items-center gap-2 text-sm font-semibold text-gray-400 mb-3">
        {icon} {title}
      </h4>
      <div className="space-y-2">{children}</div>
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-gray-500">{label}</span>
      <span className="text-white">{value}</span>
    </div>
  )
}

function ComposeMessageModal({
  recipients,
  onClose,
  onSend
}: {
  recipients: Learner[]
  onClose: () => void
  onSend: (data: { subject: string; body: string; type: 'email' | 'push' }) => void
}) {
  const [templateId, setTemplateId] = useState<string>('')
  const [subject, setSubject] = useState('')
  const [body, setBody] = useState('')
  const [type, setType] = useState<'email' | 'push'>('email')

  const handleTemplateChange = (id: string) => {
    setTemplateId(id)
    const template = MESSAGE_TEMPLATES.find(t => t.id === id)
    if (template) {
      setSubject(template.subject)
      setBody(template.body)
      setType(template.type)
    }
  }

  const insertVariable = (variable: string) => {
    setBody(prev => prev + `{{${variable}}}`)
  }

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="bg-gray-900 border-gray-700 max-w-2xl">
        <DialogHeader>
          <DialogTitle className="text-white flex items-center justify-between">
            Compose Message
            <Badge variant="outline" className="ml-2">{recipients.length} recipients</Badge>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4 mt-4">
          <div>
            <label className="text-sm text-gray-400 mb-2 block">Template</label>
            <Select value={templateId} onValueChange={handleTemplateChange}>
              <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                <SelectValue placeholder="Select a template..." />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-700">
                {MESSAGE_TEMPLATES.map(t => (
                  <SelectItem key={t.id} value={t.id} className="text-white hover:bg-gray-700">
                    {t.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-sm text-gray-400 mb-2 block">Subject</label>
            <Input 
              value={subject} 
              onChange={e => setSubject(e.target.value)}
              placeholder="Email subject..."
              className="bg-gray-800 border-gray-700 text-white"
            />
          </div>

          <div>
            <label className="text-sm text-gray-400 mb-2 block">Message</label>
            <Textarea 
              value={body} 
              onChange={e => setBody(e.target.value)}
              rows={8}
              placeholder="Write your message..."
              className="bg-gray-800 border-gray-700 text-white font-mono text-sm"
            />
          </div>

          <div>
            <label className="text-sm text-gray-400 mb-2 block">Variables</label>
            <div className="flex gap-2 flex-wrap">
              {['name', 'email', 'streak', 'lessons', 'day'].map(v => (
                <Button 
                  key={v} 
                  variant="outline" 
                  size="sm" 
                  className="border-gray-600 text-gray-300 hover:bg-gray-800 bg-transparent"
                  onClick={() => insertVariable(v)}
                >
                  {`{{${v}}}`}
                </Button>
              ))}
            </div>
          </div>

          <div className="flex justify-end gap-2 pt-4 border-t border-gray-700">
            <Button variant="ghost" onClick={onClose}>Cancel</Button>
            <Button 
              className="bg-blue-600 hover:bg-blue-700"
              onClick={() => onSend({ subject, body, type })}
              disabled={!subject || !body}
            >
              <Send className="w-4 h-4 mr-2" /> Send Message
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

// Main Page Component
export default function LearnersPage() {
  const [learners] = useState<Learner[]>(() => generateMockLearners())
  const [searchQuery, setSearchQuery] = useState('')
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [filterCountry, setFilterCountry] = useState<string>('all')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [selectedLearner, setSelectedLearner] = useState<Learner | null>(null)
  const [isComposeOpen, setIsComposeOpen] = useState(false)
  const [isLive, setIsLive] = useState(true)

  const searchParams = useSearchParams()

  // Get unique countries for filter
  const countries = useMemo(() => {
    const countrySet = new Set(learners.map(l => l.country))
    return Array.from(countrySet).sort()
  }, [learners])

  // Filter learners
  const filteredLearners = useMemo(() => {
    return learners.filter(l => {
      const matchesSearch = !searchQuery || 
        l.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        l.email?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        l.city.toLowerCase().includes(searchQuery.toLowerCase()) ||
        l.country.toLowerCase().includes(searchQuery.toLowerCase()) ||
        l.id.toLowerCase().includes(searchQuery.toLowerCase())
      
      const matchesStatus = filterStatus === 'all' || l.subscriptionStatus === filterStatus
      const matchesCountry = filterCountry === 'all' || l.country === filterCountry
      
      return matchesSearch && matchesStatus && matchesCountry
    })
  }, [learners, searchQuery, filterStatus, filterCountry])

  // Selection handlers
  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleSelectAll = () => {
    if (selectedIds.size === filteredLearners.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(filteredLearners.map(l => l.id)))
    }
  }

  const selectedLearners = learners.filter(l => selectedIds.has(l.id))

  // Export CSV
  const exportCsv = () => {
    const headers = ['Name', 'Email', 'City', 'Country', 'Status', 'Streak', 'Lessons', 'Last Seen']
    const rows = selectedLearners.map(l => [
      l.name || 'Anonymous',
      l.email || l.id,
      l.city,
      l.country,
      l.subscriptionStatus,
      l.currentStreak,
      l.lessonsCompleted,
      l.lastSeenAt
    ])
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'learners.csv'
    a.click()
  }

  // Real-time updates simulation
  useEffect(() => {
    if (!isLive) return
    const interval = setInterval(() => {
      // In real app, this would poll the API
    }, 5000)
    return () => clearInterval(interval)
  }, [isLive])

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-950 text-white">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-slate-900/95 backdrop-blur border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">👩‍🏫</span>
            <div>
              <h1 className="text-xl font-bold text-white">Curious Kelly</h1>
              <p className="text-sm text-gray-400">Student Information System</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant={isLive ? "default" : "outline"}
              size="sm"
              className={isLive ? "bg-emerald-600 hover:bg-emerald-700" : "border-gray-600"}
              onClick={() => setIsLive(!isLive)}
            >
              <span className={`w-2 h-2 rounded-full mr-2 ${isLive ? 'bg-white animate-pulse' : 'bg-gray-500'}`} />
              {isLive ? 'Live' : 'Paused'}
            </Button>
            <Button variant="ghost" size="icon">
              <RefreshCw className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon">
              <Settings className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {/* Stats Bar */}
        <StatsBar learners={learners} />

        {/* Toolbar */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <Input
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search by name, email, city, country..."
              className="pl-10 bg-gray-800/50 border-gray-700 text-white placeholder:text-gray-500"
            />
          </div>
          <div className="flex gap-2">
            <Select value={filterStatus} onValueChange={setFilterStatus}>
              <SelectTrigger className="w-36 bg-gray-800/50 border-gray-700 text-white">
                <Filter className="w-4 h-4 mr-2" />
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-700">
                <SelectItem value="all" className="text-white hover:bg-gray-700">All Status</SelectItem>
                <SelectItem value="free" className="text-white hover:bg-gray-700">Free</SelectItem>
                <SelectItem value="trial" className="text-white hover:bg-gray-700">Trial</SelectItem>
                <SelectItem value="active" className="text-white hover:bg-gray-700">Active</SelectItem>
                <SelectItem value="canceled" className="text-white hover:bg-gray-700">Canceled</SelectItem>
                <SelectItem value="past_due" className="text-white hover:bg-gray-700">Past Due</SelectItem>
              </SelectContent>
            </Select>
            <Select value={filterCountry} onValueChange={setFilterCountry}>
              <SelectTrigger className="w-36 bg-gray-800/50 border-gray-700 text-white">
                <Globe className="w-4 h-4 mr-2" />
                <SelectValue placeholder="Country" />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-700">
                <SelectItem value="all" className="text-white hover:bg-gray-700">All Countries</SelectItem>
                {countries.map(c => (
                  <SelectItem key={c} value={c} className="text-white hover:bg-gray-700">{c}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Bulk Actions */}
        {selectedIds.size > 0 && (
          <div className="flex items-center gap-4 p-3 bg-blue-600/10 border border-blue-600/30 rounded-lg">
            <span className="text-blue-400 font-medium">{selectedIds.size} selected</span>
            <Button 
              size="sm" 
              className="bg-blue-600 hover:bg-blue-700"
              onClick={() => setIsComposeOpen(true)}
            >
              <Mail className="w-4 h-4 mr-2" /> Message
            </Button>
            <Button 
              size="sm" 
              variant="outline" 
              className="border-gray-600 hover:bg-gray-800 bg-transparent"
              onClick={exportCsv}
            >
              <Download className="w-4 h-4 mr-2" /> Export
            </Button>
            <Button 
              size="sm" 
              variant="ghost"
              onClick={() => setSelectedIds(new Set())}
            >
              <X className="w-4 h-4 mr-2" /> Clear
            </Button>
          </div>
        )}

        {/* Table */}
        <div className="bg-gray-800/30 rounded-xl border border-gray-700/50 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-800/50 border-b border-gray-700/50">
                <tr>
                  <th className="p-4 text-left">
                    <Checkbox 
                      checked={selectedIds.size === filteredLearners.length && filteredLearners.length > 0}
                      onCheckedChange={toggleSelectAll}
                    />
                  </th>
                  <th className="p-4 text-left text-sm font-medium text-gray-400">Learner</th>
                  <th className="p-4 text-left text-sm font-medium text-gray-400">Location</th>
                  <th className="p-4 text-left text-sm font-medium text-gray-400">IP Address</th>
                  <th className="p-4 text-left text-sm font-medium text-gray-400">Status</th>
                  <th className="p-4 text-left text-sm font-medium text-gray-400">Streak</th>
                  <th className="p-4 text-left text-sm font-medium text-gray-400">Lessons</th>
                  <th className="p-4 text-left text-sm font-medium text-gray-400">Last Seen</th>
                  <th className="p-4"></th>
                </tr>
              </thead>
              <tbody>
                {filteredLearners.map(learner => (
                  <LearnerRow
                    key={learner.id}
                    learner={learner}
                    isSelected={selectedIds.has(learner.id)}
                    onSelect={() => toggleSelect(learner.id)}
                    onClick={() => setSelectedLearner(learner)}
                  />
                ))}
              </tbody>
            </table>
          </div>
          
          {filteredLearners.length === 0 && (
            <div className="p-12 text-center">
              <Users className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">No learners found matching your filters.</p>
            </div>
          )}
        </div>

        {/* Pagination placeholder */}
        <div className="flex items-center justify-between text-sm text-gray-400">
          <span>Showing {filteredLearners.length} of {learners.length} learners</span>
        </div>
      </main>

      {/* Detail Panel */}
      {selectedLearner && (
        <LearnerDetailPanel
          learner={selectedLearner}
          onClose={() => setSelectedLearner(null)}
          onSendMessage={(type) => {
            setSelectedIds(new Set([selectedLearner.id]))
            setIsComposeOpen(true)
          }}
        />
      )}

      {/* Compose Modal */}
      {isComposeOpen && (
        <ComposeMessageModal
          recipients={selectedLearners}
          onClose={() => setIsComposeOpen(false)}
          onSend={(data) => {
            console.log('Sending message:', data, 'to', selectedLearners.map(l => l.id))
            setIsComposeOpen(false)
            setSelectedIds(new Set())
          }}
        />
      )}
    </div>
  )
}

// Loading Component
export function Loading() {
  return null
}
