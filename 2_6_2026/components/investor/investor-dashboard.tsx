"use client"

import { useState, useEffect } from 'react'
import useSWR from 'swr'
import { 
  BarChart3, Download, FileSpreadsheet, FileText, 
  Calendar, MessageSquare, TrendingUp, Users, Eye, 
  Clock, DollarSign, Target, Zap, ChevronRight, 
  Send, CheckCircle2, XCircle, AlertCircle, RefreshCw,
  PieChart, Activity, Briefcase
} from 'lucide-react'
import { cn } from '@/lib/utils'

const fetcher = (url: string) => fetch(url).then(res => res.json())

// Formatting utilities
const fmt = (n: number, decimals = 0) => {
  if (n == null || isNaN(n)) return '—'
  const abs = Math.abs(n)
  if (abs >= 1e9) return `$${(abs / 1e9).toFixed(decimals)}B`
  if (abs >= 1e6) return `$${(abs / 1e6).toFixed(decimals)}M`
  if (abs >= 1e3) return `$${(abs / 1e3).toFixed(0)}K`
  return `$${abs.toLocaleString()}`
}

interface MetricsData {
  totalModels: number
  totalViews: number
  totalQuestions: number
  pendingQuestions: number
  scheduledReviews: number
  avgInvestmentSize: number
  investorGrowth: number
  recentActivity: Array<{
    type: string
    investor: string
    action: string
    timestamp: string
  }>
}

interface Question {
  id: string
  investor_model_id: string
  investor_name: string
  organization: string
  question: string
  reply: string | null
  created_at: string
  replied_at: string | null
}

interface Review {
  id: string
  investor_model_id: string
  investor_name: string
  organization: string
  scheduled_date: string
  status: 'scheduled' | 'completed' | 'cancelled'
  notes: string
}

export function InvestorDashboard() {
  const [activeTab, setActiveTab] = useState<'overview' | 'questions' | 'reviews' | 'exports'>('overview')
  const [replyText, setReplyText] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState<string | null>(null)

  // Fetch real metrics
  const { data: metrics, mutate: mutateMetrics } = useSWR<MetricsData>(
    '/api/ziggurat/metrics',
    fetcher,
    { refreshInterval: 30000 }
  )

  // Fetch questions
  const { data: questionsData, mutate: mutateQuestions } = useSWR<{ questions: Question[] }>(
    '/api/ziggurat/investors/questions',
    fetcher
  )

  // Fetch reviews
  const { data: reviewsData, mutate: mutateReviews } = useSWR<{ reviews: Review[] }>(
    '/api/ziggurat/investors/reviews',
    fetcher
  )

  const questions = questionsData?.questions || []
  const reviews = reviewsData?.reviews || []
  const pendingQuestions = questions.filter(q => !q.reply)
  const upcomingReviews = reviews.filter(r => r.status === 'scheduled')

  // Submit reply to question
  const submitReply = async (questionId: string) => {
    if (!replyText[questionId]?.trim()) return
    setSubmitting(questionId)
    try {
      const res = await fetch('/api/ziggurat/investors/questions', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          questionId,
          reply: replyText[questionId]
        })
      })
      if (res.ok) {
        setReplyText(prev => ({ ...prev, [questionId]: '' }))
        mutateQuestions()
      }
    } catch (error) {
      console.error('Failed to submit reply:', error)
    } finally {
      setSubmitting(null)
    }
  }

  // Export functions
  const exportData = async (type: 'csv' | 'json' | 'pdf') => {
    const res = await fetch(`/api/ziggurat/export?type=${type}`)
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ziggurat-investor-data-${new Date().toISOString().split('T')[0]}.${type}`
    a.click()
    URL.revokeObjectURL(url)
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'questions', label: 'Questions', icon: MessageSquare, badge: pendingQuestions.length },
    { id: 'reviews', label: 'Reviews', icon: Calendar, badge: upcomingReviews.length },
    { id: 'exports', label: 'Exports', icon: Download },
  ]

  return (
    <div className="bg-zinc-900/50 rounded-2xl border border-white/10 overflow-hidden">
      {/* Tab Navigation */}
      <div className="flex border-b border-white/10">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={cn(
              "flex items-center gap-2 px-6 py-4 text-sm font-medium transition-all relative",
              activeTab === tab.id
                ? "text-white bg-white/5"
                : "text-white/50 hover:text-white hover:bg-white/[0.02]"
            )}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
            {tab.badge && tab.badge > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center">
                {tab.badge}
              </span>
            )}
            {activeTab === tab.id && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-amber-500" />
            )}
          </button>
        ))}
      </div>

      <div className="p-6">
        {/* OVERVIEW TAB */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Key Metrics Grid */}
            <div className="grid grid-cols-4 gap-4">
              {[
                { label: 'Total Models', value: metrics?.totalModels || 0, icon: Briefcase, color: 'amber', trend: '+2 this week' },
                { label: 'Total Views', value: metrics?.totalViews || 0, icon: Eye, color: 'blue', trend: '+127 this week' },
                { label: 'Open Questions', value: pendingQuestions.length, icon: MessageSquare, color: 'red', trend: pendingQuestions.length > 0 ? 'Action needed' : 'All answered' },
                { label: 'Upcoming Reviews', value: upcomingReviews.length, icon: Calendar, color: 'emerald', trend: upcomingReviews.length > 0 ? 'Next: ' + (upcomingReviews[0]?.scheduled_date?.split('T')[0] || '') : 'None scheduled' },
              ].map(({ label, value, icon: Icon, color, trend }) => (
                <div
                  key={label}
                  className={cn(
                    "p-5 rounded-xl border transition-all hover:scale-[1.02]",
                    color === 'amber' && "bg-gradient-to-br from-amber-500/10 to-amber-600/5 border-amber-500/20",
                    color === 'blue' && "bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20",
                    color === 'red' && "bg-gradient-to-br from-red-500/10 to-red-600/5 border-red-500/20",
                    color === 'emerald' && "bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20"
                  )}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <Icon className={cn(
                      "w-4 h-4",
                      color === 'amber' && "text-amber-400",
                      color === 'blue' && "text-blue-400",
                      color === 'red' && "text-red-400",
                      color === 'emerald' && "text-emerald-400"
                    )} />
                    <span className="text-xs text-white/50 uppercase tracking-wider">{label}</span>
                  </div>
                  <div className="text-3xl font-bold font-mono mb-1">{value}</div>
                  <div className="text-xs text-white/40">{trend}</div>
                </div>
              ))}
            </div>

            {/* Recent Activity */}
            <div className="bg-white/5 rounded-xl p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold flex items-center gap-2">
                  <Activity className="w-4 h-4 text-amber-400" />
                  Recent Activity
                </h3>
                <button 
                  onClick={() => mutateMetrics()}
                  className="text-xs text-white/40 hover:text-white flex items-center gap-1"
                >
                  <RefreshCw className="w-3 h-3" />
                  Refresh
                </button>
              </div>
              
              <div className="space-y-3">
                {(metrics?.recentActivity || [
                  { type: 'view', investor: 'Demo Investor', action: 'viewed model', timestamp: new Date().toISOString() },
                  { type: 'question', investor: 'Growth Capital', action: 'asked a question', timestamp: new Date(Date.now() - 3600000).toISOString() },
                  { type: 'model', investor: 'Prudent Partners', action: 'created workspace', timestamp: new Date(Date.now() - 86400000).toISOString() },
                ]).slice(0, 5).map((activity, i) => (
                  <div key={i} className="flex items-center gap-3 py-2 border-b border-white/5 last:border-0">
                    <div className={cn(
                      "w-8 h-8 rounded-full flex items-center justify-center",
                      activity.type === 'view' && "bg-blue-500/20",
                      activity.type === 'question' && "bg-amber-500/20",
                      activity.type === 'model' && "bg-emerald-500/20",
                      activity.type === 'review' && "bg-purple-500/20"
                    )}>
                      {activity.type === 'view' && <Eye className="w-4 h-4 text-blue-400" />}
                      {activity.type === 'question' && <MessageSquare className="w-4 h-4 text-amber-400" />}
                      {activity.type === 'model' && <Briefcase className="w-4 h-4 text-emerald-400" />}
                      {activity.type === 'review' && <Calendar className="w-4 h-4 text-purple-400" />}
                    </div>
                    <div className="flex-1">
                      <span className="font-medium">{activity.investor}</span>
                      <span className="text-white/50"> {activity.action}</span>
                    </div>
                    <span className="text-xs text-white/30">
                      {new Date(activity.timestamp).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 border border-purple-500/20 rounded-xl p-5">
                <PieChart className="w-5 h-5 text-purple-400 mb-3" />
                <div className="text-xs text-white/50 uppercase tracking-wider mb-1">Avg Investment Size</div>
                <div className="text-2xl font-bold font-mono">{fmt(metrics?.avgInvestmentSize || 25000000)}</div>
              </div>
              <div className="bg-gradient-to-br from-cyan-500/10 to-cyan-600/5 border border-cyan-500/20 rounded-xl p-5">
                <TrendingUp className="w-5 h-5 text-cyan-400 mb-3" />
                <div className="text-xs text-white/50 uppercase tracking-wider mb-1">Investor Growth</div>
                <div className="text-2xl font-bold font-mono text-emerald-400">+{metrics?.investorGrowth || 15}%</div>
              </div>
              <div className="bg-gradient-to-br from-pink-500/10 to-pink-600/5 border border-pink-500/20 rounded-xl p-5">
                <Target className="w-5 h-5 text-pink-400 mb-3" />
                <div className="text-xs text-white/50 uppercase tracking-wider mb-1">Response Rate</div>
                <div className="text-2xl font-bold font-mono">
                  {questions.length > 0 ? Math.round((questions.filter(q => q.reply).length / questions.length) * 100) : 100}%
                </div>
              </div>
            </div>
          </div>
        )}

        {/* QUESTIONS TAB */}
        {activeTab === 'questions' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold">
                Investor Questions ({questions.length})
              </h3>
              <div className="flex gap-2">
                <span className="text-xs px-2 py-1 bg-red-500/20 text-red-300 rounded">
                  {pendingQuestions.length} pending
                </span>
                <span className="text-xs px-2 py-1 bg-emerald-500/20 text-emerald-300 rounded">
                  {questions.length - pendingQuestions.length} answered
                </span>
              </div>
            </div>

            {questions.length === 0 ? (
              <div className="text-center py-12 text-white/40">
                <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No investor questions yet</p>
                <p className="text-sm mt-1">Questions from investor workspaces will appear here</p>
              </div>
            ) : (
              <div className="space-y-4">
                {questions.map(q => (
                  <div
                    key={q.id}
                    className={cn(
                      "rounded-xl border p-5",
                      q.reply
                        ? "bg-white/5 border-white/10"
                        : "bg-amber-500/5 border-amber-500/30"
                    )}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="font-medium">{q.investor_name}</div>
                        <div className="text-sm text-white/50">{q.organization}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        {q.reply ? (
                          <span className="flex items-center gap-1 text-xs text-emerald-400">
                            <CheckCircle2 className="w-3 h-3" />
                            Answered
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-xs text-amber-400">
                            <AlertCircle className="w-3 h-3" />
                            Pending
                          </span>
                        )}
                        <span className="text-xs text-white/30">
                          {new Date(q.created_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>

                    <div className="bg-white/5 rounded-lg p-4 mb-4">
                      <p className="text-white/90">{q.question}</p>
                    </div>

                    {q.reply ? (
                      <div className="ml-4 pl-4 border-l-2 border-emerald-500/30">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-sm font-medium text-emerald-400">Reply</span>
                          <span className="text-xs text-white/30">
                            {q.replied_at ? new Date(q.replied_at).toLocaleDateString() : ''}
                          </span>
                        </div>
                        <p className="text-sm text-white/70">{q.reply}</p>
                      </div>
                    ) : (
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={replyText[q.id] || ''}
                          onChange={(e) => setReplyText(prev => ({ ...prev, [q.id]: e.target.value }))}
                          placeholder="Type your reply..."
                          className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-sm placeholder:text-white/30 focus:outline-none focus:border-amber-500/50"
                        />
                        <button
                          onClick={() => submitReply(q.id)}
                          disabled={submitting === q.id || !replyText[q.id]?.trim()}
                          className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-black rounded-lg text-sm font-medium hover:bg-amber-400 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {submitting === q.id ? (
                            <RefreshCw className="w-4 h-4 animate-spin" />
                          ) : (
                            <Send className="w-4 h-4" />
                          )}
                          Reply
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* REVIEWS TAB */}
        {activeTab === 'reviews' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold">
                Scheduled Reviews ({reviews.length})
              </h3>
            </div>

            {reviews.length === 0 ? (
              <div className="text-center py-12 text-white/40">
                <Calendar className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No reviews scheduled</p>
                <p className="text-sm mt-1">Investors can schedule reviews from their workspace</p>
              </div>
            ) : (
              <div className="space-y-3">
                {reviews.map(r => (
                  <div
                    key={r.id}
                    className={cn(
                      "flex items-center gap-4 p-4 rounded-xl border",
                      r.status === 'scheduled' && "bg-blue-500/5 border-blue-500/30",
                      r.status === 'completed' && "bg-emerald-500/5 border-emerald-500/30",
                      r.status === 'cancelled' && "bg-red-500/5 border-red-500/30"
                    )}
                  >
                    <div className={cn(
                      "w-12 h-12 rounded-xl flex items-center justify-center",
                      r.status === 'scheduled' && "bg-blue-500/20",
                      r.status === 'completed' && "bg-emerald-500/20",
                      r.status === 'cancelled' && "bg-red-500/20"
                    )}>
                      <Calendar className={cn(
                        "w-6 h-6",
                        r.status === 'scheduled' && "text-blue-400",
                        r.status === 'completed' && "text-emerald-400",
                        r.status === 'cancelled' && "text-red-400"
                      )} />
                    </div>

                    <div className="flex-1">
                      <div className="font-medium">{r.investor_name}</div>
                      <div className="text-sm text-white/50">{r.organization}</div>
                      {r.notes && <div className="text-xs text-white/40 mt-1">{r.notes}</div>}
                    </div>

                    <div className="text-right">
                      <div className="text-sm font-mono">
                        {new Date(r.scheduled_date).toLocaleDateString('en-US', {
                          weekday: 'short',
                          month: 'short',
                          day: 'numeric'
                        })}
                      </div>
                      <div className={cn(
                        "text-xs mt-1",
                        r.status === 'scheduled' && "text-blue-400",
                        r.status === 'completed' && "text-emerald-400",
                        r.status === 'cancelled' && "text-red-400"
                      )}>
                        {r.status.charAt(0).toUpperCase() + r.status.slice(1)}
                      </div>
                    </div>

                    <ChevronRight className="w-4 h-4 text-white/30" />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* EXPORTS TAB */}
        {activeTab === 'exports' && (
          <div className="space-y-6">
            <div className="grid grid-cols-3 gap-4">
              {[
                {
                  title: 'Financial Models (CSV)',
                  description: 'Export all investor workspaces with 5-year projections',
                  icon: FileSpreadsheet,
                  color: 'emerald',
                  action: () => exportData('csv'),
                },
                {
                  title: 'Full Data (JSON)',
                  description: 'Complete dataset including questions, reviews, and activity',
                  icon: FileText,
                  color: 'blue',
                  action: () => exportData('json'),
                },
                {
                  title: 'Investor Report (PDF)',
                  description: 'Formatted report with charts and analysis',
                  icon: Download,
                  color: 'purple',
                  action: () => exportData('pdf'),
                },
              ].map(({ title, description, icon: Icon, color, action }) => (
                <button
                  key={title}
                  onClick={action}
                  className={cn(
                    "text-left p-6 rounded-xl border transition-all hover:scale-[1.02] group",
                    color === 'emerald' && "bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20 hover:border-emerald-500/40",
                    color === 'blue' && "bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20 hover:border-blue-500/40",
                    color === 'purple' && "bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20 hover:border-purple-500/40"
                  )}
                >
                  <Icon className={cn(
                    "w-8 h-8 mb-4 transition-transform group-hover:scale-110",
                    color === 'emerald' && "text-emerald-400",
                    color === 'blue' && "text-blue-400",
                    color === 'purple' && "text-purple-400"
                  )} />
                  <h4 className="font-semibold mb-2">{title}</h4>
                  <p className="text-sm text-white/50">{description}</p>
                </button>
              ))}
            </div>

            <div className="bg-white/5 rounded-xl p-5">
              <h4 className="font-semibold mb-4">Export History</h4>
              <div className="text-center py-8 text-white/30">
                <Download className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Export history will appear here</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
