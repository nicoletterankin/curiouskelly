'use client'

import React from "react"

/**
 * Video Production Dashboard
 * 
 * Central command for monitoring and controlling video generation pipeline.
 * Shows status of all 365 days × 5 phases × multiple age groups.
 */

import { useState, useEffect, useCallback } from 'react'
import Link from 'next/link'
import { 
  Play, Pause, RefreshCw, CheckCircle, XCircle, 
  Clock, Zap, Video, Volume2, ArrowLeft,
  ChevronRight, AlertTriangle, Loader2
} from 'lucide-react'

interface DayStatus {
  day_number: number
  total_variants: number
  with_video: number
  with_audio: number
  phases: string[]
  age_groups: string[]
  completion_pct: number
}

interface PipelineStats {
  total_assets: number
  with_video: number
  with_audio: number
  pending: number
  in_progress: number
  failed: number
  days_complete: number
  heygen_credits: number | null
}

interface BatchJob {
  id: string
  day: number
  phase: string
  age_group: string
  status: 'queued' | 'generating' | 'completed' | 'failed'
  video_id?: string
  error?: string
}

export default function VideoProductionPage() {
  const [stats, setStats] = useState<PipelineStats | null>(null)
  const [dayStatuses, setDayStatuses] = useState<DayStatus[]>([])
  const [batchJobs, setBatchJobs] = useState<BatchJob[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [loading, setLoading] = useState(true)
  const [selectedDay, setSelectedDay] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Fetch production stats
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch('/api/admin/video-stats')
      if (res.ok) {
        const data = await res.json()
        setStats(data.stats)
        setDayStatuses(data.dayStatuses || [])
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStats()
    const interval = setInterval(fetchStats, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [fetchStats])

  // Start batch generation
  const startBatch = async (days: number[], phases: string[] = ['hook', 'story', 'wonder', 'action', 'wisdom']) => {
    setIsRunning(true)
    setError(null)
    
    try {
      const res = await fetch('/api/heygen/mega-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          days, 
          phases,
          age_groups: ['adult'],
          language: 'en',
          max_credits: 50 // Conservative limit per batch
        })
      })
      
      if (!res.ok) {
        const data = await res.json()
        setError(data.error || 'Batch start failed')
      } else {
        const data = await res.json()
        setBatchJobs(data.jobs || [])
      }
    } catch (err) {
      setError('Network error starting batch')
    } finally {
      setIsRunning(false)
      fetchStats()
    }
  }

  // Generate single day
  const generateDay = async (day: number) => {
    setSelectedDay(day)
    await startBatch([day])
    setSelectedDay(null)
  }

  // Priority days (from thesis - first week most critical)
  const priorityDays = [1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 24, 25, 26, 27, 28]
  
  // Days with complete videos
  const completeDays = dayStatuses.filter(d => d.with_video >= 5).map(d => d.day_number)
  const partialDays = dayStatuses.filter(d => d.with_video > 0 && d.with_video < 5).map(d => d.day_number)

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-zinc-400 animate-spin" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-white/10 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <Link href="/admin" className="text-zinc-500 hover:text-white transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-xl font-semibold">Video Production</h1>
              <p className="text-sm text-zinc-500">Generate Kelly lesson videos via HeyGen</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => fetchStats()}
              className="px-3 py-1.5 text-sm bg-zinc-900 border border-white/10 rounded-lg hover:bg-zinc-800 transition-colors flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <span className="text-red-300">{error}</span>
          </div>
        )}

        {/* Stats Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-8">
          <StatCard 
            label="Total Assets" 
            value={stats?.total_assets?.toLocaleString() || '0'} 
            icon={<Video className="w-4 h-4" />}
          />
          <StatCard 
            label="With Video" 
            value={stats?.with_video?.toLocaleString() || '0'} 
            icon={<CheckCircle className="w-4 h-4 text-emerald-400" />}
            highlight
          />
          <StatCard 
            label="With Audio" 
            value={stats?.with_audio?.toLocaleString() || '0'} 
            icon={<Volume2 className="w-4 h-4 text-blue-400" />}
          />
          <StatCard 
            label="Days Complete" 
            value={`${completeDays.length}/365`} 
            icon={<CheckCircle className="w-4 h-4 text-emerald-400" />}
          />
          <StatCard 
            label="Pending" 
            value={stats?.pending?.toLocaleString() || '0'} 
            icon={<Clock className="w-4 h-4 text-amber-400" />}
          />
          <StatCard 
            label="HeyGen Credits" 
            value={stats?.heygen_credits?.toString() || '?'} 
            icon={<Zap className="w-4 h-4 text-purple-400" />}
          />
        </div>

        {/* Quick Actions */}
        <div className="mb-8 p-6 bg-zinc-900/50 border border-white/10 rounded-xl">
          <h2 className="text-lg font-medium mb-4">Quick Actions</h2>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => startBatch([1, 2, 3, 4, 5, 6, 7])}
              disabled={isRunning}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
            >
              {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Generate Week 1 (Days 1-7)
            </button>
            <button
              onClick={() => startBatch(priorityDays.filter(d => !completeDays.includes(d)))}
              disabled={isRunning}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
            >
              {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              Generate Priority Days
            </button>
            <button
              onClick={() => startBatch([30, 60, 90, 100, 180, 200, 300, 365])}
              disabled={isRunning}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
            >
              Milestone Days
            </button>
          </div>
        </div>

        {/* Day Grid - Visual representation of all 365 days */}
        <div className="mb-8">
          <h2 className="text-lg font-medium mb-4">Production Status by Day</h2>
          <div className="p-6 bg-zinc-900/50 border border-white/10 rounded-xl">
            <div className="grid grid-cols-[repeat(31,1fr)] gap-1">
              {/* Month headers */}
              {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].map((month, i) => (
                <div 
                  key={month} 
                  className="col-span-2 text-[10px] text-zinc-600 text-center mb-1"
                  style={{ gridColumn: `${(i * 2) + 1} / span 2` }}
                >
                  {month}
                </div>
              ))}
              
              {/* Day cells */}
              {Array.from({ length: 365 }, (_, i) => {
                const day = i + 1
                const status = dayStatuses.find(d => d.day_number === day)
                const hasVideo = status && status.with_video > 0
                const isComplete = status && status.with_video >= 5
                const isPriority = priorityDays.includes(day)
                const isSelected = selectedDay === day
                
                return (
                  <button
                    key={day}
                    onClick={() => generateDay(day)}
                    disabled={isRunning}
                    className={`
                      aspect-square rounded text-[8px] font-mono transition-all
                      ${isComplete ? 'bg-emerald-500/80 text-white' : ''}
                      ${hasVideo && !isComplete ? 'bg-amber-500/60 text-white' : ''}
                      ${!hasVideo && isPriority ? 'bg-red-500/30 border border-red-500/50' : ''}
                      ${!hasVideo && !isPriority ? 'bg-zinc-800/50 text-zinc-600' : ''}
                      ${isSelected ? 'ring-2 ring-white' : ''}
                      hover:ring-1 hover:ring-white/50 disabled:cursor-not-allowed
                    `}
                    title={`Day ${day}: ${status?.with_video || 0} videos`}
                  >
                    {day}
                  </button>
                )
              })}
            </div>
            
            {/* Legend */}
            <div className="flex items-center gap-6 mt-4 pt-4 border-t border-white/5">
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <div className="w-3 h-3 rounded bg-emerald-500/80" />
                Complete (5 phases)
              </div>
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <div className="w-3 h-3 rounded bg-amber-500/60" />
                Partial
              </div>
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <div className="w-3 h-3 rounded bg-red-500/30 border border-red-500/50" />
                Priority (no video)
              </div>
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <div className="w-3 h-3 rounded bg-zinc-800/50" />
                Pending
              </div>
            </div>
          </div>
        </div>

        {/* Recent Jobs */}
        {batchJobs.length > 0 && (
          <div className="mb-8">
            <h2 className="text-lg font-medium mb-4">Recent Jobs</h2>
            <div className="bg-zinc-900/50 border border-white/10 rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead className="border-b border-white/10">
                  <tr className="text-zinc-500 text-xs uppercase tracking-wider">
                    <th className="text-left p-4">Day</th>
                    <th className="text-left p-4">Phase</th>
                    <th className="text-left p-4">Age</th>
                    <th className="text-left p-4">Status</th>
                    <th className="text-left p-4">Video ID</th>
                  </tr>
                </thead>
                <tbody>
                  {batchJobs.slice(0, 10).map((job, i) => (
                    <tr key={i} className="border-b border-white/5">
                      <td className="p-4 font-mono">{job.day}</td>
                      <td className="p-4">{job.phase}</td>
                      <td className="p-4">{job.age_group}</td>
                      <td className="p-4">
                        <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs ${
                          job.status === 'completed' ? 'bg-emerald-500/20 text-emerald-400' :
                          job.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                          job.status === 'generating' ? 'bg-blue-500/20 text-blue-400' :
                          'bg-zinc-500/20 text-zinc-400'
                        }`}>
                          {job.status === 'generating' && <Loader2 className="w-3 h-3 animate-spin" />}
                          {job.status}
                        </span>
                      </td>
                      <td className="p-4 font-mono text-zinc-500 text-xs">{job.video_id || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Completion Progress */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="p-6 bg-zinc-900/50 border border-white/10 rounded-xl">
            <h3 className="font-medium mb-4">Completion by Phase</h3>
            {['hook', 'story', 'wonder', 'action', 'wisdom'].map(phase => {
              const count = dayStatuses.filter(d => d.phases?.includes(phase)).length
              const pct = (count / 365) * 100
              return (
                <div key={phase} className="mb-3">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="capitalize">{phase}</span>
                    <span className="text-zinc-500">{count}/365 days</span>
                  </div>
                  <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-emerald-500 transition-all"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>

          <div className="p-6 bg-zinc-900/50 border border-white/10 rounded-xl">
            <h3 className="font-medium mb-4">Production Targets</h3>
            <div className="space-y-4 text-sm">
              <div className="flex justify-between">
                <span>Week 1 (Days 1-7)</span>
                <span className={completeDays.filter(d => d <= 7).length >= 7 ? 'text-emerald-400' : 'text-amber-400'}>
                  {completeDays.filter(d => d <= 7).length}/7 complete
                </span>
              </div>
              <div className="flex justify-between">
                <span>Month 1 (Days 1-31)</span>
                <span className={completeDays.filter(d => d <= 31).length >= 31 ? 'text-emerald-400' : 'text-amber-400'}>
                  {completeDays.filter(d => d <= 31).length}/31 complete
                </span>
              </div>
              <div className="flex justify-between">
                <span>English Adult Variants</span>
                <span className="text-zinc-400">{stats?.with_video || 0} videos</span>
              </div>
              <div className="flex justify-between">
                <span>Audio TTS Coverage</span>
                <span className="text-emerald-400">{stats?.with_audio?.toLocaleString() || 0} files</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

function StatCard({ 
  label, 
  value, 
  icon, 
  highlight 
}: { 
  label: string
  value: string
  icon?: React.ReactNode
  highlight?: boolean 
}) {
  return (
    <div className={`p-4 rounded-xl border ${highlight ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-zinc-900/50 border-white/10'}`}>
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <span className="text-xs text-zinc-500 uppercase tracking-wider">{label}</span>
      </div>
      <p className={`text-2xl font-mono ${highlight ? 'text-emerald-400' : 'text-white'}`}>{value}</p>
    </div>
  )
}
