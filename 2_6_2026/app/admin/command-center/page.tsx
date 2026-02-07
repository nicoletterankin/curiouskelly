'use client'

import React from "react"

import { useState, useEffect, useCallback, useRef } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import {
  ArrowLeft,
  RefreshCw,
  Play,
  Square,
  Loader2,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  Database,
  Volume2,
  Video,
  Layers,
  Target,
  Zap,
  TrendingUp,
} from 'lucide-react'

// ============================================================================
// TYPES
// ============================================================================

interface ExecutiveSummary {
  pipeline_health: string
  days_with_lessons: number
  days_missing_lessons: number
  days_with_full_scripts: number
  base_videos: string
  base_videos_complete: boolean
  total_slots_needed: number
  slots_seeded: number
  slots_with_audio: number
  slots_with_video: number
  audio_coverage_pct: number
  video_coverage_pct: number
}

interface GapAnalysis {
  missing_lesson_days: { count: number; days: number[] }
  slots_to_seed: { count: number }
  audio_to_generate: { count: number }
  video_to_generate: { count: number }
  base_video_gaps: { count: number; missing: string[] }
}

interface CostEstimates {
  audio: { remaining: number; cost_usd: number; time_hours: number; note: string }
  video: { remaining: number; cost_usd: number; time_hours: number; note: string }
  storage: { cost_usd: number }
  total_cost_usd: number
  total_time_hours: number
}

interface BreakdownRow {
  age_group?: string
  phase?: string
  archetype?: string
  total: number
  audio: number
  video: number
}

interface HeatmapDay {
  day_number: number
  seeded: number
  audio: number
  video: number
}

interface RecommendedAction {
  priority: number
  action: string
  description: string
  body: Record<string, unknown>
}

interface DashboardData {
  executive: ExecutiveSummary
  gaps: GapAnalysis
  estimates: CostEstimates
  breakdowns: {
    by_age: BreakdownRow[]
    by_phase: BreakdownRow[]
    by_archetype: BreakdownRow[]
  }
  heatmap: HeatmapDay[]
  actions: RecommendedAction[]
  config: {
    ages: string[]
    archetypes: string[]
    phases: string[]
    slots_per_day: number
    total_days: number
    total_slots: number
  }
  timestamp: string
}

interface ActionResult {
  success: boolean
  action: string
  result?: Record<string, unknown>
  error?: string
  elapsed?: string
}

// ============================================================================
// COMPONENT
// ============================================================================

export default function CommandCenterPage() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [runningAction, setRunningAction] = useState<string | null>(null)
  const [actionResults, setActionResults] = useState<ActionResult[]>([])
  const [autoRunning, setAutoRunning] = useState(false)
  const autoRunRef = useRef(false)
  const [autoRunStep, setAutoRunStep] = useState('')
  const [sessionStats, setSessionStats] = useState({ audio: 0, video: 0, seeded: 0 })

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch('/api/command-center')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setData(json)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch')
    } finally {
      setLoading(false)
    }
  }, [])

  // Auto-start flag
  const [autoStarted, setAutoStarted] = useState(false)

  const executeAction = useCallback(async (
    actionName: string,
    body: Record<string, unknown>
  ): Promise<ActionResult> => {
    setRunningAction(actionName)
    try {
      const res = await fetch('/api/command-center', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const result = await res.json()
      console.log(`[v0] Pipeline action "${actionName}" response:`, JSON.stringify(result).slice(0, 500))
      const actionResult: ActionResult = {
        success: result.success !== false,
        action: actionName,
        result,
        elapsed: result.elapsed,
      }
      setActionResults(prev => [actionResult, ...prev].slice(0, 20))

      // Track session stats
      if (result.result?.succeeded) {
        if (actionName.includes('audio')) {
          setSessionStats(prev => ({ ...prev, audio: prev.audio + (result.result.succeeded || 0) }))
        }
        if (actionName.includes('video')) {
          setSessionStats(prev => ({ ...prev, video: prev.video + (result.result.succeeded || 0) }))
        }
      }

      await fetchData()
      return actionResult
    } catch (err) {
      const errorResult: ActionResult = {
        success: false,
        action: actionName,
        error: err instanceof Error ? err.message : 'Unknown error',
      }
      setActionResults(prev => [errorResult, ...prev].slice(0, 20))
      return errorResult
    } finally {
      setRunningAction(null)
    }
  }, [fetchData])

  const startAutoRun = useCallback(async () => {
    autoRunRef.current = true
    setAutoRunning(true)

    const runPipeline = async () => {
      // Step 1: Sync lessons if needed
      if (data?.gaps.missing_lesson_days.count && data.gaps.missing_lesson_days.count > 0 && autoRunRef.current) {
        setAutoRunStep('Syncing lessons from Supabase...')
        await executeAction('sync-lessons', { action: 'sync-lessons' })
      }

      // Step 2: Seed slots
      if (autoRunRef.current) {
        setAutoRunStep('Seeding asset slots...')
        await executeAction('seed-all', { action: 'seed-all', dayStart: 1, dayEnd: 365 })
      }

      // Step 3: Generate audio in batches
      while (autoRunRef.current) {
        setAutoRunStep('Generating audio (ElevenLabs)...')
        const audioResult = await executeAction('generate-audio', {
          action: 'generate-audio',
          dayStart: 1,
          dayEnd: 365,
          limit: 50,
        })

        // Check if rate limited or done
        const innerResult = audioResult.result?.result as Record<string, unknown> | undefined
        if (innerResult?.rateLimited) {
          setAutoRunStep('Rate limited by ElevenLabs. Switching to video...')
          break
        }
        if (innerResult?.succeeded === 0 && innerResult?.failed === 0) {
          break
        }
        // Small pause between batches
        await new Promise(r => setTimeout(r, 2000))
      }

      // Step 4: Generate video in batches
      while (autoRunRef.current) {
        setAutoRunStep('Generating lip-sync videos (Fal.ai)...')
        const videoResult = await executeAction('generate-video', {
          action: 'generate-video',
          dayStart: 1,
          dayEnd: 365,
          limit: 25,
        })

        const innerResult = videoResult.result?.result as Record<string, unknown> | undefined
        if (innerResult?.succeeded === 0 && innerResult?.failed === 0) {
          break
        }
        // Small pause
        await new Promise(r => setTimeout(r, 3000))
      }

      setAutoRunStep('Pipeline complete.')
      autoRunRef.current = false
      setAutoRunning(false)
    }

    runPipeline()
  }, [data, executeAction])

  const stopAutoRun = useCallback(() => {
    autoRunRef.current = false
    setAutoRunning(false)
    setAutoRunStep('Stopped by user.')
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30000) // refresh every 30s
    return () => clearInterval(interval)
  }, [fetchData])

  // AUTO-EXECUTE: Start the full pipeline automatically on page load
  useEffect(() => {
    if (data && !autoStarted && !autoRunning && !runningAction) {
      setAutoStarted(true)
      console.log('[v0] AUTO-EXECUTE: Data loaded. Starting Full Pipeline in 1.5s...')
      console.log('[v0] Pipeline state:', {
        slots_seeded: data.executive.slots_seeded,
        audio: data.executive.slots_with_audio,
        video: data.executive.slots_with_video,
        audio_pct: data.executive.audio_coverage_pct,
        video_pct: data.executive.video_coverage_pct,
      })
      const timer = setTimeout(() => {
        console.log('[v0] AUTO-EXECUTING: Starting Full Pipeline NOW')
        startAutoRun()
      }, 1500)
      return () => clearTimeout(timer)
    }
  }, [data, autoStarted, autoRunning, runningAction, startAutoRun])

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="max-w-md">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              <p className="text-destructive">{error}</p>
            </div>
            <Button onClick={fetchData}>Retry</Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!data) return null

  const { executive: exec, gaps, estimates, breakdowns, heatmap, actions, config } = data

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b px-6 py-4 sticky top-0 bg-background z-10">
        <div className="flex items-center justify-between max-w-[1400px] mx-auto">
          <div className="flex items-center gap-4">
            <Link href="/admin" className="text-muted-foreground hover:text-foreground transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-xl font-bold tracking-tight">Executive Command Center</h1>
              <p className="text-sm text-muted-foreground">
                Curious Kelly Video Pipeline -- {exec.video_coverage_pct}% coverage
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant={exec.pipeline_health === 'operational' ? 'default' : 'secondary'}>
              {exec.pipeline_health}
            </Badge>
            <Button variant="outline" size="sm" onClick={fetchData} disabled={!!runningAction}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-6 py-8 space-y-8">

        {/* ========== EXECUTIVE KPIs ========== */}
        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Pipeline KPIs
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <KpiCard label="Lessons" value={`${exec.days_with_lessons}/365`} sub="days in DB" icon={<Database className="h-4 w-4" />} />
            <KpiCard label="Scripts" value={`${exec.days_with_full_scripts}`} sub="days with all 5" icon={<Layers className="h-4 w-4" />} />
            <KpiCard label="Seeded" value={exec.slots_seeded.toLocaleString()} sub={`of ${config.total_slots.toLocaleString()}`} icon={<Target className="h-4 w-4" />} />
            <KpiCard label="Audio" value={exec.slots_with_audio.toLocaleString()} sub={`${exec.audio_coverage_pct}%`} icon={<Volume2 className="h-4 w-4" />} accent="blue" />
            <KpiCard label="Video" value={exec.slots_with_video.toLocaleString()} sub={`${exec.video_coverage_pct}%`} icon={<Video className="h-4 w-4" />} accent="green" />
            <KpiCard label="Base Videos" value={exec.base_videos} sub={exec.base_videos_complete ? 'Complete' : 'Incomplete'} icon={<CheckCircle className="h-4 w-4" />} accent={exec.base_videos_complete ? 'green' : 'amber'} />
          </div>
        </section>

        {/* ========== PROGRESS BARS ========== */}
        <section className="grid md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Audio Coverage</CardTitle>
              <CardDescription>{exec.slots_with_audio.toLocaleString()} / {config.total_slots.toLocaleString()} clips</CardDescription>
            </CardHeader>
            <CardContent>
              <Progress value={exec.audio_coverage_pct} className="h-3" />
              <p className="text-xs text-muted-foreground mt-2">
                {gaps.audio_to_generate.count.toLocaleString()} remaining -- est. ${estimates.audio.cost_usd} / {estimates.audio.time_hours}h
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Video Coverage</CardTitle>
              <CardDescription>{exec.slots_with_video.toLocaleString()} / {config.total_slots.toLocaleString()} videos</CardDescription>
            </CardHeader>
            <CardContent>
              <Progress value={exec.video_coverage_pct} className="h-3" />
              <p className="text-xs text-muted-foreground mt-2">
                {gaps.video_to_generate.count.toLocaleString()} remaining -- est. ${estimates.video.cost_usd} / {estimates.video.time_hours}h
              </p>
            </CardContent>
          </Card>
        </section>

        {/* ========== COST SUMMARY ========== */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <DollarSign className="h-4 w-4" />
              Cost to Complete (100% English Coverage)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 rounded-lg bg-muted">
                <p className="text-xs text-muted-foreground">Audio (ElevenLabs)</p>
                <p className="text-xl font-bold font-mono">${estimates.audio.cost_usd.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">{estimates.audio.time_hours}h processing</p>
              </div>
              <div className="p-3 rounded-lg bg-muted">
                <p className="text-xs text-muted-foreground">Video (Fal.ai)</p>
                <p className="text-xl font-bold font-mono">${estimates.video.cost_usd.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">{estimates.video.time_hours}h processing</p>
              </div>
              <div className="p-3 rounded-lg bg-muted">
                <p className="text-xs text-muted-foreground">Storage (Blob)</p>
                <p className="text-xl font-bold font-mono">${estimates.storage.cost_usd.toFixed(2)}</p>
                <p className="text-xs text-muted-foreground">Monthly estimate</p>
              </div>
              <div className="p-3 rounded-lg bg-primary/5 border border-primary/20">
                <p className="text-xs text-muted-foreground">Total to Complete</p>
                <p className="text-xl font-bold font-mono text-primary">${estimates.total_cost_usd.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">{estimates.total_time_hours}h total time</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ========== PIPELINE CONTROLS ========== */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Pipeline Controls
            </CardTitle>
            <CardDescription>
              Execute pipeline steps manually or start automated production
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Auto-Run */}
            <div className="flex items-center gap-4 p-4 rounded-lg border bg-muted/50">
              {!autoRunning ? (
                <Button
                  onClick={startAutoRun}
                  disabled={!!runningAction}
                  className="bg-green-600 hover:bg-green-700 text-white"
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start Full Pipeline
                </Button>
              ) : (
                <Button
                  onClick={stopAutoRun}
                  variant="destructive"
                >
                  <Square className="h-4 w-4 mr-2" />
                  Stop Pipeline
                </Button>
              )}
              <div className="flex-1">
                {autoRunning && (
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-green-600" />
                    <span className="text-sm">{autoRunStep}</span>
                  </div>
                )}
                {!autoRunning && (
                  <p className="text-sm text-muted-foreground">
                    Runs: Sync Lessons {'->'} Seed Slots {'->'} Generate Audio {'->'} Generate Video
                  </p>
                )}
              </div>
              {(sessionStats.audio > 0 || sessionStats.video > 0) && (
                <div className="text-right text-xs text-muted-foreground">
                  <p>Session: {sessionStats.audio} audio, {sessionStats.video} video</p>
                </div>
              )}
            </div>

            {/* Individual Actions */}
            <div className="flex flex-wrap gap-2">
              {actions.map((action) => (
                <Button
                  key={action.action}
                  variant="outline"
                  size="sm"
                  disabled={!!runningAction || autoRunning}
                  onClick={() => executeAction(action.action, action.body)}
                >
                  {runningAction === action.action && <Loader2 className="h-3 w-3 mr-1 animate-spin" />}
                  P{action.priority}: {action.description.slice(0, 50)}{action.description.length > 50 ? '...' : ''}
                </Button>
              ))}
            </div>

            {/* Quick Actions */}
            <div className="flex flex-wrap gap-2 pt-2 border-t">
              <Button
                variant="secondary"
                size="sm"
                disabled={!!runningAction || autoRunning}
                onClick={() => executeAction('full-day', { action: 'full-day', day: 1 })}
              >
                Full Day 1
              </Button>
              <Button
                variant="secondary"
                size="sm"
                disabled={!!runningAction || autoRunning}
                onClick={() => executeAction('full-range', { action: 'full-range', dayStart: 1, dayEnd: 7, limit: 50 })}
              >
                Full Week 1
              </Button>
              <Button
                variant="secondary"
                size="sm"
                disabled={!!runningAction || autoRunning}
                onClick={() => executeAction('full-range', { action: 'full-range', dayStart: 1, dayEnd: 30, limit: 100 })}
              >
                Full Month 1
              </Button>
              <Button
                variant="secondary"
                size="sm"
                disabled={!!runningAction || autoRunning}
                onClick={() => executeAction('generate-audio', { action: 'generate-audio', dayStart: 1, dayEnd: 365, limit: 200 })}
              >
                Audio Blast (200)
              </Button>
              <Button
                variant="secondary"
                size="sm"
                disabled={!!runningAction || autoRunning}
                onClick={() => executeAction('generate-video', { action: 'generate-video', dayStart: 1, dayEnd: 365, limit: 100 })}
              >
                Video Blast (100)
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* ========== ACTION LOG ========== */}
        {actionResults.length > 0 && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Action Log</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {actionResults.map((result, i) => (
                  <div
                    key={`${result.action}-${i}`}
                    className={`p-3 rounded-lg text-sm flex items-start gap-2 ${
                      result.success ? 'bg-green-500/5 border border-green-500/20' : 'bg-destructive/5 border border-destructive/20'
                    }`}
                  >
                    {result.success ? (
                      <CheckCircle className="h-4 w-4 text-green-600 shrink-0 mt-0.5" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="font-medium">{result.action}</p>
                      {result.elapsed && <p className="text-xs text-muted-foreground">{result.elapsed}s</p>}
                      {result.error && <p className="text-xs text-destructive">{result.error}</p>}
                      {result.result && (
                        <details className="mt-1">
                          <summary className="text-xs text-muted-foreground cursor-pointer">Details</summary>
                          <pre className="text-xs mt-1 overflow-x-auto max-w-full">
                            {JSON.stringify(result.result, null, 2).slice(0, 500)}
                          </pre>
                        </details>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* ========== BREAKDOWNS ========== */}
        <div className="grid md:grid-cols-3 gap-4">
          {/* By Age */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Coverage by Age</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {breakdowns.by_age.map((row) => {
                const pct = row.total > 0 ? (row.video / row.total) * 100 : 0
                return (
                  <div key={row.age_group}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="capitalize font-medium">{row.age_group}</span>
                      <span className="text-muted-foreground">{row.video}/{row.total}</span>
                    </div>
                    <Progress value={pct} className="h-2" />
                  </div>
                )
              })}
            </CardContent>
          </Card>

          {/* By Phase */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Coverage by Phase</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {breakdowns.by_phase.map((row) => {
                const pct = row.total > 0 ? (row.video / row.total) * 100 : 0
                return (
                  <div key={row.phase}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="capitalize font-medium">{row.phase}</span>
                      <span className="text-muted-foreground">{row.video}/{row.total}</span>
                    </div>
                    <Progress value={pct} className="h-2" />
                  </div>
                )
              })}
            </CardContent>
          </Card>

          {/* By Archetype */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Coverage by Archetype</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {breakdowns.by_archetype.map((row) => {
                const pct = row.total > 0 ? (row.video / row.total) * 100 : 0
                return (
                  <div key={row.archetype} className="flex items-center gap-2">
                    <span className="text-xs capitalize w-20 truncate">{row.archetype}</span>
                    <div className="flex-1">
                      <Progress value={pct} className="h-1.5" />
                    </div>
                    <span className="text-xs text-muted-foreground w-12 text-right">{pct.toFixed(0)}%</span>
                  </div>
                )
              })}
            </CardContent>
          </Card>
        </div>

        {/* ========== 365-DAY HEATMAP ========== */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">365-Day Production Heatmap</CardTitle>
            <CardDescription>Each cell = 1 day. Color = video coverage level (of 180 slots per day).</CardDescription>
          </CardHeader>
          <CardContent>
            <DayHeatmap heatmap={heatmap} slotsPerDay={config.slots_per_day} />
          </CardContent>
        </Card>

        {/* ========== GAP ANALYSIS ========== */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Gap Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <GapCard
                label="Missing Lessons"
                count={gaps.missing_lesson_days.count}
                total={365}
                severity={gaps.missing_lesson_days.count > 100 ? 'high' : gaps.missing_lesson_days.count > 0 ? 'medium' : 'none'}
              />
              <GapCard
                label="Slots to Seed"
                count={gaps.slots_to_seed.count}
                total={config.total_slots}
                severity={gaps.slots_to_seed.count > 10000 ? 'high' : gaps.slots_to_seed.count > 0 ? 'medium' : 'none'}
              />
              <GapCard
                label="Audio to Generate"
                count={gaps.audio_to_generate.count}
                total={config.total_slots}
                severity={gaps.audio_to_generate.count > 10000 ? 'high' : gaps.audio_to_generate.count > 0 ? 'medium' : 'none'}
              />
              <GapCard
                label="Video to Generate"
                count={gaps.video_to_generate.count}
                total={config.total_slots}
                severity={gaps.video_to_generate.count > 10000 ? 'high' : gaps.video_to_generate.count > 0 ? 'medium' : 'none'}
              />
            </div>
            {gaps.missing_lesson_days.count > 0 && (
              <details className="mt-4">
                <summary className="text-sm text-muted-foreground cursor-pointer">
                  Missing lesson days ({gaps.missing_lesson_days.count})
                </summary>
                <div className="mt-2 flex flex-wrap gap-1">
                  {gaps.missing_lesson_days.days.map(d => (
                    <Badge key={d} variant="outline" className="text-xs font-mono">{d}</Badge>
                  ))}
                </div>
              </details>
            )}
          </CardContent>
        </Card>

        {/* ========== LINKS ========== */}
        <div className="flex flex-wrap gap-3 pb-8">
          <Link href="/admin/video-factory">
            <Button variant="outline" size="sm">Video Factory (Fal.ai)</Button>
          </Link>
          <Link href="/admin/video-production">
            <Button variant="outline" size="sm">Video Production (HeyGen)</Button>
          </Link>
          <Link href="/admin">
            <Button variant="outline" size="sm">Admin Home</Button>
          </Link>
        </div>
      </main>
    </div>
  )
}

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

function KpiCard({
  label,
  value,
  sub,
  icon,
  accent,
}: {
  label: string
  value: string
  sub: string
  icon: React.ReactNode
  accent?: 'blue' | 'green' | 'amber'
}) {
  const accentClass = accent === 'blue'
    ? 'text-blue-600'
    : accent === 'green'
      ? 'text-green-600'
      : accent === 'amber'
        ? 'text-amber-600'
        : 'text-foreground'

  return (
    <Card>
      <CardContent className="pt-4 pb-3 px-4">
        <div className="flex items-center gap-1.5 mb-1 text-muted-foreground">
          {icon}
          <span className="text-xs uppercase tracking-wider">{label}</span>
        </div>
        <p className={`text-2xl font-bold font-mono ${accentClass}`}>{value}</p>
        <p className="text-xs text-muted-foreground">{sub}</p>
      </CardContent>
    </Card>
  )
}

function GapCard({
  label,
  count,
  total,
  severity,
}: {
  label: string
  count: number
  total: number
  severity: 'high' | 'medium' | 'none'
}) {
  const bgClass = severity === 'high'
    ? 'bg-destructive/5 border-destructive/20'
    : severity === 'medium'
      ? 'bg-amber-500/5 border-amber-500/20'
      : 'bg-green-500/5 border-green-500/20'

  const textClass = severity === 'high'
    ? 'text-destructive'
    : severity === 'medium'
      ? 'text-amber-600'
      : 'text-green-600'

  return (
    <div className={`p-4 rounded-lg border ${bgClass}`}>
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className={`text-xl font-bold font-mono ${textClass}`}>{count.toLocaleString()}</p>
      <p className="text-xs text-muted-foreground">of {total.toLocaleString()}</p>
    </div>
  )
}

function DayHeatmap({
  heatmap,
  slotsPerDay,
}: {
  heatmap: HeatmapDay[]
  slotsPerDay: number
}) {
  // Build lookup by day
  const dayMap = new Map(heatmap.map(d => [d.day_number, d]))

  // Month labels
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  const monthStarts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

  return (
    <div>
      {/* Month labels */}
      <div className="flex gap-0.5 mb-1 text-[10px] text-muted-foreground">
        {months.map((m, i) => (
          <div key={m} className="flex-1 text-center">{m}</div>
        ))}
      </div>
      {/* Grid */}
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: 'repeat(53, 1fr)' }}>
        {Array.from({ length: 365 }, (_, i) => {
          const day = i + 1
          const info = dayMap.get(day)
          const videoPct = info ? (info.video / slotsPerDay) * 100 : 0
          const audioPct = info ? (info.audio / slotsPerDay) * 100 : 0
          const seeded = info ? info.seeded > 0 : false

          let bg = 'bg-muted'
          if (videoPct >= 90) bg = 'bg-green-500'
          else if (videoPct >= 50) bg = 'bg-green-400/70'
          else if (videoPct > 0) bg = 'bg-green-300/50'
          else if (audioPct >= 50) bg = 'bg-blue-400/60'
          else if (audioPct > 0) bg = 'bg-blue-300/40'
          else if (seeded) bg = 'bg-amber-300/30'

          return (
            <div
              key={day}
              className={`aspect-square rounded-[2px] ${bg} transition-colors`}
              title={`Day ${day}: ${info?.seeded || 0} seeded, ${info?.audio || 0} audio, ${info?.video || 0} video`}
            />
          )
        })}
      </div>
      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
        <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-[2px] bg-muted" /> No data</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-[2px] bg-amber-300/30" /> Seeded</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-[2px] bg-blue-400/60" /> Has audio</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-[2px] bg-green-300/50" /> Some video</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-[2px] bg-green-500" /> {'>='}90% video</div>
      </div>
    </div>
  )
}
