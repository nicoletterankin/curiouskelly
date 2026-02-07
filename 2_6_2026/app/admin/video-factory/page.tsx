'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Loader2, Play, Zap, RefreshCw } from 'lucide-react'

interface Stats {
  totals: {
    total_slots: string
    has_audio: string
    has_video: string
    ready_for_lipsync: string
  }
  byAgeGroup: { age_group: string; ready: string }[]
  byDay: { day_number: number; ready: string }[]
  baseVideosAvailable: number
}

interface RunResult {
  success: boolean
  batch?: {
    succeeded: number
    failed: number
    processed: number
  }
  progress?: {
    totalCompleted: number
    remaining: number
    elapsedSeconds: number
  }
  error?: string
}

export default function VideoFactoryPage() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [lastResult, setLastResult] = useState<RunResult | null>(null)
  const [autoRunning, setAutoRunning] = useState(false)
  const [runCount, setRunCount] = useState(0)
  
  // Use ref to track auto-run state for the loop
  const autoRunRef = useRef(false)
  
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch('/api/fal/mass-lipsync')
      const data = await res.json()
      setStats(data)
    } catch (error) {
      console.error('Error fetching stats:', error)
    } finally {
      setLoading(false)
    }
  }, [])
  
  const startAutoRun = useCallback(async () => {
    autoRunRef.current = true
    setAutoRunning(true)
    
    const runLoop = async () => {
      while (autoRunRef.current) {
        setRunning(true)
        try {
          const res = await fetch('/api/fal/auto-run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ batch: 25, prioritize: 'early' }),
          })
          const data = await res.json()
          setLastResult(data)
          setRunCount(prev => prev + (data.batch?.succeeded || 0))
          
          // Refresh stats every batch
          fetchStats()
          
          // Check if we should stop (no more work or error)
          if (!data.success || (data.progress?.remaining === 0)) {
            console.log('[v0] Auto-run stopping: no more work or error')
            autoRunRef.current = false
            setAutoRunning(false)
            break
          }
          
          // Small delay between batches
          await new Promise(r => setTimeout(r, 1000))
        } catch (error) {
          console.error('[v0] Auto-run error:', error)
          setLastResult({ success: false, error: 'Network error' })
          // Continue on error after a longer delay
          await new Promise(r => setTimeout(r, 5000))
        } finally {
          setRunning(false)
        }
      }
    }
    
    runLoop()
  }, [])
  
  const stopAutoRun = useCallback(() => {
    autoRunRef.current = false
    setAutoRunning(false)
  }, [])
  
  useEffect(() => {
    fetchStats()
    // Refresh every 30 seconds
    const interval = setInterval(fetchStats, 30000)
    return () => clearInterval(interval)
  }, [])
  
  // Auto-start removed - use buttons on deployed site to trigger processing
  
  const runBatch = async (batchSize: number) => {
    setRunning(true)
    try {
      const res = await fetch('/api/fal/auto-run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batch: batchSize, prioritize: 'early' }),
      })
      const data = await res.json()
      setLastResult(data)
      setRunCount(prev => prev + (data.batch?.succeeded || 0))
      await fetchStats()
    } catch (error) {
      console.error('Error running batch:', error)
      setLastResult({ success: false, error: 'Network error' })
    } finally {
      setRunning(false)
    }
  }
  
  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }
  
  const totalSlots = parseInt(stats?.totals.total_slots || '0')
  const hasVideo = parseInt(stats?.totals.has_video || '0')
  const ready = parseInt(stats?.totals.ready_for_lipsync || '0')
  const hasAudio = parseInt(stats?.totals.has_audio || '0')
  const completionPct = totalSlots > 0 ? (hasVideo / totalSlots) * 100 : 0
  const audioPct = totalSlots > 0 ? (hasAudio / totalSlots) * 100 : 0
  
  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Video Factory</h1>
            <p className="text-muted-foreground">Mass Fal.ai Lipsync Production</p>
          </div>
          <Button variant="outline" onClick={fetchStats}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
        
        {/* Progress Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Slots</CardDescription>
              <CardTitle className="text-2xl">{totalSlots.toLocaleString()}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Videos Complete</CardDescription>
              <CardTitle className="text-2xl text-green-600">{hasVideo.toLocaleString()}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Ready for Lipsync</CardDescription>
              <CardTitle className="text-2xl text-blue-600">{ready.toLocaleString()}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>This Session</CardDescription>
              <CardTitle className="text-2xl text-purple-600">{runCount}</CardTitle>
            </CardHeader>
          </Card>
        </div>
        
        {/* Progress Bars */}
        <Card>
          <CardHeader>
            <CardTitle>Production Progress</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Video Completion</span>
                <span>{completionPct.toFixed(2)}%</span>
              </div>
              <Progress value={completionPct} className="h-3" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Audio Ready (Pipeline Fuel)</span>
                <span>{audioPct.toFixed(2)}%</span>
              </div>
              <Progress value={audioPct} className="h-3 [&>div]:bg-blue-500" />
            </div>
          </CardContent>
        </Card>
        
        {/* Control Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Production Controls</CardTitle>
            <CardDescription>
              Run batches manually or start continuous production
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-3">
              <Button 
                onClick={() => runBatch(5)} 
                disabled={running || autoRunning}
              >
                {running ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
                Run 5
              </Button>
              <Button 
                onClick={() => runBatch(10)} 
                disabled={running || autoRunning}
              >
                {running ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
                Run 10
              </Button>
              <Button 
                onClick={() => runBatch(25)} 
                disabled={running || autoRunning}
              >
                {running ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
                Run 25
              </Button>
              <Button 
                onClick={() => runBatch(50)} 
                disabled={running || autoRunning}
                variant="secondary"
              >
                {running ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Zap className="h-4 w-4 mr-2" />}
                Run 50
              </Button>
              
              <div className="w-px h-8 bg-border mx-2" />
              
              {!autoRunning ? (
                <Button 
                  onClick={startAutoRun}
                  variant="default"
                  className="bg-green-600 hover:bg-green-700"
                  disabled={running}
                >
                  <Zap className="h-4 w-4 mr-2" />
                  Start Auto-Run
                </Button>
              ) : (
                <Button 
                  onClick={stopAutoRun}
                  variant="destructive"
                >
                  Stop Auto-Run
                </Button>
              )}
            </div>
            
            {lastResult && (
              <div className="mt-4 p-4 rounded-lg bg-muted">
                {lastResult.success ? (
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Badge variant="default" className="bg-green-600">Success</Badge>
                      <span className="text-sm">
                        {lastResult.batch?.succeeded}/{lastResult.batch?.processed} videos in {lastResult.progress?.elapsedSeconds}s
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {lastResult.progress?.remaining.toLocaleString()} remaining
                    </p>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <Badge variant="destructive">Error</Badge>
                    <span className="text-sm text-red-600">{lastResult.error}</span>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
        
        {/* By Age Group */}
        <Card>
          <CardHeader>
            <CardTitle>Ready by Age Group</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {stats?.byAgeGroup.map(({ age_group, ready }) => (
                <div key={age_group} className="p-3 rounded-lg bg-muted">
                  <div className="text-sm font-medium">{age_group}</div>
                  <div className="text-2xl font-bold">{parseInt(ready).toLocaleString()}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        {/* By Day */}
        <Card>
          <CardHeader>
            <CardTitle>Ready by Day (First 30)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-5 md:grid-cols-10 gap-2">
              {stats?.byDay.map(({ day_number, ready }) => (
                <div 
                  key={day_number} 
                  className={`p-2 rounded text-center text-sm ${
                    parseInt(ready) > 0 ? 'bg-blue-100 text-blue-800' : 'bg-muted text-muted-foreground'
                  }`}
                >
                  <div className="font-medium">D{day_number}</div>
                  <div>{ready}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        {/* Base Videos Status */}
        <Card>
          <CardHeader>
            <CardTitle>Base Video Coverage</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <div className="text-4xl font-bold">{stats?.baseVideosAvailable}/36</div>
              <div className="text-muted-foreground">
                {stats?.baseVideosAvailable === 36 
                  ? 'All 36 base videos ready (3 ages x 12 archetypes)'
                  : 'Some base videos missing'
                }
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
