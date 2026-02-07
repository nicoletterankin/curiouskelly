'use client'

import { useState, useEffect } from 'react'
import useSWR from 'swr'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  RefreshCw, 
  Play, 
  Pause, 
  AlertCircle, 
  CheckCircle, 
  Clock, 
  Loader2,
  Zap,
  Globe,
  Users,
  Calendar
} from 'lucide-react'

const fetcher = (url: string) => fetch(url).then(r => r.json())

interface QueueStats {
  queue: {
    total: number
    completed: number
    processing: number
    pending: number
    failed: number
    percentComplete: number
  }
  credits: {
    remaining: number
    used: number
  }
  estimatedTimeRemaining: {
    atCurrentRate: string
    description: string
  }
  byLanguage?: Array<{ language: string; status: string; count: number }>
  byAge?: Array<{ age_group: string; status: string; count: number }>
}

export default function VideoQueueDashboard() {
  const [isGenerating, setIsGenerating] = useState(false)
  const [batchSize, setBatchSize] = useState(10)
  const [lastResult, setLastResult] = useState<Record<string, unknown> | null>(null)
  
  const { data: stats, error, mutate, isLoading } = useSWR<QueueStats>(
    '/api/video-queue/batch-generate?detailed=true',
    fetcher,
    { refreshInterval: isGenerating ? 5000 : 30000 }
  )

  const startBatch = async (dryRun = false) => {
    setIsGenerating(true)
    try {
      const res = await fetch('/api/video-queue/batch-generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batchSize, dryRun }),
      })
      const result = await res.json()
      setLastResult(result)
      mutate()
    } catch (err) {
      setLastResult({ error: String(err) })
    } finally {
      setIsGenerating(false)
    }
  }

  const resetFailed = async () => {
    try {
      const res = await fetch('/api/video-queue/batch-generate', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resetFailed: true }),
      })
      const result = await res.json()
      setLastResult(result)
      mutate()
    } catch (err) {
      setLastResult({ error: String(err) })
    }
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background p-8">
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-5 w-5" />
              <span>Failed to load queue stats: {String(error)}</span>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8 px-4 max-w-7xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Video Generation Queue</h1>
            <p className="text-muted-foreground mt-1">
              Monitor and control the 197,100 video generation pipeline
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => mutate()}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        {/* Stats Overview */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Total Videos</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {stats?.queue.total.toLocaleString() || '-'}
              </div>
              <p className="text-xs text-muted-foreground">
                365 days x 5 phases x 3 ages x 6 langs
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Completed</CardTitle>
              <CheckCircle className="h-4 w-4 text-emerald-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-emerald-600">
                {stats?.queue.completed.toLocaleString() || '0'}
              </div>
              <Progress 
                value={stats?.queue.percentComplete || 0} 
                className="mt-2 h-2"
              />
              <p className="text-xs text-muted-foreground mt-1">
                {stats?.queue.percentComplete?.toFixed(2)}% complete
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Processing</CardTitle>
              <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                {stats?.queue.processing.toLocaleString() || '0'}
              </div>
              <p className="text-xs text-muted-foreground">
                Currently generating with HeyGen
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">HeyGen Credits</CardTitle>
              <Zap className="h-4 w-4 text-amber-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-amber-600">
                {stats?.credits.remaining.toLocaleString() || '-'}
              </div>
              <p className="text-xs text-muted-foreground">
                {stats?.credits.used.toLocaleString() || '0'} used
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Status Breakdown */}
        <div className="grid gap-4 md:grid-cols-2 mb-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5" />
                By Language
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {['en', 'es', 'fr', 'de', 'pt', 'zh'].map(lang => {
                  const langStats = stats?.byLanguage?.filter(s => s.language === lang) || []
                  const completed = langStats.find(s => s.status === 'completed')?.count || 0
                  const total = langStats.reduce((sum, s) => sum + s.count, 0)
                  const percent = total > 0 ? Math.round((completed / total) * 100) : 0
                  
                  return (
                    <div key={lang} className="flex items-center gap-3">
                      <span className="w-8 font-mono text-sm uppercase">{lang}</span>
                      <Progress value={percent} className="flex-1 h-2" />
                      <span className="text-sm text-muted-foreground w-16 text-right">
                        {percent}%
                      </span>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5" />
                By Age Group
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {['kid', 'adult', 'senior'].map(age => {
                  const ageStats = stats?.byAge?.filter(s => s.age_group === age) || []
                  const completed = ageStats.find(s => s.status === 'completed')?.count || 0
                  const total = ageStats.reduce((sum, s) => sum + s.count, 0)
                  const percent = total > 0 ? Math.round((completed / total) * 100) : 0
                  
                  return (
                    <div key={age} className="flex items-center gap-3">
                      <span className="w-16 text-sm capitalize">{age}</span>
                      <Progress value={percent} className="flex-1 h-2" />
                      <span className="text-sm text-muted-foreground w-16 text-right">
                        {percent}%
                      </span>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Controls */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Generation Controls</CardTitle>
            <CardDescription>
              Start or pause video generation batches
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">Batch Size:</label>
                <select 
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="border rounded px-2 py-1 text-sm bg-background"
                >
                  <option value="5">5</option>
                  <option value="10">10</option>
                  <option value="25">25</option>
                  <option value="50">50</option>
                  <option value="100">100</option>
                </select>
              </div>
              
              <Button
                onClick={() => startBatch(true)}
                variant="outline"
                disabled={isGenerating}
              >
                <Clock className="h-4 w-4 mr-2" />
                Dry Run
              </Button>
              
              <Button
                onClick={() => startBatch(false)}
                disabled={isGenerating || (stats?.credits.remaining || 0) < batchSize}
              >
                {isGenerating ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Play className="h-4 w-4 mr-2" />
                )}
                Start Batch
              </Button>
              
              {(stats?.queue.failed || 0) > 0 && (
                <Button
                  onClick={resetFailed}
                  variant="destructive"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Reset {stats?.queue.failed} Failed
                </Button>
              )}
            </div>
            
            {(stats?.queue.pending || 0) > 0 && (
              <p className="mt-4 text-sm text-muted-foreground">
                <Clock className="h-4 w-4 inline mr-1" />
                {stats?.queue.pending.toLocaleString()} videos pending.
                Estimated time: {stats?.estimatedTimeRemaining.atCurrentRate}
              </p>
            )}
          </CardContent>
        </Card>

        {/* Last Result */}
        {lastResult && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                Last Operation Result
                <Badge variant={lastResult.error ? 'destructive' : 'secondary'}>
                  {lastResult.status || (lastResult.error ? 'Error' : 'Complete')}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted p-4 rounded-lg overflow-auto text-xs max-h-96">
                {JSON.stringify(lastResult, null, 2)}
              </pre>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
