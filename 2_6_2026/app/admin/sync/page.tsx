'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { RefreshCw, Download, CheckCircle2, XCircle, Clock, Zap } from 'lucide-react'

interface SyncStatus {
  total: number
  completed: number
  alreadySynced: number
  needsSync: number
  videos: Array<{
    video_id: string
    title: string
    parsed: {
      day: number
      language: string
      age: string
      archetype: string
      phase: string
    } | null
  }>
}

interface SyncResult {
  processed: number
  successful: number
  failed: number
  remaining: number
  results: Array<{
    video_id: string
    success: boolean
    blob_url?: string
    error?: string
  }>
}

export default function SyncPage() {
  const [status, setStatus] = useState<SyncStatus | null>(null)
  const [syncing, setSyncing] = useState(false)
  const [lastResult, setLastResult] = useState<SyncResult | null>(null)
  const [autoSync, setAutoSync] = useState(false)
  const [totalSynced, setTotalSynced] = useState(0)

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/heygen/sync-completed?limit=100')
      const data = await res.json()
      // Ensure all required fields have defaults
      setStatus({
        total: data.total || 0,
        completed: data.completed || 0,
        alreadySynced: data.alreadySynced || 0,
        needsSync: data.needsSync || 0,
        videos: data.videos || []
      })
    } catch (err) {
      console.error('Failed to fetch status:', err)
      setStatus({ total: 0, completed: 0, alreadySynced: 0, needsSync: 0, videos: [] })
    }
  }

  const runSync = async (batchSize = 10) => {
    setSyncing(true)
    try {
      const res = await fetch('/api/heygen/sync-completed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batchSize })
      })
      const data: SyncResult = await res.json()
      setLastResult(data)
      setTotalSynced(prev => prev + data.successful)
      await fetchStatus()
      return data
    } catch (err) {
      console.error('Sync failed:', err)
      return null
    } finally {
      setSyncing(false)
    }
  }

  // Auto-sync loop
  useEffect(() => {
    if (!autoSync) return
    
    const loop = async () => {
      const result = await runSync(20)
      if (result && result.remaining > 0) {
        setTimeout(loop, 2000)
      } else {
        setAutoSync(false)
      }
    }
    
    loop()
  }, [autoSync])

  useEffect(() => {
    fetchStatus()
  }, [])

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">HeyGen Video Sync</h1>
            <p className="text-muted-foreground">Download from HeyGen → Upload to Vercel Blob → Update DB</p>
          </div>
          <Button variant="outline" onClick={fetchStatus}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Status Cards */}
        {status && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-blue-500">{status.total}</div>
                <div className="text-sm text-muted-foreground">Total in HeyGen</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-green-500">{status.completed}</div>
                <div className="text-sm text-muted-foreground">Completed</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-purple-500">{status.alreadySynced}</div>
                <div className="text-sm text-muted-foreground">Already Synced</div>
              </CardContent>
            </Card>
            <Card className="border-orange-500/50">
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-orange-500">{status.needsSync}</div>
                <div className="text-sm text-muted-foreground">Needs Sync</div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Sync Controls */}
        <Card>
          <CardHeader>
            <CardTitle>Sync Controls</CardTitle>
            <CardDescription>
              Download completed videos from HeyGen and upload to Vercel Blob for permanent storage
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-4 flex-wrap">
              <Button 
                onClick={() => runSync(10)} 
                disabled={syncing || autoSync}
                className="bg-blue-600 hover:bg-blue-700"
              >
                <Download className="w-4 h-4 mr-2" />
                Sync 10 Videos
              </Button>
              <Button 
                onClick={() => runSync(25)} 
                disabled={syncing || autoSync}
                className="bg-green-600 hover:bg-green-700"
              >
                <Download className="w-4 h-4 mr-2" />
                Sync 25 Videos
              </Button>
              <Button 
                onClick={() => setAutoSync(!autoSync)} 
                disabled={syncing}
                variant={autoSync ? "destructive" : "default"}
                className={autoSync ? "" : "bg-orange-600 hover:bg-orange-700"}
              >
                <Zap className="w-4 h-4 mr-2" />
                {autoSync ? 'Stop Auto-Sync' : 'Auto-Sync All'}
              </Button>
            </div>
            
            {(syncing || autoSync) && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Syncing... Total synced this session: {totalSynced}</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Last Result */}
        {lastResult && (
          <Card>
            <CardHeader>
              <CardTitle>Last Sync Result</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-4 gap-4 mb-4">
                <div className="text-center p-3 bg-muted rounded-lg">
                  <div className="text-2xl font-bold">{lastResult.processed}</div>
                  <div className="text-xs text-muted-foreground">Processed</div>
                </div>
                <div className="text-center p-3 bg-green-500/20 rounded-lg">
                  <div className="text-2xl font-bold text-green-500">{lastResult.successful}</div>
                  <div className="text-xs text-muted-foreground">Successful</div>
                </div>
                <div className="text-center p-3 bg-red-500/20 rounded-lg">
                  <div className="text-2xl font-bold text-red-500">{lastResult.failed}</div>
                  <div className="text-xs text-muted-foreground">Failed</div>
                </div>
                <div className="text-center p-3 bg-orange-500/20 rounded-lg">
                  <div className="text-2xl font-bold text-orange-500">{lastResult.remaining}</div>
                  <div className="text-xs text-muted-foreground">Remaining</div>
                </div>
              </div>
              
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {lastResult.results.map((r, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm p-2 bg-muted/50 rounded">
                    {r.success ? (
                      <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
                    )}
                    <span className="font-mono text-xs truncate">{r.video_id}</span>
                    {r.error && <span className="text-red-400 text-xs">{r.error}</span>}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Pending Videos */}
        {status && status.videos && status.videos.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Videos Needing Sync</CardTitle>
              <CardDescription>Showing first 20 of {status.needsSync}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {status.videos.map((v, i) => (
                  <div key={i} className="flex items-center gap-3 text-sm p-2 bg-muted/50 rounded">
                    <Clock className="w-4 h-4 text-orange-500 flex-shrink-0" />
                    <span className="font-medium">{v.title || v.video_id}</span>
                    {v.parsed && (
                      <span className="text-xs text-muted-foreground">
                        Day {v.parsed.day} • {v.parsed.language} • {v.parsed.age} • {v.parsed.archetype} • {v.parsed.phase}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
