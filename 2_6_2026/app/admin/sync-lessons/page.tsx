'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { AlertCircle, CheckCircle, Loader2, RefreshCw, Database, ArrowRight } from 'lucide-react'

interface SyncStatus {
  neon?: {
    total: number
    verified: number
    fromSupabase: number
    contaminated: number
  }
  supabase?: {
    table: string
    count: number
  }
}

interface SyncResult {
  success?: boolean
  error?: string
  sourceTable?: string
  totalFromSupabase?: number
  synced?: number
  updated?: number
  errors?: string[]
  dryRun?: boolean
  sample?: Array<{ day: number; title: string; theme: string }>
  checkedTables?: string[]
  hint?: string
}

export default function SyncLessonsPage() {
  const [status, setStatus] = useState<SyncStatus | null>(null)
  const [syncResult, setSyncResult] = useState<SyncResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [syncing, setSyncing] = useState(false)

  const checkStatus = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/sync/lessons?check_supabase=true')
      const data = await res.json()
      setStatus(data)
    } catch (err) {
      console.error('[v0] Status check failed:', err)
    } finally {
      setLoading(false)
    }
  }

  const runDryRun = async () => {
    setSyncing(true)
    setSyncResult(null)
    try {
      const res = await fetch('/api/sync/lessons?dry_run=true', { method: 'POST' })
      const data = await res.json()
      setSyncResult(data)
    } catch (err) {
      setSyncResult({ error: err instanceof Error ? err.message : 'Unknown error' })
    } finally {
      setSyncing(false)
    }
  }

  const runSync = async () => {
    if (!confirm('This will overwrite ALL lessons in Neon with data from Supabase. Continue?')) {
      return
    }
    setSyncing(true)
    setSyncResult(null)
    try {
      const res = await fetch('/api/sync/lessons', { method: 'POST' })
      const data = await res.json()
      setSyncResult(data)
      // Refresh status after sync
      await checkStatus()
    } catch (err) {
      setSyncResult({ error: err instanceof Error ? err.message : 'Unknown error' })
    } finally {
      setSyncing(false)
    }
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Lesson Sync Tool</h1>
          <p className="text-muted-foreground">
            Sync REAL curriculum from Supabase (source of truth) to Neon (app database)
          </p>
        </div>

        {/* Critical Warning */}
        <Card className="border-destructive bg-destructive/5">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-5 w-5" />
              Critical: Two Track System
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <p><strong>LEARN Track:</strong> 365 daily lessons (20-year curated curriculum) - PRIORITY</p>
            <p><strong>GROW Track:</strong> AI-powered track (build later, keep separate)</p>
            <p className="text-destructive font-medium">
              Current Neon data is CONTAMINATED with placeholder content (popcorn/dolphins).
              Sync from Supabase to fix.
            </p>
          </CardContent>
        </Card>

        {/* Status Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Database Status
              </span>
              <Button variant="outline" size="sm" onClick={checkStatus} disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                <span className="ml-2">Refresh</span>
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {status ? (
              <div className="grid md:grid-cols-2 gap-6">
                {/* Neon Status */}
                <div className="space-y-3">
                  <h3 className="font-semibold text-lg">Neon (App Database)</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Total Lessons:</span>
                      <span className="font-mono">{status.neon?.total || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-green-600">Verified:</span>
                      <span className="font-mono text-green-600">{status.neon?.verified || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-blue-600">From Supabase:</span>
                      <span className="font-mono text-blue-600">{status.neon?.fromSupabase || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-destructive">Contaminated:</span>
                      <span className="font-mono text-destructive">{status.neon?.contaminated || 0}</span>
                    </div>
                  </div>
                </div>

                {/* Supabase Status */}
                <div className="space-y-3">
                  <h3 className="font-semibold text-lg">Supabase (Source of Truth)</h3>
                  {status.supabase ? (
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Table:</span>
                        <span className="font-mono">{status.supabase.table}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Total Records:</span>
                        <span className="font-mono">{status.supabase.count}</span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">Click refresh to check Supabase</p>
                  )}
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground">Click refresh to load status</p>
            )}
          </CardContent>
        </Card>

        {/* Sync Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Sync Actions</CardTitle>
            <CardDescription>
              Sync the real curriculum from Supabase to Neon
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-4">
              <Button variant="outline" onClick={runDryRun} disabled={syncing}>
                {syncing ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Dry Run (Preview)
              </Button>
              <Button onClick={runSync} disabled={syncing} className="bg-green-600 hover:bg-green-700">
                {syncing ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <ArrowRight className="h-4 w-4 mr-2" />}
                Sync from Supabase
              </Button>
            </div>

            {/* Sync Result */}
            {syncResult && (
              <div className={`p-4 rounded-lg border ${syncResult.error ? 'border-destructive bg-destructive/5' : 'border-green-500 bg-green-500/5'}`}>
                {syncResult.error ? (
                  <div className="space-y-2">
                    <p className="font-semibold text-destructive flex items-center gap-2">
                      <AlertCircle className="h-4 w-4" />
                      Sync Failed
                    </p>
                    <p className="text-sm">{syncResult.error}</p>
                    {syncResult.checkedTables && (
                      <p className="text-xs text-muted-foreground">Checked tables: {syncResult.checkedTables.join(', ')}</p>
                    )}
                    {syncResult.hint && (
                      <p className="text-xs">{syncResult.hint}</p>
                    )}
                  </div>
                ) : (
                  <div className="space-y-2">
                    <p className="font-semibold text-green-600 flex items-center gap-2">
                      <CheckCircle className="h-4 w-4" />
                      {syncResult.dryRun ? 'Dry Run Complete' : 'Sync Complete'}
                    </p>
                    <div className="text-sm space-y-1">
                      <p>Source Table: <span className="font-mono">{syncResult.sourceTable}</span></p>
                      <p>Lessons Found: <span className="font-mono">{syncResult.totalFromSupabase}</span></p>
                      {!syncResult.dryRun && (
                        <>
                          <p>New Synced: <span className="font-mono">{syncResult.synced}</span></p>
                          <p>Updated: <span className="font-mono">{syncResult.updated}</span></p>
                        </>
                      )}
                    </div>
                    {syncResult.sample && syncResult.sample.length > 0 && (
                      <div className="mt-3">
                        <p className="text-xs font-semibold mb-1">Sample Lessons:</p>
                        <div className="text-xs space-y-1">
                          {syncResult.sample.map((l, i) => (
                            <div key={i} className="font-mono">
                              Day {l.day}: {l.title} [{l.theme}]
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {syncResult.errors && syncResult.errors.length > 0 && (
                      <div className="mt-3">
                        <p className="text-xs font-semibold text-destructive mb-1">Errors:</p>
                        <div className="text-xs text-destructive space-y-1">
                          {syncResult.errors.map((e, i) => (
                            <div key={i}>{e}</div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
