'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Flame, RefreshCw, CheckCircle, XCircle, Clock, Zap } from 'lucide-react'

interface FireStatus {
  credits: number
  stats: Array<{ status: string; count: string }>
  byDay: Array<{ day_of_year: number; count: string; completed: string }>
  plan: {
    priorityDays: number[]
    priorityArchetypes: string[]
    priorityAges: string[]
    priorityLangs: string[]
    videosPerFullDay: number
  }
}

interface FireResult {
  status: string
  batchSize: number
  creditsUsed: number
  summary: { submitted: number; failed: number; remainingInQueue: number }
  submitted: string[]
  failed: string[]
}

export default function FirePage() {
  const [status, setStatus] = useState<FireStatus | null>(null)
  const [result, setResult] = useState<FireResult | null>(null)
  const [firing, setFiring] = useState(false)
  const [loading, setLoading] = useState(true)
  const [autoFire, setAutoFire] = useState(false)
  const [batchCount, setBatchCount] = useState(0)
  const [totalFired, setTotalFired] = useState(0)

  const fetchStatus = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/heygen/fire-all')
      const data = await res.json()
      setStatus(data)
    } catch (e) {
      console.error('Failed to fetch status:', e)
    }
    setLoading(false)
  }

  const fireAll = async () => {
    if (!confirm('FIRE ALL CREDITS? This will spend all available HeyGen credits immediately.')) {
      return
    }
    
    setFiring(true)
    setResult(null)
    
    try {
      const res = await fetch('/api/heygen/fire-all', { method: 'POST' })
      const data = await res.json()
      setResult(data)
      await fetchStatus() // Refresh status
    } catch (e) {
      console.error('Fire failed:', e)
    }
    
    setFiring(false)
  }

  useEffect(() => {
    fetchStatus()
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Flame className="w-8 h-8 text-orange-500" />
            HeyGen Fire Control
          </h1>
          <Button onClick={fetchStatus} variant="outline" size="sm" disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        {/* Credits Display */}
        {status && (
          <div className="bg-gradient-to-r from-orange-500/20 to-red-500/20 rounded-xl p-6 mb-8 border border-orange-500/30">
            <div className="text-5xl font-bold text-orange-400 mb-2">
              {status.credits} Credits
            </div>
            <p className="text-white/60">Available to fire</p>
          </div>
        )}

        {/* Fire Button */}
        <Button
          onClick={fireAll}
          disabled={firing || !status?.credits}
          className="w-full py-8 text-2xl font-bold bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 mb-8"
        >
          {firing ? (
            <>
              <Zap className="w-8 h-8 mr-3 animate-pulse" />
              FIRING... DO NOT CLOSE
            </>
          ) : (
            <>
              <Flame className="w-8 h-8 mr-3" />
              FIRE ALL {status?.credits || 0} CREDITS NOW
            </>
          )}
        </Button>

        {/* Result */}
        {result && (
          <div className="bg-white/5 rounded-xl p-6 mb-8 border border-white/10">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              {result.summary.submitted > 0 ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <XCircle className="w-5 h-5 text-red-400" />
              )}
              Fire Result
            </h2>
            
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="bg-green-500/20 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-green-400">{result.summary.submitted}</div>
                <div className="text-sm text-white/60">Submitted</div>
              </div>
              <div className="bg-yellow-500/20 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-yellow-400">{result.summary.skipped}</div>
                <div className="text-sm text-white/60">Skipped</div>
              </div>
              <div className="bg-red-500/20 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-red-400">{result.summary.failed}</div>
                <div className="text-sm text-white/60">Failed</div>
              </div>
            </div>

            <div className="text-sm text-white/60">
              Credits: {result.startCredits} → {result.creditsRemaining} ({result.creditsUsed} used)
            </div>

            {result.failed.length > 0 && (
              <div className="mt-4 p-3 bg-red-500/10 rounded-lg">
                <div className="font-semibold text-red-400 mb-2">Failed:</div>
                <div className="text-xs font-mono text-white/60 max-h-32 overflow-y-auto">
                  {result.failed.map((f, i) => <div key={i}>{f}</div>)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Current Stats */}
        {status && (
          <>
            <div className="grid grid-cols-2 gap-4 mb-8">
              {status.stats.map((s, i) => (
                <div key={i} className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <div className="text-2xl font-bold">{s.count}</div>
                  <div className="text-sm text-white/60 capitalize">{s.status}</div>
                </div>
              ))}
            </div>

            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Videos by Day
            </h3>
            <div className="grid grid-cols-4 gap-3 mb-8">
              {status.byDay.map((d, i) => (
                <div key={i} className="bg-white/5 rounded-lg p-3 border border-white/10 text-center">
                  <div className="text-lg font-bold">Day {d.day_of_year}</div>
                  <div className="text-sm text-white/60">
                    {d.completed}/{d.count} done
                  </div>
                </div>
              ))}
            </div>

            <div className="text-xs text-white/40 space-y-1">
              <div>Priority Days: {status.plan.priorityDays.join(', ')}</div>
              <div>Archetypes: {status.plan.priorityArchetypes.join(', ')}</div>
              <div>Ages: {status.plan.priorityAges.join(', ')}</div>
              <div>Languages: {status.plan.priorityLangs.join(', ')}</div>
              <div>Videos per full day: {status.plan.videosPerFullDay}</div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
