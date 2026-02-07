'use client'

// HeyGen Queue Processor - Process 315 queued videos
// Simple one-click interface for Nicolette

import { useState, useEffect } from 'react'
import { Play, Loader2, CheckCircle2, AlertCircle, RefreshCw, Zap, Database } from 'lucide-react'

interface QueueStatus {
  total: number
  byStatusAndLanguage: Array<{
    status: string
    language: string
    count: string
  }>
}

interface ProcessResult {
  processed: number
  submitted: number
  failed: number
  errors: string[]
  remainingInQueue: number
}

interface PollResult {
  checked: number
  completed: number
  failed: number
  stillProcessing: number
}

interface BatchResult {
  success: boolean
  message: string
  videosQueued?: number
  estimatedCredits?: number
  jobs?: Array<{
    day: number
    phase: string
    age: number
    language: string
  }>
}

export default function HeyGenBatchPage() {
  const [status, setStatus] = useState<'idle' | 'loading' | 'processing' | 'polling' | 'error'>('idle')
  const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null)
  const [processResult, setProcessResult] = useState<ProcessResult | null>(null)
  const [pollResult, setPollResult] = useState<PollResult | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [result, setResult] = useState<BatchResult | null>(null)
  
  const addLog = (msg: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`])
  }
  
  // Load queue status on mount
  useEffect(() => {
    loadQueueStatus()
  }, [])
  
  // Step 1: Load current queue status
  const loadQueueStatus = async () => {
    setStatus('loading')
    addLog('Loading queue status...')
    
    try {
      const res = await fetch('/api/heygen/process-queue')
      const data = await res.json()
      setQueueStatus(data)
      
      const queued = data.byStatusAndLanguage?.filter((s: {status: string}) => s.status === 'queued')
        .reduce((sum: number, s: {count: string}) => sum + parseInt(s.count), 0) || 0
      addLog(`Queue loaded: ${queued} videos ready to process`)
      setStatus('idle')
    } catch (err) {
      addLog(`Error loading queue: ${err}`)
      setStatus('error')
    }
  }
  
  // Step 2: Process queue batch (uses real credits!)
  const processBatch = async (batchSize: number = 10) => {
    if (!confirm(`This will process ${batchSize} videos and use HeyGen credits. Continue?`)) {
      return
    }
    
    setStatus('processing')
    addLog(`PROCESSING ${batchSize} videos - Using HeyGen credits...`)
    
    try {
      const res = await fetch('/api/heygen/process-queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batchSize, dryRun: false })
      })
      
      const data = await res.json()
      setProcessResult(data)
      setResult(data)
      
      addLog(`Submitted: ${data.submitted}, Failed: ${data.failed}, Remaining: ${data.remainingInQueue}`)
      if (data.errors?.length > 0) {
        data.errors.forEach((e: string) => addLog(`  Error: ${e}`))
      }
      
      // Refresh queue status
      await loadQueueStatus()
      setStatus('idle')
    } catch (err) {
      addLog(`Error: ${err}`)
      setStatus('error')
    }
  }
  
  // Step 3: Poll for completed videos
  const pollStatus = async () => {
    setStatus('polling')
    addLog('Polling HeyGen for completed videos...')
    
    try {
      const res = await fetch('/api/heygen/poll-status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 100 })
      })
      
      const data = await res.json()
      setPollResult(data)
      addLog(`Poll complete: ${data.completed || 0} done, ${data.stillProcessing || 0} processing, ${data.failed || 0} failed`)
      
      // Refresh queue status
      await loadQueueStatus()
      setStatus('idle')
    } catch (err) {
      addLog(`Poll error: ${err}`)
      setStatus('error')
    }
  }
  
  const runDryRun = async () => {
    setStatus('planning')
    addLog('Running dry run...')
    
    try {
      const res = await fetch('/api/heygen/process-queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batchSize: 10, dryRun: true })
      })
      
      const data = await res.json()
      setResult(data)
      
      addLog(`Dry run complete: ${data.videosQueued} videos to generate`)
      setStatus('idle')
    } catch (err) {
      addLog(`Dry run error: ${err}`)
      setStatus('error')
    }
  }
  
  const executeBatch = processBatch
  
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white p-6">
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-2xl font-bold flex items-center justify-center gap-2">
            <Zap className="w-6 h-6 text-amber-400" />
            HeyGen Queue Processor
          </h1>
          <p className="text-white/60 text-sm">Process 315 queued videos using 418 credits</p>
        </div>
        
        {/* Queue Status */}
        {queueStatus && (
          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-4 h-4 text-blue-400" />
              <span className="font-medium">Queue Status</span>
              <span className="text-xs text-white/40">({queueStatus.total} total)</span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              {queueStatus.byStatusAndLanguage?.map((s, i) => (
                <div key={i} className={`p-2 rounded-lg ${
                  s.status === 'queued' ? 'bg-amber-500/10 text-amber-400' :
                  s.status === 'processing' ? 'bg-blue-500/10 text-blue-400' :
                  s.status === 'completed' ? 'bg-green-500/10 text-green-400' :
                  'bg-red-500/10 text-red-400'
                }`}>
                  <div className="font-bold">{s.count}</div>
                  <div className="text-[10px] opacity-60">{s.language} {s.status}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Action Buttons */}
        <div className="grid grid-cols-4 gap-2">
          <button
            onClick={loadQueueStatus}
            disabled={status !== 'idle' && status !== 'error'}
            className="flex flex-col items-center gap-1 p-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 disabled:opacity-50 transition-all"
          >
            {status === 'loading' ? (
              <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
            ) : (
              <Database className="w-5 h-5 text-blue-400" />
            )}
            <span className="text-xs font-medium">Refresh</span>
          </button>
          
          <button
            onClick={() => processBatch(10)}
            disabled={status !== 'idle'}
            className="flex flex-col items-center gap-1 p-3 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 disabled:opacity-50 transition-all"
          >
            {status === 'processing' ? (
              <Loader2 className="w-5 h-5 animate-spin text-amber-400" />
            ) : (
              <Play className="w-5 h-5 text-amber-400" />
            )}
            <span className="text-xs font-medium">Process 10</span>
          </button>
          
          <button
            onClick={() => processBatch(50)}
            disabled={status !== 'idle'}
            className="flex flex-col items-center gap-1 p-3 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 disabled:opacity-50 transition-all"
          >
            <Zap className="w-5 h-5 text-amber-400" />
            <span className="text-xs font-medium">Process 50</span>
          </button>
          
          <button
            onClick={pollStatus}
            disabled={status === 'processing' || status === 'loading'}
            className="flex flex-col items-center gap-1 p-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 disabled:opacity-50 transition-all"
          >
            {status === 'polling' ? (
              <Loader2 className="w-5 h-5 animate-spin text-green-400" />
            ) : (
              <RefreshCw className="w-5 h-5 text-green-400" />
            )}
            <span className="text-xs font-medium">Poll</span>
          </button>
        </div>
        
        {/* Process Results */}
        {processResult && (
          <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-3">
            <div className="flex items-center gap-2">
              {processResult.failed === 0 ? (
                <CheckCircle2 className="w-5 h-5 text-green-400" />
              ) : (
                <AlertCircle className="w-5 h-5 text-amber-400" />
              )}
              <span className="font-medium">Batch Processed</span>
            </div>
            
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="p-3 rounded-lg bg-green-500/10">
                <div className="text-2xl font-bold text-green-400">{processResult.submitted}</div>
                <div className="text-xs text-white/40">Submitted</div>
              </div>
              <div className="p-3 rounded-lg bg-red-500/10">
                <div className="text-2xl font-bold text-red-400">{processResult.failed}</div>
                <div className="text-xs text-white/40">Failed</div>
              </div>
              <div className="p-3 rounded-lg bg-amber-500/10">
                <div className="text-2xl font-bold text-amber-400">{processResult.remainingInQueue}</div>
                <div className="text-xs text-white/40">Remaining</div>
              </div>
            </div>
            
            {processResult.errors?.length > 0 && (
              <details className="text-xs">
                <summary className="cursor-pointer text-red-400 hover:text-red-300">
                  View {processResult.errors.length} errors
                </summary>
                <div className="mt-2 max-h-32 overflow-y-auto space-y-1 text-red-400/60">
                  {processResult.errors.map((err, i) => (
                    <div key={i}>{err}</div>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}
        
        {/* Poll Results */}
        {pollResult && (
          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            <div className="grid grid-cols-4 gap-2 text-center">
              <div>
                <div className="text-lg font-bold text-white">{pollResult.checked}</div>
                <div className="text-[10px] text-white/40">Checked</div>
              </div>
              <div>
                <div className="text-lg font-bold text-green-400">{pollResult.completed}</div>
                <div className="text-[10px] text-white/40">Done</div>
              </div>
              <div>
                <div className="text-lg font-bold text-amber-400">{pollResult.stillProcessing}</div>
                <div className="text-[10px] text-white/40">Processing</div>
              </div>
              <div>
                <div className="text-lg font-bold text-red-400">{pollResult.failed}</div>
                <div className="text-[10px] text-white/40">Failed</div>
              </div>
            </div>
          </div>
        )}
        
        {/* Logs */}
        <div className="p-4 rounded-xl bg-black/50 border border-white/10 font-mono text-xs space-y-1 max-h-60 overflow-y-auto">
          {logs.length === 0 ? (
            <div className="text-white/30">Click "Plan" to start...</div>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="text-white/60">{log}</div>
            ))
          )}
        </div>
        
        {/* Instructions */}
        <div className="text-xs text-white/40 space-y-1">
          <p><strong>1. Plan</strong> - See what videos will be generated (no credits used)</p>
          <p><strong>2. Execute</strong> - Start generating videos (USES CREDITS)</p>
          <p><strong>3. Poll</strong> - Check progress and download completed videos (run multiple times)</p>
        </div>
      </div>
    </div>
  )
}
