'use client'

import { useState, useEffect, useRef } from 'react'

interface PipelineResult {
  success: boolean
  elapsed_seconds: string
  iterations_completed: number
  rate_limited: boolean
  audio: { generated: number; failed: number }
  video: { generated: number; failed: number }
  initial_status: Record<string, number>
  final_status: Record<string, number>
  log: Array<{ timestamp: string; step: string; message: string }>
  error?: string
  dryRun?: boolean
  status?: Record<string, number>
  plan?: Record<string, unknown>
}

export default function PipelineGoPage() {
  const [status, setStatus] = useState<'preflight' | 'idle' | 'running' | 'done' | 'error'>('preflight')
  const [result, setResult] = useState<PipelineResult | null>(null)
  const [preflightData, setPreflightData] = useState<PipelineResult | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [runCount, setRunCount] = useState(0)
  const [totalAudio, setTotalAudio] = useState(0)
  const [totalVideo, setTotalVideo] = useState(0)
  const started = useRef(false)

  const runPipeline = async (dryRun = false) => {
    if (!dryRun) setStatus('running')
    setErrorMsg(null)
    try {
      const res = await fetch('/api/pipeline/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audioBatchSize: 50,
          videoBatchSize: 25,
          maxIterations: 10,
          dayStart: 1,
          dayEnd: 365,
          dryRun,
        }),
      })
      const data = await res.json()

      if (dryRun) {
        setPreflightData(data)
        setStatus('idle')
        return data
      }

      setResult(data)
      if (data.success) {
        setTotalAudio(prev => prev + (data.audio?.generated || 0))
        setTotalVideo(prev => prev + (data.video?.generated || 0))
        setRunCount(prev => prev + 1)
        setStatus('done')
      } else {
        setErrorMsg(data.error || 'Unknown pipeline error')
        setStatus('error')
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Network error'
      setErrorMsg(msg)
      setStatus('error')
    }
  }

  // Pre-flight check on mount, then auto-fire
  useEffect(() => {
    if (!started.current) {
      started.current = true
      // Do a dry run first to confirm DB connectivity
      runPipeline(true).then((data) => {
        if (data?.success) {
          // DB is reachable, auto-fire real pipeline after 2s
          setTimeout(() => runPipeline(false), 2000)
        }
      })
    }
  }, [])

  // Auto-continue if there's more work to do
  useEffect(() => {
    if (status === 'done' && result?.success) {
      const needsMore =
        (result.final_status?.need_audio > 0 || result.final_status?.need_video > 0) &&
        !result.rate_limited &&
        (result.audio.generated > 0 || result.video.generated > 0)

      if (needsMore && runCount < 100) {
        const timer = setTimeout(() => {
          runPipeline(false)
        }, 3000)
        return () => clearTimeout(timer)
      }
    }
  }, [status, result, runCount])

  return (
    <div className="min-h-screen bg-background p-8 font-mono text-sm">
      <h1 className="text-2xl font-bold mb-4 font-sans">Pipeline Executor</h1>

      <div className="mb-6 p-4 rounded-lg border bg-muted/50">
        <div className="flex items-center gap-3 mb-2">
          <span className={`inline-block w-3 h-3 rounded-full ${
            status === 'preflight' ? 'bg-blue-500 animate-pulse' :
            status === 'idle' ? 'bg-muted-foreground' :
            status === 'running' ? 'bg-amber-500 animate-pulse' :
            status === 'done' ? 'bg-green-500' :
            'bg-red-500'
          }`} />
          <span className="font-semibold font-sans">
            {status === 'preflight' && 'Running pre-flight check...'}
            {status === 'idle' && 'Ready to fire...'}
            {status === 'running' && `Running batch ${runCount + 1}...`}
            {status === 'done' && `Batch ${runCount} complete`}
            {status === 'error' && 'Error occurred'}
          </span>
        </div>
        <p className="text-muted-foreground">
          Session totals: {totalAudio} audio generated, {totalVideo} video generated
        </p>
      </div>

      {preflightData?.success && preflightData.status && (
        <div className="mb-6 p-4 rounded-lg border border-blue-500/30 bg-blue-500/5">
          <h3 className="font-semibold font-sans mb-2">Pre-flight Status (DB Connected)</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
            <div>Total Slots: <strong>{Number(preflightData.status.total || 0).toLocaleString()}</strong></div>
            <div>With Audio: <strong>{Number(preflightData.status.with_audio || 0).toLocaleString()}</strong></div>
            <div>With Video: <strong>{Number(preflightData.status.with_video || 0).toLocaleString()}</strong></div>
            <div>Need Audio: <strong>{Number(preflightData.status.need_audio || 0).toLocaleString()}</strong></div>
            <div>Need Video: <strong>{Number(preflightData.status.need_video || 0).toLocaleString()}</strong></div>
          </div>
          {preflightData.plan && (
            <div className="mt-2 text-xs text-muted-foreground">
              Est. audio batches: {String(preflightData.plan.estimatedAudioBatches)} | 
              Est. video batches: {String(preflightData.plan.estimatedVideoBatches)}
            </div>
          )}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Audio Generated" value={result.audio?.generated || 0} />
            <StatCard label="Audio Failed" value={result.audio?.failed || 0} />
            <StatCard label="Video Generated" value={result.video?.generated || 0} />
            <StatCard label="Video Failed" value={result.video?.failed || 0} />
          </div>

          {result.final_status && (
            <div className="p-4 rounded-lg border">
              <h3 className="font-semibold font-sans mb-2">Current Coverage</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                <div>Total: <strong>{result.final_status.total?.toLocaleString()}</strong></div>
                <div>Audio: <strong>{result.final_status.with_audio?.toLocaleString()}</strong></div>
                <div>Video: <strong>{result.final_status.with_video?.toLocaleString()}</strong></div>
                <div>Need Audio: <strong>{result.final_status.need_audio?.toLocaleString()}</strong></div>
                <div>Need Video: <strong>{result.final_status.need_video?.toLocaleString()}</strong></div>
              </div>
            </div>
          )}

          <div className="p-4 rounded-lg border max-h-96 overflow-auto">
            <h3 className="font-semibold font-sans mb-2">Log ({result.elapsed_seconds}s)</h3>
            {result.log?.map((entry, i) => (
              <div key={i} className="py-0.5 text-xs">
                <span className="text-muted-foreground">[{entry.step}]</span>{' '}
                <span>{entry.message}</span>
              </div>
            ))}
          </div>

          {result.rate_limited && (
            <div className="p-3 rounded-lg border border-amber-500/50 bg-amber-500/10 text-amber-700">
              Rate limited by ElevenLabs. The pipeline will pause audio generation. 
              Video generation for existing audio will continue.
            </div>
          )}

          {status === 'done' && result.final_status?.need_audio === 0 && result.final_status?.need_video === 0 && (
            <div className="p-3 rounded-lg border border-green-500/50 bg-green-500/10 text-green-700 font-semibold">
              Pipeline complete! All 65,700 slots have audio and video.
            </div>
          )}

          <button
            onClick={() => runPipeline(false)}
            disabled={status === 'running'}
            className="px-4 py-2 rounded-lg bg-foreground text-background font-sans font-medium disabled:opacity-50"
          >
            {status === 'running' ? 'Running...' : 'Run Another Batch'}
          </button>
        </div>
      )}

      {status === 'error' && (
        <div className="p-4 rounded-lg border border-red-500/50">
          <p className="text-red-600 mb-1 font-semibold font-sans">Pipeline Error</p>
          <p className="text-red-500 text-xs mb-3 font-mono">{errorMsg || result?.error || 'Unknown error. Check environment variables (DATABASE_URL, ELEVENLABS_API_KEY, FAL_KEY).'}</p>
          <button
            onClick={() => runPipeline(false)}
            className="px-4 py-2 rounded-lg bg-foreground text-background font-sans font-medium"
          >
            Retry
          </button>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="p-3 rounded-lg border">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-xl font-bold">{value}</p>
    </div>
  )
}
