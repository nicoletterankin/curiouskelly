'use client'

import { useState, useEffect } from 'react'

interface GenerationResult {
  success?: boolean
  day?: number
  processed?: number
  successful?: number
  failed?: number
  elapsedMs?: number
  message?: string
  error?: string
  results?: Array<{
    id: string
    phase: string
    status: string
    audioUrl?: string
    error?: string
  }>
}

interface StatusResult {
  day: number
  lesson: { title: string; hook_script: string }
  assets: { total: string; with_audio: string; with_video: string }
  ready: boolean
}

export default function GenerateDay34Page() {
  const [status, setStatus] = useState<StatusResult | null>(null)
  const [generating, setGenerating] = useState(false)
  const [result, setResult] = useState<GenerationResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Check status on load
  useEffect(() => {
    fetch('/api/admin/generate-day-34')
      .then(res => res.json())
      .then(setStatus)
      .catch(err => setError(err.message))
  }, [])

  const startGeneration = async () => {
    setGenerating(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch('/api/admin/generate-day-34', {
        method: 'POST',
      })
      const data = await res.json()
      setResult(data)
      
      // Refresh status
      const statusRes = await fetch('/api/admin/generate-day-34')
      const newStatus = await statusRes.json()
      setStatus(newStatus)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <h1 className="text-3xl font-bold">Day 34 Audio Generation</h1>
        
        {/* Current Status */}
        {status && (
          <div className="bg-card border rounded-lg p-6 space-y-4">
            <h2 className="text-xl font-semibold">Current Status</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-muted-foreground">Lesson</p>
                <p className="font-medium">{status.lesson?.title || 'Loading...'}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Hook Script</p>
                <p className="text-sm">{status.lesson?.hook_script || '-'}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Total Assets</p>
                <p className="font-medium">{status.assets?.total || '0'}</p>
              </div>
              <div>
                <p className="text-muted-foreground">With Audio</p>
                <p className="font-medium">{status.assets?.with_audio || '0'}</p>
              </div>
            </div>
          </div>
        )}

        {/* Generate Button */}
        <div className="bg-card border rounded-lg p-6">
          <button
            onClick={startGeneration}
            disabled={generating}
            className="w-full bg-primary text-primary-foreground py-4 rounded-lg font-semibold text-lg disabled:opacity-50"
          >
            {generating ? 'Generating Audio...' : 'Generate All Audio for Day 34'}
          </button>
          
          {generating && (
            <p className="text-center mt-4 text-muted-foreground">
              This may take a few minutes. Please wait...
            </p>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="bg-destructive/10 border border-destructive rounded-lg p-4">
            <p className="text-destructive font-medium">Error: {error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="bg-card border rounded-lg p-6 space-y-4">
            <h2 className="text-xl font-semibold">Generation Results</h2>
            
            {result.success ? (
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-4 bg-green-500/10 rounded-lg">
                  <p className="text-2xl font-bold text-green-600">{result.successful}</p>
                  <p className="text-sm text-muted-foreground">Successful</p>
                </div>
                <div className="text-center p-4 bg-red-500/10 rounded-lg">
                  <p className="text-2xl font-bold text-red-600">{result.failed}</p>
                  <p className="text-sm text-muted-foreground">Failed</p>
                </div>
                <div className="text-center p-4 bg-blue-500/10 rounded-lg">
                  <p className="text-2xl font-bold text-blue-600">{result.elapsedMs}ms</p>
                  <p className="text-sm text-muted-foreground">Time</p>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground">{result.message || result.error}</p>
            )}

            {result.results && result.results.length > 0 && (
              <div className="mt-4">
                <h3 className="font-medium mb-2">Details:</h3>
                <div className="max-h-64 overflow-y-auto space-y-2">
                  {result.results.map((r, i) => (
                    <div 
                      key={i} 
                      className={`p-2 rounded text-sm ${
                        r.status === 'success' ? 'bg-green-500/10' : 'bg-red-500/10'
                      }`}
                    >
                      <span className="font-medium">{r.phase}</span>: {r.status}
                      {r.error && <span className="text-red-600 ml-2">{r.error}</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
