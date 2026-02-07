"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { ArrowLeft, CheckCircle, Circle, AlertCircle, RefreshCw, Play, Download } from "lucide-react"

interface ProductionStatus {
  kellyLessonAssets: {
    total: number
    withAudio: number
    withVideo: number
    daysCovered: number
  }
  heygenVideos: {
    completed: number
    queued: number
    failed: number
    daysCovered: number
  }
  generatedAssets: {
    total: number
    audio: number
    video: number
  }
  coverage: {
    audioPercent: number
    videoPercent: number
    totalLessons: number
  }
}

export default function ProductionDashboard() {
  const [status, setStatus] = useState<ProductionStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/production/status')
      if (!res.ok) throw new Error('Failed to fetch status')
      const data = await res.json()
      setStatus(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStatus()
    // Refresh every 30 seconds
    const interval = setInterval(fetchStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-white/10 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/admin" className="text-zinc-500 hover:text-white transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <h1 
              className="text-xl"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
            >
              Production Dashboard
            </h1>
          </div>
          <button
            onClick={fetchStatus}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 text-sm border border-white/20 hover:bg-white/5 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-12">
        {error && (
          <div className="mb-8 p-4 border border-red-500/30 bg-red-500/10 text-red-400">
            {error}
          </div>
        )}

        {/* Coverage Overview */}
        <section className="mb-12">
          <h2 className="text-zinc-500 text-sm uppercase tracking-wider mb-6">Coverage Overview</h2>
          <div className="grid grid-cols-3 gap-6">
            <div className="p-6 border border-white/10">
              <p className="text-zinc-500 text-xs uppercase tracking-wider mb-2">Audio Coverage</p>
              <p className="text-4xl font-mono text-emerald-400">
                {status?.coverage.audioPercent.toFixed(1) || '—'}%
              </p>
              <p className="text-zinc-600 text-sm mt-2">
                {status?.kellyLessonAssets.withAudio.toLocaleString() || '—'} audio files
              </p>
            </div>
            <div className="p-6 border border-white/10">
              <p className="text-zinc-500 text-xs uppercase tracking-wider mb-2">Video Coverage</p>
              <p className="text-4xl font-mono text-blue-400">
                {status?.coverage.videoPercent.toFixed(1) || '—'}%
              </p>
              <p className="text-zinc-600 text-sm mt-2">
                {status?.heygenVideos.completed.toLocaleString() || '—'} videos ready
              </p>
            </div>
            <div className="p-6 border border-white/10">
              <p className="text-zinc-500 text-xs uppercase tracking-wider mb-2">Days Covered</p>
              <p className="text-4xl font-mono text-white">
                {status?.kellyLessonAssets.daysCovered || '—'}/365
              </p>
              <p className="text-zinc-600 text-sm mt-2">
                All phases with audio
              </p>
            </div>
          </div>
        </section>

        {/* Data Sources */}
        <section className="mb-12">
          <h2 className="text-zinc-500 text-sm uppercase tracking-wider mb-6">Data Sources</h2>
          <div className="space-y-4">
            {/* kelly_lesson_assets */}
            <div className="p-6 border border-white/10">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-white font-medium">kelly_lesson_assets</h3>
                  <p className="text-zinc-500 text-sm">ElevenLabs TTS audio and visual URLs</p>
                </div>
                <CheckCircle className="w-5 h-5 text-emerald-400" />
              </div>
              <div className="grid grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-zinc-600">Total Rows</p>
                  <p className="text-white font-mono">{status?.kellyLessonAssets.total.toLocaleString() || '—'}</p>
                </div>
                <div>
                  <p className="text-zinc-600">With Audio</p>
                  <p className="text-emerald-400 font-mono">{status?.kellyLessonAssets.withAudio.toLocaleString() || '—'}</p>
                </div>
                <div>
                  <p className="text-zinc-600">With Video</p>
                  <p className="text-blue-400 font-mono">{status?.kellyLessonAssets.withVideo.toLocaleString() || '—'}</p>
                </div>
                <div>
                  <p className="text-zinc-600">Days</p>
                  <p className="text-white font-mono">{status?.kellyLessonAssets.daysCovered || '—'}</p>
                </div>
              </div>
            </div>

            {/* heygen_videos */}
            <div className="p-6 border border-white/10">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-white font-medium">heygen_videos</h3>
                  <p className="text-zinc-500 text-sm">HeyGen lip-synced avatar videos</p>
                </div>
                {(status?.heygenVideos.completed || 0) > 0 ? (
                  <Circle className="w-5 h-5 text-yellow-400" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-zinc-600" />
                )}
              </div>
              <div className="grid grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-zinc-600">Completed</p>
                  <p className="text-emerald-400 font-mono">{status?.heygenVideos.completed.toLocaleString() || '—'}</p>
                </div>
                <div>
                  <p className="text-zinc-600">Queued</p>
                  <p className="text-yellow-400 font-mono">{status?.heygenVideos.queued.toLocaleString() || '—'}</p>
                </div>
                <div>
                  <p className="text-zinc-600">Failed</p>
                  <p className="text-red-400 font-mono">{status?.heygenVideos.failed || '—'}</p>
                </div>
                <div>
                  <p className="text-zinc-600">Days</p>
                  <p className="text-white font-mono">{status?.heygenVideos.daysCovered || '—'}</p>
                </div>
              </div>
            </div>

            {/* generated_assets */}
            <div className="p-6 border border-white/10">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-white font-medium">generated_assets</h3>
                  <p className="text-zinc-500 text-sm">Fal.ai and other generated media</p>
                </div>
                {(status?.generatedAssets.total || 0) > 0 ? (
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                ) : (
                  <Circle className="w-5 h-5 text-zinc-600" />
                )}
              </div>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-zinc-600">Total</p>
                  <p className="text-white font-mono">{status?.generatedAssets.total.toLocaleString() || '—'}</p>
                </div>
                <div>
                  <p className="text-zinc-600">Audio</p>
                  <p className="text-emerald-400 font-mono">{status?.generatedAssets.audio.toLocaleString() || '—'}</p>
                </div>
                <div>
                  <p className="text-zinc-600">Video</p>
                  <p className="text-blue-400 font-mono">{status?.generatedAssets.video.toLocaleString() || '—'}</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Actions */}
        <section>
          <h2 className="text-zinc-500 text-sm uppercase tracking-wider mb-6">Actions</h2>
          <div className="grid grid-cols-2 gap-4">
            <Link
              href="/play"
              className="flex items-center gap-3 p-4 border border-white/10 hover:bg-white/5 transition-colors"
            >
              <Play className="w-5 h-5 text-emerald-400" />
              <div>
                <p className="text-white font-medium">Test Learner Experience</p>
                <p className="text-zinc-500 text-sm">Play today's lesson</p>
              </div>
            </Link>
            <Link
              href="/api/lesson/download/32?age=adult&language=en"
              className="flex items-center gap-3 p-4 border border-white/10 hover:bg-white/5 transition-colors"
            >
              <Download className="w-5 h-5 text-blue-400" />
              <div>
                <p className="text-white font-medium">Test Download</p>
                <p className="text-zinc-500 text-sm">Download Day 32 bundle</p>
              </div>
            </Link>
          </div>
        </section>
      </main>
    </div>
  )
}
