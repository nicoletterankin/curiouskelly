'use client'

import { useState, useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'
import { Play, Pause, Volume2, VolumeX, ThumbsUp, ThumbsDown, RefreshCw, Check } from 'lucide-react'

type VideoEngine = 'heygen' | 'fal' | 'sync' | 'musetalk'

interface EngineVideo {
  engine: VideoEngine
  label: string
  color: string
  videoUrl: string | null
  audioUrl: string | null
  status: 'ready' | 'generating' | 'pending' | 'none'
  jobId?: string
}

interface CompareResponse {
  day: number
  phase: string
  engines: EngineVideo[]
  coverage: { total: number; ready: number; percent?: number }
  error?: string
}

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const

export function VideoEngineComparison({ day = 1 }: { day?: number }) {
  const [data, setData] = useState<CompareResponse | null>(null)
  const [selectedPhase, setSelectedPhase] = useState<string>('hook')
  const [loading, setLoading] = useState(true)
  const [playingEngine, setPlayingEngine] = useState<VideoEngine | null>(null)
  const [mutedEngines, setMutedEngines] = useState<Set<VideoEngine>>(new Set())
  const [ratings, setRatings] = useState<Record<VideoEngine, 'best' | 'worst' | null>>({
    heygen: null,
    fal: null,
    sync: null,
    musetalk: null,
  })
  const [currentDay, setCurrentDay] = useState(day)
  
  const videoRefs = useRef<Record<VideoEngine, HTMLVideoElement | null>>({
    heygen: null,
    fal: null,
    sync: null,
    musetalk: null,
  })

  // Fetch comparison data
  const fetchData = async () => {
    setLoading(true)
    try {
      const res = await fetch(`/api/video/compare?day=${currentDay}&phase=${selectedPhase}`)
      const json = await res.json()
      setData(json)
    } catch (err) {
      console.error('Failed to fetch comparison:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [currentDay, selectedPhase])

  // Play/pause handlers
  const handlePlayPause = (engine: VideoEngine) => {
    const video = videoRefs.current[engine]
    if (!video) return

    if (playingEngine === engine) {
      video.pause()
      setPlayingEngine(null)
    } else {
      // Pause all other videos
      Object.entries(videoRefs.current).forEach(([e, v]) => {
        if (v && e !== engine) v.pause()
      })
      video.play()
      setPlayingEngine(engine)
    }
  }

  const toggleMute = (engine: VideoEngine) => {
    const video = videoRefs.current[engine]
    if (!video) return
    
    const newMuted = new Set(mutedEngines)
    if (mutedEngines.has(engine)) {
      newMuted.delete(engine)
      video.muted = false
    } else {
      newMuted.add(engine)
      video.muted = true
    }
    setMutedEngines(newMuted)
  }

  const handleRate = async (engine: VideoEngine, rating: 'best' | 'worst') => {
    setRatings(prev => ({ ...prev, [engine]: prev[engine] === rating ? null : rating }))
    
    // Submit rating to API
    try {
      await fetch('/api/video/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ day: currentDay, phase: selectedPhase, engine, rating }),
      })
    } catch (err) {
      console.error('Failed to submit rating:', err)
    }
  }

  const playAll = () => {
    Object.values(videoRefs.current).forEach(v => {
      if (v) {
        v.currentTime = 0
        v.play()
      }
    })
    setPlayingEngine('heygen') // Just to show playing state
  }

  const pauseAll = () => {
    Object.values(videoRefs.current).forEach(v => v?.pause())
    setPlayingEngine(null)
  }

  return (
    <div className="space-y-6">
      {/* Header with day selector */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-bold text-white">Day {currentDay}</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentDay(Math.max(1, currentDay - 1))}
              className="w-8 h-8 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 flex items-center justify-center"
            >
              -
            </button>
            <input
              type="number"
              value={currentDay}
              onChange={(e) => setCurrentDay(Math.max(1, Math.min(365, parseInt(e.target.value) || 1)))}
              className="w-16 px-2 py-1 rounded-lg bg-white/10 text-white text-center border border-white/10"
            />
            <button
              onClick={() => setCurrentDay(Math.min(365, currentDay + 1))}
              className="w-8 h-8 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 flex items-center justify-center"
            >
              +
            </button>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={playAll}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/10 text-white hover:bg-white/20 transition-colors"
          >
            <Play className="w-4 h-4" />
            Play All
          </button>
          <button
            onClick={pauseAll}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/10 text-white hover:bg-white/20 transition-colors"
          >
            <Pause className="w-4 h-4" />
            Pause All
          </button>
          <button
            onClick={fetchData}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/10 text-white hover:bg-white/20 transition-colors"
          >
            <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
          </button>
        </div>
      </div>

      {/* Phase selector - Apple glass pills */}
      <div className="flex gap-2 p-1 rounded-xl bg-white/5 w-fit">
        {PHASES.map(phase => (
          <button
            key={phase}
            onClick={() => setSelectedPhase(phase)}
            className={cn(
              'px-5 py-2 rounded-lg text-sm font-medium capitalize transition-all duration-200',
              selectedPhase === phase
                ? 'bg-white/15 text-white shadow-sm'
                : 'text-white/50 hover:text-white/80 hover:bg-white/5'
            )}
          >
            {phase}
          </button>
        ))}
      </div>

      {/* 4-up video grid */}
      {loading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="aspect-[9/16] rounded-2xl bg-white/5 animate-pulse" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {data?.engines.map(engine => (
            <div
              key={engine.engine}
              className="relative rounded-2xl overflow-hidden bg-black/40 backdrop-blur-sm border border-white/10"
            >
              {/* Engine label badge */}
              <div
                className="absolute top-3 left-3 z-10 px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider"
                style={{ backgroundColor: engine.color + '30', color: engine.color }}
              >
                {engine.label}
              </div>

              {/* Video container - portrait aspect ratio for talking head */}
              <div className="aspect-[9/16] relative">
                {engine.status === 'ready' && engine.videoUrl ? (
                  <>
                    <video
                      ref={el => { videoRefs.current[engine.engine] = el }}
                      src={engine.videoUrl}
                      loop
                      playsInline
                      muted={mutedEngines.has(engine.engine)}
                      className="w-full h-full object-cover"
                      onEnded={() => setPlayingEngine(null)}
                    />
                    
                    {/* Play/pause overlay */}
                    <button
                      onClick={() => handlePlayPause(engine.engine)}
                      className="absolute inset-0 flex items-center justify-center bg-black/20 opacity-0 hover:opacity-100 transition-opacity"
                    >
                      {playingEngine === engine.engine ? (
                        <Pause className="w-12 h-12 text-white" />
                      ) : (
                        <Play className="w-12 h-12 text-white fill-white" />
                      )}
                    </button>

                    {/* Mute button */}
                    <button
                      onClick={() => toggleMute(engine.engine)}
                      className="absolute bottom-3 right-3 z-10 w-8 h-8 rounded-full bg-black/50 flex items-center justify-center text-white hover:bg-black/70 transition-colors"
                    >
                      {mutedEngines.has(engine.engine) ? (
                        <VolumeX className="w-4 h-4" />
                      ) : (
                        <Volume2 className="w-4 h-4" />
                      )}
                    </button>
                  </>
                ) : engine.status === 'generating' ? (
                  <div className="w-full h-full flex items-center justify-center bg-white/[0.02]">
                    <div className="text-center">
                      <div className="w-10 h-10 rounded-full border-2 border-white/20 border-t-orange-500 animate-spin mx-auto mb-3" />
                      <span className="text-xs text-white/50 font-medium">Generating...</span>
                    </div>
                  </div>
                ) : engine.status === 'pending' ? (
                  <div className="w-full h-full flex items-center justify-center bg-white/[0.02]">
                    <div className="text-center">
                      <div className="w-10 h-10 rounded-full border-2 border-white/10 mx-auto mb-3 flex items-center justify-center">
                        <span className="text-white/30 text-lg">?</span>
                      </div>
                      <span className="text-xs text-white/40 font-medium">Queued</span>
                    </div>
                  </div>
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-white/[0.02]">
                    <div className="text-center px-4">
                      <div className="w-10 h-10 rounded-full border-2 border-dashed border-white/10 mx-auto mb-3 flex items-center justify-center">
                        <span className="text-white/20 text-xl">-</span>
                      </div>
                      <span className="text-xs text-white/30 font-medium block">No Video</span>
                      <span className="text-[10px] text-white/20 mt-1 block">Not generated yet</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Rating buttons */}
              <div className="p-3 flex justify-center gap-2 border-t border-white/5">
                <button
                  onClick={() => handleRate(engine.engine, 'best')}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
                    ratings[engine.engine] === 'best'
                      ? 'bg-emerald-500/30 text-emerald-400 ring-1 ring-emerald-500/50'
                      : 'bg-white/5 text-white/50 hover:bg-emerald-500/10 hover:text-emerald-400'
                  )}
                >
                  {ratings[engine.engine] === 'best' ? <Check className="w-3 h-3" /> : <ThumbsUp className="w-3 h-3" />}
                  Best
                </button>
                <button
                  onClick={() => handleRate(engine.engine, 'worst')}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
                    ratings[engine.engine] === 'worst'
                      ? 'bg-rose-500/30 text-rose-400 ring-1 ring-rose-500/50'
                      : 'bg-white/5 text-white/50 hover:bg-rose-500/10 hover:text-rose-400'
                  )}
                >
                  {ratings[engine.engine] === 'worst' ? <Check className="w-3 h-3" /> : <ThumbsDown className="w-3 h-3" />}
                  Worst
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Coverage summary - HONEST about what exists */}
      {data && (
        <div className={cn(
          "p-4 rounded-xl border",
          data.coverage.ready === 0 
            ? "bg-rose-500/5 border-rose-500/20" 
            : data.coverage.ready < 4 
              ? "bg-amber-500/5 border-amber-500/20"
              : "bg-emerald-500/5 border-emerald-500/20"
        )}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-white">Coverage Status</span>
            <span className={cn(
              "px-2 py-0.5 rounded text-xs font-bold",
              data.coverage.ready === 0 
                ? "bg-rose-500/20 text-rose-400" 
                : data.coverage.ready < 4 
                  ? "bg-amber-500/20 text-amber-400"
                  : "bg-emerald-500/20 text-emerald-400"
            )}>
              {data.coverage.ready}/{data.coverage.total} READY
            </span>
          </div>
          <p className="text-sm text-white/50">
            <span className="text-white font-medium">Day {currentDay}</span> / {' '}
            <span className="text-white font-medium capitalize">{selectedPhase}</span> phase: {' '}
            {data.coverage.ready === 0 ? (
              <span className="text-rose-400">No videos generated yet for any engine.</span>
            ) : data.coverage.ready < 4 ? (
              <span className="text-amber-400">Partial coverage. {4 - data.coverage.ready} engine(s) need generation.</span>
            ) : (
              <span className="text-emerald-400">All engines ready for comparison!</span>
            )}
          </p>
        </div>
      )}
    </div>
  )
}
