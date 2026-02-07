'use client'

import { useRef, useState, useEffect, useCallback } from 'react'
import { cn } from '@/lib/utils'
import type { VideoState, Caption } from '@/lib/types'
import { Play, Pause, Volume2, VolumeX, Maximize, Minimize, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'

interface VideoPlayerProps {
  videoUrl: string | null
  audioUrl?: string | null
  fallbackImageUrl: string | null
  status: 'ready' | 'generating' | 'error'
  captions?: Caption[]
  autoPlay?: boolean
  onStateChange?: (state: VideoState) => void
  onTimeUpdate?: (time: number) => void
  onPhaseComplete?: () => void
  className?: string
}

export function VideoPlayer({
  videoUrl,
  audioUrl,
  fallbackImageUrl,
  status,
  captions = [],
  autoPlay = false,
  onStateChange,
  onTimeUpdate,
  onPhaseComplete,
  className,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  
  // Determine if we're in audio-only mode (no video, but have audio)
  const isAudioOnly = !videoUrl && !!audioUrl
  
  const [playerState, setPlayerState] = useState<VideoState>('idle')
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [buffered, setBuffered] = useState(0)
  const [volume, setVolume] = useState(1)
  const [muted, setMuted] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showControls, setShowControls] = useState(true)
  const [currentCaption, setCurrentCaption] = useState<string>('')
  
  const controlsTimeoutRef = useRef<NodeJS.Timeout>()

  // Update state and notify parent
  const updateState = useCallback((state: VideoState) => {
    setPlayerState(state)
    onStateChange?.(state)
  }, [onStateChange])

  // Handle video events
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleLoadStart = () => updateState('loading')
    const handleCanPlay = () => updateState('ready')
    const handlePlaying = () => updateState('playing')
    const handlePause = () => updateState('paused')
    const handleWaiting = () => updateState('buffering')
    const handleSeeking = () => updateState('seeking')
    const handleError = () => updateState('error')
    const handleEnded = () => {
      updateState('idle')
      onPhaseComplete?.()
    }

    const handleTimeUpdate = () => {
      const time = video.currentTime
      setCurrentTime(time)
      onTimeUpdate?.(time)

      // Update caption
      const activeCaption = captions.find(
        (c) => time >= c.startTime && time <= c.endTime
      )
      setCurrentCaption(activeCaption?.text || '')
    }

    const handleDurationChange = () => setDuration(video.duration)
    
    const handleProgress = () => {
      if (video.buffered.length > 0) {
        setBuffered(video.buffered.end(video.buffered.length - 1))
      }
    }

    video.addEventListener('loadstart', handleLoadStart)
    video.addEventListener('canplay', handleCanPlay)
    video.addEventListener('playing', handlePlaying)
    video.addEventListener('pause', handlePause)
    video.addEventListener('waiting', handleWaiting)
    video.addEventListener('seeking', handleSeeking)
    video.addEventListener('error', handleError)
    video.addEventListener('ended', handleEnded)
    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('durationchange', handleDurationChange)
    video.addEventListener('progress', handleProgress)

    return () => {
      video.removeEventListener('loadstart', handleLoadStart)
      video.removeEventListener('canplay', handleCanPlay)
      video.removeEventListener('playing', handlePlaying)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('waiting', handleWaiting)
      video.removeEventListener('seeking', handleSeeking)
      video.removeEventListener('error', handleError)
      video.removeEventListener('ended', handleEnded)
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('durationchange', handleDurationChange)
      video.removeEventListener('progress', handleProgress)
    }
  }, [captions, onTimeUpdate, onPhaseComplete, updateState])

  // Audio-only event handlers
  useEffect(() => {
    const audio = audioRef.current
    if (!audio || !isAudioOnly) return

    const handleLoadStart = () => updateState('loading')
    const handleCanPlay = () => updateState('ready')
    const handlePlaying = () => updateState('playing')
    const handlePause = () => updateState('paused')
    const handleWaiting = () => updateState('buffering')
    const handleError = () => updateState('error')
    const handleEnded = () => {
      updateState('idle')
      onPhaseComplete?.()
    }
    const handleTimeUpdate = () => {
      const time = audio.currentTime
      setCurrentTime(time)
      onTimeUpdate?.(time)
      const activeCaption = captions.find(
        (c) => time >= c.startTime && time <= c.endTime
      )
      setCurrentCaption(activeCaption?.text || '')
    }
    const handleDurationChange = () => setDuration(audio.duration)

    audio.addEventListener('loadstart', handleLoadStart)
    audio.addEventListener('canplay', handleCanPlay)
    audio.addEventListener('playing', handlePlaying)
    audio.addEventListener('pause', handlePause)
    audio.addEventListener('waiting', handleWaiting)
    audio.addEventListener('error', handleError)
    audio.addEventListener('ended', handleEnded)
    audio.addEventListener('timeupdate', handleTimeUpdate)
    audio.addEventListener('durationchange', handleDurationChange)

    return () => {
      audio.removeEventListener('loadstart', handleLoadStart)
      audio.removeEventListener('canplay', handleCanPlay)
      audio.removeEventListener('playing', handlePlaying)
      audio.removeEventListener('pause', handlePause)
      audio.removeEventListener('waiting', handleWaiting)
      audio.removeEventListener('error', handleError)
      audio.removeEventListener('ended', handleEnded)
      audio.removeEventListener('timeupdate', handleTimeUpdate)
      audio.removeEventListener('durationchange', handleDurationChange)
    }
  }, [isAudioOnly, captions, onTimeUpdate, onPhaseComplete, updateState])

  // Auto-play when video/audio is ready
  useEffect(() => {
    if (autoPlay && playerState === 'ready') {
      if (isAudioOnly) {
        audioRef.current?.play()
      } else if (videoUrl) {
        videoRef.current?.play()
      }
    }
  }, [autoPlay, videoUrl, isAudioOnly, playerState])

  // Handle status changes
  useEffect(() => {
    if (status === 'generating') {
      updateState('generating')
    } else if (status === 'error' && !videoUrl) {
      updateState('error')
    }
  }, [status, videoUrl, updateState])

  // Auto-hide controls
  const showControlsTemporarily = useCallback(() => {
    setShowControls(true)
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current)
    }
    if (playerState === 'playing') {
      controlsTimeoutRef.current = setTimeout(() => {
        setShowControls(false)
      }, 3000)
    }
  }, [playerState])

  // Toggle play/pause
  const togglePlay = useCallback(() => {
    const media = isAudioOnly ? audioRef.current : videoRef.current
    if (!media) return

    if (media.paused) {
      media.play()
    } else {
      media.pause()
    }
  }, [isAudioOnly])

  // Toggle mute
  const toggleMute = useCallback(() => {
    const media = isAudioOnly ? audioRef.current : videoRef.current
    if (!media) return

    media.muted = !media.muted
    setMuted(media.muted)
  }, [isAudioOnly])

  // Handle volume change
  const handleVolumeChange = useCallback((value: number[]) => {
    const media = isAudioOnly ? audioRef.current : videoRef.current
    if (!media) return

    const newVolume = value[0]
    media.volume = newVolume
    setVolume(newVolume)
    setMuted(newVolume === 0)
  }, [isAudioOnly])

  // Handle seek
  const handleSeek = useCallback((value: number[]) => {
    const media = isAudioOnly ? audioRef.current : videoRef.current
    if (!media) return

    media.currentTime = value[0]
  }, [isAudioOnly])

  // Toggle fullscreen
  const toggleFullscreen = useCallback(async () => {
    const container = containerRef.current
    if (!container) return

    try {
      if (!document.fullscreenElement) {
        await container.requestFullscreen()
        setIsFullscreen(true)
      } else {
        await document.exitFullscreen()
        setIsFullscreen(false)
      }
    } catch (error) {
      console.error('Fullscreen error:', error)
    }
  }, [])

  // Format time display
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Render loading/generating state
  if (status === 'generating' || playerState === 'generating') {
    return (
      <div
        ref={containerRef}
        className={cn(
          'relative aspect-[9/16] md:aspect-video bg-muted rounded-lg overflow-hidden flex items-center justify-center',
          className
        )}
      >
        {fallbackImageUrl && (
          <img
            src={fallbackImageUrl || "/placeholder.svg"}
            alt="Kelly"
            className="absolute inset-0 w-full h-full object-cover opacity-50"
          />
        )}
        <div className="relative z-10 flex flex-col items-center gap-4 text-center p-6">
          <Loader2 className="w-12 h-12 animate-spin text-primary" />
          <div>
            <p className="text-lg font-medium">Generating video...</p>
            <p className="text-sm text-muted-foreground">Kelly is preparing your lesson</p>
          </div>
        </div>
      </div>
    )
  }

  // Render audio-only mode (Kelly image + audio)
  if (isAudioOnly && audioUrl) {
    return (
      <div
        ref={containerRef}
        className={cn(
          'relative aspect-[9/16] md:aspect-video bg-black rounded-lg overflow-hidden group',
          className
        )}
        onMouseMove={showControlsTemporarily}
        onMouseLeave={() => playerState === 'playing' && setShowControls(false)}
      >
        {/* Kelly Image */}
        <img
          src={fallbackImageUrl || '/images/kelly-fallback.jpg'}
          alt="Kelly"
          className="w-full h-full object-cover cursor-pointer"
          onClick={togglePlay}
        />

        {/* Hidden Audio Element */}
        <audio ref={audioRef} src={audioUrl} preload="auto" />

        {/* Caption Overlay */}
        {currentCaption && (
          <div className="absolute bottom-20 left-4 right-4 pointer-events-none">
            <div className="bg-black/70 backdrop-blur-sm rounded-lg px-4 py-2 mx-auto max-w-2xl">
              <p className="text-white text-center text-lg md:text-xl font-medium">
                {currentCaption}
              </p>
            </div>
          </div>
        )}

        {/* Loading Overlay */}
        {(playerState === 'loading' || playerState === 'buffering') && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30">
            <Loader2 className="w-12 h-12 animate-spin text-white" />
          </div>
        )}

        {/* Play/Pause Overlay */}
        {(playerState === 'paused' || playerState === 'idle' || playerState === 'ready') && (
          <button
            onClick={togglePlay}
            className="absolute inset-0 flex items-center justify-center bg-black/20 hover:bg-black/30 transition-colors"
          >
            <div className="w-20 h-20 rounded-full bg-white/90 flex items-center justify-center">
              <Play className="w-10 h-10 text-black ml-1" />
            </div>
          </button>
        )}

        {/* Audio Indicator */}
        {playerState === 'playing' && (
          <div className="absolute top-4 right-4 bg-black/50 rounded-full px-3 py-1 flex items-center gap-2">
            <Volume2 className="w-4 h-4 text-white animate-pulse" />
            <span className="text-white text-sm">Audio Playing</span>
          </div>
        )}

        {/* Controls */}
        <div
          className={cn(
            'absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 via-black/40 to-transparent transition-opacity duration-300',
            showControls || playerState !== 'playing' ? 'opacity-100' : 'opacity-0'
          )}
        >
          <div className="mb-3">
            <Slider
              value={[currentTime]}
              max={duration || 100}
              step={0.1}
              onValueChange={handleSeek}
              className="cursor-pointer"
            />
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={togglePlay}
              className="text-white hover:bg-white/20"
            >
              {playerState === 'playing' ? (
                <Pause className="w-5 h-5" />
              ) : (
                <Play className="w-5 h-5" />
              )}
            </Button>
            <span className="text-white text-sm font-medium min-w-[80px]">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleMute}
                className="text-white hover:bg-white/20"
              >
                {muted || volume === 0 ? (
                  <VolumeX className="w-5 h-5" />
                ) : (
                  <Volume2 className="w-5 h-5" />
                )}
              </Button>
              <Slider
                value={[muted ? 0 : volume]}
                max={1}
                step={0.01}
                onValueChange={handleVolumeChange}
                className="w-20"
              />
            </div>
            <div className="flex-1" />
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleFullscreen}
              className="text-white hover:bg-white/20"
            >
              {isFullscreen ? (
                <Minimize className="w-5 h-5" />
              ) : (
                <Maximize className="w-5 h-5" />
              )}
            </Button>
          </div>
        </div>
      </div>
    )
  }

  // Render error/fallback state
  if (status === 'error' || (!videoUrl && !audioUrl && fallbackImageUrl)) {
    return (
      <div
        ref={containerRef}
        className={cn(
          'relative aspect-[9/16] md:aspect-video bg-muted rounded-lg overflow-hidden',
          className
        )}
      >
        <img
          src={fallbackImageUrl || '/images/kelly-fallback.jpg'}
          alt="Kelly"
          className="w-full h-full object-cover"
        />
        {currentCaption && (
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
            <p className="text-white text-center text-lg font-medium">{currentCaption}</p>
          </div>
        )}
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        'relative aspect-[9/16] md:aspect-video bg-black rounded-lg overflow-hidden group',
        className
      )}
      onMouseMove={showControlsTemporarily}
      onMouseLeave={() => playerState === 'playing' && setShowControls(false)}
    >
      {/* Video Element */}
      <video
        ref={videoRef}
        src={videoUrl || undefined}
        className="w-full h-full object-cover"
        playsInline
        onClick={togglePlay}
      />

      {/* Caption Overlay */}
      {currentCaption && (
        <div className="absolute bottom-20 left-4 right-4 pointer-events-none">
          <div className="bg-black/70 backdrop-blur-sm rounded-lg px-4 py-2 mx-auto max-w-2xl">
            <p className="text-white text-center text-lg md:text-xl font-medium">
              {currentCaption}
            </p>
          </div>
        </div>
      )}

      {/* Loading Overlay */}
      {(playerState === 'loading' || playerState === 'buffering') && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/30">
          <Loader2 className="w-12 h-12 animate-spin text-white" />
        </div>
      )}

      {/* Play/Pause Overlay */}
      {playerState === 'paused' && (
        <button
          onClick={togglePlay}
          className="absolute inset-0 flex items-center justify-center bg-black/20 hover:bg-black/30 transition-colors"
        >
          <div className="w-20 h-20 rounded-full bg-white/90 flex items-center justify-center">
            <Play className="w-10 h-10 text-black ml-1" />
          </div>
        </button>
      )}

      {/* Controls */}
      <div
        className={cn(
          'absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 via-black/40 to-transparent transition-opacity duration-300',
          showControls || playerState !== 'playing' ? 'opacity-100' : 'opacity-0'
        )}
      >
        {/* Progress Bar */}
        <div className="mb-3">
          <Slider
            value={[currentTime]}
            max={duration || 100}
            step={0.1}
            onValueChange={handleSeek}
            className="cursor-pointer"
          />
        </div>

        {/* Controls Row */}
        <div className="flex items-center gap-3">
          {/* Play/Pause */}
          <Button
            variant="ghost"
            size="icon"
            onClick={togglePlay}
            className="text-white hover:bg-white/20"
          >
            {playerState === 'playing' ? (
              <Pause className="w-5 h-5" />
            ) : (
              <Play className="w-5 h-5" />
            )}
          </Button>

          {/* Time */}
          <span className="text-white text-sm font-medium min-w-[80px]">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>

          {/* Volume */}
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleMute}
              className="text-white hover:bg-white/20"
            >
              {muted || volume === 0 ? (
                <VolumeX className="w-5 h-5" />
              ) : (
                <Volume2 className="w-5 h-5" />
              )}
            </Button>
            <Slider
              value={[muted ? 0 : volume]}
              max={1}
              step={0.01}
              onValueChange={handleVolumeChange}
              className="w-20"
            />
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Fullscreen */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleFullscreen}
            className="text-white hover:bg-white/20"
          >
            {isFullscreen ? (
              <Minimize className="w-5 h-5" />
            ) : (
              <Maximize className="w-5 h-5" />
            )}
          </Button>
        </div>
      </div>
    </div>
  )
}
