'use client'

import { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { Play, Pause, RefreshCw, CheckCircle, XCircle, Loader2 } from 'lucide-react'

const TOTAL_VIDEOS = 180
const BATCH_SIZE = 10

interface BatchResult {
  index: number
  age: string
  archetype: string
  phase: string
  videoId?: string
  error?: string
}

export default function HeyGenBatchPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [results, setResults] = useState<BatchResult[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const abortRef = useRef(false)
  
  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [`[${timestamp}] ${message}`, ...prev.slice(0, 99)])
  }
  
  const runBatch = async (startIndex: number) => {
    if (abortRef.current) return
    
    addLog(`Starting batch from index ${startIndex}...`)
    
    try {
      const res = await fetch(`/api/heygen/batch-day18?action=generate&start=${startIndex}&batch=${BATCH_SIZE}`)
      const data = await res.json()
      
      if (data.success && data.results) {
        setResults(prev => [...prev, ...data.results])
        const successful = data.results.filter((r: BatchResult) => r.videoId).length
        const failed = data.results.filter((r: BatchResult) => r.error).length
        addLog(`Batch complete: ${successful} success, ${failed} failed`)
        
        const nextIndex = startIndex + BATCH_SIZE
        setCurrentIndex(nextIndex)
        
        if (nextIndex < TOTAL_VIDEOS && !abortRef.current) {
          // Small delay before next batch
          await new Promise(resolve => setTimeout(resolve, 2000))
          runBatch(nextIndex)
        } else {
          addLog(`All batches complete! Total: ${results.length + data.results.length} videos`)
          setIsRunning(false)
        }
      } else {
        addLog(`Batch failed: ${data.error || 'Unknown error'}`)
        setIsRunning(false)
      }
    } catch (error) {
      addLog(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`)
      setIsRunning(false)
    }
  }
  
  const startGeneration = (fromIndex: number = 0) => {
    abortRef.current = false
    setIsRunning(true)
    setCurrentIndex(fromIndex)
    if (fromIndex === 0) {
      setResults([])
      setLogs([])
    }
    addLog(`Starting Day 18 video generation from index ${fromIndex}`)
    runBatch(fromIndex)
  }
  
  const pauseGeneration = () => {
    abortRef.current = true
    setIsRunning(false)
    addLog('Generation paused')
  }
  
  const successCount = results.filter(r => r.videoId).length
  const failedCount = results.filter(r => r.error).length
  const progress = (currentIndex / TOTAL_VIDEOS) * 100
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-950 text-white p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">HeyGen Batch Generator</h1>
          <p className="text-slate-400">Day 18: Why Cats Purr - 180 Videos (3 ages × 12 archetypes × 5 phases)</p>
        </div>
        
        {/* Progress Card */}
        <div className="bg-slate-800/50 rounded-2xl p-6 mb-6 border border-slate-700">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-4xl font-bold">{Math.round(progress)}%</div>
              <div className="text-slate-400">{currentIndex} / {TOTAL_VIDEOS} videos</div>
            </div>
            <div className="flex gap-3">
              {!isRunning ? (
                <>
                  <Button 
                    onClick={() => startGeneration(currentIndex || 0)}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    <Play className="w-4 h-4 mr-2" />
                    {currentIndex > 0 ? 'Resume' : 'Start'}
                  </Button>
                  {currentIndex > 0 && (
                    <Button 
                      onClick={() => startGeneration(0)}
                      variant="outline"
                      className="border-slate-600"
                    >
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Restart
                    </Button>
                  )}
                </>
              ) : (
                <Button 
                  onClick={pauseGeneration}
                  className="bg-yellow-600 hover:bg-yellow-700"
                >
                  <Pause className="w-4 h-4 mr-2" />
                  Pause
                </Button>
              )}
            </div>
          </div>
          
          {/* Progress Bar */}
          <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          
          {/* Stats */}
          <div className="grid grid-cols-3 gap-4 mt-4">
            <div className="text-center p-3 bg-slate-700/50 rounded-xl">
              <div className="text-2xl font-bold text-green-400">{successCount}</div>
              <div className="text-xs text-slate-400">Success</div>
            </div>
            <div className="text-center p-3 bg-slate-700/50 rounded-xl">
              <div className="text-2xl font-bold text-red-400">{failedCount}</div>
              <div className="text-xs text-slate-400">Failed</div>
            </div>
            <div className="text-center p-3 bg-slate-700/50 rounded-xl">
              <div className="text-2xl font-bold text-blue-400">{TOTAL_VIDEOS - currentIndex}</div>
              <div className="text-xs text-slate-400">Remaining</div>
            </div>
          </div>
        </div>
        
        {/* Status Indicator */}
        {isRunning && (
          <div className="flex items-center justify-center gap-3 mb-6 p-4 bg-blue-500/20 rounded-xl border border-blue-500/30">
            <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
            <span className="text-blue-300">Generating videos... Do not close this page.</span>
          </div>
        )}
        
        {/* Logs */}
        <div className="bg-slate-800/50 rounded-2xl border border-slate-700 overflow-hidden">
          <div className="p-4 border-b border-slate-700 flex items-center justify-between">
            <h2 className="font-semibold">Activity Log</h2>
            <span className="text-xs text-slate-500">{logs.length} entries</span>
          </div>
          <div className="h-64 overflow-y-auto p-4 font-mono text-sm">
            {logs.length === 0 ? (
              <div className="text-slate-500 text-center py-8">
                Click "Start" to begin generating videos
              </div>
            ) : (
              logs.map((log, i) => (
                <div key={i} className={cn(
                  "py-1",
                  log.includes('success') && 'text-green-400',
                  log.includes('failed') || log.includes('Error') && 'text-red-400',
                  log.includes('Starting') && 'text-blue-400'
                )}>
                  {log}
                </div>
              ))
            )}
          </div>
        </div>
        
        {/* Recent Results */}
        {results.length > 0 && (
          <div className="mt-6 bg-slate-800/50 rounded-2xl border border-slate-700 overflow-hidden">
            <div className="p-4 border-b border-slate-700">
              <h2 className="font-semibold">Recent Results</h2>
            </div>
            <div className="max-h-64 overflow-y-auto">
              {results.slice(-20).reverse().map((result, i) => (
                <div key={i} className="flex items-center gap-3 p-3 border-b border-slate-700/50 last:border-0">
                  {result.videoId ? (
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="text-sm truncate">
                      {result.age} / {result.archetype} / {result.phase}
                    </div>
                    <div className="text-xs text-slate-500 truncate">
                      {result.videoId || result.error}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
