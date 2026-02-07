'use client'

import { useEffect, useState } from 'react'

interface TestResult {
  step: string
  status: 'pending' | 'running' | 'success' | 'error'
  data?: unknown
  error?: string
}

export default function HeyGenAutoTest() {
  const [results, setResults] = useState<TestResult[]>([
    { step: 'Check Credits', status: 'pending' },
    { step: 'List Avatars', status: 'pending' },
    { step: 'Generate Test Video', status: 'pending' },
  ])
  const [avatarId, setAvatarId] = useState<string | null>(null)
  const [videoId, setVideoId] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)

  const updateResult = (index: number, update: Partial<TestResult>) => {
    setResults(prev => prev.map((r, i) => i === index ? { ...r, ...update } : r))
  }

  const runTests = async () => {
    setIsRunning(true)
    
    // Step 1: Check Credits
    updateResult(0, { status: 'running' })
    try {
      const creditsRes = await fetch('/api/heygen/simple?action=credits')
      const creditsData = await creditsRes.json()
      if (creditsData.error) throw new Error(creditsData.error)
      updateResult(0, { status: 'success', data: creditsData })
    } catch (e) {
      updateResult(0, { status: 'error', error: e instanceof Error ? e.message : 'Failed' })
      setIsRunning(false)
      return
    }

    // Step 2: List Avatars
    updateResult(1, { status: 'running' })
    try {
      const avatarsRes = await fetch('/api/heygen/simple?action=avatars')
      const avatarsData = await avatarsRes.json()
      if (avatarsData.error) throw new Error(avatarsData.error)
      
      // Find first avatar or Kelly
      const avatars = avatarsData.avatars || []
      const kelly = avatars.find((a: { avatar_name?: string }) => 
        a.avatar_name?.toLowerCase().includes('kelly')
      ) || avatars[0]
      
      if (kelly) {
        setAvatarId(kelly.avatar_id)
      }
      
      updateResult(1, { status: 'success', data: { 
        count: avatars.length, 
        selectedAvatar: kelly?.avatar_name || 'None',
        avatarId: kelly?.avatar_id,
        allAvatars: avatars.slice(0, 5).map((a: { avatar_name?: string; avatar_id?: string }) => ({
          name: a.avatar_name,
          id: a.avatar_id
        }))
      }})
    } catch (e) {
      updateResult(1, { status: 'error', error: e instanceof Error ? e.message : 'Failed' })
      setIsRunning(false)
      return
    }

    // Step 3: Generate Test Video (only if we have an avatar)
    if (avatarId) {
      updateResult(2, { status: 'running' })
      try {
        const generateRes = await fetch('/api/heygen/simple?action=generate&avatar_id=' + avatarId)
        const generateData = await generateRes.json()
        if (generateData.error) throw new Error(generateData.error)
        setVideoId(generateData.video_id)
        updateResult(2, { status: 'success', data: generateData })
      } catch (e) {
        updateResult(2, { status: 'error', error: e instanceof Error ? e.message : 'Failed' })
      }
    } else {
      updateResult(2, { status: 'error', error: 'No avatar ID found - please provide Kelly avatar ID' })
    }

    setIsRunning(false)
  }

  const checkVideoStatus = async () => {
    if (!videoId) return
    try {
      const res = await fetch(`/api/heygen/simple?action=status&video_id=${videoId}`)
      const data = await res.json()
      updateResult(2, { status: 'success', data })
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    // Auto-run on mount
    runTests()
  }, [])

  // Poll for video status if we have a video ID
  useEffect(() => {
    if (!videoId) return
    const interval = setInterval(checkVideoStatus, 5000)
    return () => clearInterval(interval)
  }, [videoId])

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <h1 className="text-3xl font-bold mb-8">HeyGen API Test</h1>
      
      <div className="space-y-6">
        {results.map((result, i) => (
          <div key={i} className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center gap-4 mb-4">
              <div className={`w-4 h-4 rounded-full ${
                result.status === 'pending' ? 'bg-gray-500' :
                result.status === 'running' ? 'bg-yellow-500 animate-pulse' :
                result.status === 'success' ? 'bg-green-500' :
                'bg-red-500'
              }`} />
              <h2 className="text-xl font-semibold">{result.step}</h2>
              <span className="text-sm text-gray-400 uppercase">{result.status}</span>
            </div>
            
            {result.error && (
              <pre className="bg-red-900/30 text-red-300 p-4 rounded overflow-auto text-sm">
                {result.error}
              </pre>
            )}
            
            {result.data && (
              <pre className="bg-gray-700 p-4 rounded overflow-auto text-sm max-h-96">
                {JSON.stringify(result.data, null, 2)}
              </pre>
            )}
          </div>
        ))}
      </div>

      <div className="mt-8 flex gap-4">
        <button 
          onClick={runTests}
          disabled={isRunning}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg font-semibold"
        >
          {isRunning ? 'Running...' : 'Run Tests Again'}
        </button>
        
        {videoId && (
          <button 
            onClick={checkVideoStatus}
            className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-semibold"
          >
            Check Video Status
          </button>
        )}
      </div>

      {avatarId && (
        <div className="mt-8 p-4 bg-gray-800 rounded-lg">
          <p className="text-sm text-gray-400">Selected Avatar ID:</p>
          <code className="text-green-400">{avatarId}</code>
        </div>
      )}

      {videoId && (
        <div className="mt-4 p-4 bg-gray-800 rounded-lg">
          <p className="text-sm text-gray-400">Video ID (polling for status):</p>
          <code className="text-yellow-400">{videoId}</code>
        </div>
      )}
    </div>
  )
}
