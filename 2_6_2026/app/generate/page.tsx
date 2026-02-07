"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

interface LanguageStats {
  language: string
  total: number
  withAudio: number
  pending: number
  percentComplete: number
}

interface VideoStats {
  total: number
  byStatus: Record<string, number>
  percentComplete: number
}

const LANGUAGES = [
  { code: "es", name: "Spanish", speakers: "559M" },
  { code: "hi", name: "Hindi", speakers: "602M" },
  { code: "zh", name: "Chinese", speakers: "1.1B" },
  { code: "pt", name: "Portuguese", speakers: "264M" },
  { code: "ar", name: "Arabic", speakers: "274M" },
  { code: "fr", name: "French", speakers: "280M" },
  { code: "ru", name: "Russian", speakers: "255M" },
  { code: "ja", name: "Japanese", speakers: "123M" },
  { code: "de", name: "German", speakers: "134M" },
  { code: "ko", name: "Korean", speakers: "81M" },
  { code: "it", name: "Italian", speakers: "68M" },
]

export default function GeneratePage() {
  const [languageStats, setLanguageStats] = useState<Record<string, LanguageStats>>({})
  const [videoStats, setVideoStats] = useState<VideoStats | null>(null)
  const [generating, setGenerating] = useState<string | null>(null)
  const [log, setLog] = useState<string[]>([])

  useEffect(() => {
    // Fetch stats for all languages
    LANGUAGES.forEach(async (lang) => {
      try {
        const res = await fetch(`/api/audio/generate-language?language=${lang.code}`)
        const data = await res.json()
        setLanguageStats(prev => ({ ...prev, [lang.code]: data }))
      } catch (err) {
        console.error(`Failed to fetch ${lang.code} stats:`, err)
      }
    })

    // Fetch video queue stats
    fetch("/api/video/process-queue")
      .then(res => res.json())
      .then(data => setVideoStats(data))
      .catch(err => console.error("Failed to fetch video stats:", err))
  }, [])

  const generateAudio = async (language: string) => {
    setGenerating(language)
    setLog(prev => [...prev, `Starting ${language} audio generation...`])

    try {
      const res = await fetch("/api/audio/generate-language", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language, dayStart: 1, dayEnd: 365, batchSize: 50 }),
      })
      const data = await res.json()
      setLog(prev => [...prev, `Generated ${data.generated} audio files for ${language}`])
      
      // Refresh stats
      const statsRes = await fetch(`/api/audio/generate-language?language=${language}`)
      const stats = await statsRes.json()
      setLanguageStats(prev => ({ ...prev, [language]: stats }))
    } catch (err) {
      setLog(prev => [...prev, `Error generating ${language}: ${err}`])
    } finally {
      setGenerating(null)
    }
  }

  const processVideos = async () => {
    setGenerating("videos")
    setLog(prev => [...prev, "Processing video queue..."])

    try {
      const res = await fetch("/api/video/process-queue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batchSize: 10 }),
      })
      const data = await res.json()
      setLog(prev => [...prev, `Processed ${data.processed} videos, ${data.errors} errors`])
      
      // Refresh stats
      const statsRes = await fetch("/api/video/process-queue")
      const stats = await statsRes.json()
      setVideoStats(stats)
    } catch (err) {
      setLog(prev => [...prev, `Error processing videos: ${err}`])
    } finally {
      setGenerating(null)
    }
  }

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">Content Generation Dashboard</h1>
            <p className="text-zinc-400 mt-1">Generate audio and video for all languages</p>
          </div>
          <Link href="/">
            <Button variant="outline">Back to Home</Button>
          </Link>
        </div>

        {/* English Status - Already Complete */}
        <Card className="bg-emerald-950/30 border-emerald-500/30 mb-8">
          <CardHeader>
            <CardTitle className="text-emerald-400">English Audio - COMPLETE</CardTitle>
            <CardDescription>9,135 audio files ready</CardDescription>
          </CardHeader>
          <CardContent>
            <Progress value={100} className="h-3 bg-emerald-950" />
          </CardContent>
        </Card>

        {/* Language Audio Generation */}
        <h2 className="text-xl font-semibold mb-4">Multi-Language Audio Generation</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {LANGUAGES.map((lang) => {
            const stats = languageStats[lang.code]
            const percent = stats?.percentComplete || 0
            const isGenerating = generating === lang.code

            return (
              <Card key={lang.code} className="bg-zinc-900 border-zinc-800">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{lang.name}</CardTitle>
                    <span className="text-xs text-zinc-500">{lang.speakers} speakers</span>
                  </div>
                  <CardDescription>
                    {stats ? `${stats.withAudio} / ${stats.total} files` : "Loading..."}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Progress value={percent} className="h-2 mb-3" />
                  <Button
                    size="sm"
                    onClick={() => generateAudio(lang.code)}
                    disabled={isGenerating || percent === 100}
                    className="w-full"
                  >
                    {isGenerating ? "Generating..." : percent === 100 ? "Complete" : "Generate Audio"}
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>

        {/* Video Generation */}
        <h2 className="text-xl font-semibold mb-4">Video Generation (HeyGen)</h2>
        <Card className="bg-zinc-900 border-zinc-800 mb-8">
          <CardHeader>
            <CardTitle>Video Queue Status</CardTitle>
            <CardDescription>
              {videoStats ? `${videoStats.total} total videos` : "Loading..."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {videoStats && (
              <>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  {Object.entries(videoStats.byStatus).map(([status, count]) => (
                    <div key={status} className="text-center p-3 bg-zinc-800 rounded-lg">
                      <div className="text-2xl font-bold">{count}</div>
                      <div className="text-xs text-zinc-400 capitalize">{status}</div>
                    </div>
                  ))}
                </div>
                <Progress value={videoStats.percentComplete} className="h-2 mb-3" />
                <Button
                  onClick={processVideos}
                  disabled={generating === "videos"}
                  className="w-full"
                >
                  {generating === "videos" ? "Processing..." : "Process Next Batch (10 videos)"}
                </Button>
              </>
            )}
          </CardContent>
        </Card>

        {/* Generation Log */}
        <h2 className="text-xl font-semibold mb-4">Generation Log</h2>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="p-4">
            <div className="font-mono text-sm text-zinc-400 h-48 overflow-y-auto">
              {log.length === 0 ? (
                <p>No generation activity yet. Click a button above to start.</p>
              ) : (
                log.map((entry, i) => (
                  <p key={i} className="mb-1">{entry}</p>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
