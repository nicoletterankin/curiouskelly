"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, Play, CheckCircle, XCircle, Clock } from "lucide-react"

type VideoJob = {
  id: string
  avatarKey: string
  status: "pending" | "processing" | "completed" | "failed"
  videoUrl?: string
  error?: string
}

export default function HeyGenTestPage() {
  const [credits, setCredits] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [testVideoId, setTestVideoId] = useState<string | null>(null)
  const [testVideoStatus, setTestVideoStatus] = useState<string | null>(null)
  const [testVideoUrl, setTestVideoUrl] = useState<string | null>(null)
  const [batchJobs, setBatchJobs] = useState<VideoJob[]>([])
  const [error, setError] = useState<string | null>(null)

  const checkCredits = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("/api/heygen/credits")
      
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`API error ${res.status}: ${text.substring(0, 100)}`)
      }
      
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setCredits(data.data?.remaining ?? data.credits ?? 0)
    } catch (e: unknown) {
      console.error("[v0] Credits error:", e)
      setError(e instanceof Error ? e.message : "Failed to check credits")
    } finally {
      setLoading(false)
    }
  }

  const generateTestVideo = async () => {
    setLoading(true)
    setError(null)
    setTestVideoId(null)
    setTestVideoStatus(null)
    setTestVideoUrl(null)
    
    try {
      const res = await fetch("/api/heygen/generate-video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          age: "kid",
          archetype: "scientist",
          script: "Hello! I am Kelly, your curious learning companion. Today we are going to explore something amazing together!",
          title: "Kelly Test Video"
        })
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`API error ${res.status}: ${text.substring(0, 100)}`)
      }
      
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      
      // Handle nested response structure
      const videoId = data.data?.video_id || data.videoId || data.video_id
      if (!videoId) throw new Error("No video ID in response")
      
      setTestVideoId(videoId)
      setTestVideoStatus("processing")
      
      // Start polling
      pollVideoStatus(videoId)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to generate video")
    } finally {
      setLoading(false)
    }
  }

  const pollVideoStatus = async (videoId: string) => {
    const maxAttempts = 60 // 5 minutes max
    let attempts = 0
    
    const poll = async () => {
      try {
        const res = await fetch(`/api/heygen/video-status/${videoId}`)
        const data = await res.json()
        
        // Handle nested response: data.data.status or data.status
        const status = data.data?.status || data.status
        const videoUrl = data.data?.video_url || data.videoUrl || data.video_url
        
        if (status === "completed") {
          setTestVideoStatus("completed")
          setTestVideoUrl(videoUrl)
          return
        }
        
        if (status === "failed") {
          setTestVideoStatus("failed")
          setError(data.data?.error || data.error || "Video generation failed")
          return
        }
        
        setTestVideoStatus(status || "processing")
        
        attempts++
        if (attempts < maxAttempts) {
          setTimeout(poll, 5000) // Poll every 5 seconds
        } else {
          setError("Timeout waiting for video")
        }
      } catch (e) {
        console.error("[v0] Poll error:", e)
        setError("Failed to check video status")
      }
    }
    
    poll()
  }

  const kickoffBaseVideos = async (limit?: number) => {
    setLoading(true)
    setError(null)
    setBatchJobs([])
    
    try {
      const url = limit 
        ? `/api/heygen/kickoff-base-videos?limit=${limit}`
        : "/api/heygen/kickoff-base-videos"
      
      const res = await fetch(url, { method: "POST" })
      const data = await res.json()
      
      if (data.error) throw new Error(data.error)
      
      // Transform to VideoJob format - handle both 'results' and 'jobs' response shapes
      const jobsData = data.jobs || data.results || []
      const jobs: VideoJob[] = jobsData.map((r: { avatarName?: string; avatarKey?: string; videoId?: string; error?: string; status?: string }) => ({
        id: r.videoId || "pending",
        avatarKey: r.avatarName || r.avatarKey || "unknown",
        status: r.error ? "failed" : r.status === "failed" ? "failed" : "processing",
        error: r.error
      }))
      
      setBatchJobs(jobs)
      
      // Start polling batch
      if (jobs.some(j => j.status === "processing")) {
        pollBatchStatus(jobs.filter(j => j.status === "processing").map(j => j.id))
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to kickoff base videos")
    } finally {
      setLoading(false)
    }
  }

  const pollBatchStatus = async (videoIds: string[]) => {
    const poll = async () => {
      try {
        const res = await fetch(`/api/heygen/check-base-videos?ids=${videoIds.join(",")}`)
        const data = await res.json()
        
        setBatchJobs(prev => prev.map(job => {
          const status = data.statuses?.find((s: { videoId: string }) => s.videoId === job.id)
          if (!status) return job
          
          return {
            ...job,
            status: status.status === "completed" ? "completed" 
                  : status.status === "failed" ? "failed" 
                  : "processing",
            videoUrl: status.videoUrl,
            error: status.error
          }
        }))
        
        // Continue polling if any still processing
        const stillProcessing = data.statuses?.filter((s: { status: string }) => 
          s.status !== "completed" && s.status !== "failed"
        )
        
        if (stillProcessing?.length > 0) {
          setTimeout(poll, 5000)
        }
      } catch {
        console.error("Batch poll error")
      }
    }
    
    setTimeout(poll, 5000)
  }

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">HeyGen Video Generation Test</h1>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <div className="grid gap-6">
        {/* Credits Check */}
        <Card>
          <CardHeader>
            <CardTitle>Step 1: Check API Credits</CardTitle>
            <CardDescription>Verify HeyGen API connection and available credits</CardDescription>
          </CardHeader>
          <CardContent className="flex items-center gap-4">
            <Button onClick={checkCredits} disabled={loading}>
              {loading ? <Loader2 className="animate-spin mr-2" /> : null}
              Check Credits
            </Button>
            {credits !== null && (
              <Badge variant="secondary" className="text-lg px-4 py-2">
                {credits} credits available
              </Badge>
            )}
          </CardContent>
        </Card>

        {/* Single Test Video */}
        <Card>
          <CardHeader>
            <CardTitle>Step 2: Generate Test Video</CardTitle>
            <CardDescription>Generate a single test video with Kelly (kid scientist)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4 mb-4">
              <Button onClick={generateTestVideo} disabled={loading || testVideoStatus === "processing"}>
                {loading ? <Loader2 className="animate-spin mr-2" /> : <Play className="mr-2" />}
                Generate Test Video
              </Button>
              {testVideoStatus && (
                <Badge variant={
                  testVideoStatus === "completed" ? "default" :
                  testVideoStatus === "failed" ? "destructive" : "secondary"
                }>
                  {testVideoStatus === "processing" && <Loader2 className="animate-spin mr-1 h-3 w-3" />}
                  {testVideoStatus}
                </Badge>
              )}
            </div>
            
            {testVideoId && (
              <p className="text-sm text-muted-foreground mb-2">Video ID: {testVideoId}</p>
            )}
            
            {testVideoUrl && (
              <div className="mt-4">
                <video 
                  src={testVideoUrl} 
                  controls 
                  className="w-full max-w-md rounded-lg shadow-lg"
                />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Batch Base Videos */}
        <Card>
          <CardHeader>
            <CardTitle>Step 3: Generate Base Videos</CardTitle>
            <CardDescription>Generate neutral talking base videos for all 36 avatar combinations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2 mb-4">
              <Button onClick={() => kickoffBaseVideos(1)} disabled={loading} variant="outline">
                Test 1 Video
              </Button>
              <Button onClick={() => kickoffBaseVideos(3)} disabled={loading} variant="outline">
                Test 3 Videos
              </Button>
              <Button onClick={() => kickoffBaseVideos()} disabled={loading}>
                Generate All 36
              </Button>
            </div>
            
            {batchJobs.length > 0 && (
              <div className="space-y-2 mt-4">
                <div className="flex gap-4 text-sm mb-2">
                  <span>Completed: {batchJobs.filter(j => j.status === "completed").length}</span>
                  <span>Processing: {batchJobs.filter(j => j.status === "processing").length}</span>
                  <span>Failed: {batchJobs.filter(j => j.status === "failed").length}</span>
                </div>
                
                <div className="grid gap-2 max-h-96 overflow-y-auto">
                  {batchJobs.map(job => (
                    <div key={job.id} className="flex items-center gap-2 p-2 bg-muted rounded text-sm">
                      {job.status === "completed" && <CheckCircle className="h-4 w-4 text-green-500" />}
                      {job.status === "failed" && <XCircle className="h-4 w-4 text-red-500" />}
                      {job.status === "processing" && <Clock className="h-4 w-4 text-yellow-500 animate-pulse" />}
                      <span className="font-mono">{job.avatarKey}</span>
                      {job.videoUrl && (
                        <a href={job.videoUrl} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline ml-auto">
                          View
                        </a>
                      )}
                      {job.error && <span className="text-red-500 ml-auto">{job.error}</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
