"use client"

import { cn } from "@/lib/utils"
import { ArrowLeft, Heart, Share2, MapPin, Info, ChevronUp, ImageIcon, Download, FileText, ExternalLink } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { AnimatePresence, motion } from "framer-motion"

const SCRIPT_SEGMENTS = [
  { id: 1, speaker: "kelly", text: "Let me show you something." },
  { id: 2, speaker: "vo", text: "In 1968, William Pereira designed this facility for 7,000 engineers." },
  { id: 3, speaker: "kelly", text: "Facility for 7,000 engineers at Rocketdyne." },
  { id: 4, speaker: "vo", text: "92 acres. 1 million square feet." },
  { id: 5, speaker: "kelly", text: "Never moved in." },
  { id: 6, speaker: "vo", text: "50 years of paperwork, zoning, permits." },
  { id: 7, speaker: "kelly", text: "Not what it was made for." },
  { id: 8, speaker: "vo", text: "Three auctions. No deal." },
  { id: 9, speaker: "kelly", text: "Different idea." },
  { id: 10, speaker: "vo", text: "18-inch raised floors, fiber optic conduits." },
  { id: 11, speaker: "kelly", text: "The infrastructure I need." },
  { id: 12, speaker: "vo", text: "8 billion learners worldwide." },
  { id: 13, speaker: "vo", text: "Pereira designed this for the future." },
  { id: 14, speaker: "kelly", text: "Finish what he started." },
  { id: 15, speaker: "kelly", text: "Will you help me?" },
  { id: 16, speaker: "vo", text: "Kelly teaches the world." },
]

const VISION_GALLERY = [
  { id: "original", image: "/ziggurat/heroes/original-day.jpg", name: "Original", label: "Concept" },
  { id: "living-building", image: "/ziggurat/heroes/living-building.jpg", name: "Living Building", label: "Vision 1" },
  { id: "kelly-blue-campus", image: "/ziggurat/heroes/kelly-blue-day.jpg", name: "Kelly Blue Campus", label: "Vision 2" },
  { id: "ocean-beacon-night", image: "/ziggurat/heroes/ocean-beacon-night.jpg", name: "Ocean Beacon Night", label: "Vision 3" },
  { id: "la2028-olympics", image: "/ziggurat/heroes/la2028-olympics.jpg", name: "LA 2028 Olympics", label: "Vision 4" },
]

const LESSON = {
  title: "Ziggurat Vision",
  subtitle: "Explore the future of Rocketdyne",
  location: "Orange County, CA",
  specs: [
    { label: "Acres", value: "92" },
    { label: "Square Feet", value: "1,000,000" },
    { label: "Engineers", value: "7,000" },
    { label: "Auctions", value: "3" },
  ],
  phases: [
    { name: "phase1", label: "Concept" },
    { name: "phase2", label: "Vision 1" },
    { name: "phase3", label: "Vision 2" },
    { name: "phase4", label: "Vision 3" },
    { name: "phase5", label: "Vision 4" },
  ],
}

// Production completed 2026-02-01 - all videos rendered and deployed
const FINAL_VIDEOS = {
  full: "https://pub-ae8248f6a4f44c61a5de0d2f19b8dcd1.r2.dev/ziggurat/THE_ZIGGURAT_FINAL.mp4",
  twitter: "https://pub-ae8248f6a4f44c61a5de0d2f19b8dcd1.r2.dev/ziggurat/twitter_60sec.mp4",
  square: "https://pub-ae8248f6a4f44c61a5de0d2f19b8dcd1.r2.dev/ziggurat/THE_ZIGGURAT_SQUARE.mp4",
  vertical: "https://pub-ae8248f6a4f44c61a5de0d2f19b8dcd1.r2.dev/ziggurat/THE_ZIGGURAT_VERTICAL.mp4",
}

const SOCIAL_COPY = `The Ziggurat: 92 acres, 1M sq ft—designed as the world's largest electronic marketplace.

Three auctions. No deal closed. The last bidder wanted to demolish it.

We see something else.

Kelly presents: thedailylesson.com/ziggurat`

export default function ZigguratPage() {
  const [kellyImageUrl, setKellyImageUrl] = useState(
    "https://pub-3e97ff7b44214b609b373dd49fbaa1a2.r2.dev/kelly-headshot.jpg"
  )
  const [status, setStatus] = useState<"ready" | "starting" | "running" | "complete" | "error">(
    "complete" // Production finished!
  )
  const [workflowId, setWorkflowId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [liked, setLiked] = useState(false)
  const [selectedVision, setSelectedVision] = useState(VISION_GALLERY[0])
  const [showSpecs, setShowSpecs] = useState(false)
  const [currentPhase, setCurrentPhase] = useState(0)
  const phase = LESSON.phases[currentPhase]

  const startProduction = async () => {
    setStatus("starting")
    setError(null)

    try {
      const response = await fetch("/api/ziggurat/produce", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kellyImageUrl }),
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`)
      }

      const data = await response.json()
      setWorkflowId(data.workflowId)
      setStatus("running")
    } catch (err) {
      setError(String(err))
      setStatus("error")
    }
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <div className="border-b border-zinc-800 px-6 py-8">
        <h1 className="text-4xl font-bold text-balance">Ziggurat Video Production</h1>
        <p className="mt-2 text-zinc-400">Laguna Ridge Investor Pitch Generator</p>
      </div>

      <div className="mx-auto max-w-4xl px-6 py-12">
        {/* Status Card */}
        <Card className="mb-8 border-zinc-800 bg-zinc-900 p-6">
          <div className="mb-4">
            <h2 className="text-lg font-semibold">Production Status</h2>
            <div className="mt-2 flex items-center gap-2">
              <div
                className={`h-3 w-3 rounded-full ${
                  status === "ready"
                    ? "bg-zinc-500"
                    : status === "starting" || status === "running"
                      ? "bg-yellow-500 animate-pulse"
                      : status === "complete"
                        ? "bg-green-500"
                        : "bg-red-500"
                }`}
              />
              <span className="capitalize">{status}</span>
            </div>
          </div>

          {workflowId && (
            <div className="mt-4 rounded bg-black p-3 font-mono text-sm text-zinc-400">
              Workflow ID: {workflowId}
              <br />
              <span className="text-zinc-500">
                Monitor progress: <code className="text-white">npx workflow web</code>
              </span>
            </div>
          )}

          {error && <div className="mt-4 text-sm text-red-400">{error}</div>}
        </Card>

        {/* PRODUCTION COMPLETE - Final Video & Downloads */}
        {status === "complete" && (
          <Card className="mb-8 border-green-800 bg-green-950/30 p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="h-4 w-4 rounded-full bg-green-500" />
              <h2 className="text-xl font-bold text-green-400">Production Complete!</h2>
            </div>
            
            {/* Video Player */}
            <div className="mb-6 rounded-xl overflow-hidden aspect-video bg-black">
              <video
                src={FINAL_VIDEOS.full}
                controls
                className="w-full h-full"
                poster="/ziggurat/heroes/original-day.jpg"
              />
            </div>

            {/* Download Links */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              <a href={FINAL_VIDEOS.full} target="_blank" rel="noopener noreferrer"
                className="flex flex-col items-center p-3 rounded-lg bg-black/50 border border-zinc-800 hover:border-zinc-600 transition-colors">
                <Download className="w-5 h-5 mb-1 text-zinc-400" />
                <span className="text-xs text-zinc-400">Full (LinkedIn)</span>
              </a>
              <a href={FINAL_VIDEOS.twitter} target="_blank" rel="noopener noreferrer"
                className="flex flex-col items-center p-3 rounded-lg bg-black/50 border border-zinc-800 hover:border-zinc-600 transition-colors">
                <Download className="w-5 h-5 mb-1 text-zinc-400" />
                <span className="text-xs text-zinc-400">Twitter/X (60s)</span>
              </a>
              <a href={FINAL_VIDEOS.square} target="_blank" rel="noopener noreferrer"
                className="flex flex-col items-center p-3 rounded-lg bg-black/50 border border-zinc-800 hover:border-zinc-600 transition-colors">
                <Download className="w-5 h-5 mb-1 text-zinc-400" />
                <span className="text-xs text-zinc-400">Instagram Feed</span>
              </a>
              <a href={FINAL_VIDEOS.vertical} target="_blank" rel="noopener noreferrer"
                className="flex flex-col items-center p-3 rounded-lg bg-black/50 border border-zinc-800 hover:border-zinc-600 transition-colors">
                <Download className="w-5 h-5 mb-1 text-zinc-400" />
                <span className="text-xs text-zinc-400">Reels/TikTok/Shorts</span>
              </a>
            </div>

            {/* Social Copy */}
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-zinc-400 mb-2">Social Copy</h3>
              <div className="bg-black rounded-lg p-4 font-mono text-sm text-zinc-300 whitespace-pre-wrap">
                {SOCIAL_COPY}
              </div>
            </div>
            <Button 
              onClick={() => navigator.clipboard.writeText(SOCIAL_COPY)}
              variant="outline"
              className="w-full bg-transparent border-zinc-700 hover:bg-zinc-800"
            >
              Copy to Clipboard
            </Button>
          </Card>
        )}

        {/* Input Section */}
        {status === "ready" && (
          <Card className="mb-8 border-zinc-800 bg-zinc-900 p-6">
            <h2 className="mb-4 text-lg font-semibold">Configure Production</h2>

            <div className="mb-6">
              <label className="block text-sm text-zinc-400">Kelly Headshot URL</label>
              <Input
                value={kellyImageUrl}
                onChange={(e) => setKellyImageUrl(e.target.value)}
                placeholder="https://..."
                className="mt-2 border-zinc-700 bg-black text-white"
              />
              <p className="mt-2 text-xs text-zinc-500">
                Source image for lip-sync generation (fal.ai sadtalker)
              </p>
            </div>

            <Button
              onClick={startProduction}
              className="w-full bg-white text-black hover:bg-zinc-200"
            >
              Start Production
            </Button>
          </Card>
        )}

        {/* Script Grid */}
        <div className="mb-12">
          <h2 className="mb-6 text-2xl font-bold">16-Segment Script</h2>
          <div className="grid gap-4 md:grid-cols-2">
            {SCRIPT_SEGMENTS.map((seg) => (
              <div
                key={seg.id}
                className={`rounded border p-4 ${
                  seg.speaker === "kelly"
                    ? "border-zinc-700 bg-zinc-950"
                    : "border-zinc-800 bg-black"
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="text-sm font-mono text-zinc-500">
                      {String(seg.id).padStart(2, "0")}_{seg.speaker}
                    </div>
                    <p className="mt-2 text-sm leading-relaxed">{seg.text}</p>
                  </div>
                  {seg.speaker === "kelly" && (
                    <div className="ml-3 flex-shrink-0 rounded-full bg-zinc-700 px-2 py-1 text-xs font-mono text-zinc-300">
                      K
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Instructions */}
        <Card className="border-zinc-800 bg-zinc-900 p-6">
          <h2 className="mb-4 font-semibold">How It Works</h2>
          <ol className="space-y-3 text-sm text-zinc-300">
            <li>
              <strong>1. Audio Generation:</strong> ElevenLabs generates 16 audio segments (88 seconds
              total)
            </li>
            <li>
              <strong>2. Lip-Sync Video:</strong> fal.ai sadtalker creates 8 on-camera Kelly videos
              (segments 1,3,5,7,9,11,14,15)
            </li>
            <li>
              <strong>3. Manifest:</strong> All URLs saved to manifest.json in Vercel Blob
            </li>
            <li>
              <strong>4. Monitor:</strong> Watch real-time progress with{" "}
              <code className="rounded bg-black px-2 py-1">npx workflow web</code>
            </li>
          </ol>
        </Card>
      </div>

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-[#0A0A0C] via-[#0A0A0C]/90 to-transparent">
        <div className="max-w-4xl mx-auto flex gap-3">
          <button
            onClick={() => setCurrentPhase(Math.max(0, currentPhase - 1))}
            disabled={currentPhase === 0}
            className="flex-1 py-4 rounded-xl bg-black/40 backdrop-blur-md border border-white/20 text-white/60 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-black/50 transition-colors"
          >
            Previous
          </button>
          <button
            onClick={() => setCurrentPhase(Math.min(LESSON.phases.length - 1, currentPhase + 1))}
            disabled={currentPhase === LESSON.phases.length - 1}
            className="flex-1 py-4 rounded-xl bg-white text-black font-semibold disabled:opacity-30 disabled:cursor-not-allowed hover:bg-white/90 transition-colors"
          >
            {currentPhase === LESSON.phases.length - 1 ? "Complete" : "Next"}
          </button>
        </div>
      </nav>
    </div>
  )
}
