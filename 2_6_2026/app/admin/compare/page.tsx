'use client'

import { VideoEngineComparison } from '@/components/video-engine-comparison'
import Link from 'next/link'
import { ArrowLeft, Sparkles } from 'lucide-react'

export default function ComparePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/40 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/admin"
              className="flex items-center gap-2 text-white/60 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Admin
            </Link>
            <div className="w-px h-6 bg-white/10" />
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-amber-400" />
              <h1 className="text-xl font-bold text-white">Video Engine Comparison</h1>
            </div>
          </div>
          
          <div className="text-sm text-white/40">
            Compare HeyGen, Fal.ai, Sync.so, and MuseTalk
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <VideoEngineComparison day={1} />
        
        {/* Instructions panel */}
        <div className="mt-8 p-6 rounded-2xl bg-white/5 border border-white/10">
          <h3 className="text-lg font-semibold text-white mb-4">How to use this tool</h3>
          <div className="grid md:grid-cols-3 gap-6 text-sm text-white/70">
            <div>
              <div className="w-8 h-8 rounded-lg bg-sky-500/20 text-sky-400 flex items-center justify-center font-bold mb-2">1</div>
              <p>Select a day and phase to compare videos from all 4 generation engines.</p>
            </div>
            <div>
              <div className="w-8 h-8 rounded-lg bg-orange-500/20 text-orange-400 flex items-center justify-center font-bold mb-2">2</div>
              <p>Play videos side-by-side to evaluate lip sync quality, naturalness, and expression.</p>
            </div>
            <div>
              <div className="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold mb-2">3</div>
              <p>Rate each engine as "Best" or "Worst" to help determine the winner for batch generation.</p>
            </div>
          </div>
        </div>

        {/* Engine details */}
        <div className="mt-8 grid md:grid-cols-4 gap-4">
          {[
            { name: 'HeyGen', color: '#0ea5e9', desc: 'Premium quality, natural expressions, best lip sync' },
            { name: 'Fal.ai', color: '#f97316', desc: 'Fast generation, good quality, cost-effective' },
            { name: 'Sync.so', color: '#22c55e', desc: 'Excellent lip sync accuracy, natural head movement' },
            { name: 'MuseTalk', color: '#ec4899', desc: 'Open source, runs locally, full control' },
          ].map(engine => (
            <div
              key={engine.name}
              className="p-4 rounded-xl bg-white/5 border border-white/10"
            >
              <div
                className="text-sm font-bold mb-2"
                style={{ color: engine.color }}
              >
                {engine.name}
              </div>
              <p className="text-xs text-white/50">{engine.desc}</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  )
}
