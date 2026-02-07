'use client'

/**
 * KellyOS Admin Overlay
 * Founder-only admin controls within the 2-layer Kelly architecture
 */

import { useState, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel, ListItem, SectionHeader } from './index'
import { 
  Flame, RefreshCw, Users, Video, BarChart3, Settings, Shield,
  Upload, Database, Clock, CheckCircle, XCircle, AlertTriangle,
} from 'lucide-react'

interface AdminOverlayProps {
  isOpen: boolean
  onClose: () => void
}

interface SystemStatus {
  heygenCredits: number
  heygenQueued: number
  heygenCompleted: number
  totalLearners: number
  totalLessons: number
  totalPerspectives: number
}

export function AdminOverlay({ isOpen, onClose }: AdminOverlayProps) {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activePanel, setActivePanel] = useState<'main' | 'fire' | 'sync' | 'learners'>('main')
  const [fireResult, setFireResult] = useState<{ submitted: number; failed: number } | null>(null)
  const [syncResult, setSyncResult] = useState<{ synced: number; failed: number } | null>(null)

  useEffect(() => {
    if (isOpen) fetchStatus()
  }, [isOpen])

  const fetchStatus = async () => {
    setIsLoading(true)
    try {
      const [fireRes, syncRes] = await Promise.all([
        fetch('/api/heygen/fire-all').then(r => r.json()).catch(() => ({})),
        fetch('/api/heygen/sync-completed?limit=100').then(r => r.json()).catch(() => ({})),
      ])
      setStatus({
        heygenCredits: fireRes.credits || 999999,
        heygenQueued: fireRes.queued || 0,
        heygenCompleted: syncRes.completed || 0,
        totalLearners: 0,
        totalLessons: 365,
        totalPerspectives: 39420,
      })
    } catch (e) {
      console.error('Failed to fetch status:', e)
    } finally {
      setIsLoading(false)
    }
  }

  const handleFire = async (count: number) => {
    setIsLoading(true)
    try {
      const res = await fetch('/api/heygen/fire-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batchSize: count }),
      })
      const data = await res.json()
      setFireResult({ submitted: data.submitted || 0, failed: data.failed || 0 })
      fetchStatus()
    } catch (e) {
      console.error('Fire failed:', e)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSync = async (count: number) => {
    setIsLoading(true)
    try {
      const res = await fetch('/api/heygen/sync-completed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: count }),
      })
      const data = await res.json()
      setSyncResult({ synced: data.synced || 0, failed: data.failed || 0 })
      fetchStatus()
    } catch (e) {
      console.error('Sync failed:', e)
    } finally {
      setIsLoading(false)
    }
  }

  const renderMainPanel = () => (
    <div className="space-y-4">
      {/* Status Cards */}
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-white/5 rounded-xl p-3">
          <Video className="w-4 h-4 text-green-400 mb-1" />
          <p className="text-lg font-bold">{status?.heygenCompleted || 0}</p>
          <p className="text-white/40 text-xs">Videos Done</p>
        </div>
        <div className="bg-white/5 rounded-xl p-3">
          <Clock className="w-4 h-4 text-amber-400 mb-1" />
          <p className="text-lg font-bold">{status?.heygenQueued || 0}</p>
          <p className="text-white/40 text-xs">Queued</p>
        </div>
        <div className="bg-white/5 rounded-xl p-3">
          <Users className="w-4 h-4 text-blue-400 mb-1" />
          <p className="text-lg font-bold">{status?.totalLearners || 0}</p>
          <p className="text-white/40 text-xs">Learners</p>
        </div>
        <div className="bg-white/5 rounded-xl p-3">
          <BarChart3 className="w-4 h-4 text-purple-400 mb-1" />
          <p className="text-lg font-bold">{status?.totalPerspectives?.toLocaleString() || 0}</p>
          <p className="text-white/40 text-xs">Perspectives</p>
        </div>
      </div>

      <SectionHeader title="Quick Actions" />

      <ListItem
        icon={<Flame className="w-4 h-4" />}
        title="Fire HeyGen"
        subtitle="Submit videos for generation"
        onClick={() => setActivePanel('fire')}
      />
      <ListItem
        icon={<RefreshCw className="w-4 h-4" />}
        title="Sync Videos"
        subtitle="Download completed videos"
        onClick={() => setActivePanel('sync')}
      />
      <ListItem
        icon={<Users className="w-4 h-4" />}
        title="Learners"
        subtitle="View learner data"
        onClick={() => setActivePanel('learners')}
      />
    </div>
  )

  const renderFirePanel = () => (
    <div className="space-y-4">
      <div className="bg-gradient-to-br from-red-500/20 to-orange-500/20 rounded-2xl p-4 border border-red-500/30">
        <div className="flex items-center gap-3 mb-3">
          <Flame className="w-8 h-8 text-red-400" />
          <div>
            <h3 className="font-bold text-lg">Fire Control</h3>
            <p className="text-white/60 text-sm">{status?.heygenCredits?.toLocaleString()} credits available</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {[10, 25, 50, 100].map((count) => (
            <button
              key={count}
              onClick={() => handleFire(count)}
              disabled={isLoading}
              className="py-3 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 font-semibold disabled:opacity-50"
            >
              Fire {count}
            </button>
          ))}
        </div>
      </div>

      {fireResult && (
        <div className="bg-white/5 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            {fireResult.failed === 0 ? (
              <CheckCircle className="w-5 h-5 text-green-400" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            )}
            <span className="font-semibold">Result</span>
          </div>
          <p className="text-sm text-white/70">
            Submitted: {fireResult.submitted} | Failed: {fireResult.failed}
          </p>
        </div>
      )}

      <button
        onClick={() => setActivePanel('main')}
        className="w-full py-2 text-white/50 hover:text-white text-sm"
      >
        Back to Admin
      </button>
    </div>
  )

  const renderSyncPanel = () => (
    <div className="space-y-4">
      <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-2xl p-4 border border-blue-500/30">
        <div className="flex items-center gap-3 mb-3">
          <RefreshCw className="w-8 h-8 text-blue-400" />
          <div>
            <h3 className="font-bold text-lg">Sync Videos</h3>
            <p className="text-white/60 text-sm">Download from HeyGen to Vercel Blob</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {[5, 10, 25, 50].map((count) => (
            <button
              key={count}
              onClick={() => handleSync(count)}
              disabled={isLoading}
              className="py-3 rounded-xl bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 font-semibold disabled:opacity-50"
            >
              Sync {count}
            </button>
          ))}
        </div>
      </div>

      {syncResult && (
        <div className="bg-white/5 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            {syncResult.failed === 0 ? (
              <CheckCircle className="w-5 h-5 text-green-400" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            )}
            <span className="font-semibold">Result</span>
          </div>
          <p className="text-sm text-white/70">
            Synced: {syncResult.synced} | Failed: {syncResult.failed}
          </p>
        </div>
      )}

      <button
        onClick={() => setActivePanel('main')}
        className="w-full py-2 text-white/50 hover:text-white text-sm"
      >
        Back to Admin
      </button>
    </div>
  )

  const renderLearnersPanel = () => (
    <div className="space-y-4">
      <div className="text-center py-8">
        <Users className="w-12 h-12 text-white/20 mx-auto mb-3" />
        <p className="text-white/50">No learners yet</p>
        <p className="text-white/30 text-xs mt-1">Waiting for first signups</p>
      </div>
      <button
        onClick={() => setActivePanel('main')}
        className="w-full py-2 text-white/50 hover:text-white text-sm"
      >
        Back to Admin
      </button>
    </div>
  )

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose}>
      <OverlayPanel
        title={activePanel === 'main' ? 'Admin Panel' : activePanel === 'fire' ? 'Fire HeyGen' : activePanel === 'sync' ? 'Sync Videos' : 'Learners'}
        subtitle="Founder Controls"
        onClose={onClose}
        position="right"
      >
        <div className="p-4">
          {isLoading && activePanel === 'main' ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin w-6 h-6 border-2 border-white/20 border-t-white rounded-full" />
            </div>
          ) : (
            <>
              {activePanel === 'main' && renderMainPanel()}
              {activePanel === 'fire' && renderFirePanel()}
              {activePanel === 'sync' && renderSyncPanel()}
              {activePanel === 'learners' && renderLearnersPanel()}
            </>
          )}
        </div>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
