'use client'

// Offline fallback page - shows when no network and no cache
import { WifiOff, RefreshCw } from 'lucide-react'

export default function OfflinePage() {
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col items-center justify-center p-6">
      <div className="max-w-sm text-center space-y-6">
        {/* Icon */}
        <div className="w-20 h-20 mx-auto rounded-full bg-white/5 flex items-center justify-center">
          <WifiOff className="w-10 h-10 text-white/40" />
        </div>
        
        {/* Message */}
        <div className="space-y-2">
          <h1 className="text-2xl font-bold">You're Offline</h1>
          <p className="text-white/60 text-sm">
            Kelly needs an internet connection to load new lessons. 
            But don't worry - any lessons you've viewed before are saved!
          </p>
        </div>
        
        {/* Tip */}
        <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-left">
          <p className="text-xs text-white/40 uppercase tracking-wider mb-2">Pro tip</p>
          <p className="text-sm text-white/80">
            When you're online, Kelly automatically saves lessons for offline viewing. 
            The more you learn, the more you can access anywhere!
          </p>
        </div>
        
        {/* Retry button */}
        <button
          onClick={() => window.location.reload()}
          className="flex items-center gap-2 mx-auto px-6 py-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Try Again</span>
        </button>
        
        {/* Footer */}
        <p className="text-xs text-white/30">
          Curious Kelly - Learning never stops
        </p>
      </div>
    </div>
  )
}
