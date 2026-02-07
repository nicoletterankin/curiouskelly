'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

/**
 * Dashboard redirects to main app with profile overlay
 * All dashboard functionality is unified in the main page ProfileOverlay
 */
export default function DashboardRedirect() {
  const router = useRouter()
  
  useEffect(() => {
    // Redirect to main app - profile can be opened via the Profile button
    router.replace('/?view=profile')
  }, [router])
  
  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-white/20 border-t-white rounded-full animate-spin mx-auto mb-3" />
        <p className="text-white/50 text-sm">Redirecting...</p>
      </div>
    </div>
  )
}
