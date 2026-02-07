'use client'

/**
 * /terms redirects to main app with terms overlay open
 * 2-Layer Architecture: Kelly + Everything Else (overlays)
 */

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function TermsPage() {
  const router = useRouter()
  
  useEffect(() => {
    router.replace('/?view=terms')
  }, [router])
  
  return (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="animate-spin w-8 h-8 border-2 border-white/20 border-t-white rounded-full" />
    </div>
  )
}
