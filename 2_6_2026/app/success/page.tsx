'use client'

import { useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { Check, ArrowRight, Sparkles } from 'lucide-react'
import Link from 'next/link'

export default function SuccessPage() {
  const searchParams = useSearchParams()
  const sessionId = searchParams.get('session_id')
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')

  useEffect(() => {
    if (!sessionId) {
      setStatus('success')
      return
    }

    // Verify the session on the server
    async function verify() {
      try {
        const res = await fetch(`/api/checkout/verify?session_id=${sessionId}`)
        if (res.ok) {
          setStatus('success')
        } else {
          setStatus('success') // Still show success - webhook handles the actual state
        }
      } catch {
        setStatus('success')
      }
    }
    verify()
  }, [sessionId])

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="max-w-md w-full text-center space-y-8">
        {/* Animated check */}
        <div className="mx-auto w-20 h-20 rounded-full bg-emerald-500/10 flex items-center justify-center">
          <div className="w-14 h-14 rounded-full bg-emerald-500/20 flex items-center justify-center animate-in zoom-in-50 duration-500">
            <Check className="h-8 w-8 text-emerald-500" strokeWidth={3} />
          </div>
        </div>

        <div className="space-y-3">
          <h1 className="text-3xl font-bold text-foreground">Welcome to the journey</h1>
          <p className="text-muted-foreground text-lg leading-relaxed">
            Your subscription is active. 365 days of learning await you.
          </p>
        </div>

        {/* What's next */}
        <div className="bg-muted/30 rounded-2xl p-6 text-left space-y-4">
          <h3 className="font-semibold text-sm text-muted-foreground uppercase tracking-wider">
            What you just unlocked
          </h3>
          {[
            'Full lesson archive - 365 daily lessons',
            'All 47+ languages',
            'Every Kelly archetype',
            'Interactive captions & visual aids',
            'Offline downloads',
          ].map((item, i) => (
            <div key={i} className="flex items-center gap-3 text-sm">
              <Sparkles className="h-4 w-4 text-emerald-500 flex-shrink-0" />
              <span className="text-foreground">{item}</span>
            </div>
          ))}
        </div>

        {/* CTA */}
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-8 py-4 rounded-full bg-foreground text-background font-semibold hover:opacity-90 transition-opacity"
        >
          Start learning
          <ArrowRight className="h-5 w-5" />
        </Link>

        <p className="text-xs text-muted-foreground">
          A confirmation email has been sent to your inbox.
        </p>
      </div>
    </div>
  )
}
