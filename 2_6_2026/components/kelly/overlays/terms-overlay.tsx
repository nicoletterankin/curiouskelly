'use client'

/**
 * KellyOS Terms Overlay
 * Terms of Service and Privacy Policy within the 2-layer Kelly architecture
 */

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel } from './index'
import { FileText, Shield, Scale } from 'lucide-react'

interface TermsOverlayProps {
  isOpen: boolean
  onClose: () => void
  initialTab?: 'terms' | 'privacy'
}

export function TermsOverlay({ isOpen, onClose, initialTab = 'terms' }: TermsOverlayProps) {
  const [activeTab, setActiveTab] = useState<'terms' | 'privacy'>(initialTab)

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose}>
      <OverlayPanel
        title={activeTab === 'terms' ? 'Terms of Service' : 'Privacy Policy'}
        subtitle="The Daily Lesson, Inc."
        onClose={onClose}
        position="right"
      >
        <div className="p-4 space-y-4">
          {/* Tab Selector */}
          <div className="flex gap-2 p-1 bg-white/5 rounded-xl">
            <button
              onClick={() => setActiveTab('terms')}
              className={cn(
                'flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2',
                activeTab === 'terms' ? 'bg-white/10 text-white' : 'text-white/50'
              )}
            >
              <Scale className="w-4 h-4" /> Terms
            </button>
            <button
              onClick={() => setActiveTab('privacy')}
              className={cn(
                'flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2',
                activeTab === 'privacy' ? 'bg-white/10 text-white' : 'text-white/50'
              )}
            >
              <Shield className="w-4 h-4" /> Privacy
            </button>
          </div>

          {activeTab === 'terms' ? (
            <div className="space-y-4 text-sm text-white/70">
              <p className="text-white/40 text-xs">Last updated: January 2026</p>
              
              <section>
                <h3 className="font-semibold text-white mb-2">1. Acceptance of Terms</h3>
                <p>By accessing or using Curious Kelly ("Service"), you agree to be bound by these Terms of Service. If you do not agree, please do not use the Service.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">2. Description of Service</h3>
                <p>Curious Kelly is an educational platform delivering daily micro-lessons through AI-generated video content. The Service is designed for learners of all ages.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">3. User Accounts</h3>
                <p>You may use the Service without an account in limited capacity. Creating an account enables progress tracking, personalization, and subscription features.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">4. Subscriptions</h3>
                <p>Paid subscriptions provide full access to all content. Subscriptions auto-renew unless cancelled. Refunds are available within 30 days of purchase.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">5. Affiliate Program</h3>
                <p>Affiliates earn commission on referred subscriptions. Commission rates vary by tier. Payouts are processed monthly for balances over $50.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">6. Content</h3>
                <p>All content is owned by The Daily Lesson, Inc. Users may not redistribute, modify, or commercially use content without permission.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">7. Contact</h3>
                <p>Questions? Email us at hello@curiouskelly.com</p>
              </section>
            </div>
          ) : (
            <div className="space-y-4 text-sm text-white/70">
              <p className="text-white/40 text-xs">Last updated: January 2026</p>
              
              <section>
                <h3 className="font-semibold text-white mb-2">1. Information We Collect</h3>
                <p>We collect: email address, learning preferences, usage data, and device fingerprints for personalization. We do not collect data from children under 13 without parental consent.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">2. How We Use Information</h3>
                <p>We use your information to: deliver personalized lessons, track progress, process payments, and improve the Service. We never sell your data.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">3. Data Storage</h3>
                <p>Your data is stored securely on encrypted servers. We retain data while your account is active and for 30 days after deletion requests.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">4. Third Parties</h3>
                <p>We use: Stripe (payments), HeyGen (video generation), ElevenLabs (voice), Vercel (hosting). These providers have their own privacy policies.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">5. Cookies</h3>
                <p>We use cookies for authentication, preferences, and analytics. You can disable cookies in your browser settings.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">6. Your Rights</h3>
                <p>You can: access your data, request deletion, opt out of marketing, and export your learning history. Contact hello@curiouskelly.com for requests.</p>
              </section>

              <section>
                <h3 className="font-semibold text-white mb-2">7. COPPA Compliance</h3>
                <p>We comply with the Children&apos;s Online Privacy Protection Act. Parents can review, delete, or refuse further collection of their child&apos;s data.</p>
              </section>
            </div>
          )}
        </div>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
