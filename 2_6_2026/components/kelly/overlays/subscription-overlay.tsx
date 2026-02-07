'use client'

/**
 * KellyOS Subscription Management Overlay
 * Apple-quality subscription status and management panel
 */

import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel, ListItem, SectionHeader } from './index'
import { 
  Crown, 
  Calendar, 
  CreditCard, 
  Receipt, 
  Gift,
  ChevronRight,
  Check,
  X,
  AlertCircle,
  Sparkles,
} from 'lucide-react'
import { Button } from '@/components/ui/button'

interface SubscriptionOverlayProps {
  isOpen: boolean
  onClose: () => void
  currentPlan: 'free' | 'learner' | 'family' | 'lifetime'
  billingCycle?: 'monthly' | 'annual'
  nextBillingDate?: string
  onUpgrade?: () => void
  onManageBilling?: () => void
  onCancelSubscription?: () => void
}

const PLAN_DETAILS = {
  free: {
    name: 'Explorer',
    description: 'Free plan',
    features: ['Today\'s lesson only', '1 language', 'Basic captions'],
    limitations: ['No archive access', 'No offline downloads', 'Limited archetypes'],
    color: 'bg-muted',
  },
  learner: {
    name: 'Learner',
    description: 'Individual plan',
    features: ['365 daily lessons', 'All 12 languages', 'All archetypes', 'Interactive captions', 'Visual aids', 'Offline downloads'],
    limitations: [],
    color: 'bg-primary',
  },
  family: {
    name: 'Family',
    description: 'Up to 6 members',
    features: ['Everything in Learner', 'Up to 6 profiles', 'Parental controls', 'Family discussions', 'Priority support'],
    limitations: [],
    color: 'bg-white/20',
  },
  lifetime: {
    name: 'Lifetime',
    description: 'One-time purchase',
    features: ['Everything in Learner', 'Lifetime access', 'All future updates', 'Founding member badge'],
    limitations: [],
    color: 'bg-[var(--kelly-blue)]',
  },
}

export function SubscriptionOverlay({
  isOpen,
  onClose,
  currentPlan,
  billingCycle,
  nextBillingDate,
  onUpgrade,
  onManageBilling,
  onCancelSubscription,
}: SubscriptionOverlayProps) {
  if (!isOpen) return null

  const plan = PLAN_DETAILS[currentPlan]
  const isPremium = currentPlan !== 'free'
  const isLifetime = currentPlan === 'lifetime'

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose}>
      <OverlayPanel
        title="Subscription"
        onClose={onClose}
        size="md"
      >
        <div className="p-6 space-y-6">
          {/* Current Plan Card */}
          <div className={cn(
            'rounded-3xl p-6 text-white',
            currentPlan === 'free' ? 'bg-muted text-foreground' : plan.color
          )}>
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  {isPremium && <Crown className="h-5 w-5" />}
                  <h3 className="text-xl font-bold">{plan.name}</h3>
                </div>
                <p className={cn('text-sm', isPremium ? 'opacity-80' : 'text-muted-foreground')}>
                  {plan.description}
                </p>
              </div>
              {!isPremium && (
                <Button 
                  size="sm"
                  className="bg-primary text-primary-foreground hover:bg-primary/90"
                  onClick={onUpgrade}
                >
                  Upgrade
                </Button>
              )}
            </div>

            {/* Billing info for premium users */}
            {isPremium && !isLifetime && (
              <div className="mt-4 pt-4 border-t border-white/20">
                <div className="flex items-center justify-between text-sm">
                  <span className="opacity-70">
                    {billingCycle === 'monthly' ? 'Billed monthly' : 'Billed annually'}
                  </span>
                  {nextBillingDate && (
                    <span>Next: {nextBillingDate}</span>
                  )}
                </div>
              </div>
            )}

            {isLifetime && (
              <div className="mt-4 pt-4 border-t border-white/20">
                <div className="flex items-center gap-2 text-sm">
                  <Sparkles className="h-4 w-4" />
                  <span>Lifetime access - no renewal needed</span>
                </div>
              </div>
            )}
          </div>

          {/* Plan Features */}
          <div className="space-y-3">
            <h4 className="font-semibold text-sm text-muted-foreground uppercase tracking-wider">
              What's Included
            </h4>
            <div className="space-y-2">
              {plan.features.map((feature, i) => (
                <div key={i} className="flex items-center gap-3 text-sm">
                  <Check className="h-4 w-4 text-white/70 flex-shrink-0" />
                  <span>{feature}</span>
                </div>
              ))}
              {plan.limitations.map((limit, i) => (
                <div key={i} className="flex items-center gap-3 text-sm text-muted-foreground">
                  <X className="h-4 w-4 text-muted-foreground/50 flex-shrink-0" />
                  <span>{limit}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Upgrade prompt for free users */}
          {!isPremium && (
            <div className="p-4 rounded-2xl bg-primary/5 border border-primary/10">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-sm">Unlock the full iLearn experience</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    365 lessons, 12 languages, all archetypes, visual aids, and more.
                  </p>
                  <Button 
                    size="sm" 
                    className="mt-3"
                    onClick={onUpgrade}
                  >
                    View Plans
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Management options for premium users */}
          {isPremium && !isLifetime && (
            <>
              <SectionHeader title="Manage" />
              <div className="rounded-2xl border overflow-hidden">
                <button
                  onClick={onManageBilling}
                  className="w-full flex items-center gap-4 p-4 hover:bg-muted/50 transition-colors border-b"
                >
                  <CreditCard className="h-5 w-5 text-muted-foreground" />
                  <span className="flex-1 text-left">Payment Method</span>
                  <ChevronRight className="h-5 w-5 text-muted-foreground" />
                </button>
                <button
                  onClick={onManageBilling}
                  className="w-full flex items-center gap-4 p-4 hover:bg-muted/50 transition-colors border-b"
                >
                  <Receipt className="h-5 w-5 text-muted-foreground" />
                  <span className="flex-1 text-left">Billing History</span>
                  <ChevronRight className="h-5 w-5 text-muted-foreground" />
                </button>
                <button
                  onClick={() => { window.open('https://curiouskelly.com/gift', '_blank'); }}
                  className="w-full flex items-center gap-4 p-4 hover:bg-muted/50 transition-colors"
                >
                  <Gift className="h-5 w-5 text-muted-foreground" />
                  <span className="flex-1 text-left">Gift a Subscription</span>
                  <ChevronRight className="h-5 w-5 text-muted-foreground" />
                </button>
              </div>

              {/* Cancel option */}
              <div className="pt-4">
                <button
                  onClick={onCancelSubscription}
                  className="w-full text-center text-sm text-muted-foreground hover:text-destructive transition-colors"
                >
                  Cancel Subscription
                </button>
              </div>
            </>
          )}

          {/* Footer */}
          <div className="text-center text-xs text-muted-foreground pt-4 border-t">
            <p>Questions? Contact support@lessonoftheday.com</p>
          </div>
        </div>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
