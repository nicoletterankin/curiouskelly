'use client';

import { Learner } from '../../lib/sis-types';
import { formatTimeAgo, formatDuration, getSubscriptionBadge } from '../../lib/sis-utils';
import { X, Mail, Bell, Globe, Smartphone, CreditCard, Award, ExternalLink, Copy } from 'lucide-react';
import { useState } from 'react';

interface LearnerDetailPanelProps {
  learner: Learner;
  onClose: () => void;
  onSendEmail: () => void;
  onSendPush: () => void;
}

export function LearnerDetailPanel({ learner, onClose, onSendEmail, onSendPush }: LearnerDetailPanelProps) {
  const [copied, setCopied] = useState(false);
  const badge = getSubscriptionBadge(learner.subscriptionStatus, learner.subscriptionPlan);
  
  const copyIP = () => {
    if (learner.ipFull) {
      navigator.clipboard.writeText(learner.ipFull);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50 z-40" onClick={onClose} />
      
      {/* Panel */}
      <div className="fixed top-0 right-0 h-full w-full max-w-md bg-gray-900 border-l border-gray-700/50 z-50 overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-gray-900 border-b border-gray-700/50 p-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Learner Profile</h2>
          <button onClick={onClose} aria-label="Close panel" className="p-2 hover:bg-gray-800 rounded-lg">
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>
        
        <div className="p-4 space-y-6">
          {/* Profile Header */}
          <div className="flex items-center gap-4">
            <div className="relative">
              {learner.avatar ? (
                <img src={learner.avatar} alt="" className="w-16 h-16 rounded-full bg-gray-700" />
              ) : (
                <div className="w-16 h-16 rounded-full bg-gray-700 flex items-center justify-center text-gray-400 text-xl">?</div>
              )}
              {learner.isOnline && (
                <span className="absolute bottom-0 right-0 w-4 h-4 bg-green-500 rounded-full border-2 border-gray-900"></span>
              )}
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">
                {learner.name || 'Anonymous'}
                {learner.isVIP && <span className="ml-2">⭐</span>}
              </h3>
              <p className="text-gray-400">{learner.email || learner.id}</p>
              <div className="flex items-center gap-2 mt-1">
                <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${badge.color}`}>
                  {badge.label}
                </span>
                {learner.isOnline && <span className="text-green-400 text-sm">● Online</span>}
              </div>
            </div>
          </div>
          
          {/* Quick Actions */}
          <div className="flex gap-2">
            <button
              onClick={onSendEmail}
              disabled={!learner.email || !learner.emailOptIn}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors"
            >
              <Mail className="w-4 h-4" />
              Send Email
            </button>
            <button
              onClick={onSendPush}
              disabled={!learner.pushOptIn}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-lg transition-colors"
            >
              <Bell className="w-4 h-4" />
              Send Push
            </button>
          </div>
          
          {/* Location & Device */}
          <Section title="Location & Device" icon={Globe}>
            <Row label="Location" value={`${learner.countryFlag} ${learner.city || 'Unknown'}, ${learner.region || ''} ${learner.country || ''}`} />
            <Row label="Timezone" value={learner.timezone || '—'} />
            <Row 
              label="IP Address" 
              value={
                <span className="flex items-center gap-2">
                  <code className="font-mono">{learner.ipFull || '—'}</code>
                  {learner.ipFull && (
                    <button onClick={copyIP} className="p-1 hover:bg-gray-700 rounded" title={copied ? 'Copied!' : 'Copy IP'}>
                      <Copy className={`w-3 h-3 ${copied ? 'text-green-400' : 'text-gray-400'}`} />
                    </button>
                  )}
                </span>
              } 
            />
            <Row label="Device" value={`${learner.device || '—'} • ${learner.browser || '—'} • ${learner.os || '—'}`} />
          </Section>
          
          {/* Subscription */}
          <Section title="Subscription" icon={CreditCard}>
            <Row label="Status" value={<span className={`px-2 py-0.5 rounded-full text-xs font-medium ${badge.color}`}>{badge.label}</span>} />
            {learner.stripeCustomerId && (
              <Row 
                label="Stripe ID" 
                value={
                  <a
                    href={`https://dashboard.stripe.com/customers/${learner.stripeCustomerId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-blue-400 hover:text-blue-300"
                  >
                    {learner.stripeCustomerId}
                    <ExternalLink className="w-3 h-3" />
                  </a>
                }
              />
            )}
            {learner.trialEndsAt && <Row label="Trial Ends" value={new Date(learner.trialEndsAt).toLocaleDateString()} />}
            {learner.currentPeriodEnd && <Row label="Renews" value={new Date(learner.currentPeriodEnd).toLocaleDateString()} />}
          </Section>
          
          {/* Learning Progress */}
          <Section title="Learning Progress" icon={Award}>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <Stat label="Lessons" value={learner.lessonsCompleted} />
              <Stat label="Streak" value={learner.currentStreak > 0 ? `🔥 ${learner.currentStreak}` : '—'} />
              <Stat label="Best" value={learner.longestStreak} />
            </div>
            <Row label="Current Day" value={`Day ${learner.currentDay} / 365`} />
            <Row label="Total Time" value={formatDuration(learner.totalTimeSpent)} />
            <Row label="Kelly Age" value={`${learner.kellyAge} years`} />
            <Row label="Language" value={learner.preferredLanguage.toUpperCase()} />
            {learner.lastLessonTopic && <Row label="Last Lesson" value={learner.lastLessonTopic} />}
          </Section>
          
          {/* Communication */}
          <Section title="Communication" icon={Mail}>
            <Row label="Email Opt-in" value={learner.emailOptIn ? '✓ Subscribed' : '✗ Opted out'} />
            <Row label="Push Opt-in" value={learner.pushOptIn ? '✓ Enabled' : '✗ Disabled'} />
            <Row label="Emails Sent" value={learner.emailsSent} />
            <Row label="Open Rate" value={learner.emailsSent > 0 ? `${Math.round((learner.emailsOpened / learner.emailsSent) * 100)}%` : '—'} />
          </Section>
          
          {/* Timeline */}
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-400 uppercase tracking-wider">Timeline</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-gray-500"></span>
                <span className="text-gray-400">Last seen:</span>
                <span className="text-white">{formatTimeAgo(learner.lastSeenAt)}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                <span className="text-gray-400">Signed up via {learner.authProvider}:</span>
                <span className="text-white">{new Date(learner.createdAt).toLocaleDateString()}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function Section({ title, icon: Icon, children }: { title: string; icon: React.ElementType; children: React.ReactNode }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Icon className="w-4 h-4 text-gray-400" />
        <h4 className="text-sm font-medium text-gray-400 uppercase tracking-wider">{title}</h4>
      </div>
      <div className="bg-gray-800/50 rounded-lg p-4 space-y-2">
        {children}
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between items-center text-sm">
      <span className="text-gray-400">{label}</span>
      <span className="text-white">{value}</span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="text-center">
      <div className="text-xl font-semibold text-white">{value}</div>
      <div className="text-xs text-gray-400">{label}</div>
    </div>
  );
}
