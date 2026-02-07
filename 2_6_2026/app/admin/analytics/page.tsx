'use client'

import React, { useEffect, useState } from 'react'
import useSWR from 'swr'

// Curious Kelly Analytics Dashboard & Tracking System
// Real-time visitor tracking for curiouskelly.com

interface Visitor {
  id: string
  timestamp: string
  ip: string
  location: { city: string; region: string; country: string }
  device: { type: string; os: string; browser: string }
  screen: { width: number; height: number }
  referrer: string
  timeOnPage: number
  scrollDepth: number
  interactions: string[]
}

interface AnalyticsStats {
  liveVisitors: number
  todayVisits: number
  avgTimeOnPage: number
  avgScrollDepth: number
  deviceBreakdown: { desktop: number; mobile: number; tablet: number }
  topReferrers: { source: string; count: number }[]
  recentVisitors: Visitor[]
}

const fetcher = (url: string) => fetch(url).then(r => r.json()).catch(() => null)

export default function CuriousKellyAnalytics() {
  const { data: stats } = useSWR<AnalyticsStats>(
    '/api/analytics/stats',
    fetcher,
    { refreshInterval: 5000 }
  )

  const [copied, setCopied] = useState<string | null>(null)

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }

  const trackingScript = `<!-- Curious Kelly Analytics Tracking -->
<script>
(function() {
  const CK_ANALYTICS = {
    endpoint: '${typeof window !== 'undefined' ? window.location.origin : ''}/api/track',
    sessionId: crypto.randomUUID(),
    startTime: Date.now(),
    maxScrollDepth: 0,
    interactions: [],
    
    init() {
      this.trackPageView();
      this.trackScrollDepth();
      this.trackInteractions();
      this.trackTimeOnPage();
      this.trackVisibility();
    },
    
    getDeviceInfo() {
      const ua = navigator.userAgent;
      return {
        type: /Mobile|Android|iPhone|iPad/.test(ua) ? 
          (/iPad/.test(ua) ? 'Tablet' : 'Mobile') : 'Desktop',
        os: navigator.platform,
        browser: ua.match(/(Chrome|Firefox|Safari|Edge)\\/[\\d.]+/)?.[0] || 'Unknown',
        screen: { width: screen.width, height: screen.height },
        viewport: { width: window.innerWidth, height: window.innerHeight },
        language: navigator.language,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
      };
    },
    
    async trackPageView() {
      const data = {
        type: 'pageview',
        sessionId: this.sessionId,
        url: window.location.href,
        referrer: document.referrer,
        timestamp: new Date().toISOString(),
        device: this.getDeviceInfo()
      };
      this.send(data);
    },
    
    trackScrollDepth() {
      let ticking = false;
      window.addEventListener('scroll', () => {
        if (!ticking) {
          requestAnimationFrame(() => {
            const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrolled = Math.round((window.scrollY / scrollHeight) * 100);
            if (scrolled > this.maxScrollDepth) {
              this.maxScrollDepth = scrolled;
            }
            ticking = false;
          });
          ticking = true;
        }
      });
    },
    
    trackInteractions() {
      document.addEventListener('click', (e) => {
        const target = e.target.closest('[data-track]');
        if (target) {
          this.interactions.push({
            action: target.dataset.track,
            timestamp: Date.now() - this.startTime,
            element: target.tagName
          });
          this.send({
            type: 'interaction',
            sessionId: this.sessionId,
            action: target.dataset.track,
            timestamp: new Date().toISOString()
          });
        }
      });
      
      document.querySelectorAll('video, audio').forEach(el => {
        el.addEventListener('play', () => {
          this.send({
            type: 'media_play',
            sessionId: this.sessionId,
            element: el.tagName,
            src: el.currentSrc
          });
        });
      });
    },
    
    trackTimeOnPage() {
      window.addEventListener('beforeunload', () => {
        const timeOnPage = Math.round((Date.now() - this.startTime) / 1000);
        navigator.sendBeacon(this.endpoint, JSON.stringify({
          type: 'session_end',
          sessionId: this.sessionId,
          timeOnPage,
          scrollDepth: this.maxScrollDepth,
          interactions: this.interactions
        }));
      });
    },
    
    trackVisibility() {
      document.addEventListener('visibilitychange', () => {
        this.send({
          type: 'visibility',
          sessionId: this.sessionId,
          visible: !document.hidden,
          timestamp: new Date().toISOString()
        });
      });
    },
    
    async send(data) {
      try {
        await fetch(this.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
      } catch (e) { console.debug('Analytics:', e); }
    }
  };
  
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => CK_ANALYTICS.init());
  } else {
    CK_ANALYTICS.init();
  }
})();
</script>`

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-blue-400">Curious Kelly Analytics</h1>
        <p className="text-gray-400">Real-time visitor tracking for curiouskelly.com</p>
      </div>

      {/* Live Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <StatCard 
          title="Live Visitors" 
          value={stats?.liveVisitors?.toString() || '0'} 
          subtitle="Right now" 
          color="green" 
        />
        <StatCard 
          title="Today's Visits" 
          value={stats?.todayVisits?.toString() || '0'} 
          subtitle="Since midnight PT" 
          color="blue" 
        />
        <StatCard 
          title="Avg Time on Page" 
          value={`${stats?.avgTimeOnPage || 0}s`} 
          subtitle="Last 24 hours" 
          color="purple" 
        />
        <StatCard 
          title="Scroll Depth" 
          value={`${stats?.avgScrollDepth || 0}%`} 
          subtitle="Average" 
          color="orange" 
        />
      </div>

      {/* Main Dashboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Live Visitor Feed */}
        <div className="lg:col-span-2 bg-gray-900 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            Live Visitor Feed
          </h2>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {stats?.recentVisitors && stats.recentVisitors.length > 0 ? (
              stats.recentVisitors.map((visitor) => (
                <VisitorCard key={visitor.id} visitor={visitor} />
              ))
            ) : (
              <div className="text-gray-500 text-center py-8">
                Waiting for visitors...
                <br />
                <span className="text-sm">Analytics will populate when visitors arrive</span>
              </div>
            )}
          </div>
        </div>

        {/* Device Breakdown */}
        <div className="bg-gray-900 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Device Types</h2>
          <div className="space-y-3">
            <DeviceBar 
              label="Desktop" 
              percentage={stats?.deviceBreakdown?.desktop || 0} 
              color="blue" 
            />
            <DeviceBar 
              label="Mobile" 
              percentage={stats?.deviceBreakdown?.mobile || 0} 
              color="green" 
            />
            <DeviceBar 
              label="Tablet" 
              percentage={stats?.deviceBreakdown?.tablet || 0} 
              color="purple" 
            />
          </div>
          
          {/* Top Referrers */}
          <h3 className="text-lg font-semibold mt-6 mb-3">Top Referrers</h3>
          <div className="space-y-2">
            {stats?.topReferrers?.map((ref, i) => (
              <div key={i} className="flex justify-between text-sm">
                <span className="text-gray-300 truncate">{ref.source || 'Direct'}</span>
                <span className="text-gray-500">{ref.count}</span>
              </div>
            )) || (
              <p className="text-gray-500 text-sm">No data yet</p>
            )}
          </div>
        </div>
      </div>

      {/* Tracking Code Section */}
      <div className="mt-8 bg-gray-900 rounded-xl p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Tracking Code for curiouskelly.com</h2>
          <button
            onClick={() => copyToClipboard(trackingScript, 'tracking')}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm transition-colors"
          >
            {copied === 'tracking' ? 'Copied!' : 'Copy Code'}
          </button>
        </div>
        <p className="text-gray-400 mb-4">Add this script to your site's &lt;head&gt; tag:</p>
        <pre className="bg-black p-4 rounded-lg overflow-x-auto text-sm text-green-400 max-h-64 overflow-y-auto">
          {trackingScript}
        </pre>
      </div>

      {/* Data Attributes Guide */}
      <div className="mt-6 bg-gray-900 rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">Track Kelly Interactions</h2>
        <p className="text-gray-400 mb-4">Add data-track attributes to elements you want to monitor:</p>
        <pre className="bg-black p-4 rounded-lg overflow-x-auto text-sm text-blue-400">
{`<!-- Examples for curiouskelly.com -->
<button data-track="clicked_start_lesson">Start Lesson</button>
<div data-track="viewed_kelly_avatar">Kelly Avatar Container</div>
<video data-track="played_lesson_video">...</video>
<a data-track="clicked_signup" href="/signup">Sign Up</a>
<button data-track="clicked_heygen_cta">Powered by HeyGen</button>`}
        </pre>
      </div>

      {/* What You'll See */}
      <div className="mt-6 bg-blue-900/30 border border-blue-500/50 rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4 text-blue-400">What You'll Know When They Visit</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h3 className="font-semibold text-white mb-2">Visitor Profile</h3>
            <ul className="text-gray-300 space-y-1">
              <li>City, Region, Country (via IP geolocation)</li>
              <li>Device type (Desktop/Mobile/Tablet)</li>
              <li>Operating system & browser</li>
              <li>Screen resolution & viewport size</li>
              <li>Language & timezone</li>
              <li>Referrer (heygen.com, direct, email, etc.)</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-white mb-2">Engagement Metrics</h3>
            <ul className="text-gray-300 space-y-1">
              <li>Time on page (seconds)</li>
              <li>Max scroll depth (%)</li>
              <li>Every click on tracked elements</li>
              <li>Video/audio play events</li>
              <li>Tab visibility (active vs backgrounded)</li>
              <li>Session timeline reconstruction</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

// Sub-components
function StatCard({ title, value, subtitle, color }: { 
  title: string
  value: string
  subtitle: string
  color: 'green' | 'blue' | 'purple' | 'orange'
}) {
  const colors = {
    green: 'text-green-400',
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    orange: 'text-orange-400'
  }
  return (
    <div className="bg-gray-900 rounded-xl p-5">
      <p className="text-gray-400 text-sm">{title}</p>
      <p className={`text-3xl font-bold ${colors[color]}`}>{value}</p>
      <p className="text-gray-500 text-xs">{subtitle}</p>
    </div>
  )
}

function DeviceBar({ label, percentage, color }: {
  label: string
  percentage: number
  color: 'blue' | 'green' | 'purple'
}) {
  const colors = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500'
  }
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span className="text-gray-400">{percentage}%</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div 
          className={`h-full ${colors[color]} rounded-full transition-all`} 
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

function VisitorCard({ visitor }: { visitor: Visitor }) {
  const timeAgo = (timestamp: string) => {
    const seconds = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000)
    if (seconds < 60) return `${seconds}s ago`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
    return `${Math.floor(seconds / 3600)}h ago`
  }
  
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg">
            {visitor.device.type === 'Desktop' ? '💻' : 
             visitor.device.type === 'Mobile' ? '📱' : '📱'}
          </span>
          <div>
            <p className="text-white font-medium">
              {visitor.location.city}, {visitor.location.region}
            </p>
            <p className="text-gray-400 text-xs">
              {visitor.device.browser} on {visitor.device.os}
            </p>
          </div>
        </div>
        <span className="text-gray-500 text-xs">{timeAgo(visitor.timestamp)}</span>
      </div>
      <div className="flex gap-4 text-xs text-gray-400">
        <span>Time: {visitor.timeOnPage}s</span>
        <span>Scroll: {visitor.scrollDepth}%</span>
        <span>Actions: {visitor.interactions.length}</span>
      </div>
      {visitor.referrer && (
        <p className="text-xs text-blue-400 mt-1">via {visitor.referrer}</p>
      )}
    </div>
  )
}
