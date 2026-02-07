'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { CheckCircle2, XCircle, AlertCircle, Copy, ExternalLink, RefreshCw } from 'lucide-react'

interface WebhookStatus {
  production_url: string
  overall_status: 'READY' | 'SETUP_INCOMPLETE'
  webhooks: Array<{
    name: string
    endpoint: string
    fullUrl: string
    method: string
    description: string
    configureAt?: string
    envVars: string[]
    status: 'configured' | 'missing_env' | 'unknown'
    events?: string[]
  }>
  cron_jobs: Array<{
    name: string
    endpoint: string
    schedule: string
    description: string
    envVars: string[]
    status: 'configured' | 'missing_env' | 'unknown'
  }>
  database: {
    status: string
    stats: {
      completed_videos: number
      queued_videos: number
      processing_videos: number
      active_subscribers: number
      total_affiliates: number
    } | null
  }
  setup_checklist: Array<{
    item: string
    status: 'complete' | 'incomplete'
    action: string | null
  }>
  quick_setup: {
    stripe: Record<string, string>
    heygen: Record<string, string>
    cron: Record<string, string>
  }
}

export default function WebhooksPage() {
  const [status, setStatus] = useState<WebhookStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [copied, setCopied] = useState<string | null>(null)

  const fetchStatus = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/webhooks/status')
      const data = await res.json()
      setStatus(data)
    } catch (e) {
      console.error('Failed to fetch status:', e)
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchStatus()
  }, [])

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white p-8 flex items-center justify-center">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (!status) {
    return (
      <div className="min-h-screen bg-black text-white p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold text-red-500">Failed to load webhook status</h1>
        </div>
      </div>
    )
  }

  const StatusIcon = ({ s }: { s: string }) => {
    if (s === 'configured' || s === 'complete' || s === 'connected') {
      return <CheckCircle2 className="w-5 h-5 text-green-500" />
    }
    if (s === 'missing_env' || s === 'incomplete') {
      return <XCircle className="w-5 h-5 text-red-500" />
    }
    return <AlertCircle className="w-5 h-5 text-yellow-500" />
  }

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Webhook Configuration</h1>
            <p className="text-zinc-400 mt-1">Production: {status.production_url}</p>
          </div>
          <div className="flex items-center gap-4">
            <Badge 
              variant={status.overall_status === 'READY' ? 'default' : 'destructive'}
              className="text-lg px-4 py-2"
            >
              {status.overall_status === 'READY' ? 'All Systems Ready' : 'Setup Incomplete'}
            </Badge>
            <Button variant="outline" onClick={fetchStatus} disabled={loading}>
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>

        {/* Database Status */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <StatusIcon s={status.database.status} />
              Database Connection
            </CardTitle>
          </CardHeader>
          <CardContent>
            {status.database.stats && (
              <div className="grid grid-cols-5 gap-4">
                <div className="text-center p-4 bg-zinc-800 rounded-lg">
                  <div className="text-3xl font-bold text-green-500">{status.database.stats.completed_videos}</div>
                  <div className="text-sm text-zinc-400">Completed Videos</div>
                </div>
                <div className="text-center p-4 bg-zinc-800 rounded-lg">
                  <div className="text-3xl font-bold text-yellow-500">{status.database.stats.queued_videos}</div>
                  <div className="text-sm text-zinc-400">Queued</div>
                </div>
                <div className="text-center p-4 bg-zinc-800 rounded-lg">
                  <div className="text-3xl font-bold text-blue-500">{status.database.stats.processing_videos}</div>
                  <div className="text-sm text-zinc-400">Processing</div>
                </div>
                <div className="text-center p-4 bg-zinc-800 rounded-lg">
                  <div className="text-3xl font-bold text-purple-500">{status.database.stats.active_subscribers}</div>
                  <div className="text-sm text-zinc-400">Subscribers</div>
                </div>
                <div className="text-center p-4 bg-zinc-800 rounded-lg">
                  <div className="text-3xl font-bold text-pink-500">{status.database.stats.total_affiliates}</div>
                  <div className="text-sm text-zinc-400">Affiliates</div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Webhooks */}
        <div className="grid gap-6">
          <h2 className="text-xl font-semibold">Webhooks</h2>
          {status.webhooks.map((webhook, i) => (
            <Card key={i} className="bg-zinc-900 border-zinc-800">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    <StatusIcon s={webhook.status} />
                    {webhook.name}
                  </CardTitle>
                  {webhook.configureAt && (
                    <Button variant="outline" size="sm" asChild>
                      <a href={webhook.configureAt} target="_blank" rel="noopener noreferrer">
                        Configure <ExternalLink className="w-4 h-4 ml-2" />
                      </a>
                    </Button>
                  )}
                </div>
                <CardDescription>{webhook.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-2">
                  <code className="flex-1 bg-zinc-800 px-4 py-2 rounded-lg text-sm font-mono text-green-400">
                    {webhook.fullUrl}
                  </code>
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={() => copyToClipboard(webhook.fullUrl, `url-${i}`)}
                  >
                    {copied === `url-${i}` ? <CheckCircle2 className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                  </Button>
                </div>
                
                {webhook.events && (
                  <div>
                    <p className="text-sm text-zinc-400 mb-2">Events to subscribe:</p>
                    <div className="flex flex-wrap gap-2">
                      {webhook.events.map((event, j) => (
                        <Badge key={j} variant="secondary" className="font-mono text-xs">
                          {event}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                <div>
                  <p className="text-sm text-zinc-400 mb-2">Required environment variables:</p>
                  <div className="flex flex-wrap gap-2">
                    {webhook.envVars.map((env, j) => (
                      <Badge 
                        key={j} 
                        variant={webhook.status === 'configured' ? 'default' : 'destructive'}
                        className="font-mono text-xs"
                      >
                        {env}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Cron Jobs */}
        <div className="grid gap-6">
          <h2 className="text-xl font-semibold">Cron Jobs</h2>
          {status.cron_jobs.map((cron, i) => (
            <Card key={i} className="bg-zinc-900 border-zinc-800">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <StatusIcon s={cron.status} />
                  {cron.name}
                </CardTitle>
                <CardDescription>{cron.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-4">
                  <div>
                    <p className="text-sm text-zinc-400">Endpoint</p>
                    <code className="text-green-400 font-mono">{cron.endpoint}</code>
                  </div>
                  <div>
                    <p className="text-sm text-zinc-400">Schedule</p>
                    <code className="text-blue-400 font-mono">{cron.schedule}</code>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Setup Steps */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader>
            <CardTitle>Quick Setup Guide</CardTitle>
            <CardDescription>Follow these steps to complete your webhook configuration</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Stripe */}
            <div className="space-y-3">
              <h3 className="font-semibold text-lg flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-purple-600 text-white text-sm flex items-center justify-center">1</span>
                Stripe Webhook
              </h3>
              <ol className="list-decimal list-inside space-y-2 text-zinc-300 ml-8">
                {Object.entries(status.quick_setup.stripe).map(([key, value]) => (
                  <li key={key}>{value}</li>
                ))}
              </ol>
            </div>

            {/* HeyGen */}
            <div className="space-y-3">
              <h3 className="font-semibold text-lg flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-blue-600 text-white text-sm flex items-center justify-center">2</span>
                HeyGen Webhook (Optional)
              </h3>
              <ol className="list-decimal list-inside space-y-2 text-zinc-300 ml-8">
                {Object.entries(status.quick_setup.heygen).map(([key, value]) => (
                  <li key={key}>{value}</li>
                ))}
              </ol>
            </div>

            {/* Cron */}
            <div className="space-y-3">
              <h3 className="font-semibold text-lg flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-green-600 text-white text-sm flex items-center justify-center">3</span>
                Cron Jobs
              </h3>
              <div className="ml-8 text-zinc-300">
                <p>{status.quick_setup.cron.note}</p>
                <p className="text-sm text-zinc-400 mt-1">{status.quick_setup.cron.verify}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Checklist */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader>
            <CardTitle>Setup Checklist</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {status.setup_checklist.map((item, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-zinc-800 rounded-lg">
                  <div className="flex items-center gap-3">
                    <StatusIcon s={item.status} />
                    <span>{item.item}</span>
                  </div>
                  {item.action && (
                    <span className="text-sm text-zinc-400">{item.action}</span>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
