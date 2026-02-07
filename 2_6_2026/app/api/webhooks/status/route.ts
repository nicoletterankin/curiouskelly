import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * WEBHOOK STATUS DASHBOARD
 * 
 * Returns the status of all configured webhooks and cron jobs.
 * Use this to verify your production setup is complete.
 * 
 * GET /api/webhooks/status
 * 
 * Last updated: 2026-02-03
 */

interface WebhookConfig {
  name: string
  endpoint: string
  fullUrl: string
  method: string
  description: string
  configureAt?: string
  envVars: string[]
  status: 'configured' | 'missing_env' | 'unknown'
  events?: string[]
}

export async function GET() {
  const baseUrl = process.env.VERCEL_PROJECT_PRODUCTION_URL 
    ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
    : process.env.VERCEL_URL 
      ? `https://${process.env.VERCEL_URL}`
      : 'https://thedailylesson.com'

  // Define all webhooks
  const webhooks: WebhookConfig[] = [
    {
      name: 'Stripe Payments',
      endpoint: '/api/webhooks/stripe',
      fullUrl: `${baseUrl}/api/webhooks/stripe`,
      method: 'POST',
      description: 'Handles payment completions, subscriptions, and affiliate commissions',
      configureAt: 'https://dashboard.stripe.com/webhooks',
      envVars: ['STRIPE_SECRET_KEY', 'STRIPE_WEBHOOK_SECRET'],
      status: process.env.STRIPE_SECRET_KEY && process.env.STRIPE_WEBHOOK_SECRET 
        ? 'configured' 
        : 'missing_env',
      events: [
        'checkout.session.completed',
        'customer.subscription.created',
        'invoice.paid',
      ],
    },
    {
      name: 'HeyGen Video Callbacks',
      endpoint: '/api/webhooks/heygen',
      fullUrl: `${baseUrl}/api/webhooks/heygen`,
      method: 'POST',
      description: 'Receives video completion notifications from HeyGen',
      configureAt: 'https://app.heygen.com/settings/api',
      envVars: ['HEYGEN_API_KEY'],
      status: process.env.HEYGEN_API_KEY ? 'configured' : 'missing_env',
      events: [
        'avatar_video.completed',
        'avatar_video.failed',
      ],
    },
    {
      name: 'Antigravity Assets',
      endpoint: '/api/antigravity/webhook',
      fullUrl: `${baseUrl}/api/antigravity/webhook`,
      method: 'POST',
      description: 'Receives asset completion notifications from Antigravity system',
      envVars: ['SUPABASE_WEBHOOK_SECRET'],
      status: process.env.SUPABASE_WEBHOOK_SECRET ? 'configured' : 'missing_env',
      events: ['asset_complete', 'status'],
    },
  ]

  // Define cron jobs
  const cronJobs = [
    {
      name: 'Process HeyGen Queue',
      endpoint: '/api/cron/process-videos',
      schedule: '*/5 * * * *',
      description: 'Processes queued videos and polls for completions every 5 minutes',
      envVars: ['CRON_SECRET'],
      status: process.env.CRON_SECRET ? 'configured' : 'missing_env',
    },
  ]

  // Check database connectivity
  let dbStatus = 'unknown'
  let dbStats = null
  try {
    const result = await sql`
      SELECT 
        (SELECT COUNT(*) FROM heygen_videos WHERE status = 'completed') as completed_videos,
        (SELECT COUNT(*) FROM heygen_videos WHERE status = 'queued') as queued_videos,
        (SELECT COUNT(*) FROM heygen_videos WHERE status = 'processing') as processing_videos,
        (SELECT COUNT(*) FROM subscribers WHERE status = 'active') as active_subscribers,
        (SELECT COUNT(*) FROM affiliates) as total_affiliates
    `
    dbStatus = 'connected'
    dbStats = result[0]
  } catch (e) {
    dbStatus = 'error: ' + String(e)
  }

  // Build setup checklist
  const setupChecklist = [
    {
      item: 'Database Connection',
      status: dbStatus === 'connected' ? 'complete' : 'incomplete',
      action: dbStatus === 'connected' ? null : 'Check DATABASE_URL environment variable',
    },
    {
      item: 'Stripe Webhook',
      status: process.env.STRIPE_WEBHOOK_SECRET ? 'complete' : 'incomplete',
      action: process.env.STRIPE_WEBHOOK_SECRET 
        ? null 
        : `Add webhook at stripe.com pointing to ${baseUrl}/api/webhooks/stripe`,
    },
    {
      item: 'HeyGen API Key',
      status: process.env.HEYGEN_API_KEY ? 'complete' : 'incomplete',
      action: process.env.HEYGEN_API_KEY ? null : 'Add HEYGEN_API_KEY to environment variables',
    },
    {
      item: 'Cron Secret',
      status: process.env.CRON_SECRET ? 'complete' : 'incomplete',
      action: process.env.CRON_SECRET ? null : 'Add CRON_SECRET to environment variables',
    },
  ]

  const allConfigured = webhooks.every(w => w.status === 'configured') &&
                        cronJobs.every(c => c.status === 'configured') &&
                        dbStatus === 'connected'

  return NextResponse.json({
    production_url: baseUrl,
    overall_status: allConfigured ? 'READY' : 'SETUP_INCOMPLETE',
    
    webhooks,
    cron_jobs: cronJobs,
    
    database: {
      status: dbStatus,
      stats: dbStats,
    },

    setup_checklist: setupChecklist,
    
    quick_setup: {
      stripe: {
        step1: 'Go to https://dashboard.stripe.com/webhooks',
        step2: 'Click "Add endpoint"',
        step3: `Enter URL: ${baseUrl}/api/webhooks/stripe`,
        step4: 'Select events: checkout.session.completed, customer.subscription.created, invoice.paid',
        step5: 'Copy the signing secret to STRIPE_WEBHOOK_SECRET in Vercel',
      },
      heygen: {
        step1: 'Go to https://app.heygen.com/settings/api',
        step2: 'Find webhook configuration section',
        step3: `Enter URL: ${baseUrl}/api/webhooks/heygen`,
        step4: '(Optional) Add HEYGEN_WEBHOOK_SECRET for signature verification',
      },
      cron: {
        note: 'Cron jobs are auto-configured via vercel.json',
        verify: 'Check Vercel Dashboard > Project > Cron Jobs',
      },
    },

    test_endpoints: {
      stripe: `curl -X POST ${baseUrl}/api/webhooks/stripe -d '{"type":"test"}' # Returns 400 (expected - no signature)`,
      heygen: `curl ${baseUrl}/api/webhooks/heygen # Returns status`,
      antigravity: `curl "${baseUrl}/api/antigravity/webhook?day=34" # Returns day status`,
      cron: `curl ${baseUrl}/api/cron/process-videos -H "Authorization: Bearer YOUR_CRON_SECRET"`,
    },
  })
}
