import { NextRequest, NextResponse } from 'next/server'

// Cron job to auto-process HeyGen video queue
// Run every 5 minutes via Vercel Cron
// vercel.json: { "crons": [{ "path": "/api/cron/process-videos", "schedule": "*/5 * * * *" }] }

export async function GET(request: NextRequest) {
  // Verify cron secret
  const authHeader = request.headers.get('authorization')
  if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  try {
    // Call the process-queue API internally
    const baseUrl = process.env.VERCEL_URL 
      ? `https://${process.env.VERCEL_URL}` 
      : 'http://localhost:3000'
    
    const response = await fetch(`${baseUrl}/api/heygen/process-queue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ batchSize: 10, dryRun: false })
    })

    const result = await response.json()

    // Also poll for completed videos
    const pollResponse = await fetch(`${baseUrl}/api/heygen/poll-status`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ limit: 50 })
    })

    const pollResult = await pollResponse.json()

    return NextResponse.json({
      processed: result,
      polled: pollResult,
      timestamp: new Date().toISOString()
    })

  } catch (error) {
    console.error('Cron process-videos error:', error)
    return NextResponse.json({ error: 'Cron failed' }, { status: 500 })
  }
}
