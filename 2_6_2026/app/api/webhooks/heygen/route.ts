import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * HEYGEN WEBHOOK ENDPOINT
 * 
 * Receives callbacks from HeyGen when video generation completes.
 * Configure in HeyGen Dashboard: https://app.heygen.com/settings/api
 * 
 * Endpoint: https://thedailylesson.com/api/webhooks/heygen
 * 
 * HeyGen sends:
 * {
 *   "event_type": "avatar_video.completed" | "avatar_video.failed",
 *   "event_data": {
 *     "video_id": "xxx",
 *     "status": "completed" | "failed",
 *     "video_url": "https://...",
 *     "thumbnail_url": "https://...",
 *     "duration": 30,
 *     "callback_id": "optional-custom-id"
 *   }
 * }
 */

interface HeyGenWebhookPayload {
  event_type: 'avatar_video.completed' | 'avatar_video.failed' | 'avatar_video.processing'
  event_data: {
    video_id: string
    status: 'completed' | 'failed' | 'processing'
    video_url?: string
    thumbnail_url?: string
    duration?: number
    callback_id?: string
    error?: string
  }
}

export async function POST(request: NextRequest) {
  try {
    // Optional: Verify webhook signature if HeyGen provides one
    const signature = request.headers.get('x-heygen-signature')
    const webhookSecret = process.env.HEYGEN_WEBHOOK_SECRET
    
    // If we have a secret configured, verify it
    if (webhookSecret && signature !== webhookSecret) {
      console.error('[HeyGen Webhook] Invalid signature')
      return NextResponse.json({ error: 'Invalid signature' }, { status: 401 })
    }

    const body = await request.json() as HeyGenWebhookPayload
    const { event_type, event_data } = body

    console.log('[HeyGen Webhook] Received:', {
      event_type,
      video_id: event_data.video_id,
      status: event_data.status,
    })

    // Handle video completion
    if (event_type === 'avatar_video.completed' && event_data.video_url) {
      // Update heygen_videos table (legacy)
      const updated = await sql`
        UPDATE heygen_videos
        SET 
          status = 'completed',
          video_url = ${event_data.video_url},
          thumbnail_url = ${event_data.thumbnail_url || null},
          duration = ${event_data.duration || null},
          updated_at = NOW()
        WHERE heygen_video_id = ${event_data.video_id}
        RETURNING id, day_of_year, phase, archetype, age_category, language
      `

      // ALSO update video_queue table (new master queue)
      const queueUpdated = await sql`
        UPDATE video_queue
        SET 
          status = 'completed',
          video_url = ${event_data.video_url},
          video_duration = ${event_data.duration || null},
          completed_at = NOW(),
          updated_at = NOW()
        WHERE heygen_video_id = ${event_data.video_id}
        RETURNING id, day_number, phase, age_group, language
      `.catch(() => [])

      if (updated.length > 0 || queueUpdated.length > 0) {
        const video = updated[0] || queueUpdated[0]
        const source = updated.length > 0 ? 'heygen_videos' : 'video_queue'
        
        console.log('[HeyGen Webhook] Video updated:', {
          source,
          id: video.id,
          day: video.day_of_year || video.day_number,
          phase: video.phase,
        })

        return NextResponse.json({
          received: true,
          updated: true,
          source,
          video: {
            day: video.day_of_year || video.day_number,
            phase: video.phase,
            archetype: video.archetype,
            ageCategory: video.age_category || video.age_group,
            language: video.language,
          }
        })
      } else {
        // Video ID not found in our database - log but don't error
        console.warn('[HeyGen Webhook] Video ID not found in database:', event_data.video_id)
        return NextResponse.json({
          received: true,
          updated: false,
          message: 'Video ID not found in database'
        })
      }
    }

    // Handle video failure
    if (event_type === 'avatar_video.failed') {
      // Update legacy table
      await sql`
        UPDATE heygen_videos
        SET 
          status = 'failed',
          error_message = ${event_data.error || 'Unknown error'},
          updated_at = NOW()
        WHERE heygen_video_id = ${event_data.video_id}
      `
      
      // Update new video_queue table
      await sql`
        UPDATE video_queue
        SET 
          status = 'failed',
          error_message = ${event_data.error || 'Unknown error'},
          last_error_at = NOW(),
          updated_at = NOW()
        WHERE heygen_video_id = ${event_data.video_id}
      `.catch(() => {})

      console.error('[HeyGen Webhook] Video failed:', {
        video_id: event_data.video_id,
        error: event_data.error,
      })

      return NextResponse.json({
        received: true,
        status: 'failed',
        error: event_data.error,
      })
    }

    // Handle processing status (optional update)
    if (event_type === 'avatar_video.processing') {
      await sql`
        UPDATE heygen_videos
        SET 
          status = 'processing',
          updated_at = NOW()
        WHERE heygen_video_id = ${event_data.video_id}
      `

      return NextResponse.json({
        received: true,
        status: 'processing',
      })
    }

    return NextResponse.json({ received: true })

  } catch (error) {
    console.error('[HeyGen Webhook] Error:', error)
    return NextResponse.json(
      { error: 'Webhook handler failed' },
      { status: 500 }
    )
  }
}

// GET - Health check and status
export async function GET() {
  try {
    // Get recent webhook activity
    const stats = await sql`
      SELECT 
        status,
        COUNT(*) as count,
        MAX(updated_at) as last_update
      FROM heygen_videos
      WHERE updated_at > NOW() - INTERVAL '24 hours'
      GROUP BY status
      ORDER BY count DESC
    `

    return NextResponse.json({
      endpoint: '/api/webhooks/heygen',
      status: 'active',
      description: 'HeyGen video completion webhook',
      last_24h: stats,
      configure_at: 'https://app.heygen.com/settings/api',
    })
  } catch (error) {
    return NextResponse.json({
      endpoint: '/api/webhooks/heygen',
      status: 'error',
      error: String(error),
    })
  }
}
