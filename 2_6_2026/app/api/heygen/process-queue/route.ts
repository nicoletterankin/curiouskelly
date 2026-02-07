import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY

// Kelly's avatar and voice configurations by language
const KELLY_CONFIG = {
  en: { avatar_id: process.env.KELLY_AVATAR_ID || 'Kristin_public_2_20240108', voice_id: 'en-US-AvaNeural' },
  es: { avatar_id: process.env.KELLY_AVATAR_ID || 'Kristin_public_2_20240108', voice_id: 'es-ES-ElviraNeural' },
  fr: { avatar_id: process.env.KELLY_AVATAR_ID || 'Kristin_public_2_20240108', voice_id: 'fr-FR-DeniseNeural' },
  de: { avatar_id: process.env.KELLY_AVATAR_ID || 'Kristin_public_2_20240108', voice_id: 'de-DE-KatjaNeural' },
  pt: { avatar_id: process.env.KELLY_AVATAR_ID || 'Kristin_public_2_20240108', voice_id: 'pt-BR-FranciscaNeural' },
  zh: { avatar_id: process.env.KELLY_AVATAR_ID || 'Kristin_public_2_20240108', voice_id: 'zh-CN-XiaoxiaoNeural' },
}

// Process queued videos and submit to HeyGen
export async function POST(request: NextRequest) {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }

  try {
    const body = await request.json().catch(() => ({}))
    const batchSize = Math.min(body.batchSize || 10, 50) // Process up to 50 at a time
    const dryRun = body.dryRun === true

    // Get queued videos
    const queued = await sql`
      SELECT hv.id, hv.day_of_year, hv.phase, hv.language, hv.age_category, hv.archetype,
             l.title, 
             CASE hv.phase 
               WHEN 'hook' THEN COALESCE(lt.hook_script, l.hook_script)
               WHEN 'story' THEN COALESCE(lt.story_script, l.story_script)
               WHEN 'wonder' THEN COALESCE(lt.wonder_script, l.wonder_script)
               WHEN 'action' THEN COALESCE(lt.action_script, l.action_script)
               WHEN 'wisdom' THEN COALESCE(lt.wisdom_script, l.wisdom_script)
             END as script
      FROM heygen_videos hv
      JOIN lessons l ON l.day_of_year = hv.day_of_year
      LEFT JOIN lesson_translations lt ON lt.day_of_year = hv.day_of_year AND lt.language_code = hv.language
      WHERE hv.status = 'queued'
      ORDER BY hv.day_of_year ASC, hv.phase ASC
      LIMIT ${batchSize}
    `

    if (queued.length === 0) {
      return NextResponse.json({ message: 'No queued videos to process', processed: 0 })
    }

    if (dryRun) {
      return NextResponse.json({
        dryRun: true,
        wouldProcess: queued.length,
        videos: queued.map(v => ({
          day: v.day_of_year,
          phase: v.phase,
          language: v.language,
          script: v.script?.substring(0, 50) + '...'
        }))
      })
    }

    const results = { submitted: 0, failed: 0, errors: [] as string[] }

    for (const video of queued) {
      if (!video.script) {
        results.failed++
        results.errors.push(`Day ${video.day_of_year} ${video.phase}: No script found`)
        continue
      }

      const config = KELLY_CONFIG[video.language as keyof typeof KELLY_CONFIG] || KELLY_CONFIG.en

      try {
        // Submit to HeyGen
        const heygenResponse = await fetch('https://api.heygen.com/v2/video/generate', {
          method: 'POST',
          headers: {
            'X-Api-Key': HEYGEN_API_KEY,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            video_inputs: [{
              character: {
                type: 'avatar',
                avatar_id: config.avatar_id,
                avatar_style: 'normal',
              },
              voice: {
                type: 'text',
                input_text: video.script,
                voice_id: config.voice_id,
                speed: 1.0,
              },
              background: {
                type: 'color',
                value: '#000000',
              },
            }],
            dimension: { width: 1080, height: 1920 }, // Portrait for mobile
            aspect_ratio: '9:16',
            test: false,
          }),
        })

        const heygenData = await heygenResponse.json()

        if (heygenData.data?.video_id) {
          // Update status to processing
          await sql`
            UPDATE heygen_videos 
            SET status = 'processing', 
                heygen_video_id = ${heygenData.data.video_id},
                updated_at = NOW()
            WHERE id = ${video.id}
          `
          results.submitted++
        } else {
          results.failed++
          results.errors.push(`Day ${video.day_of_year} ${video.phase}: ${heygenData.error?.message || 'Unknown error'}`)
          
          // Mark as failed
          await sql`
            UPDATE heygen_videos 
            SET status = 'failed', 
                error_message = ${heygenData.error?.message || 'Unknown error'},
                updated_at = NOW()
            WHERE id = ${video.id}
          `
        }

        // Small delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 500))

      } catch (err) {
        results.failed++
        results.errors.push(`Day ${video.day_of_year} ${video.phase}: ${err instanceof Error ? err.message : 'Network error'}`)
      }
    }

    // Get remaining queue count
    const remaining = await sql`SELECT COUNT(*) as count FROM heygen_videos WHERE status = 'queued'`

    return NextResponse.json({
      processed: results.submitted + results.failed,
      submitted: results.submitted,
      failed: results.failed,
      errors: results.errors.slice(0, 10),
      remainingInQueue: Number(remaining[0]?.count || 0)
    })

  } catch (error) {
    console.error('Process queue error:', error)
    return NextResponse.json({ error: 'Failed to process queue' }, { status: 500 })
  }
}

// GET: Show queue status
export async function GET() {
  const stats = await sql`
    SELECT 
      status, 
      language,
      COUNT(*) as count 
    FROM heygen_videos 
    GROUP BY status, language 
    ORDER BY status, language
  `

  const total = await sql`SELECT COUNT(*) as total FROM heygen_videos`

  return NextResponse.json({
    total: Number(total[0]?.total || 0),
    byStatusAndLanguage: stats
  })
}
