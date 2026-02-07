/**
 * PIPELINE STATUS & ORCHESTRATION ENDPOINT
 * 
 * Shows complete status of video pipeline for any day.
 * 
 * GET /api/pipeline/status?day=18
 * 
 * Returns:
 * - Audio generation progress (ElevenLabs)
 * - Video generation progress per engine (HeyGen, Fal, Sync)
 * - Which combinations are missing
 * - Estimated time to completion
 */

import { sql } from '@/lib/db'

const AGES = ['baby', 'kid', 'teen', 'adult', 'elder']
const ARCHETYPES = ['storyteller', 'explorer', 'scientist', 'philosopher', 'coach', 'mystic']
const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom']

const TOTAL_PER_DAY = AGES.length * ARCHETYPES.length * PHASES.length // 150

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const day = searchParams.get('day')
  
  // Overall pipeline stats
  let audioStats
  let videoStats
  
  if (day) {
    const dayNum = parseInt(day)
    audioStats = await sql`
      SELECT day_number, status, COUNT(*)::int as count
      FROM kelly_audio
      WHERE day_number = ${dayNum}
      GROUP BY day_number, status
      ORDER BY day_number, status
    `.catch(() => [])
    videoStats = await sql`
      SELECT day_number, video_engine, status, COUNT(*)::int as count
      FROM kelly_video_jobs
      WHERE day_number = ${dayNum}
      GROUP BY day_number, video_engine, status
      ORDER BY day_number, video_engine, status
    `.catch(() => [])
  } else {
    audioStats = await sql`
      SELECT status, COUNT(*)::int as count
      FROM kelly_audio
      GROUP BY status
      ORDER BY status
    `.catch(() => [])
    videoStats = await sql`
      SELECT video_engine, status, COUNT(*)::int as count
      FROM kelly_video_jobs
      GROUP BY video_engine, status
      ORDER BY video_engine, status
    `.catch(() => [])
  }
  
  // If specific day requested, show detailed breakdown
  let detailed = null
  if (day) {
    const dayNum = parseInt(day)
    
    // Get lesson info
    const lesson = await sql`
      SELECT day_of_year, title FROM lessons WHERE day_of_year = ${dayNum}
    `.catch(() => [])
    
    // Get completed audio
    const completedAudio = await sql`
      SELECT phase, age_group, archetype, audio_url
      FROM kelly_audio
      WHERE day_number = ${dayNum} AND status = 'completed'
    `.catch(() => [])
    
    // Get completed videos
    const completedVideos = await sql`
      SELECT phase, age_group, archetype, video_engine, video_url
      FROM kelly_video_jobs
      WHERE day_number = ${dayNum} AND status = 'completed'
    `.catch(() => [])
    
    // Build matrix of what's done
    const audioSet = new Set(completedAudio.map(a => `${a.phase}-${a.age_group}-${a.archetype}`))
    const videoSets: Record<string, Set<string>> = {
      heygen: new Set(),
      fal: new Set(),
      sync: new Set(),
    }
    for (const v of completedVideos) {
      videoSets[v.video_engine]?.add(`${v.phase}-${v.age_group}-${v.archetype}`)
    }
    
    // Find what's missing
    const missingAudio: string[] = []
    const missingVideo: { combo: string; engines: string[] }[] = []
    
    for (const phase of PHASES) {
      for (const age of AGES) {
        for (const archetype of ARCHETYPES) {
          const key = `${phase}-${age}-${archetype}`
          
          if (!audioSet.has(key)) {
            missingAudio.push(key)
          }
          
          const missingEngines = []
          if (!videoSets.heygen.has(key)) missingEngines.push('heygen')
          if (!videoSets.fal.has(key)) missingEngines.push('fal')
          if (!videoSets.sync.has(key)) missingEngines.push('sync')
          
          if (missingEngines.length === 3) {
            missingVideo.push({ combo: key, engines: missingEngines })
          }
        }
      }
    }
    
    detailed = {
      day: dayNum,
      lesson: lesson[0] || null,
      audio: {
        completed: completedAudio.length,
        total: TOTAL_PER_DAY,
        percent: Math.round((completedAudio.length / TOTAL_PER_DAY) * 100),
        missing_count: missingAudio.length,
        missing_samples: missingAudio.slice(0, 10),
      },
      video: {
        heygen: {
          completed: videoSets.heygen.size,
          total: TOTAL_PER_DAY,
          percent: Math.round((videoSets.heygen.size / TOTAL_PER_DAY) * 100),
        },
        fal: {
          completed: videoSets.fal.size,
          total: TOTAL_PER_DAY,
          percent: Math.round((videoSets.fal.size / TOTAL_PER_DAY) * 100),
        },
        sync: {
          completed: videoSets.sync.size,
          total: TOTAL_PER_DAY,
          percent: Math.round((videoSets.sync.size / TOTAL_PER_DAY) * 100),
        },
      },
      missing_video_count: missingVideo.length,
      missing_video_samples: missingVideo.slice(0, 10),
    }
  }
  
  // Summary across all days
  const summary = {
    audio: {
      by_status: audioStats.reduce((acc, row) => {
        acc[row.status] = (acc[row.status] || 0) + row.count
        return acc
      }, {} as Record<string, number>),
    },
    video: {
      by_engine_status: videoStats,
    },
    config: {
      ages: AGES,
      archetypes: ARCHETYPES,
      phases: PHASES,
      combinations_per_day: TOTAL_PER_DAY,
      total_days: 365,
      total_combinations: TOTAL_PER_DAY * 365,
    },
  }
  
  return Response.json({
    pipeline: 'Curious Kelly Video Pipeline',
    summary,
    detailed,
    endpoints: {
      generate_audio: 'POST /api/audio/generate { day, batch }',
      generate_video_heygen: 'POST /api/heygen/generate-day { day, batch }',
      poll_heygen: 'POST /api/heygen/poll-status { limit }',
      identify_orphans: 'GET /api/heygen/identify-orphans',
    },
  })
}
