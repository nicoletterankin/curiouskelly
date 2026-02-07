/**
 * HEYGEN ORPHAN VIDEO IDENTIFICATION
 * 
 * Fetches all videos from HeyGen account and helps identify which Kelly variant they are.
 * For orphaned videos generated without proper tracking.
 */

import { getSql } from "@/lib/db"
import { NextResponse } from "next/server"

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

// Kelly avatar ID to (age, archetype) mapping
// This is the SINGLE SOURCE OF TRUTH for Kelly identification
const KELLY_AVATAR_MAP: Record<string, { age: string; archetype: string }> = {
  // Kid variants
  "kid_storyteller_id": { age: "kid", archetype: "storyteller" },
  "kid_scientist_id": { age: "kid", archetype: "scientist" },
  "kid_coach_id": { age: "kid", archetype: "coach" },
  "kid_mystic_id": { age: "kid", archetype: "mystic" },
  "kid_philosopher_id": { age: "kid", archetype: "philosopher" },
  "kid_explorer_id": { age: "kid", archetype: "explorer" },
  "kid_nurturer_id": { age: "kid", archetype: "nurturer" },
  "kid_challenger_id": { age: "kid", archetype: "challenger" },
  "kid_humorist_id": { age: "kid", archetype: "humorist" },
  "kid_sage_id": { age: "kid", archetype: "sage" },
  "kid_innovator_id": { age: "kid", archetype: "innovator" },
  "kid_guardian_id": { age: "kid", archetype: "guardian" },
  
  // Adult variants (example - add all 36)
  "3d6a9d6f91b444469dae87ebb3d9eba6": { age: "adult", archetype: "storyteller" },
  
  // TODO: Add all 36 Kelly avatar IDs from Antigravity
}

export async function GET() {
  try {
    // Fetch all videos from HeyGen
    const response = await fetch("https://api.heygen.com/v1/video.list", {
      headers: { "X-Api-Key": HEYGEN_API_KEY }
    })
    
    if (!response.ok) {
      return NextResponse.json({ error: "HeyGen API error", status: response.status }, { status: 500 })
    }
    
    const data = await response.json()
    const videos = data.data?.videos || []
    
    const sql = getSql()
    // Check which videos are already tracked in heygen_videos table
    const trackedIds = await sql`
      SELECT heygen_video_id FROM heygen_videos WHERE heygen_video_id IS NOT NULL
    `.catch(() => [])
    const trackedSet = new Set(trackedIds.map((r: any) => r.heygen_video_id))
    
    // Categorize videos
    const orphans: any[] = []
    const tracked: any[] = []
    
    for (const video of videos) {
      const isTracked = trackedSet.has(video.video_id)
      const avatarInfo = KELLY_AVATAR_MAP[video.talking_photo_id] || null
      
      const videoInfo = {
        video_id: video.video_id,
        status: video.status,
        video_url: video.video_url,
        thumbnail_url: video.thumbnail_url,
        duration: video.duration,
        created_at: video.created_at,
        talking_photo_id: video.talking_photo_id,
        identified_kelly: avatarInfo,
        is_tracked: isTracked
      }
      
      if (isTracked) {
        tracked.push(videoInfo)
      } else {
        orphans.push(videoInfo)
      }
    }
    
    return NextResponse.json({
      summary: {
        total_in_heygen: videos.length,
        tracked: tracked.length,
        orphans: orphans.length,
        identified_orphans: orphans.filter(v => v.identified_kelly).length,
        unidentified_orphans: orphans.filter(v => !v.identified_kelly).length
      },
      orphans,
      tracked
    })
    
  } catch (error) {
    console.error("[v0] Error fetching HeyGen videos:", error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

// POST: Save identified orphan to database
export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { video_id, video_url, day, phase, age, archetype } = body
    
    if (!video_id || !day || !phase || !age || !archetype) {
      return NextResponse.json({ 
        error: "Missing required fields: video_id, day, phase, age, archetype" 
      }, { status: 400 })
    }
    
    const sql = getSql()
    // Save to heygen_videos table
    await sql`
      INSERT INTO heygen_videos (
        day_of_year, phase, age_category, archetype, 
        heygen_video_id, video_url, status, completed_at
      ) VALUES (
        ${day}, ${phase}, ${age}, ${archetype},
        ${video_id}, ${video_url}, 'completed', NOW()
      )
      ON CONFLICT (day_of_year, phase, age_category, archetype) DO UPDATE SET
        video_url = EXCLUDED.video_url,
        heygen_video_id = EXCLUDED.heygen_video_id,
        status = 'completed',
        completed_at = NOW()
    `.catch(() => {})
    
    // Also update kelly_lesson_assets for v0 app to display
    await sql`
      INSERT INTO kelly_lesson_assets (
        id, day_number, phase, age_group, archetype, language,
        video_url, video_id, video_source, status, created_at, updated_at
      ) VALUES (
        gen_random_uuid(), ${day}, ${phase}, ${age}, ${archetype}, 'en',
        ${video_url}, ${video_id}, 'heygen', 'completed', NOW(), NOW()
      )
      ON CONFLICT (day_number, phase, age_group, archetype, language) DO UPDATE SET
        video_url = EXCLUDED.video_url,
        video_id = EXCLUDED.video_id,
        status = 'completed',
        updated_at = NOW()
    `
    
    return NextResponse.json({
      success: true,
      saved: { day, phase, age, archetype, video_id }
    })
    
  } catch (error) {
    console.error("[v0] Error saving orphan:", error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}
