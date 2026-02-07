import { NextResponse } from "next/server"
import { sql } from '@/lib/db'

// Map HeyGen avatar IDs to our age/archetype combos
// These need to be filled in with actual HeyGen talking_photo_ids
const AVATAR_MAP: Record<string, { age: string; archetype: string }> = {
  // Kid avatars (the ones with blue dot/mystic look)
  // Adult avatars (professional Kelly)
  // Elder avatars (wise Kelly)
  // We'll identify these from the videos
}

export async function GET() {
  const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY
  
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: "HEYGEN_API_KEY not configured" }, { status: 500 })
  }

  try {
    // Fetch all videos from HeyGen
    const response = await fetch("https://api.heygen.com/v1/video.list", {
      headers: {
        "X-Api-Key": HEYGEN_API_KEY,
      },
    })

    if (!response.ok) {
      const error = await response.text()
      return NextResponse.json({ error: `HeyGen API error: ${error}` }, { status: response.status })
    }

    const data = await response.json()
    const videos = data.data?.videos || []

    // Get existing videos in our database
    const existing = await sql`SELECT heygen_video_id FROM heygen_videos WHERE heygen_video_id IS NOT NULL`
    const existingIds = new Set(existing.map((r: { heygen_video_id: string }) => r.heygen_video_id))

    // Separate into synced and unsynced
    const synced = videos.filter((v: { video_id: string }) => existingIds.has(v.video_id))
    const unsynced = videos.filter((v: { video_id: string }) => !existingIds.has(v.video_id))

    // For each unsynced video, get full details including video_url
    const unsyncedWithUrls = await Promise.all(
      unsynced.slice(0, 20).map(async (v: { video_id: string; status: string }) => {
        try {
          const detailRes = await fetch(
            `https://api.heygen.com/v1/video_status.get?video_id=${v.video_id}`,
            { headers: { "X-Api-Key": HEYGEN_API_KEY } }
          )
          const detail = await detailRes.json()
          return {
            video_id: v.video_id,
            status: detail.data?.status || v.status,
            video_url: detail.data?.video_url || null,
            thumbnail_url: detail.data?.thumbnail_url || null,
            duration: detail.data?.duration || null,
          }
        } catch {
          return { video_id: v.video_id, status: v.status, video_url: null }
        }
      })
    )

    return NextResponse.json({
      total: videos.length,
      synced: synced.length,
      unsynced: unsynced.length,
      unsyncedVideos: unsyncedWithUrls,
    })
  } catch (error) {
    console.error("Error fetching HeyGen videos:", error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

export async function POST(request: Request) {
  const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY
  
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: "HEYGEN_API_KEY not configured" }, { status: 500 })
  }

  try {
    const body = await request.json()
    const { mappings } = body as { 
      mappings: Array<{
        video_id: string
        video_url: string
        day: number
        phase: string
        age: string
        archetype: string
      }>
    }

    if (!mappings || !Array.isArray(mappings)) {
      return NextResponse.json({ error: "mappings array required" }, { status: 400 })
    }

    const results = []

    for (const mapping of mappings) {
      const { video_id, video_url, day, phase, age, archetype } = mapping

      // Insert into heygen_videos
      await sql`
        INSERT INTO heygen_videos (
          day_of_year, phase, age_category, archetype,
          heygen_video_id, video_url, status, completed_at
        ) VALUES (
          ${day}, ${phase}, ${age}, ${archetype},
          ${video_id}, ${video_url}, 'completed', NOW()
        )
        ON CONFLICT (day_of_year, phase, age_category, archetype) 
        DO UPDATE SET 
          video_url = ${video_url},
          heygen_video_id = ${video_id},
          status = 'completed',
          completed_at = NOW()
      `

      // Also insert into kelly_lesson_assets for the app to find
      await sql`
        INSERT INTO kelly_lesson_assets (
          id, day_number, phase, age_group, archetype, language,
          video_url, video_id, video_source, status, created_at, updated_at
        ) VALUES (
          gen_random_uuid(), ${day}, ${phase}, ${age}, ${archetype}, 'en',
          ${video_url}, ${video_id}, 'heygen', 'completed', NOW(), NOW()
        )
        ON CONFLICT (day_number, phase, age_group, archetype, language)
        DO UPDATE SET 
          video_url = ${video_url},
          video_id = ${video_id},
          status = 'completed',
          updated_at = NOW()
      `

      results.push({ video_id, day, phase, age, archetype, status: "saved" })
    }

    return NextResponse.json({
      saved: results.length,
      results,
    })
  } catch (error) {
    console.error("Error saving video mappings:", error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}
