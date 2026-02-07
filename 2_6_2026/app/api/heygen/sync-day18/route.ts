import { NextResponse } from "next/server"
import { sql } from '@/lib/db'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY

// Video IDs that need updating (from database query)
const VIDEO_IDS = [
  "0b376958108947e9befbc07c221aa94c",
  "0cc3eef6c2e040bda5cf4507b57858db",
  "1eadb862da0a47d8b9d9d9952d6f8855",
  "2902689e59342c7a5b2a1443905e70a",
  "42eab18b45344710bec361e7ef65194d",
  "51dea0cab4134fd880c500f2074c573e",
  "546fe85cf7684e8b9e20340ee2a5ea14",
  "6466de612d1a4b51905087af300d9790",
  "727985baefd243158a700b2b1a0b2f83",
  "74237b19a3c84c2f9a97cebc59d1b70d",
  "a31aaa9ab87a486396b30ae9db19b10a",
  "a54ee7f61f94ff0ba49afa7ee402cc2",
  "b790be6057b43dda097b0ae4284ee29",
  "c412bedcda4149dfa4f8ef218b6d2e85",
]

async function getHeyGenVideoUrl(videoId: string): Promise<string | null> {
  try {
    const response = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
      {
        headers: { "X-Api-Key": HEYGEN_API_KEY! },
      }
    )

    if (!response.ok) {
      console.log(`[v0] HeyGen API error for ${videoId}: ${response.status}`)
      return null
    }

    const data = await response.json()
    console.log(`[v0] Video ${videoId}: status=${data.data?.status}`)
    return data.data?.video_url || null
  } catch (error) {
    console.error(`[v0] Error fetching ${videoId}:`, error)
    return null
  }
}

export async function POST() {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: "Missing HEYGEN_API_KEY" }, { status: 500 })
  }

  const results: { video_id: string; success: boolean; url?: string }[] = []

  for (const videoId of VIDEO_IDS) {
    console.log(`[v0] Processing ${videoId}...`)
    
    const realUrl = await getHeyGenVideoUrl(videoId)

    if (realUrl) {
      try {
        await sql`
          UPDATE kelly_lesson_assets
          SET video_url = ${realUrl}, updated_at = NOW()
          WHERE video_id = ${videoId}
        `
        results.push({ video_id: videoId, success: true, url: realUrl.substring(0, 60) })
      } catch (dbError) {
        console.error(`[v0] DB error for ${videoId}:`, dbError)
        results.push({ video_id: videoId, success: false })
      }
    } else {
      results.push({ video_id: videoId, success: false })
    }

    // Rate limit - 300ms between calls
    await new Promise((resolve) => setTimeout(resolve, 300))
  }

  const updated = results.filter((r) => r.success).length
  const failed = results.filter((r) => !r.success).length

  return NextResponse.json({
    message: `Updated ${updated}/${VIDEO_IDS.length} videos`,
    updated,
    failed,
    results,
  })
}

export async function GET() {
  return NextResponse.json({
    message: "POST to this endpoint to sync all 14 Day 18 HeyGen videos",
    video_count: VIDEO_IDS.length,
    video_ids: VIDEO_IDS,
  })
}
