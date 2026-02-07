import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { put } from '@vercel/blob'

// POST: Upload a HeyGen video file and register in database
// This bypasses R2 SSL issues by using Vercel Blob instead
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    const dayOfYear = parseInt(formData.get('day') as string)
    const phase = formData.get('phase') as string
    const language = formData.get('language') as string
    const ageCategory = formData.get('age') as string
    const archetype = formData.get('archetype') as string

    if (!file || !dayOfYear || !phase) {
      return NextResponse.json({ error: 'Missing required fields: file, day, phase' }, { status: 400 })
    }

    // Upload to Vercel Blob (bypasses R2 SSL issues)
    const filename = `heygen/day-${String(dayOfYear).padStart(3, '0')}/${language}/${ageCategory || 'adult'}/${archetype || 'default'}/${phase}.mp4`
    
    const blob = await put(filename, file, {
      access: 'public',
      contentType: 'video/mp4',
    })

    const sql = getSql()
    // Register in database - Note: heygen_videos table does NOT have language column
    await sql`
      INSERT INTO heygen_videos (id, day_of_year, phase, age_category, archetype, status, video_url, created_at, updated_at)
      VALUES (gen_random_uuid(), ${dayOfYear}, ${phase}, ${ageCategory || 'adult'}, ${archetype || 'default'}, 'completed', ${blob.url}, NOW(), NOW())
      ON CONFLICT (day_of_year, phase, age_category, archetype) 
      DO UPDATE SET video_url = ${blob.url}, status = 'completed', updated_at = NOW()
    `

    return NextResponse.json({
      success: true,
      url: blob.url,
      day: dayOfYear,
      phase,
      language,
      age: ageCategory,
      archetype,
    })
  } catch (error) {
    console.error('Upload error:', error)
    return NextResponse.json({ error: 'Upload failed', details: String(error) }, { status: 500 })
  }
}

// GET: Check upload status for a day
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = searchParams.get('day')

  const sql = getSql()
  if (!day) {
    // Return overall stats
    const stats = await sql`
      SELECT 
        status, 
        COUNT(*) as count,
        COUNT(DISTINCT day_of_year) as days
      FROM heygen_videos 
      GROUP BY status
    `
    return NextResponse.json({ stats })
  }

  // Return status for specific day
  const dayVideos = await sql`
    SELECT phase, age_category, archetype, status, video_url
    FROM heygen_videos 
    WHERE day_of_year = ${parseInt(day)}
    ORDER BY age_category, archetype, phase
  `

  return NextResponse.json({ 
    day: parseInt(day),
    videos: dayVideos,
    total: dayVideos.length,
    completed: dayVideos.filter((v: Record<string, unknown>) => v.status === 'completed').length
  })
}
