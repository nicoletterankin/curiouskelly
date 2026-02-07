import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'

// Debug endpoint to see exactly what video data exists
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '18')
  
  try {
    const sql = getSql()
    // Check heygen_videos
    const heygen = await sql`
      SELECT * FROM heygen_videos 
      WHERE day_of_year = ${day} 
      ORDER BY phase
    `
    
    // Check generated_assets for video
    const assets = await sql`
      SELECT * FROM generated_assets 
      WHERE lesson_id LIKE ${'day-' + String(day).padStart(3, '0') + '%'}
      AND asset_type IN ('video', 'audio')
      ORDER BY phase, asset_type
    `
    
    return NextResponse.json({
      day,
      heygen_videos: heygen,
      generated_assets: assets,
      message: heygen.length > 0 
        ? `Found ${heygen.length} video(s) in heygen_videos` 
        : 'No videos found - Antigravity needs to generate them'
    })
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}
