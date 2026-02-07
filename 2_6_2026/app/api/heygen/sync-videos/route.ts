import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY
const HEYGEN_API_BASE = 'https://api.heygen.com'

// Map HeyGen talking_photo_id back to age/archetype
const AVATAR_REVERSE_MAP: Record<string, { age: string; archetype: string }> = {
  // Kid avatars
  '1024bc304a1146998bc4c360173b2c48': { age: 'kid', archetype: 'storyteller' },
  'fa4a6780e25a49699ee4f75cb1f03103': { age: 'kid', archetype: 'explorer' },
  '82813816115c4fbe93b3f3f211bd9931': { age: 'kid', archetype: 'scientist' },
  'cc1dd0e9e2fd432099985c9b036ed836': { age: 'kid', archetype: 'architect' },
  '6249632f58ce479891de00b4da5fb88d': { age: 'kid', archetype: 'strategist' },
  '48bddc41ae94473caa645ce9ab93136d': { age: 'kid', archetype: 'diplomat' },
  '5cff601bfb344015a65ff46c6b8cd70a': { age: 'kid', archetype: 'mystic' },
  'd4e960f7a3424d869877f3a951adfae7': { age: 'kid', archetype: 'rebel' },
  '7b6ab196f2c7430b945411df51a84c58': { age: 'kid', archetype: 'macgyver' },
  'deeb27f2648848b48c5c1ce59059bd54': { age: 'kid', archetype: 'empath' },
  'deaa213342944dc2bf671abe1442e316': { age: 'kid', archetype: 'provider' },
  'bd579e4ca77444aca2bfea8ee9070830': { age: 'kid', archetype: 'survivor' },
  // Adult avatars  
  '3d6a9d6f91b444469dae87ebb3d9eba6': { age: 'adult', archetype: 'storyteller' },
  '62516885ca4b4eae8f63b87b8c060e25': { age: 'adult', archetype: 'explorer' },
  '277aba5b86a14ff2a4eca2eab2402ab3': { age: 'adult', archetype: 'scientist' },
  '35d0115505824e3182eb9d2ee8cfe73d': { age: 'adult', archetype: 'architect' },
  '08d53d1b065041bda2e5b6bc32962a8a': { age: 'adult', archetype: 'strategist' },
  'c3cdbe48fe274420a7f45a4da7e366aa': { age: 'adult', archetype: 'diplomat' },
  'dfaf9fbd644a475595b178f0be65a39a': { age: 'adult', archetype: 'mystic' },
  '390be3fb2b064883bb2304fc3968fd87': { age: 'adult', archetype: 'rebel' },
  'c5aab6ab13d940f8ae4700d546bd6b6b': { age: 'adult', archetype: 'macgyver' },
  '6bb1a05678c64213a1ed3a4dc790b81e': { age: 'adult', archetype: 'empath' },
  '9a143feeb2994989b034cebeb78753be': { age: 'adult', archetype: 'provider' },
  '831c8d6048104ba0b03a74a36543cfb9': { age: 'adult', archetype: 'survivor' },
  // Senior avatars
  '98178c87897e4421884b535b7864ba86': { age: 'senior', archetype: 'storyteller' },
  'c38e30f2a3cf4e81b0365abf41579f22': { age: 'senior', archetype: 'explorer' },
  '97e1c9dc1ed04e8fa357c69bde34e58e': { age: 'senior', archetype: 'scientist' },
  '42e9197ab9d84961915b00d5cc780190': { age: 'senior', archetype: 'architect' },
  'e4ab0d4d1f1b4dc9b81a1076b018557f': { age: 'senior', archetype: 'strategist' },
  'a82183881e284e3782db75b755c3f080': { age: 'senior', archetype: 'diplomat' },
  'c6d104b2ca354b0a9593cb840988bf6e': { age: 'senior', archetype: 'mystic' },
  'dc835263eaa247f5b0e06106b848df18': { age: 'senior', archetype: 'rebel' },
  'cb5b025506284d64b696e296ca2feead': { age: 'senior', archetype: 'macgyver' },
  '493dac2cf2ba4509b3cc048ff819765e': { age: 'senior', archetype: 'empath' },
  '12582467e9ff48889d7b2435642e2d65': { age: 'senior', archetype: 'provider' },
}

interface HeyGenVideo {
  video_id: string
  status: string
  video_url?: string
  thumbnail_url?: string
  duration?: number
  created_at?: number
}

async function fetchRecentVideos(): Promise<HeyGenVideo[]> {
  const res = await fetch(`${HEYGEN_API_BASE}/v1/video.list?limit=100`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY! }
  })
  
  if (!res.ok) {
    throw new Error(`HeyGen API error: ${res.status}`)
  }
  
  const data = await res.json()
  return data.data?.videos || []
}

async function getVideoDetails(videoId: string): Promise<HeyGenVideo | null> {
  const res = await fetch(`${HEYGEN_API_BASE}/v1/video_status.get?video_id=${videoId}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY! }
  })
  
  if (!res.ok) return null
  
  const data = await res.json()
  return data.data
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action')
  
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }
  
  try {
    // Fetch all recent videos from HeyGen
    const videos = await fetchRecentVideos()
    
    // Filter to completed videos
    const completedVideos = videos.filter(v => v.status === 'completed' && v.video_url)
    
    if (action === 'save') {
      // Save completed videos to database
      let saved = 0
      let skipped = 0
      const errors: string[] = []
      
      for (const video of completedVideos) {
        try {
          // Try to get more details
          const details = await getVideoDetails(video.video_id)
          
          // For now, save to kelly_lesson_assets with day 18
          // We'll need metadata to determine the exact phase/age/archetype
          // Check if already exists
          const existing = await sql`
            SELECT id FROM kelly_lesson_assets WHERE video_id = ${video.video_id}
          `
          
          if (existing.length === 0) {
            await sql`
              INSERT INTO kelly_lesson_assets (
                id, day_number, phase, age_group, archetype, language,
                video_url, video_id, video_source, status, created_at, updated_at
              ) VALUES (
                gen_random_uuid(),
                18,
                'hook',
                'adult',
                'storyteller',
                'en',
                ${video.video_url},
                ${video.video_id},
                'heygen',
                'completed',
                NOW(),
                NOW()
              )
            `
          } else {
            await sql`
              UPDATE kelly_lesson_assets 
              SET video_url = ${video.video_url}, status = 'completed', updated_at = NOW()
              WHERE video_id = ${video.video_id}
            `
          }
          saved++
        } catch (e) {
          errors.push(`${video.video_id}: ${e instanceof Error ? e.message : 'unknown error'}`)
          skipped++
        }
      }
      
      return NextResponse.json({
        success: true,
        message: `Synced ${saved} videos, skipped ${skipped}`,
        saved,
        skipped,
        errors: errors.slice(0, 10)
      })
    }
    
    // Default: just list videos
    return NextResponse.json({
      success: true,
      total: videos.length,
      completed: completedVideos.length,
      pending: videos.filter(v => v.status === 'pending').length,
      processing: videos.filter(v => v.status === 'processing').length,
      failed: videos.filter(v => v.status === 'failed').length,
      videos: completedVideos.slice(0, 20).map(v => ({
        id: v.video_id,
        url: v.video_url,
        status: v.status,
        duration: v.duration
      })),
      usage: {
        list: '/api/heygen/sync-videos',
        save: '/api/heygen/sync-videos?action=save'
      }
    })
  } catch (error) {
    return NextResponse.json({
      error: error instanceof Error ? error.message : 'Unknown error',
      success: false
    }, { status: 500 })
  }
}
