import { NextResponse } from 'next/server'
import { getAllAvatarConfigs } from '@/lib/heygen-client'

export async function GET() {
  try {
    const configs = getAllAvatarConfigs()
    
    // Group by age category
    const grouped = {
      kid: configs.filter(c => c.ageCategory === 'kid'),
      adult: configs.filter(c => c.ageCategory === 'adult'),
      senior: configs.filter(c => c.ageCategory === 'senior'),
    }
    
    return NextResponse.json({
      success: true,
      data: {
        total: configs.length,
        byAge: {
          kid: grouped.kid.length,
          adult: grouped.adult.length,
          senior: grouped.senior.length,
        },
        avatars: configs.map(c => ({
          name: c.avatarName,
          ageCategory: c.ageCategory,
          archetype: c.archetype,
          avatarGroupId: c.avatarGroupId,
          lookId: c.lookId,
        })),
      },
    })
  } catch (error) {
    console.error('[HeyGen] Avatars list error:', error)
    
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Failed to list avatars',
        success: false,
      },
      { status: 500 }
    )
  }
}
