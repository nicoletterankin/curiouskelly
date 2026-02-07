import { NextResponse } from 'next/server'
import { getCurrentUser } from '@/lib/auth'

export async function GET() {
  try {
    const user = await getCurrentUser()
    
    if (!user) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      )
    }
    
    return NextResponse.json({
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        age: user.age,
        language: user.language,
        archetype: user.archetype,
        dayOfYear: user.dayOfYear,
        subscriptionStatus: user.subscriptionStatus,
      }
    })
    
  } catch (error) {
    console.error('[auth/me] Error:', error)
    return NextResponse.json(
      { error: 'Failed to get user' },
      { status: 500 }
    )
  }
}
