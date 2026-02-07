import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { sql } from '@/lib/db'

// Default guest profile
const GUEST_PROFILE = {
  id: null,
  name: null,
  email: null,
  birthday: null,
  preferred_timezone: null,
  preferred_language: 'en',
  preferred_avatar_age: 30,
  preferred_archetype: 'storyteller',
  subscription_status: null,
}

export async function GET() {
  try {
    // Get session from cookie (simplified - in production use proper auth)
    const cookieStore = await cookies()
    const sessionToken = cookieStore.get('session_token')?.value

    // Return guest profile if no session
    if (!sessionToken) {
      return NextResponse.json(GUEST_PROFILE)
    }

    // Fetch user from session
    const sessions = await sql`
      SELECT u.id, u.name, u.email, u.image, s.token
      FROM neon_auth.session s
      JOIN neon_auth.user u ON s."userId" = u.id
      WHERE s.token = ${sessionToken}
        AND s."expiresAt" > NOW()
      LIMIT 1
    `

    if (sessions.length === 0) {
      return NextResponse.json(GUEST_PROFILE)
    }

    const user = sessions[0]

    // Fetch user preferences from user_perspective_preferences
    const preferences = await sql`
      SELECT 
        preferred_avatar_age,
        preferred_language,
        preferred_detail_level,
        days_completed
      FROM user_perspective_preferences
      WHERE user_id = ${user.id}
      LIMIT 1
    `
    const prefs = preferences[0] || {}

    // Fetch subscription status
    const subscribers = await sql`
      SELECT subscription_status, subscription_tier
      FROM subscribers
      WHERE email = ${user.email}
      LIMIT 1
    `
    const subscription = subscribers[0] || {}

    // Fetch age verification for birthday (if available)
    const ageVerification = await sql`
      SELECT stated_age, created_at
      FROM age_verifications
      WHERE user_id = ${user.id}
        AND verified = true
      ORDER BY created_at DESC
      LIMIT 1
    `
    
    // Calculate approximate birthday from stated age if available
    let birthday = null
    if (ageVerification.length > 0 && ageVerification[0].stated_age) {
      const verifiedAt = new Date(ageVerification[0].created_at)
      const birthYear = verifiedAt.getFullYear() - ageVerification[0].stated_age
      // We don't know exact birthday, so we won't set it
      // Birthday feature requires explicit user input
      birthday = null
    }

    return NextResponse.json({
      id: user.id,
      name: user.name,
      email: user.email,
      birthday,
      preferred_timezone: null, // Can be added to user_perspective_preferences
      preferred_language: prefs.preferred_language || 'en',
      preferred_avatar_age: prefs.preferred_avatar_age || 30,
      preferred_archetype: 'storyteller', // Default - can be added to prefs table
      subscription_status: subscription.subscription_status || null,
    })
  } catch (error) {
    console.error('Error fetching user profile:', error)
    // Return guest context on error
    return NextResponse.json(GUEST_PROFILE)
  }
}
