import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'

/**
 * Pioneer Waitlist API
 * 
 * POST /api/coalition/pioneer
 * Body: { email }
 * 
 * Adds email to pioneer waitlist
 * Handles duplicates silently (ON CONFLICT DO NOTHING)
 * Returns { success: true, message }
 */

export async function POST(request: NextRequest) {
  try {
    const { email } = await request.json()

    // Validation
    if (!email || !email.includes('@')) {
      return NextResponse.json({ 
        error: 'Valid email is required' 
      }, { status: 400 })
    }

    const sql = getSql()
    // Insert into database (duplicate emails are silently ignored)
    await sql`
      INSERT INTO pioneer_waitlist (email)
      VALUES (${email.toLowerCase().trim()})
      ON CONFLICT (email) DO NOTHING
    `

    return NextResponse.json({
      success: true,
      email: email.toLowerCase().trim(),
      message: 'Welcome to the Pioneer movement! You\'ll be among the first to join.',
    })
  } catch (error) {
    console.error('Pioneer waitlist error:', error)
    return NextResponse.json({ 
      error: 'Failed to join waitlist' 
    }, { status: 500 })
  }
}

export async function GET() {
  try {
    const sql = getSql()
    // Return pioneer waitlist stats (public endpoint)
    const result = await sql`
      SELECT 
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as today,
        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as this_week
      FROM pioneer_waitlist
    `

    const stats = result[0] || { total: 0, today: 0, this_week: 0 }

    return NextResponse.json({
      total: Number(stats.total),
      today: Number(stats.today || 0),
      this_week: Number(stats.this_week || 0),
    })
  } catch {
    return NextResponse.json({ 
      total: 0, 
      today: 0, 
      this_week: 0 
    })
  }
}
