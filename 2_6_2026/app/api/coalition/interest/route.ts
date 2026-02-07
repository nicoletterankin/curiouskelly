import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * Coalition Interest API
 * 
 * POST /api/coalition/interest
 * Body: { name, email, phone?, tier, amount_range?, message? }
 * 
 * Inserts coalition partner interest into database
 * Validates tier and email format
 * Returns { success: true, id, message }
 */

export async function POST(request: NextRequest) {
  try {
    const { name, email, phone, tier, amount_range, message } = await request.json()

    // Validation
    if (!name || !name.trim()) {
      return NextResponse.json({ error: 'Name is required' }, { status: 400 })
    }

    if (!email || !email.includes('@')) {
      return NextResponse.json({ error: 'Valid email is required' }, { status: 400 })
    }

    const validTiers = ['founding', 'anchor', 'builder', 'pioneer']
    if (!tier || !validTiers.includes(tier)) {
      return NextResponse.json({ 
        error: 'Valid tier required: founding, anchor, builder, or pioneer' 
      }, { status: 400 })
    }

    // Insert into database
    const result = await sql`
      INSERT INTO coalition_interest (name, email, phone, tier, amount_range, message)
      VALUES (
        ${name.trim()},
        ${email.toLowerCase().trim()},
        ${phone?.trim() || null},
        ${tier},
        ${amount_range || null},
        ${message?.trim() || null}
      )
      ON CONFLICT (email) DO UPDATE SET
        name = EXCLUDED.name,
        phone = COALESCE(EXCLUDED.phone, coalition_interest.phone),
        tier = EXCLUDED.tier,
        amount_range = COALESCE(EXCLUDED.amount_range, coalition_interest.amount_range),
        message = COALESCE(EXCLUDED.message, coalition_interest.message),
        updated_at = NOW()
      RETURNING id, email, tier, created_at
    `

    const entry = result[0]

    return NextResponse.json({
      success: true,
      id: entry.id,
      email: entry.email,
      tier: entry.tier,
      message: `Thank you! We've received your interest in joining as a ${tier} partner.`,
      created_at: entry.created_at,
    })
  } catch (error) {
    console.error('Coalition interest error:', error)
    return NextResponse.json({ 
      error: 'Failed to submit interest' 
    }, { status: 500 })
  }
}

export async function GET() {
  try {
    // Return aggregate stats (public endpoint)
    const result = await sql`
      SELECT 
        tier,
        COUNT(*) as count,
        SUM(CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 1 ELSE 0 END) as today
      FROM coalition_interest
      GROUP BY tier
      ORDER BY 
        CASE tier 
          WHEN 'founding' THEN 1
          WHEN 'anchor' THEN 2
          WHEN 'builder' THEN 3
          WHEN 'pioneer' THEN 4
        END
    `

    const stats = {
      byTier: result.map(row => ({
        tier: row.tier,
        count: Number(row.count),
        today: Number(row.today || 0),
      })),
      total: result.reduce((sum, row) => sum + Number(row.count), 0),
      today: result.reduce((sum, row) => sum + Number(row.today || 0), 0),
    }

    return NextResponse.json(stats)
  } catch {
    return NextResponse.json({ 
      byTier: [], 
      total: 0, 
      today: 0 
    })
  }
}
