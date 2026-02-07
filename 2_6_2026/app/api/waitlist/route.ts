import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

export async function POST(request: NextRequest) {
  try {
    const { email, name, language, source, referrerId } = await request.json()
    
    if (!email || !email.includes('@')) {
      return NextResponse.json({ error: 'Valid email required' }, { status: 400 })
    }
    
    const result = await sql`
      INSERT INTO waitlist (email, name, language, source, referrer_id)
      VALUES (
        ${email.toLowerCase().trim()},
        ${name || null},
        ${language || 'en'},
        ${source || 'organic'},
        ${referrerId || null}
      )
      ON CONFLICT (email) DO UPDATE SET
        name = COALESCE(EXCLUDED.name, waitlist.name),
        language = COALESCE(EXCLUDED.language, waitlist.language)
      RETURNING id, email, created_at
    `
    
    const entry = result[0]
    const referralLink = `https://curiouskelly.com?ref=${entry.id}`
    
    return NextResponse.json({
      success: true,
      message: "You're on the list!",
      position: await getPosition(entry.id),
      referralLink
    })
  } catch (error) {
    console.error('Waitlist error:', error)
    return NextResponse.json({ error: 'Something went wrong' }, { status: 500 })
  }
}

async function getPosition(id: string): Promise<number> {
  const result = await sql`
    SELECT COUNT(*) as position FROM waitlist 
    WHERE created_at <= (SELECT created_at FROM waitlist WHERE id = ${id})
  `
  return Number(result[0]?.position || 1)
}

export async function GET() {
  try {
    const result = await sql`
      SELECT COUNT(*) as total,
        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as today
      FROM waitlist
    `
    return NextResponse.json({
      total: Number(result[0]?.total || 0),
      today: Number(result[0]?.today || 0)
    })
  } catch {
    return NextResponse.json({ total: 0, today: 0 })
  }
}
