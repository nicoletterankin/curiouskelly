import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * GET /api/affiliate/leaderboard
 * Returns top 10 affiliates who opted into the leaderboard
 */
export async function GET() {
  try {
    const leaderboard = await sql`
      SELECT 
        a.id,
        a.display_name,
        a.total_referrals,
        a.tier,
        a.is_language_guardian,
        t.badge_icon,
        t.badge_color
      FROM affiliates a
      LEFT JOIN affiliate_tiers t ON LOWER(a.tier) = LOWER(t.name)
      WHERE a.show_on_leaderboard = true
        AND a.total_referrals > 0
      ORDER BY a.total_referrals DESC
      LIMIT 10
    `

    const formattedLeaderboard = leaderboard.map((entry, index) => ({
      rank: index + 1,
      displayName: entry.display_name || `Affiliate #${entry.id}`,
      referralCount: entry.total_referrals || 0,
      tier: entry.tier || 'Starter',
      isLanguageGuardian: entry.is_language_guardian || false,
      badgeIcon: entry.badge_icon,
      badgeColor: entry.badge_color,
    }))

    return NextResponse.json({ 
      leaderboard: formattedLeaderboard,
      updatedAt: new Date().toISOString(),
    })

  } catch (error) {
    console.error('[affiliate/leaderboard] Error:', error)
    
    // Return mock data if DB fails
    return NextResponse.json({ 
      leaderboard: [
        { rank: 1, displayName: 'Top Affiliate', referralCount: 50, tier: 'Legend', isLanguageGuardian: true },
        { rank: 2, displayName: 'Rising Star', referralCount: 35, tier: 'Champion', isLanguageGuardian: false },
        { rank: 3, displayName: 'Community Hero', referralCount: 28, tier: 'Champion', isLanguageGuardian: true },
      ],
      updatedAt: new Date().toISOString(),
      isMock: true,
    })
  }
}
