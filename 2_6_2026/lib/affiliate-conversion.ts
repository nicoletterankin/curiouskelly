// lib/affiliate-conversion.ts
// Track and calculate affiliate earnings from conversions

interface ConversionData {
  refCode: string
  amountCents: number
  tierId: string
  customerId: string
  timestamp?: Date
}

export async function trackAffiliateConversion(data: ConversionData): Promise<{
  success: boolean
  commissionCents?: number
  error?: string
}> {
  const { refCode, amountCents, tierId, customerId, timestamp = new Date() } = data

  try {
    // TODO: Query affiliate_programs to get affiliate_id
    // const affiliate = await db.query(
    //   'SELECT id FROM affiliate_programs WHERE ref_code = $1 AND status = "active"',
    //   [refCode]
    // )

    // TODO: Calculate commission based on tier
    // const commissionRate = COMMISSION_RATES[tierId] || 0.30
    // const commissionCents = Math.round(amountCents * commissionRate)

    // TODO: Insert into affiliate_earnings table
    // await db.query(
    //   `INSERT INTO affiliate_earnings
    //     (affiliate_id, stripe_customer_id, subscription_tier, amount_cents, commission_cents, status)
    //    VALUES ($1, $2, $3, $4, $5, 'pending')`,
    //   [affiliate.id, customerId, tierId, amountCents, commissionCents]
    // )

    // TODO: Update affiliate_programs conversion_count and pending_payout
    // await db.query(
    //   `UPDATE affiliate_programs SET
    //      conversion_count = conversion_count + 1,
    //      pending_payout = pending_payout + $1
    //    WHERE id = $2`,
    //   [commissionCents, affiliate.id]
    // )

    console.log(`[affiliate-conversion] Tracked conversion: ${refCode} earned ${Math.round(amountCents * 0.30)} cents`)

    return {
      success: true,
      commissionCents: Math.round(amountCents * 0.30),
    }
  } catch (error) {
    console.error('[affiliate-conversion] Error:', error)
    return {
      success: false,
      error: String(error),
    }
  }
}

export async function getAffiliateEarnings(affiliateId: string): Promise<{
  totalEarned: number
  conversions: number
  pendingPayout: number
  lastPayoutDate: Date | null
}> {
  try {
    // TODO: Query affiliate_earnings and aggregate
    // SELECT
    //   SUM(commission_cents) as total_earned,
    //   COUNT(*) as conversions,
    //   SUM(CASE WHEN status = 'pending' THEN commission_cents ELSE 0 END) as pending_payout,
    //   MAX(paid_at) as last_payout_date
    // FROM affiliate_earnings
    // WHERE affiliate_id = $1

    return {
      totalEarned: 0,
      conversions: 0,
      pendingPayout: 0,
      lastPayoutDate: null,
    }
  } catch (error) {
    console.error('[affiliate-earnings] Error:', error)
    throw error
  }
}

export const COMMISSION_RATES: Record<string, number> = {
  explorer: 0.0,
  learner: 0.3,
  family: 0.35,
  school: 0.25,
}
