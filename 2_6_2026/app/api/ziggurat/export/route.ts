import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * Export API for Ziggurat Financial Model
 * 
 * Supports CSV, JSON, and PDF exports
 * 
 * Generates downloadable CSV files for investor analysis:
 * - 5-year P&L projections
 * - Break-even analysis
 * - Learner-affiliate economics
 */

interface ModelInputs {
  startingLearners: number
  monthlyGrowthRate: number
  conversionRate: number
  avgRevenuePerUser: number
  affiliateShare: number
}

function generateProjections(inputs: ModelInputs) {
  const {
    startingLearners = 1000,
    monthlyGrowthRate = 15,
    conversionRate = 5,
    avgRevenuePerUser = 8,
    affiliateShare = 70,
  } = inputs

  const years = []
  let learners = startingLearners
  let cumulativeLearners = startingLearners

  for (let year = 1; year <= 5; year++) {
    // Monthly compounding for learner growth
    for (let month = 0; month < 12; month++) {
      learners = Math.floor(learners * (1 + monthlyGrowthRate / 100))
      cumulativeLearners += learners
    }

    const subscribers = Math.floor(cumulativeLearners * (conversionRate / 100))
    const grossRevenue = subscribers * avgRevenuePerUser * 12
    const affiliatePayout = grossRevenue * (affiliateShare / 100)
    const netRevenue = grossRevenue - affiliatePayout
    
    // Operating costs scale with learners
    const contentCosts = 50000 + (year - 1) * 25000 // Growing content library
    const infraCosts = 10000 + subscribers * 0.50 // ~$0.50 per subscriber
    const teamCosts = 150000 + (year - 1) * 50000 // Growing team
    const marketingCosts = grossRevenue * 0.15 // 15% of gross
    const totalCosts = contentCosts + infraCosts + teamCosts + marketingCosts
    
    const operatingIncome = netRevenue - totalCosts
    const margin = netRevenue > 0 ? (operatingIncome / netRevenue * 100) : 0

    years.push({
      year,
      learners,
      cumulativeLearners,
      subscribers,
      grossRevenue: Math.round(grossRevenue),
      affiliatePayout: Math.round(affiliatePayout),
      netRevenue: Math.round(netRevenue),
      contentCosts: Math.round(contentCosts),
      infraCosts: Math.round(infraCosts),
      teamCosts: Math.round(teamCosts),
      marketingCosts: Math.round(marketingCosts),
      totalCosts: Math.round(totalCosts),
      operatingIncome: Math.round(operatingIncome),
      margin: Math.round(margin * 10) / 10,
    })
  }

  return years
}

function generatePLCsv(inputs: ModelInputs): string {
  const projections = generateProjections(inputs)
  
  const headers = [
    'Year',
    'Active Learners',
    'Cumulative Learners',
    'Subscribers',
    'Gross Revenue ($)',
    'Affiliate Payout ($)',
    'Net Revenue ($)',
    'Content Costs ($)',
    'Infrastructure ($)',
    'Team Costs ($)',
    'Marketing ($)',
    'Total Costs ($)',
    'Operating Income ($)',
    'Margin (%)',
  ]

  const rows = projections.map(p => [
    `Year ${p.year}`,
    p.learners.toLocaleString(),
    p.cumulativeLearners.toLocaleString(),
    p.subscribers.toLocaleString(),
    p.grossRevenue.toLocaleString(),
    p.affiliatePayout.toLocaleString(),
    p.netRevenue.toLocaleString(),
    p.contentCosts.toLocaleString(),
    p.infraCosts.toLocaleString(),
    p.teamCosts.toLocaleString(),
    p.marketingCosts.toLocaleString(),
    p.totalCosts.toLocaleString(),
    p.operatingIncome.toLocaleString(),
    `${p.margin}%`,
  ])

  // Add totals row
  const totals = projections.reduce((acc, p) => ({
    grossRevenue: acc.grossRevenue + p.grossRevenue,
    affiliatePayout: acc.affiliatePayout + p.affiliatePayout,
    netRevenue: acc.netRevenue + p.netRevenue,
    totalCosts: acc.totalCosts + p.totalCosts,
    operatingIncome: acc.operatingIncome + p.operatingIncome,
  }), { grossRevenue: 0, affiliatePayout: 0, netRevenue: 0, totalCosts: 0, operatingIncome: 0 })

  rows.push([
    '5-Year Total',
    '',
    '',
    '',
    totals.grossRevenue.toLocaleString(),
    totals.affiliatePayout.toLocaleString(),
    totals.netRevenue.toLocaleString(),
    '',
    '',
    '',
    '',
    totals.totalCosts.toLocaleString(),
    totals.operatingIncome.toLocaleString(),
    '',
  ])

  return [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
}

function generateBreakEvenCsv(inputs: ModelInputs): string {
  const projections = generateProjections(inputs)
  
  let cumulativeIncome = 0
  let breakEvenYear: number | null = null
  
  const headers = [
    'Year',
    'Operating Income ($)',
    'Cumulative Income ($)',
    'Break-Even Status',
  ]

  const rows = projections.map(p => {
    cumulativeIncome += p.operatingIncome
    const status = cumulativeIncome >= 0 
      ? (breakEvenYear === null ? (breakEvenYear = p.year, 'BREAK-EVEN ACHIEVED') : 'Profitable')
      : 'Pre-profit'
    
    return [
      `Year ${p.year}`,
      p.operatingIncome.toLocaleString(),
      cumulativeIncome.toLocaleString(),
      status,
    ]
  })

  return [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
}

function generateAffiliateCsv(inputs: ModelInputs): string {
  const { affiliateShare = 70, avgRevenuePerUser = 8, conversionRate = 5 } = inputs
  
  const avgCommission = avgRevenuePerUser * 12 * (affiliateShare / 100) * (conversionRate / 100)
  
  const tiers = [
    { name: 'Curious', shares: 5, bonus: 1.0 },
    { name: 'Ambassador', shares: 25, bonus: 1.1 },
    { name: 'Champion', shares: 100, bonus: 1.25 },
    { name: 'Legend', shares: 250, bonus: 1.5 },
  ]

  const headers = [
    'Tier',
    'Shares Required',
    'Bonus Multiplier',
    'Est. Conversions (5%)',
    'Commission per Convert ($)',
    'Annual Earnings ($)',
  ]

  const rows = tiers.map(t => {
    const conversions = Math.floor(t.shares * 0.05)
    const annualEarnings = conversions * avgCommission * t.bonus
    return [
      t.name,
      t.shares.toString(),
      `${t.bonus}x`,
      conversions.toString(),
      `$${(avgCommission * t.bonus).toFixed(2)}`,
      `$${annualEarnings.toFixed(2)}`,
    ]
  })

  return [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
}

// GET endpoint for direct browser download
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const type = searchParams.get('type') || 'pl'
  const inputs = {
    startingLearners: Number(searchParams.get('learners')) || 1000,
    monthlyGrowthRate: Number(searchParams.get('growth')) || 15,
    conversionRate: Number(searchParams.get('conversion')) || 5,
    avgRevenuePerUser: Number(searchParams.get('arpu')) || 8,
    affiliateShare: Number(searchParams.get('affiliate')) || 70,
  }

  let csv: string
  let filename: string

  // Handle JSON export - full investor data from database
  if (type === 'json') {
    try {
      const [models, questions, reviews] = await Promise.all([
        sql`SELECT * FROM investor_models ORDER BY created_at DESC`.catch(() => []),
        sql`SELECT * FROM investor_questions ORDER BY created_at DESC`.catch(() => []),
        sql`SELECT * FROM investor_reviews ORDER BY scheduled_at DESC`.catch(() => []),
      ])
      
      const exportData = {
        exportedAt: new Date().toISOString(),
        investorModels: models,
        questions,
        reviews,
        projections: generateProjections(inputs),
      }
      
      return new NextResponse(JSON.stringify(exportData, null, 2), {
        headers: {
          'Content-Type': 'application/json',
          'Content-Disposition': `attachment; filename="ziggurat-investor-data-${new Date().toISOString().split('T')[0]}.json"`,
        },
      })
    } catch {
      return NextResponse.json({ error: 'Failed to export JSON' }, { status: 500 })
    }
  }

  // Handle CSV exports
  switch (type) {
    case 'csv':
    case 'pl':
      csv = generatePLCsv(inputs)
      filename = 'lotd-5year-pl.csv'
      break
    case 'breakeven':
      csv = generateBreakEvenCsv(inputs)
      filename = 'lotd-breakeven-analysis.csv'
      break
    case 'affiliate':
      csv = generateAffiliateCsv(inputs)
      filename = 'lotd-affiliate-economics.csv'
      break
    default:
      csv = generatePLCsv(inputs)
      filename = 'lotd-5year-pl.csv'
  }

  return new NextResponse(csv, {
    headers: {
      'Content-Type': 'text/csv',
      'Content-Disposition': `attachment; filename="${filename}"`,
    },
  })
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { type = 'pl', inputs = {} } = body

    let csv: string
    let filename: string

    switch (type) {
      case 'pl':
        csv = generatePLCsv(inputs)
        filename = 'lotd-5year-pl.csv'
        break
      case 'breakeven':
        csv = generateBreakEvenCsv(inputs)
        filename = 'lotd-breakeven-analysis.csv'
        break
      case 'affiliate':
        csv = generateAffiliateCsv(inputs)
        filename = 'lotd-affiliate-economics.csv'
        break
      default:
        return NextResponse.json({ error: 'Invalid export type' }, { status: 400 })
    }

    return new NextResponse(csv, {
      headers: {
        'Content-Type': 'text/csv',
        'Content-Disposition': `attachment; filename="${filename}"`,
      },
    })
    
  } catch (error) {
    console.error('Error generating export:', error)
    return NextResponse.json({ error: 'Failed to generate export' }, { status: 500 })
  }
}
