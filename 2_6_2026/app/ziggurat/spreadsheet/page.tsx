"use client"

import { useState, useMemo, useCallback } from 'react'
import Link from 'next/link'
import useSWR from 'swr'
import { 
  ArrowLeft, ArrowRight, Download, Copy, Check, RefreshCw,
  TrendingUp, TrendingDown, DollarSign, Users,
  Calculator, FileSpreadsheet, ChevronDown, ChevronUp,
  Zap, Globe, GraduationCap, Target, Layers, BarChart3,
  Activity, AlertCircle, Loader2, Building2
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { DEFAULT_INPUTS, GROWTH, SCENARIOS, AFFILIATE_MODEL, PRODUCT_LINES, GROWTH_EVENTS, PRODUCT_REVENUE_PROJECTIONS } from '@/lib/ziggurat/data'
import type { FinancialInputs, Scenario } from '@/lib/ziggurat/types'

// Fetcher for SWR
const fetcher = (url: string) => fetch(url).then(res => res.json())

// Metrics type from API
interface RealMetrics {
  timestamp: string
  learners: { total: number; new30d: number; active7d: number; active24h: number }
  subscribers: { total: number; monthly: number; annual: number; lifetime: number; family: number }
  revenue: { mrr: number; arr: number; arpu: number }
  engagement: { totalCompletions: number; uniqueLearners: number; avgCompletionRate: number }
  content: { totalLessons: number; completeAssets: number; daysWithAudio: number }
  investors: { totalModels: number; totalViews: number; totalQuestions: number; upcomingReviews: number }
  ratios: { conversionRate: number; retentionRate7d: number; contentCompletionRate: number }
}

// Formatting utilities
const fmt = (n: number, decimals = 0) => {
  if (n == null || isNaN(n)) return '—'
  const neg = n < 0
  const abs = Math.abs(n)
  let result: string
  if (abs >= 1e9) result = `$${(abs / 1e9).toFixed(decimals)}B`
  else if (abs >= 1e6) result = `$${(abs / 1e6).toFixed(decimals)}M`
  else if (abs >= 1e3) result = `$${(abs / 1e3).toFixed(0)}K`
  else result = `$${abs.toLocaleString()}`
  return neg ? `(${result})` : result
}

const fmtRaw = (n: number) => (n ?? 0).toLocaleString('en-US', { maximumFractionDigits: 0 })
const fmtPct = (n: number) => `${((n ?? 0) * 100).toFixed(1)}%`

// Calculate full 5-year model
function calculateModel(inputs: FinancialInputs, scenario: Scenario) {
  const mult = SCENARIOS[scenario].mult
  const g = GROWTH
  const i = inputs

  return [1, 2, 3, 4, 5].map(year => {
    const y = year - 1
    
    // Staffing
    const headcount = Math.round(i.headcountY1 + (i.headcountY5 - i.headcountY1) * (y / 4))
    const payroll = headcount * i.avgSalary * Math.pow(1.03, y)
    const contractors = 3000000 * g.contractors[y]
    
    // Revenue streams
    const monthlyRev = i.monthlySubsY1 * g.subs[y] * mult * i.monthlyPrice * 12
    const annualRev = i.annualSubsY1 * g.subs[y] * mult * i.annualPrice
    const districtRev = i.districtDealsY1 * g.district[y] * mult * i.studentsPerDistrict * i.pricePerStudent
    const onlineTotal = monthlyRev + annualRev + districtRev
    
    const digitalRev = (i.monthlyPackY1 * 9 + i.completeYearY1 * 29 + i.premiumYearY1 * 49) * g.digital[y] * mult
    const printRev = (i.essentialsY1 * 79 + i.standardY1 * 99 + i.collectorsY1 * 149) * g.print[y] * mult
    const liveRev = [5, 15, 30, 50, 75][y] * 52 * [50, 75, 100, 150, 200][y] * [0.3, 0.25, 0.2, 0.15, 0.1][y] * 5 * mult
    const hwRev = year >= 3 ? (i.iLearnMiniY3 * 149 + i.iLearnStdY3 * 299 + i.iLearnProY3 * 499) * g.hardware[y] * mult : 0
    const licenseRev = i.enterpriseDealsY1 * g.licensing[y] * mult * i.enterpriseDealSize
    
    const totalProductRev = onlineTotal + digitalRev + printRev + liveRev + hwRev + licenseRev
    const philanthropy = i.philanthroYearlyY1 * g.philanthropy[y]
    const totalRevenue = totalProductRev + philanthropy
    
    // Costs
    const onlineCosts = onlineTotal * 0.18
    const digitalCosts = digitalRev * 0.05
    const printUnits = (i.essentialsY1 + i.standardY1 + i.collectorsY1) * g.print[y] * mult
    const printCOGS = printUnits * (45 * g.cogsScale[y] + 8) + printRev * 0.08
    const liveCosts = [5, 15, 30, 50, 75][y] * 52 * 100 + liveRev * 0.1
    const hwCOGS = hwRev * (1 - g.hwMargin[y])
    const licenseCosts = licenseRev * 0.2
    
    const totalCOR = onlineCosts + digitalCosts + printCOGS + liveCosts + hwCOGS + licenseCosts
    const grossProfit = totalRevenue - totalCOR
    
    const totalOpex = payroll + contractors +
      i.buildingOps * g.building[y] +
      i.techInfra * g.tech[y] +
      i.contentProd * g.content[y] +
      i.marketing * g.marketing[y] +
      i.gaLegal * g.ga[y]
    
    const netOperating = grossProfit - totalOpex
    const selfSustaining = totalProductRev - totalCOR - totalOpex
    
    // Learner-Affiliate Model
    const dailyLearners = 50000 * g.learners[y] * mult
    const affiliateRevenue = dailyLearners * 0.05 * 12 * 2.50 // 5% affiliate rate, $2.50 avg commission
    const affiliatePayouts = affiliateRevenue * 0.70 // 70% to learner-affiliates
    const netAffiliateRev = affiliateRevenue - affiliatePayouts
    
    return {
      year,
      headcount,
      
      // Revenue Detail
      monthlyRev,
      annualRev,
      districtRev,
      onlineTotal,
      digitalRev,
      printRev,
      liveRev,
      hwRev,
      licenseRev,
      totalProductRev,
      philanthropy,
      totalRevenue,
      
      // Costs Detail
      onlineCosts,
      digitalCosts,
      printCOGS,
      liveCosts,
      hwCOGS,
      licenseCosts,
      totalCOR,
      grossProfit,
      payroll,
      contractors,
      totalOpex,
      
      // Results
      netOperating,
      selfSustaining,
      grossMargin: totalRevenue > 0 ? grossProfit / totalRevenue : 0,
      philanthroDep: totalRevenue > 0 ? philanthropy / totalRevenue : 0,
      
      // Learner-Affiliate
      dailyLearners,
      affiliateRevenue,
      affiliatePayouts,
      netAffiliateRev,
      
      // Break-even metrics
      revenuePerLearner: dailyLearners > 0 ? totalProductRev / (dailyLearners * 365) : 0,
      costPerLearner: dailyLearners > 0 ? (totalCOR + totalOpex) / (dailyLearners * 365) : 0,
    }
  })
}

// Capital structure calculation - Coalition Model ($309M Phase 1)
function calculateCapital(inputs: FinancialInputs) {
  // Phase 1 Uses: $200M purchase + $75M reno + $4M tech + $30M reserve = $309M
  const totalCapitalNeeded = (inputs.purchasePrice ?? 200_000_000) + 
    (inputs.renovationBuildout ?? 75_000_000) + 
    (inputs.techInfrastructure ?? 4_000_000) + 
    (inputs.operatingReserve ?? 30_000_000)
  
  // Coalition Sources: PRIs + Grants + Tax Credits + Debt = $290M invited
  const coalitionInvited = 
    (inputs.emersonCollective ?? 50_000_000) +      // PRI
    (inputs.czi ?? 50_000_000) +                     // PRI
    (inputs.gatesFoundation ?? 40_000_000) +        // Grant
    (inputs.legoFoundation ?? 15_000_000) +         // Grant
    (inputs.otherFoundations ?? 20_000_000) +       // Grants
    (inputs.historicTaxCredits ?? 20_000_000) +     // Tax credits
    (inputs.govGrants ?? 40_000_000) +              // Gov grants
    (inputs.missionDebt ?? 55_000_000)              // Mission debt
  
  return {
    totalCapitalNeeded,
    totalCapitalRaised: coalitionInvited,
    fundingGap: totalCapitalNeeded - coalitionInvited,
    fundingPercent: totalCapitalNeeded > 0 ? (coalitionInvited / totalCapitalNeeded) * 100 : 0,
  }
}

// CSV Export
function generateCSV(data: ReturnType<typeof calculateModel>, inputs: FinancialInputs, capital: ReturnType<typeof calculateCapital>) {
  const rows: string[][] = []
  
  // Header
  rows.push(['LESSON OF THE DAY, PBC - ZIGGURAT FINANCIAL MODEL'])
  rows.push(['Generated', new Date().toISOString()])
  rows.push([])
  
  // Capital Structure - Coalition Model ($309M Phase 1)
  rows.push(['=== PHASE 1 CAPITAL STRUCTURE ($309M) ==='])
  rows.push(['Uses of Capital', 'Amount'])
  rows.push(['Pre-Auction Purchase Price', String(inputs.purchasePrice ?? 200_000_000)])
  rows.push(['Phase 1 Renovation (Levels 1-4)', String(inputs.renovationBuildout ?? 75_000_000)])
  rows.push(['Initial Technology Infrastructure', String(inputs.techInfrastructure ?? 4_000_000)])
  rows.push(['Operating Reserve (18 months)', String(inputs.operatingReserve ?? 30_000_000)])
  rows.push(['TOTAL PHASE 1 CAPITAL NEEDED', String(capital.totalCapitalNeeded ?? 309_000_000)])
  rows.push([])
  rows.push(['Coalition Sources (Invited)', 'Amount', 'Type'])
  rows.push(['Emerson Collective', String(inputs.emersonCollective ?? 50_000_000), 'PRI'])
  rows.push(['Chan Zuckerberg Initiative', String(inputs.czi ?? 50_000_000), 'PRI'])
  rows.push(['Gates Foundation', String(inputs.gatesFoundation ?? 40_000_000), 'Grant'])
  rows.push(['Lego Foundation', String(inputs.legoFoundation ?? 15_000_000), 'Grant'])
  rows.push(['Other Foundations (Hewlett, Bezos)', String(inputs.otherFoundations ?? 20_000_000), 'Grant'])
  rows.push(['Historic Tax Credits (20% of QRE)', String(inputs.historicTaxCredits ?? 20_000_000), 'Tax Credit'])
  rows.push(['Government Grants (NSF, DOE, NEA)', String(inputs.govGrants ?? 40_000_000), 'Grant'])
  rows.push(['Mission-Aligned Debt (CDFIs)', String(inputs.missionDebt ?? 55_000_000), 'Debt'])
  rows.push(['TOTAL COALITION INVITED', String(capital.totalCapitalRaised ?? 290_000_000)])
  rows.push(['FUNDING GAP', String(capital.fundingGap ?? 19_000_000)])
  rows.push([])
  
  // 5-Year Path (from investor memo)
  rows.push(['=== FIVE-YEAR PATH ==='])
  rows.push(['Metric', '2026', '2027', '2028', '2029', '2030'])
  rows.push(['Revenue', '$1.7M', '$7.8M', '$33M', '$90M', '$202M'])
  rows.push(['EBITDA', '($0.65M)', '($2.4M)', '($3M)', '$15M', '$56M'])
  rows.push(['Daily Learners', '50K', '500K', '5M', '25M', '100M'])
  rows.push([])
  
  // 5-Year P&L
  rows.push(['=== 5-YEAR PROJECTION ==='])
  const header = ['Metric', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
  rows.push(header)
  
  const metrics = [
    { label: 'Headcount', key: 'headcount' },
    { label: '--- REVENUE ---', key: null },
    { label: 'Online Subscriptions', key: 'onlineTotal' },
    { label: 'Digital Downloads', key: 'digitalRev' },
    { label: 'Print Edition', key: 'printRev' },
    { label: 'Live Events', key: 'liveRev' },
    { label: 'Hardware (Y3+)', key: 'hwRev' },
    { label: 'Licensing', key: 'licenseRev' },
    { label: 'TOTAL PRODUCT REVENUE', key: 'totalProductRev' },
    { label: 'Philanthropic Operating', key: 'philanthropy' },
    { label: 'TOTAL REVENUE', key: 'totalRevenue' },
    { label: '--- COSTS ---', key: null },
    { label: 'Total Cost of Revenue', key: 'totalCOR' },
    { label: 'GROSS PROFIT', key: 'grossProfit' },
    { label: 'Payroll', key: 'payroll' },
    { label: 'Contractors', key: 'contractors' },
    { label: 'Total Operating Expenses', key: 'totalOpex' },
    { label: '--- RESULTS ---', key: null },
    { label: 'NET OPERATING', key: 'netOperating' },
    { label: 'SELF-SUSTAINING (excl. philanthropy)', key: 'selfSustaining' },
    { label: '--- METRICS ---', key: null },
    { label: 'Gross Margin', key: 'grossMargin', format: 'pct' },
    { label: 'Philanthropy Dependency', key: 'philanthroDep', format: 'pct' },
    { label: '--- LEARNER-AFFILIATE MODEL ---', key: null },
    { label: 'Daily Active Learners', key: 'dailyLearners' },
    { label: 'Affiliate Revenue', key: 'affiliateRevenue' },
    { label: 'Affiliate Payouts (70%)', key: 'affiliatePayouts' },
    { label: 'Net Affiliate Revenue', key: 'netAffiliateRev' },
    { label: 'Revenue Per Learner/Day', key: 'revenuePerLearner', format: 'decimal' },
    { label: 'Cost Per Learner/Day', key: 'costPerLearner', format: 'decimal' },
  ]
  
  metrics.forEach(({ label, key, format }) => {
    if (!key) {
      rows.push([label, '', '', '', '', ''])
    } else {
      const values = data.map(d => {
        const val = (d[key as keyof typeof d] as number) ?? 0
        if (format === 'pct') return ((val ?? 0) * 100).toFixed(1) + '%'
        if (format === 'decimal') return (val ?? 0).toFixed(4)
        return String(val ?? 0)
      })
      rows.push([label, ...values])
    }
  })
  
  // Break-even analysis
  rows.push([])
  rows.push(['=== BREAK-EVEN ANALYSIS ==='])
  const breakEvenYear = data.findIndex(d => d.selfSustaining >= 0) + 1
  rows.push(['Self-Sustaining Break-Even Year', breakEvenYear > 0 ? `Year ${breakEvenYear}` : 'Beyond Year 5'])
  rows.push(['Year 5 Self-Sustaining', String(data[4]?.selfSustaining ?? 0)])
  rows.push(['Year 5 Philanthropy Dependency', ((data[4]?.philanthroDep ?? 0) * 100).toFixed(1) + '%'])
  
  return rows.map(row => row.join(',')).join('\n')
}

// Editable cell component
function EditableCell({ 
  value, 
  onChange, 
  format = 'currency',
  min = 0,
  max = Infinity,
  step = 1000000
}: {
  value: number
  onChange: (v: number) => void
  format?: 'currency' | 'number' | 'percent'
  min?: number
  max?: number
  step?: number
}) {
  const [editing, setEditing] = useState(false)
  const [tempValue, setTempValue] = useState(String(value ?? 0))
  
  const display = format === 'currency' ? fmt(value, 1) : 
                  format === 'percent' ? fmtPct(value) : 
                  fmtRaw(value)
  
  const handleBlur = () => {
    const parsed = parseFloat(tempValue.replace(/[,$%]/g, ''))
    if (!isNaN(parsed)) {
      const clamped = Math.max(min, Math.min(max, format === 'percent' ? parsed / 100 : parsed))
      onChange(clamped)
    }
    setEditing(false)
  }
  
  if (editing) {
    return (
      <input
        type="text"
        value={tempValue}
        onChange={e => setTempValue(e.target.value)}
        onBlur={handleBlur}
        onKeyDown={e => e.key === 'Enter' && handleBlur()}
        autoFocus
        className="w-full bg-amber-500/20 border border-amber-500 rounded px-2 py-1 text-right text-sm font-mono focus:outline-none"
      />
    )
  }
  
  return (
    <button
      onClick={() => { setTempValue(String(value ?? 0)); setEditing(true) }}
      className="w-full text-right font-mono text-sm hover:bg-white/10 px-2 py-1 rounded transition-colors"
    >
      {display}
    </button>
  )
}

export default function SpreadsheetPage() {
  const [inputs, setInputs] = useState<FinancialInputs>(DEFAULT_INPUTS)
  const [scenario, setScenario] = useState<Scenario>('baseline')
  const [copied, setCopied] = useState(false)
  const [expandedSections, setExpandedSections] = useState<string[]>(['actuals', 'capital', 'revenue', 'costs', 'affiliate', 'products'])
  
  // Fetch real metrics from database
  const { data: metrics, isLoading: metricsLoading, error: metricsError } = useSWR<RealMetrics>(
    '/api/ziggurat/metrics',
    fetcher,
    { refreshInterval: 60000, revalidateOnFocus: false }
  )
  
  const data = useMemo(() => calculateModel(inputs, scenario), [inputs, scenario])
  const capital = useMemo(() => calculateCapital(inputs), [inputs])
  const breakEvenYear = data.findIndex(d => d.selfSustaining >= 0) + 1
  
  const updateInput = useCallback((key: keyof FinancialInputs, value: number) => {
    setInputs(prev => ({ ...prev, [key]: value }))
  }, [])
  
  const toggleSection = (section: string) => {
    setExpandedSections(prev => 
      prev.includes(section) 
        ? prev.filter(s => s !== section)
        : [...prev, section]
    )
  }
  
  const downloadCSV = () => {
    const csv = generateCSV(data, inputs, capital)
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ziggurat-model-${scenario}-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }
  
  const copyToClipboard = async () => {
    const csv = generateCSV(data, inputs, capital)
    await navigator.clipboard.writeText(csv)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  
  const resetToDefaults = () => {
    setInputs(DEFAULT_INPUTS)
    setScenario('baseline')
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-black via-black to-zinc-950 text-white selection:bg-amber-500/30">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-black/90 backdrop-blur-xl border-b border-white/[0.06]">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-5">
            <Link 
              href="/ziggurat" 
              className="group flex items-center gap-2 text-white/50 hover:text-white transition-colors duration-200"
            >
              <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
              <span className="text-sm font-medium">Ziggurat</span>
            </Link>
            <div className="h-5 w-px bg-white/10" />
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-600/20 border border-amber-500/20 flex items-center justify-center">
                <FileSpreadsheet className="w-4 h-4 text-amber-400" />
              </div>
              <div>
                <h1 className="font-semibold tracking-tight">Financial Model</h1>
                <p className="text-[11px] text-white/40">5-Year Projections</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {/* Scenario Selector */}
            <div className="flex bg-white/[0.04] rounded-xl p-1 border border-white/[0.06]">
              {(['conservative', 'baseline', 'optimistic'] as Scenario[]).map(s => (
                <button
                  key={s}
                  onClick={() => setScenario(s)}
                  className={cn(
                    "px-4 py-2 text-xs font-medium rounded-lg transition-all duration-200",
                    scenario === s 
                      ? "bg-amber-500 text-black shadow-lg shadow-amber-500/20" 
                      : "text-white/50 hover:text-white hover:bg-white/[0.04]"
                  )}
                >
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </button>
              ))}
            </div>
            
            <div className="h-6 w-px bg-white/10" />
            
            <button 
              onClick={resetToDefaults} 
              className="p-2.5 text-white/40 hover:text-white hover:bg-white/[0.06] rounded-lg transition-all duration-200"
              title="Reset to defaults"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button 
              onClick={copyToClipboard} 
              className="p-2.5 text-white/40 hover:text-white hover:bg-white/[0.06] rounded-lg transition-all duration-200"
              title="Copy to clipboard"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            </button>
            <button 
              onClick={downloadCSV} 
              className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-amber-500 to-amber-600 text-black rounded-xl text-sm font-semibold hover:from-amber-400 hover:to-amber-500 shadow-lg shadow-amber-500/20 transition-all duration-200"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-4 py-6">
        {/* INVESTOR EXECUTIVE SUMMARY - One glance view */}
        <div className="mb-8 p-6 rounded-2xl border border-amber-500/20 bg-gradient-to-r from-amber-500/[0.05] via-transparent to-amber-500/[0.03]">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold text-amber-100">Coalition Investment Opportunity</h2>
              <p className="text-sm text-white/40 mt-1">Confidential Memorandum - January 29, 2026</p>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold font-mono text-amber-400">$309M</div>
              <div className="text-xs text-white/40">Phase 1 Capital</div>
            </div>
          </div>
          
          <div className="grid grid-cols-5 gap-4 py-4 border-y border-white/10">
            <div>
              <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1">Purchase</div>
              <div className="text-lg font-semibold text-white font-mono">$200M</div>
              <div className="text-[10px] text-white/30">Pre-auction offer</div>
            </div>
            <div>
              <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1">Phase 1 Reno</div>
              <div className="text-lg font-semibold text-white font-mono">$75M</div>
              <div className="text-[10px] text-white/30">Levels 1-4</div>
            </div>
            <div>
              <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1">Reserve</div>
              <div className="text-lg font-semibold text-white font-mono">$30M</div>
              <div className="text-[10px] text-white/30">18-month ops</div>
            </div>
            <div>
              <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1">Y5 Revenue</div>
              <div className="text-lg font-semibold text-emerald-400 font-mono">$202M</div>
              <div className="text-[10px] text-white/30">100M daily learners</div>
            </div>
            <div>
              <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1">Y5 EBITDA</div>
              <div className="text-lg font-semibold text-emerald-400 font-mono">$56M</div>
              <div className="text-[10px] text-white/30">Self-sustaining</div>
            </div>
          </div>
          
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-white/40 text-xs uppercase tracking-wider">Coalition Invited</span>
              <div className="mt-1 flex flex-wrap gap-2">
                {['Emerson ($50M PRI)', 'CZI ($50M PRI)', 'Gates ($40M)', 'Lego ($15M)', 'Historic Tax Credits ($20M)'].map(p => (
                  <span key={p} className="px-2 py-0.5 rounded bg-white/5 text-white/70 text-xs">{p}</span>
                ))}
              </div>
            </div>
            <div className="text-right flex flex-col justify-end">
              <Link 
                href="/ziggurat/investors" 
                className="text-amber-400 hover:text-amber-300 font-medium transition-colors inline-flex items-center gap-1 justify-end"
              >
                Full Investment Memo <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
          </div>
        </div>

        {/* Key Metrics Dashboard */}
        <div className="grid grid-cols-6 gap-4 mb-8">
          {[
            { label: 'Total Capital', value: capital.totalCapitalNeeded, icon: Building2, color: 'amber' },
            { label: 'Funding Gap', value: capital.fundingGap, icon: Target, color: capital.fundingGap > 0 ? 'red' : 'emerald' },
            { label: 'Break-Even', value: breakEvenYear > 0 ? `Year ${breakEvenYear}` : 'Y5+', icon: TrendingUp, isText: true, color: 'blue' },
            { label: 'Y5 Revenue', value: data[4]?.totalRevenue ?? 0, icon: DollarSign, color: 'emerald' },
            { label: 'Y5 DAL', value: data[4]?.dailyLearners ?? 0, icon: Users, format: 'number', color: 'purple', sublabel: 'Daily Active Learners' },
            { label: 'Y5 Independence', value: 1 - (data[4]?.philanthroDep ?? 0), icon: TrendingUp, format: 'percent', color: (data[4]?.philanthroDep ?? 1) < 0.3 ? 'emerald' : 'amber' },
          ].map(({ label, value, icon: Icon, color, format, isText, sublabel }) => (
            <div 
              key={label} 
              className={cn(
                "group relative p-5 rounded-2xl border backdrop-blur-sm transition-all duration-300 hover:scale-[1.02]", 
                color === 'amber' && "bg-gradient-to-br from-amber-500/10 to-amber-600/5 border-amber-500/20 hover:border-amber-500/40",
                color === 'emerald' && "bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20 hover:border-emerald-500/40",
                color === 'red' && "bg-gradient-to-br from-red-500/10 to-red-600/5 border-red-500/20 hover:border-red-500/40",
                color === 'blue' && "bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20 hover:border-blue-500/40",
                color === 'purple' && "bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20 hover:border-purple-500/40"
              )}
            >
              <div className="flex items-center gap-2 mb-3">
                <div className={cn(
                  "w-7 h-7 rounded-lg flex items-center justify-center",
                  color === 'amber' && "bg-amber-500/20",
                  color === 'emerald' && "bg-emerald-500/20",
                  color === 'red' && "bg-red-500/20",
                  color === 'blue' && "bg-blue-500/20",
                  color === 'purple' && "bg-purple-500/20"
                )}>
                  <Icon className={cn(
                    "w-3.5 h-3.5",
                    color === 'amber' && "text-amber-400",
                    color === 'emerald' && "text-emerald-400",
                    color === 'red' && "text-red-400",
                    color === 'blue' && "text-blue-400",
                    color === 'purple' && "text-purple-400"
                  )} />
                </div>
                <span className="text-[11px] uppercase tracking-wider text-white/40 font-medium">{label}</span>
              </div>
              <div className={cn(
                "text-2xl font-bold font-mono tracking-tight",
                color === 'amber' && "text-amber-100",
                color === 'emerald' && "text-emerald-100",
                color === 'red' && "text-red-100",
                color === 'blue' && "text-blue-100",
                color === 'purple' && "text-purple-100"
              )}>
                {isText ? value : format === 'percent' ? fmtPct(value as number) : format === 'number' ? fmtRaw(value as number) : fmt(value as number, 1)}
              </div>
              {sublabel && (
                <div className="absolute bottom-2 right-3 text-[9px] text-white/20 opacity-0 group-hover:opacity-100 transition-opacity">
                  {sublabel}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* ACTUALS VS PROJECTED - Real Data from Database */}
        <div className="border border-emerald-500/20 rounded-2xl overflow-hidden mb-8 bg-gradient-to-br from-emerald-500/[0.03] to-transparent">
          <button
            onClick={() => toggleSection('actuals')}
            className="w-full flex items-center justify-between px-6 py-4 bg-emerald-500/[0.06] hover:bg-emerald-500/[0.08] transition-colors duration-200"
          >
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                <Activity className="w-5 h-5 text-emerald-400" />
              </div>
              <div className="text-left">
                <div className="flex items-center gap-2">
                  <span className="font-semibold tracking-tight">Live Metrics</span>
                  {metricsLoading && <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-400/60" />}
                  {metricsError && <AlertCircle className="w-3.5 h-3.5 text-red-400" />}
                  {metrics && !metricsLoading && (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/20 text-[10px] text-emerald-400 font-medium">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      LIVE
                    </span>
                  )}
                </div>
                <p className="text-xs text-white/40">
                  {metrics ? `Updated ${new Date(metrics.timestamp).toLocaleTimeString()}` : 'Real-time database metrics'}
                </p>
              </div>
            </div>
            <ChevronDown className={cn(
              "w-5 h-5 text-white/40 transition-transform duration-200",
              expandedSections.includes('actuals') && "rotate-180"
            )} />
          </button>
          
          {expandedSections.includes('actuals') && (
            <div className="px-6 py-5 border-t border-emerald-500/10">
              {metricsLoading ? (
                <div className="flex flex-col items-center justify-center py-12 text-white/40">
                  <Loader2 className="w-8 h-8 animate-spin mb-3 text-emerald-400/40" />
                  <span className="text-sm">Loading live metrics...</span>
                </div>
              ) : metricsError ? (
                <div className="flex flex-col items-center justify-center py-12 text-red-400/80">
                  <AlertCircle className="w-8 h-8 mb-3" />
                  <span className="text-sm font-medium">Failed to load metrics</span>
                  <span className="text-xs text-white/30 mt-1">Using projections only</span>
                </div>
              ) : metrics ? (
                <div className="grid grid-cols-4 gap-6">
                  {/* Learners */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] uppercase tracking-widest text-emerald-400/50 font-semibold">Learners</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/50 text-sm">Total</span>
                        <span className="font-mono font-bold text-lg text-emerald-300">{fmtRaw(metrics.learners.total)}</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/30 text-xs">New (30d)</span>
                        <span className="font-mono text-sm text-white/60">{fmtRaw(metrics.learners.new30d)}</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/30 text-xs">Active (7d)</span>
                        <span className="font-mono text-sm text-white/60">{fmtRaw(metrics.learners.active7d)}</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/30 text-xs">Active (24h)</span>
                        <span className="font-mono text-sm text-white/60">{fmtRaw(metrics.learners.active24h)}</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Subscribers */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] uppercase tracking-widest text-emerald-400/50 font-semibold">Subscribers</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/50 text-sm">Paying</span>
                        <span className="font-mono font-bold text-lg text-emerald-300">{fmtRaw(metrics.subscribers.total)}</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/30 text-xs">Monthly</span>
                        <span className="font-mono text-sm text-white/60">{fmtRaw(metrics.subscribers.monthly)}</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/30 text-xs">Annual</span>
                        <span className="font-mono text-sm text-white/60">{fmtRaw(metrics.subscribers.annual)}</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/30 text-xs">Lifetime</span>
                        <span className="font-mono text-sm text-white/70">{fmtRaw(metrics.subscribers.lifetime)}</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Revenue */}
                  <div className="space-y-3">
                    <h4 className="text-xs uppercase tracking-wider text-emerald-400/60 font-medium">Revenue</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/60 text-sm">MRR</span>
                        <span className="font-mono font-bold text-emerald-400">{fmt(metrics.revenue.mrr)}</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/40 text-xs">ARR</span>
                        <span className="font-mono text-sm text-white/70">{fmt(metrics.revenue.arr)}</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/40 text-xs">ARPU</span>
                        <span className="font-mono text-sm text-white/70">${(metrics?.revenue?.arpu ?? 0).toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/40 text-xs">vs Y1 Target</span>
                        <span className={cn("font-mono text-sm", 
                          (metrics?.revenue?.arr ?? 0) >= (data[0]?.totalRevenue ?? 1) * 0.1 ? "text-emerald-400" : "text-amber-400"
                        )}>
                          {(((metrics?.revenue?.arr ?? 0) / (data[0]?.totalRevenue || 1)) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Key Ratios */}
                  <div className="space-y-3">
                    <h4 className="text-xs uppercase tracking-wider text-emerald-400/60 font-medium">Key Ratios</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/60 text-sm">Conversion Rate</span>
                        <span className="font-mono font-bold text-emerald-400">{(metrics?.ratios?.conversionRate ?? 0).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/40 text-xs">7d Retention</span>
                        <span className="font-mono text-sm text-white/70">{(metrics?.ratios?.retentionRate7d ?? 0).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/40 text-xs">Content Ready</span>
                        <span className="font-mono text-sm text-white/70">{(metrics?.ratios?.contentCompletionRate ?? 0).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between items-baseline">
                        <span className="text-white/40 text-xs">Investor Models</span>
                        <span className="font-mono text-sm text-white/70">{metrics.investors.totalModels}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </div>

        {/* Spreadsheet Grid */}
        <div className="space-y-4">
          {/* Capital Structure */}
          <div className="border border-white/10 rounded-xl overflow-hidden">
            <button
              onClick={() => toggleSection('capital')}
              className="w-full flex items-center justify-between p-4 bg-white/5 hover:bg-white/[0.07]"
            >
              <div className="flex items-center gap-3">
                <Building2 className="w-5 h-5 text-amber-400" />
                <span className="font-semibold">Capital Structure</span>
                <span className="text-xs text-white/40 font-mono">{fmt(capital.totalCapitalNeeded, 0)}</span>
              </div>
              {expandedSections.includes('capital') ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {expandedSections.includes('capital') && (
              <div className="p-4">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-white/40 text-xs uppercase tracking-wider">
                      <th className="text-left py-2 px-3">Uses of Capital</th>
                      <th className="text-right py-2 px-3 w-40">Amount</th>
                      <th className="text-right py-2 px-3 w-24">% of Total</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {[
                      { label: 'Purchase Price', key: 'purchasePrice' as const },
                      { label: 'Renovation & Buildout', key: 'renovationBuildout' as const },
                      { label: 'LED Display System', key: 'ledDisplay' as const },
                      { label: 'Technology Infrastructure', key: 'techInfrastructure' as const },
                      { label: 'Soft Costs & Contingency', key: 'softCostsContingency' as const },
                      { label: 'Operating Reserve', key: 'operatingReserve' as const },
                    ].map(({ label, key }) => (
                      <tr key={key} className="hover:bg-white/5">
                        <td className="py-2 px-3">{label}</td>
                        <td className="py-2 px-3">
                          <EditableCell value={inputs[key]} onChange={v => updateInput(key, v)} />
                        </td>
                        <td className="py-2 px-3 text-right text-white/50 font-mono">
                          {capital.totalCapitalNeeded > 0 ? (((inputs[key] ?? 0) / capital.totalCapitalNeeded) * 100).toFixed(1) + '%' : '—'}
                        </td>
                      </tr>
                    ))}
                    <tr className="bg-amber-500/10 font-semibold">
                      <td className="py-2 px-3">TOTAL CAPITAL NEEDED</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(capital.totalCapitalNeeded, 0)}</td>
                      <td className="py-2 px-3 text-right font-mono">100%</td>
                    </tr>
                  </tbody>
                </table>
                
                <div className="mt-4 pt-4 border-t border-white/10">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-white/40 text-xs uppercase tracking-wider">
                        <th className="text-left py-2 px-3">Sources of Capital</th>
                        <th className="text-right py-2 px-3 w-40">Amount</th>
                        <th className="text-right py-2 px-3 w-24">Target %</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {[
                        { label: 'Strategic Partner Equity', key: 'strategicPartnerEquity' as const, target: '40%' },
                        { label: 'Institutional Impact Equity', key: 'institutionalEquity' as const, target: '30%' },
                        { label: 'Historic Tax Credits', key: 'historicTaxCredits' as const, target: '10%' },
                        { label: 'Debt Financing', key: 'debtFinancing' as const, target: '20%' },
                      ].map(({ label, key, target }) => (
                        <tr key={key} className="hover:bg-white/5">
                          <td className="py-2 px-3">{label}</td>
                          <td className="py-2 px-3">
                            <EditableCell value={inputs[key]} onChange={v => updateInput(key, v)} />
                          </td>
                          <td className="py-2 px-3 text-right text-white/50 font-mono">{target}</td>
                        </tr>
                      ))}
                      <tr className={cn("font-semibold", capital.fundingGap <= 0 ? "bg-emerald-500/10" : "bg-red-500/10")}>
                        <td className="py-2 px-3">FUNDING GAP</td>
                        <td className="py-2 px-3 text-right font-mono">{fmt(capital.fundingGap, 0)}</td>
                        <td className="py-2 px-3 text-right font-mono">
                          {capital.fundingGap <= 0 ? 'FUNDED' : `${(((capital.fundingGap ?? 0) / (capital.totalCapitalNeeded || 1)) * 100).toFixed(1)}% needed`}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {/* 5-Year Revenue */}
          <div className="border border-white/10 rounded-xl overflow-hidden">
            <button
              onClick={() => toggleSection('revenue')}
              className="w-full flex items-center justify-between p-4 bg-white/5 hover:bg-white/[0.07]"
            >
              <div className="flex items-center gap-3">
                <TrendingUp className="w-5 h-5 text-emerald-400" />
                <span className="font-semibold">5-Year Revenue Projection</span>
                <span className="text-xs text-white/40 font-mono">{fmt(data.reduce((s, d) => s + d.totalRevenue, 0), 0)} total</span>
              </div>
              {expandedSections.includes('revenue') ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {expandedSections.includes('revenue') && (
              <div className="p-4 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-white/40 text-xs uppercase tracking-wider">
                      <th className="text-left py-2 px-3">Revenue Stream</th>
                      {[1, 2, 3, 4, 5].map(y => (
                        <th key={y} className="text-right py-2 px-3 w-28">Year {y}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {[
                      { label: 'Online Subscriptions', key: 'onlineTotal' },
                      { label: 'Digital Downloads', key: 'digitalRev' },
                      { label: 'Print Edition', key: 'printRev' },
                      { label: 'Live Events', key: 'liveRev' },
                      { label: 'Hardware (Y3+)', key: 'hwRev' },
                      { label: 'Licensing', key: 'licenseRev' },
                    ].map(({ label, key }) => (
                      <tr key={key} className="hover:bg-white/5">
                        <td className="py-2 px-3">{label}</td>
                        {data.map(d => (
                          <td key={d.year} className="py-2 px-3 text-right font-mono text-white/80">
                            {fmt(d[key as keyof typeof d] as number, 1)}
                          </td>
                        ))}
                      </tr>
                    ))}
                    <tr className="bg-emerald-500/10 font-semibold">
                      <td className="py-2 px-3">TOTAL PRODUCT REVENUE</td>
                      {data.map(d => (
                        <td key={d.year} className="py-2 px-3 text-right font-mono">{fmt(d.totalProductRev, 1)}</td>
                      ))}
                    </tr>
                    <tr className="hover:bg-white/5 text-white/60">
                      <td className="py-2 px-3">Philanthropic Operating</td>
                      {data.map(d => (
                        <td key={d.year} className="py-2 px-3 text-right font-mono">{fmt(d.philanthropy, 1)}</td>
                      ))}
                    </tr>
                    <tr className="bg-amber-500/10 font-bold">
                      <td className="py-2 px-3">TOTAL REVENUE</td>
                      {data.map(d => (
                        <td key={d.year} className="py-2 px-3 text-right font-mono">{fmt(d.totalRevenue, 1)}</td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Costs & Profitability */}
          <div className="border border-white/10 rounded-xl overflow-hidden">
            <button
              onClick={() => toggleSection('costs')}
              className="w-full flex items-center justify-between p-4 bg-white/5 hover:bg-white/[0.07]"
            >
              <div className="flex items-center gap-3">
                <Calculator className="w-5 h-5 text-red-400" />
                <span className="font-semibold">Costs & Profitability</span>
                <span className={cn("text-xs font-mono px-2 py-0.5 rounded", 
                  (data[4]?.selfSustaining ?? 0) >= 0 ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"
                )}>
                  Y5: {fmt(data[4]?.selfSustaining ?? 0, 1)}
                </span>
              </div>
              {expandedSections.includes('costs') ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {expandedSections.includes('costs') && (
              <div className="p-4 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-white/40 text-xs uppercase tracking-wider">
                      <th className="text-left py-2 px-3">Line Item</th>
                      {[1, 2, 3, 4, 5].map(y => (
                        <th key={y} className="text-right py-2 px-3 w-28">Year {y}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    <tr className="hover:bg-white/5">
                      <td className="py-2 px-3">Total Cost of Revenue</td>
                      {data.map(d => (
                        <td key={d.year} className="py-2 px-3 text-right font-mono text-red-400/80">{fmt(d.totalCOR, 1)}</td>
                      ))}
                    </tr>
                    <tr className="bg-white/5 font-semibold">
                      <td className="py-2 px-3">Gross Profit</td>
                      {data.map(d => (
                        <td key={d.year} className="py-2 px-3 text-right font-mono">{fmt(d.grossProfit, 1)}</td>
                      ))}
                    </tr>
                    <tr className="hover:bg-white/5 text-white/60">
                      <td className="py-2 px-3 pl-6">Gross Margin</td>
                      {data.map(d => (
                        <td key={d.year} className="py-2 px-3 text-right font-mono">{fmtPct(d.grossMargin)}</td>
                      ))}
                    </tr>
                    <tr className="hover:bg-white/5">
                      <td className="py-2 px-3">Payroll</td>
                      {data.map(d => (
                        <td key={d.year} className="py-2 px-3 text-right font-mono text-red-400/80">{fmt(d.payroll, 1)}</td>
                      ))}
                    </tr>
                    <tr className="hover:bg-white/5">
                      <td className="py-2 px-3">Total Operating Expenses</td>
                      {data.map(d => (
                        <td key={d.year} className="py-2 px-3 text-right font-mono text-red-400/80">{fmt(d.totalOpex, 1)}</td>
                      ))}
                    </tr>
                    <tr className="bg-amber-500/10 font-semibold">
                      <td className="py-2 px-3">Net Operating</td>
                      {data.map(d => (
                        <td key={d.year} className={cn("py-2 px-3 text-right font-mono", d.netOperating >= 0 ? "text-emerald-400" : "text-red-400")}>
                          {fmt(d.netOperating, 1)}
                        </td>
                      ))}
                    </tr>
                    <tr className={cn("font-bold", (data[4]?.selfSustaining ?? 0) >= 0 ? "bg-emerald-500/10" : "bg-red-500/10")}>
                      <td className="py-2 px-3">Self-Sustaining (excl. philanthropy)</td>
                      {data.map(d => (
                        <td key={d.year} className={cn("py-2 px-3 text-right font-mono", d.selfSustaining >= 0 ? "text-emerald-400" : "text-red-400")}>
                          {fmt(d.selfSustaining, 1)}
                        </td>
                      ))}
                    </tr>
                    <tr className="hover:bg-white/5 text-white/60">
                      <td className="py-2 px-3 pl-6">Philanthropy Dependency</td>
                      {data.map(d => (
                        <td key={d.year} className="py-2 px-3 text-right font-mono">{fmtPct(d.philanthroDep)}</td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Learner-Affiliate Model */}
          <div className="border border-purple-500/30 rounded-xl overflow-hidden bg-purple-500/5">
            <button
              onClick={() => toggleSection('affiliate')}
              className="w-full flex items-center justify-between p-4 bg-purple-500/10 hover:bg-purple-500/15"
            >
              <div className="flex items-center gap-3">
                <Zap className="w-5 h-5 text-purple-400" />
                <span className="font-semibold">Learner-Affiliate Economic Model</span>
                <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded">NEW</span>
              </div>
              {expandedSections.includes('affiliate') ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {expandedSections.includes('affiliate') && (
              <div className="p-4">
                <p className="text-sm text-white/60 mb-4">
                  The learner-affiliate model creates economic opportunity for every learner who shares Curious Kelly. 
                  Learners earn 70% of affiliate revenue they generate, creating a virtuous cycle of learning and earning.
                </p>
                
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-white/40 text-xs uppercase tracking-wider">
                        <th className="text-left py-2 px-3">Metric</th>
                        {[1, 2, 3, 4, 5].map(y => (
                          <th key={y} className="text-right py-2 px-3 w-28">Year {y}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      <tr className="hover:bg-white/5">
                        <td className="py-2 px-3">Daily Active Learners</td>
                        {data.map(d => (
                          <td key={d.year} className="py-2 px-3 text-right font-mono">{fmtRaw(d.dailyLearners)}</td>
                        ))}
                      </tr>
                      <tr className="hover:bg-white/5">
                        <td className="py-2 px-3">Affiliate Revenue Generated</td>
                        {data.map(d => (
                          <td key={d.year} className="py-2 px-3 text-right font-mono text-emerald-400/80">{fmt(d.affiliateRevenue, 1)}</td>
                        ))}
                      </tr>
                      <tr className="hover:bg-white/5">
                        <td className="py-2 px-3">Paid to Learner-Affiliates (70%)</td>
                        {data.map(d => (
                          <td key={d.year} className="py-2 px-3 text-right font-mono text-purple-400">{fmt(d.affiliatePayouts, 1)}</td>
                        ))}
                      </tr>
                      <tr className="bg-purple-500/10 font-semibold">
                        <td className="py-2 px-3">Net Affiliate Revenue to LOTD</td>
                        {data.map(d => (
                          <td key={d.year} className="py-2 px-3 text-right font-mono">{fmt(d.netAffiliateRev, 1)}</td>
                        ))}
                      </tr>
                      <tr className="hover:bg-white/5 text-white/60">
                        <td className="py-2 px-3">Revenue Per Learner/Day</td>
                        {data.map(d => (
                          <td key={d.year} className="py-2 px-3 text-right font-mono">${(d.revenuePerLearner ?? 0).toFixed(4)}</td>
                        ))}
                      </tr>
                      <tr className="hover:bg-white/5 text-white/60">
                        <td className="py-2 px-3">Cost Per Learner/Day</td>
                        {data.map(d => (
                          <td key={d.year} className="py-2 px-3 text-right font-mono">${(d.costPerLearner ?? 0).toFixed(4)}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
                
                <div className="mt-4 p-4 bg-white/5 rounded-lg">
                  <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                    <Globe className="w-4 h-4 text-purple-400" />
                    How It Works
                  </h4>
                  <ol className="text-xs text-white/60 space-y-1 list-decimal list-inside">
                    <li>Learner completes daily lessons and earns "Share" credits</li>
                    <li>Learner shares Kelly with friends/family via unique affiliate link</li>
                    <li>New learners who sign up generate affiliate revenue</li>
                    <li>70% of revenue goes directly to the referring learner</li>
                    <li>Creates economic opportunity while expanding learning reach</li>
                  </ol>
                </div>
              </div>
            )}
          </div>
          
          {/* Product Revenue Breakdown */}
          <div className="border border-white/10 rounded-xl overflow-hidden">
            <button
              onClick={() => toggleSection('products')}
              className="w-full flex items-center justify-between p-4 bg-white/5 hover:bg-white/[0.07]"
            >
              <div className="flex items-center gap-3">
                <Layers className="w-5 h-5 text-cyan-400" />
                <span className="font-semibold">Product Line Revenue (5-Year)</span>
                <span className="text-xs text-cyan-400/60">Square DNA: Kelly, iLearn, Print</span>
              </div>
              {expandedSections.includes('products') ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {expandedSections.includes('products') && (
              <div className="p-4">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-white/40 text-xs uppercase tracking-wider">
                      <th className="text-left py-2 px-3">Product Line</th>
                      <th className="text-right py-2 px-3">Y1</th>
                      <th className="text-right py-2 px-3">Y2</th>
                      <th className="text-right py-2 px-3">Y3</th>
                      <th className="text-right py-2 px-3">Y4</th>
                      <th className="text-right py-2 px-3">Y5</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    <tr className="hover:bg-white/5">
                      <td className="py-2 px-3 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-400" />
                        Kelly App (Digital)
                      </td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year1.kellyApp.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year2.kellyApp.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year3.kellyApp.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year4.kellyApp.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year5.kellyApp.revenue)}</td>
                    </tr>
                    <tr className="hover:bg-white/5">
                      <td className="py-2 px-3 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-amber-400" />
                        Daily Lesson Print
                      </td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year1.dailyLessonPrint.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year2.dailyLessonPrint.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year3.dailyLessonPrint.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year4.dailyLessonPrint.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year5.dailyLessonPrint.revenue)}</td>
                    </tr>
                    <tr className="hover:bg-white/5">
                      <td className="py-2 px-3 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-purple-400" />
                        Annual Book
                      </td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year1.annualBook.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year2.annualBook.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year3.annualBook.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year4.annualBook.revenue)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year5.annualBook.revenue)}</td>
                    </tr>
                    <tr className="hover:bg-white/5">
                      <td className="py-2 px-3 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-cyan-400" />
                        iLearn Mini (Square)
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-white/30">-</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year2.iLearnMini?.revenue || 0)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year3.iLearnMini?.revenue || 0)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year4.iLearnMini?.revenue || 0)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year5.iLearnMini?.revenue || 0)}</td>
                    </tr>
                    <tr className="hover:bg-white/5">
                      <td className="py-2 px-3 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-blue-400" />
                        iLearn Pro (Square)
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-white/30">-</td>
                      <td className="py-2 px-3 text-right font-mono text-white/30">-</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year3.iLearnPro?.revenue || 0)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year4.iLearnPro?.revenue || 0)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year5.iLearnPro?.revenue || 0)}</td>
                    </tr>
                    <tr className="hover:bg-white/5">
                      <td className="py-2 px-3 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-indigo-400" />
                        iLearn Hub (Square)
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-white/30">-</td>
                      <td className="py-2 px-3 text-right font-mono text-white/30">-</td>
                      <td className="py-2 px-3 text-right font-mono text-white/30">-</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year4.iLearnHub?.revenue || 0)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year5.iLearnHub?.revenue || 0)}</td>
                    </tr>
                    <tr className="hover:bg-white/5">
                      <td className="py-2 px-3 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-orange-400" />
                        Kelly Enterprise (B2B)
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-white/30">-</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year2.kellyEnterprise?.revenue || 0)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year3.kellyEnterprise?.revenue || 0)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year4.kellyEnterprise?.revenue || 0)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year5.kellyEnterprise?.revenue || 0)}</td>
                    </tr>
                    <tr className="bg-cyan-500/10 font-semibold">
                      <td className="py-2 px-3">TOTAL PRODUCT REVENUE</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year1.total)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year2.total)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year3.total)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year4.total)}</td>
                      <td className="py-2 px-3 text-right font-mono">{fmt(PRODUCT_REVENUE_PROJECTIONS.year5.total)}</td>
                    </tr>
                  </tbody>
                </table>
                
                {/* Growth Events */}
                <div className="mt-4 p-4 bg-white/5 rounded-lg">
                  <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-amber-400" />
                    Growth Catalyst Events
                  </h4>
                  <div className="grid grid-cols-3 gap-4">
                    {Object.entries(GROWTH_EVENTS).map(([key, event]) => (
                      <div key={key} className="p-3 bg-black/30 rounded-lg border border-white/5">
                        <div className="text-sm font-medium text-white/90">{event.name}</div>
                        <div className="text-xs text-white/40 mt-1">{event.date} - {event.location}</div>
                        <div className="mt-2 space-y-1 text-xs">
                          {event.revenueImpact.subscriptions && (
                            <div className="flex justify-between">
                              <span className="text-white/50">New Subs</span>
                              <span className="font-mono text-emerald-400">+{fmtRaw(event.revenueImpact.subscriptions)}</span>
                            </div>
                          )}
                          {event.revenueImpact.hardware && (
                            <div className="flex justify-between">
                              <span className="text-white/50">iLearn Units</span>
                              <span className="font-mono text-cyan-400">+{fmtRaw(event.revenueImpact.hardware)}</span>
                            </div>
                          )}
                          {event.revenueImpact.licensing && (
                            <div className="flex justify-between">
                              <span className="text-white/50">Licensing</span>
                              <span className="font-mono text-amber-400">{fmt(event.revenueImpact.licensing)}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 pt-8 border-t border-white/[0.04]">
          <div className="flex items-center justify-between text-xs text-white/30">
            <div>
              <p className="font-medium text-white/40">Lesson of the Day, PBC</p>
              <p className="mt-0.5">Ziggurat Financial Model v2.0</p>
            </div>
            <div className="text-right">
              <p>All projections are estimates based on market research</p>
              <p className="mt-0.5">
                Scenario: <span className="text-white/50">{scenario.charAt(0).toUpperCase() + scenario.slice(1)}</span> ({SCENARIOS[scenario].mult}x multiplier)
              </p>
            </div>
          </div>
        </footer>
      </main>
    </div>
  )
}
