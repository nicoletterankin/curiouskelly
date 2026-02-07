"use client"

// The Ziggurat × Curious Kelly - Global Education HQ Financial Model
// Overlay version for 2-layer architecture

import React, { useState, useMemo, useEffect, useRef } from 'react'
import { 
  Building2, Users, DollarSign, TrendingUp, Package, Monitor, 
  FileText, MessageSquare, ChevronDown, Info, Target, Layers, 
  BarChart3, CheckCircle2, AlertTriangle, Zap, Globe, GraduationCap, 
  Heart, Building, Landmark, X, Sparkles, Clock, ExternalLink
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

import { 
  BUILDING, 
  KELLY_VISION, 
  DEFAULT_INPUTS, 
  GROWTH, 
  SCENARIOS, 
  SDG_GOALS,
  SLIDER_CONFIG,
  COPY
} from '@/lib/ziggurat/data'

import type { 
  Scenario, 
  FinancialInputs, 
  YearCalculation, 
  FullCalculation,
  FeedbackForm 
} from '@/lib/ziggurat/types'

interface ZigOverlayProps {
  isOpen: boolean
  onClose: () => void
}

// ============================================================
// UTILITIES
// ============================================================

const fmt = (n: number | null | undefined, decimals = 1): string => {
  if (n === null || n === undefined || isNaN(n)) return '—'
  const neg = n < 0
  const abs = Math.abs(n)
  let result: string
  if (abs >= 1e9) result = `$${(abs / 1e9).toFixed(decimals)}B`
  else if (abs >= 1e6) result = `$${(abs / 1e6).toFixed(decimals)}M`
  else if (abs >= 1e3) result = `$${(abs / 1e3).toFixed(0)}K`
  else result = `$${abs.toLocaleString()}`
  return neg ? `(${result})` : result
}

const fmtNum = (n: number): string => Math.round(n).toLocaleString()
const fmtPct = (n: number): string => `${(n * 100).toFixed(1)}%`

function useAnimatedValue(target: number, duration = 800): number {
  const [value, setValue] = useState(0)
  const prev = useRef(0)
  
  useEffect(() => {
    const start = prev.current
    const startTime = Date.now()
    
    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setValue(start + (target - start) * eased)
      if (progress < 1) requestAnimationFrame(animate)
      else prev.current = target
    }
    
    requestAnimationFrame(animate)
  }, [target, duration])
  
  return value
}

// ============================================================
// CALCULATION ENGINE
// ============================================================

function calculateFinancials(
  inputs: FinancialInputs, 
  viewYear: number, 
  scenario: Scenario
): FullCalculation {
  const mult = SCENARIOS[scenario].mult
  const i = inputs
  const g = GROWTH

  const years: YearCalculation[] = [1, 2, 3, 4, 5].map(year => {
    const y = year - 1
    
    const headcount = Math.round(i.headcountY1 + (i.headcountY5 - i.headcountY1) * (y / 4))
    const salaryGrowth = Math.pow(1.03, y)
    const payroll = headcount * i.avgSalary * salaryGrowth
    const contractors = 3000000 * g.contractors[y]

    const monthlyRev = i.monthlySubsY1 * g.subs[y] * mult * i.monthlyPrice * 12
    const annualRev = i.annualSubsY1 * g.subs[y] * mult * i.annualPrice
    const cumDistricts = i.districtDealsY1 * g.district[y] * mult
    const districtRev = cumDistricts * i.studentsPerDistrict * i.pricePerStudent
    const onlineTotal = monthlyRev + annualRev + districtRev
    const onlineCosts = onlineTotal * 0.18

    const digitalRev = (i.monthlyPackY1 * 9 + i.completeYearY1 * 29 + i.premiumYearY1 * 49) * g.digital[y] * mult
    const digitalCosts = digitalRev * 0.05

    const printUnits = (i.essentialsY1 + i.standardY1 + i.collectorsY1) * g.print[y] * mult
    const printRev = (i.essentialsY1 * 79 + i.standardY1 * 99 + i.collectorsY1 * 149) * g.print[y] * mult
    const printCOGS = printUnits * (45 * g.cogsScale[y] + 8) + printRev * 0.08

    const liveRev = [5, 15, 30, 50, 75][y] * 52 * [50, 75, 100, 150, 200][y] * [0.3, 0.25, 0.2, 0.15, 0.1][y] * 5 * mult
    const liveCosts = [5, 15, 30, 50, 75][y] * 52 * 100 + liveRev * 0.1

    const hwRev = year >= 3 ? (i.iLearnMiniY3 * 149 + i.iLearnStdY3 * 299 + i.iLearnProY3 * 499) * g.hardware[y] * mult : 0
    const hwCOGS = hwRev * (1 - g.hwMargin[y])

    const licenseRev = i.enterpriseDealsY1 * g.licensing[y] * mult * i.enterpriseDealSize * Math.pow(1.1, y) +
      (year >= 2 ? i.govDealsY3 * [0, 0.33, 1, 1.67, 2.67][y] * mult * i.govDealSize * Math.pow(1.2, Math.max(0, y - 1)) : 0)
    const licenseCosts = licenseRev * 0.2

    const totalProductRev = onlineTotal + digitalRev + printRev + liveRev + hwRev + licenseRev
    const philanthroOperating = i.philanthroYearlyY1 * g.philanthropy[y]
    const totalRevenue = totalProductRev + philanthroOperating

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

    return {
      year,
      headcount,
      onlineTotal,
      digitalRev,
      printRev,
      liveRev,
      hwRev,
      licenseRev,
      totalProductRev,
      philanthroOperating,
      totalRevenue,
      totalCOR,
      grossProfit,
      payroll,
      contractors,
      totalOpex,
      netOperating,
      selfSustaining,
      grossMargin: totalRevenue > 0 ? grossProfit / totalRevenue : 0,
      philanthroDependency: totalRevenue > 0 ? philanthroOperating / totalRevenue : 0,
      dailyActiveLearners: 50000 * g.learners[y] * mult,
      languages: g.languages[y],
      commonsVisitors: g.commonsVisitors[y],
      cumDistricts,
    }
  })

  const philanthropyTotal = i.emersonCollective + i.czi + i.legoFoundation + i.gatesFoundation + i.waltonFoundation + i.otherFoundations
  const acquisitionTotal = i.purchasePrice * 1.085
  const totalCapitalNeeded = acquisitionTotal + i.phase1Reno + i.operatingReserve
  const totalCapitalRaised = philanthropyTotal + i.historicTaxCredits + i.govGrants + i.missionDebt
  const fundingGap = totalCapitalNeeded - totalCapitalRaised
  const fundingPercent = (totalCapitalRaised / totalCapitalNeeded) * 100

  const current = viewYear === 0 ? years[4] : years[viewYear - 1]
  const breakEvenYear = years.findIndex(y => y.selfSustaining >= 0) + 1 || null

  return {
    years,
    current,
    philanthropyTotal,
    acquisitionTotal,
    totalCapitalNeeded,
    totalCapitalRaised,
    fundingGap,
    fundingPercent,
    totalRenovation: i.phase1Reno + i.phase2Reno + i.phase3Reno,
    breakEvenYear,
    fiveYearProductRev: years.reduce((s, y) => s + y.totalProductRev, 0),
    fiveYearPhilanthropy: years.reduce((s, y) => s + y.philanthroOperating, 0),
  }
}

// ============================================================
// COMPACT COMPONENTS FOR OVERLAY
// ============================================================

function GlassCard({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`relative overflow-hidden rounded-xl bg-white/5 backdrop-blur-xl border border-white/10 ${className}`}>
      {children}
    </div>
  )
}

function MetricCard({ label, value, format = 'currency', color = 'amber' }: {
  label: string; value: number | string; format?: 'currency' | 'number' | 'text'; color?: string
}) {
  const animated = useAnimatedValue(typeof value === 'number' ? value : 0)
  const display = format === 'currency' ? fmt(animated) : format === 'text' ? value : fmtNum(animated)
  
  const colors: Record<string, string> = {
    amber: 'text-amber-400',
    emerald: 'text-emerald-400',
    red: 'text-red-400',
    blue: 'text-blue-400',
  }
  
  return (
    <div className="p-3 bg-white/5 rounded-lg">
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <div className={`text-lg font-bold font-mono ${colors[color] || 'text-white'}`}>{display}</div>
    </div>
  )
}

function CompactSlider({ label, value, onChange, min, max, step = 1, format = 'currency' }: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step?: number; format?: string
}) {
  const display = format === 'currency' ? fmt(value) : fmtNum(value)
  
  return (
    <div className="mb-4">
      <div className="flex justify-between text-xs mb-2">
        <span className="text-slate-400">{label}</span>
        <span className="text-amber-400 font-mono">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-slate-700 rounded-full appearance-none cursor-pointer accent-amber-500"
      />
    </div>
  )
}

// ============================================================
// MAIN OVERLAY COMPONENT
// ============================================================

export function ZigOverlay({ isOpen, onClose }: ZigOverlayProps) {
  const [tab, setTab] = useState<'vision' | 'model' | 'impact'>('vision')
  const [year, setYear] = useState(5)
  const [scenario, setScenario] = useState<Scenario>('baseline')
  const [inputs, setInputs] = useState<FinancialInputs>(DEFAULT_INPUTS)

  const updateInput = (key: keyof FinancialInputs, value: number) => {
    setInputs(prev => ({ ...prev, [key]: value }))
  }

  const calc = useMemo(
    () => calculateFinancials(inputs, year, scenario),
    [inputs, year, scenario]
  )

  // Prevent body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => { document.body.style.overflow = '' }
  }, [isOpen])

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[100] flex items-end justify-center bg-black/80 backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.div
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="w-full max-w-4xl h-[90vh] bg-[#0a0f1a] rounded-t-3xl overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex-shrink-0 p-4 border-b border-white/10 bg-slate-900/80">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
                    <Building2 size={20} className="text-white" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold">
                      <span className="text-blue-400">The Ziggurat</span>
                      <span className="text-slate-500 mx-2">x</span>
                      <span className="text-amber-400">Curious Kelly</span>
                    </h2>
                    <p className="text-xs text-slate-500">Open Financial Model</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={scenario}
                    onChange={(e) => setScenario(e.target.value as Scenario)}
                    className="bg-slate-800 border border-white/10 rounded-lg px-3 py-1.5 text-xs"
                  >
                    {Object.entries(SCENARIOS).map(([key, { name }]) => (
                      <option key={key} value={key}>{name}</option>
                    ))}
                  </select>
                  <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-lg">
                    <X size={20} />
                  </button>
                </div>
              </div>
              
              {/* Tabs */}
              <div className="flex gap-2">
                {(['vision', 'model', 'impact'] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      tab === t 
                        ? 'bg-amber-500 text-slate-900' 
                        : 'text-slate-400 hover:bg-white/5'
                    }`}
                  >
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Metrics Bar */}
            <div className="flex-shrink-0 p-3 bg-slate-900/50 border-b border-white/5">
              <div className="grid grid-cols-4 gap-2">
                <MetricCard label="Capital Needed" value={calc.totalCapitalNeeded} color="amber" />
                <MetricCard label="Funding Gap" value={calc.fundingGap} color={calc.fundingGap <= 0 ? 'emerald' : 'red'} />
                <MetricCard label={`Y${year} Revenue`} value={calc.current.totalRevenue} color="emerald" />
                <MetricCard label="Break-Even" value={calc.breakEvenYear ? `Year ${calc.breakEvenYear}` : 'N/A'} format="text" color="blue" />
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
              {tab === 'vision' && (
                <div className="space-y-4">
                  <GlassCard className="p-5">
                    <div className="text-xs text-blue-400 font-semibold mb-2">THE VISION</div>
                    <h3 className="text-2xl font-bold mb-2">A Temple of Learning</h3>
                    <p className="text-slate-300 text-sm leading-relaxed">{COPY.hero.body}</p>
                  </GlassCard>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <GlassCard className="p-4">
                      <div className="text-2xl font-bold text-amber-400">{fmtNum(BUILDING.sqft)}</div>
                      <div className="text-xs text-slate-400">Square Feet</div>
                    </GlassCard>
                    <GlassCard className="p-4">
                      <div className="text-2xl font-bold text-blue-400">{BUILDING.floors}</div>
                      <div className="text-xs text-slate-400">Floors</div>
                    </GlassCard>
                    <GlassCard className="p-4">
                      <div className="text-2xl font-bold text-emerald-400">{calc.current.headcount}</div>
                      <div className="text-xs text-slate-400">Jobs Created (Y{year})</div>
                    </GlassCard>
                    <GlassCard className="p-4">
                      <div className="text-2xl font-bold text-purple-400">{fmtNum(calc.current.dailyActiveLearners)}</div>
                      <div className="text-xs text-slate-400">Daily Learners (Y{year})</div>
                    </GlassCard>
                  </div>
                  
                  <GlassCard className="p-4">
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <Clock size={16} className="text-amber-400" />
                      Building Timeline
                    </h4>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {BUILDING.history.slice(-5).map((event, i) => (
                        <div key={i} className="flex gap-3 items-start">
                          <span className="text-xs font-mono text-amber-400 w-10">{event.year}</span>
                          <span className={`text-xs ${event.type === 'opportunity' ? 'text-emerald-400' : 'text-slate-400'}`}>
                            {event.event}
                          </span>
                        </div>
                      ))}
                    </div>
                  </GlassCard>
                </div>
              )}

              {tab === 'model' && (
                <div className="space-y-4">
                  <div className="flex gap-2 mb-4">
                    {[1, 2, 3, 4, 5].map((y) => (
                      <button
                        key={y}
                        onClick={() => setYear(y)}
                        className={`flex-1 py-2 rounded-lg text-sm font-medium ${
                          year === y ? 'bg-amber-500 text-slate-900' : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        Y{y}
                      </button>
                    ))}
                  </div>
                  
                  <GlassCard className="p-4">
                    <h4 className="text-sm font-semibold mb-3">Capital Stack</h4>
                    <div className="h-3 bg-slate-700 rounded-full overflow-hidden mb-2">
                      <div 
                        className="h-full bg-gradient-to-r from-amber-500 to-emerald-500"
                        style={{ width: `${Math.min(calc.fundingPercent, 100)}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400">{fmt(calc.totalCapitalRaised)} raised</span>
                      <span className="text-amber-400">{Math.round(calc.fundingPercent)}% funded</span>
                    </div>
                  </GlassCard>

                  <GlassCard className="p-4">
                    <h4 className="text-sm font-semibold mb-3">Key Inputs</h4>
                    <CompactSlider 
                      label="Purchase Price" 
                      value={inputs.purchasePrice} 
                      onChange={v => updateInput('purchasePrice', v)} 
                      min={70000000} max={180000000} step={5000000}
                    />
                    <CompactSlider 
                      label="Emerson Collective" 
                      value={inputs.emersonCollective} 
                      onChange={v => updateInput('emersonCollective', v)} 
                      min={0} max={50000000} step={5000000}
                    />
                    <CompactSlider 
                      label="Monthly Subscribers (Y1)" 
                      value={inputs.monthlySubsY1} 
                      onChange={v => updateInput('monthlySubsY1', v)} 
                      min={100} max={5000} step={100}
                      format="number"
                    />
                    <CompactSlider 
                      label="District Deals (Y1)" 
                      value={inputs.districtDealsY1} 
                      onChange={v => updateInput('districtDealsY1', v)} 
                      min={1} max={50} step={1}
                      format="number"
                    />
                  </GlassCard>

                  <GlassCard className="p-4">
                    <h4 className="text-sm font-semibold mb-3">P&L Summary (Y{year})</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Revenue</span>
                        <span className="font-mono text-emerald-400">{fmt(calc.current.totalRevenue)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Expenses</span>
                        <span className="font-mono text-red-400">({fmt(calc.current.totalCOR + calc.current.totalOpex)})</span>
                      </div>
                      <div className="flex justify-between pt-2 border-t border-white/10">
                        <span className="font-semibold">Net Operating</span>
                        <span className={`font-mono font-semibold ${calc.current.netOperating >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {fmt(calc.current.netOperating)}
                        </span>
                      </div>
                    </div>
                  </GlassCard>
                </div>
              )}

              {tab === 'impact' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <GlassCard className="p-4">
                      <GraduationCap size={20} className="text-emerald-400 mb-2" />
                      <div className="text-2xl font-bold">{fmtNum(calc.current.dailyActiveLearners)}</div>
                      <div className="text-xs text-slate-400">Daily Learners</div>
                    </GlassCard>
                    <GlassCard className="p-4">
                      <Globe size={20} className="text-purple-400 mb-2" />
                      <div className="text-2xl font-bold">{calc.current.languages}</div>
                      <div className="text-xs text-slate-400">Languages</div>
                    </GlassCard>
                    <GlassCard className="p-4">
                      <Users size={20} className="text-blue-400 mb-2" />
                      <div className="text-2xl font-bold">{calc.current.headcount}</div>
                      <div className="text-xs text-slate-400">Jobs Created</div>
                    </GlassCard>
                    <GlassCard className="p-4">
                      <Building size={20} className="text-amber-400 mb-2" />
                      <div className="text-2xl font-bold">{fmtNum(calc.current.commonsVisitors)}</div>
                      <div className="text-xs text-slate-400">Commons Visitors/yr</div>
                    </GlassCard>
                  </div>

                  <GlassCard className="p-4">
                    <h4 className="text-sm font-semibold mb-3">UN Sustainable Development Goals</h4>
                    {SDG_GOALS.map((goal, i) => (
                      <div key={i} className="mb-3 last:mb-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded">{goal.goal}</span>
                          <span className="text-xs font-medium">{goal.title}</span>
                        </div>
                        <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                          <div className="h-full bg-emerald-500" style={{ width: `${goal.progress}%` }} />
                        </div>
                      </div>
                    ))}
                  </GlassCard>

                  <GlassCard className="p-4 bg-gradient-to-br from-blue-500/10 to-amber-500/5">
                    <h4 className="text-sm font-semibold mb-2">The Mission</h4>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      Transform a vacant federal building into the global headquarters for Curious Kelly, 
                      serving 3.7 billion learners without reliable internet access through offline-first 
                      educational technology.
                    </p>
                  </GlassCard>
                </div>
              )}
            </div>

            {/* Footer with CTAs */}
            <div className="flex-shrink-0 p-4 border-t border-white/10 bg-slate-900/80">
              <div className="flex items-center justify-between gap-4">
                <span className="text-xs text-slate-500">Lesson of the Day, PBC</span>
                <div className="flex items-center gap-3">
                  <a 
                    href="/strategic"
                    className="text-xs text-slate-400 hover:text-white flex items-center gap-1 transition-colors"
                  >
                    Strategic Overview <ExternalLink size={12} />
                  </a>
                  <a 
                    href="/ziggurat/investors"
                    className="text-xs px-3 py-1.5 bg-amber-500 text-slate-900 rounded-lg font-medium hover:bg-amber-400 flex items-center gap-1 transition-colors"
                  >
                    Full Model <ExternalLink size={12} />
                  </a>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
