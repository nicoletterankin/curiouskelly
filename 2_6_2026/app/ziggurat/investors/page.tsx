"use client"

import { useState, useMemo } from 'react'
import Link from 'next/link'
import useSWR from 'swr'
import { 
  ArrowLeft, Plus, Users, Eye, MessageSquare, Calendar, 
  TrendingUp, Building2, Share2, Clock, Download, Copy, Check,
  Sparkles, Lock, Globe, FileSpreadsheet, Target, DollarSign,
  ChevronDown, ChevronUp, Calculator, Zap, GraduationCap, AlertTriangle, Loader2
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { DEFAULT_INPUTS, GROWTH, SCENARIOS, AFFILIATE_MODEL } from '@/lib/ziggurat/data'
import type { FinancialInputs, Scenario, InvestorModel } from '@/lib/ziggurat/types'
import { InvestorDashboard } from '@/components/investor/investor-dashboard'

const fetcher = (url: string) => fetch(url).then(res => res.json())

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

// Calculate model for workspace
function calculateModel(inputs: FinancialInputs, scenario: Scenario) {
  const mult = SCENARIOS[scenario].mult
  const g = GROWTH

  return [1, 2, 3, 4, 5].map(year => {
    const y = year - 1
    
    const headcount = Math.round(inputs.headcountY1 + (inputs.headcountY5 - inputs.headcountY1) * (y / 4))
    const payroll = headcount * inputs.avgSalary * Math.pow(1.03, y)
    const contractors = 3000000 * g.contractors[y]
    
    const monthlyRev = inputs.monthlySubsY1 * g.subs[y] * mult * inputs.monthlyPrice * 12
    const annualRev = inputs.annualSubsY1 * g.subs[y] * mult * inputs.annualPrice
    const districtRev = inputs.districtDealsY1 * g.district[y] * mult * inputs.studentsPerDistrict * inputs.pricePerStudent
    const onlineTotal = monthlyRev + annualRev + districtRev
    
    const digitalRev = (inputs.monthlyPackY1 * 9 + inputs.completeYearY1 * 29 + inputs.premiumYearY1 * 49) * g.digital[y] * mult
    const printRev = (inputs.essentialsY1 * 79 + inputs.standardY1 * 99 + inputs.collectorsY1 * 149) * g.print[y] * mult
    const liveRev = [5, 15, 30, 50, 75][y] * 52 * [50, 75, 100, 150, 200][y] * [0.3, 0.25, 0.2, 0.15, 0.1][y] * 5 * mult
    const hwRev = year >= 3 ? (inputs.iLearnMiniY3 * 149 + inputs.iLearnStdY3 * 299 + inputs.iLearnProY3 * 499) * g.hardware[y] * mult : 0
    const licenseRev = inputs.enterpriseDealsY1 * g.licensing[y] * mult * inputs.enterpriseDealSize
    
    const totalProductRev = onlineTotal + digitalRev + printRev + liveRev + hwRev + licenseRev
    const philanthropy = inputs.philanthroYearlyY1 * g.philanthropy[y]
    const totalRevenue = totalProductRev + philanthropy
    
    const totalCOR = onlineTotal * 0.18 + digitalRev * 0.05 + printRev * 0.15 + liveRev * 0.25 + hwRev * (1 - g.hwMargin[y]) + licenseRev * 0.2
    const grossProfit = totalRevenue - totalCOR
    
    const totalOpex = payroll + contractors +
      inputs.buildingOps * g.building[y] +
      inputs.techInfra * g.tech[y] +
      inputs.contentProd * g.content[y] +
      inputs.marketing * g.marketing[y] +
      inputs.gaLegal * g.ga[y]
    
    const netOperating = grossProfit - totalOpex
    const selfSustaining = totalProductRev - totalCOR - totalOpex
    
    const dailyLearners = 50000 * g.learners[y] * mult
    const affiliateRevenue = dailyLearners * AFFILIATE_MODEL.avgConversionRate * 12 * AFFILIATE_MODEL.avgCommissionPerConvert
    
    return {
      year,
      headcount,
      totalProductRev,
      philanthropy,
      totalRevenue,
      totalCOR,
      grossProfit,
      totalOpex,
      netOperating,
      selfSustaining,
      grossMargin: totalRevenue > 0 ? grossProfit / totalRevenue : 0,
      philanthroDep: totalRevenue > 0 ? philanthropy / totalRevenue : 0,
      dailyLearners,
      affiliateRevenue,
    }
  })
}

// Types for investor workspaces
interface InvestorWorkspace {
  id: string
  investor_name: string
  investor_email: string
  organization: string
  created_at: string
  updated_at: string
  is_public: boolean
  view_count: number
  model_inputs: Partial<FinancialInputs>
  scenario: Scenario
  notes: string
  questions: Question[]
  focus_areas: string[]
}

interface Question {
  id: string
  author: string
  content: string
  created_at: string
  replies: Reply[]
}

interface Reply {
  id: string
  author: string
  content: string
  created_at: string
}

// Mock data - DEPRECATED - Use real database only
const MOCK_MODELS: InvestorWorkspace[] = []

export default function InvestorModelsPage() {
  // Fetch real investor models from database
  const { data: apiModels, isLoading: apiLoading, error } = useSWR<{ models: InvestorWorkspace[] }>(
    '/api/ziggurat/investors',
    fetcher,
    { revalidateOnFocus: false }
  )
  
  // Use real API data only - no mocks
  const models = useMemo(() => {
    if (apiModels?.models?.length) {
      // Transform API response to match InvestorWorkspace interface
      return apiModels.models.map(m => ({
        ...m,
        scenario: (m.model_inputs as Record<string, unknown>)?.scenario as Scenario || 'baseline',
        questions: [],
        focus_areas: [],
      }))
    }
    return []
  }, [apiModels])
  
  const [selectedModel, setSelectedModel] = useState<InvestorWorkspace | null>(null)
  const [showNewModelForm, setShowNewModelForm] = useState(false)
  const [newQuestion, setNewQuestion] = useState('')
  const [copied, setCopied] = useState(false)
  const [expandedSections, setExpandedSections] = useState<string[]>(['summary', 'breakeven'])
  const isLoading = apiLoading
  
  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric',
    })
  }

  const formatRelativeTime = (dateStr: string) => {
    const diff = Date.now() - new Date(dateStr).getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    if (days === 0) return 'Today'
    if (days === 1) return 'Yesterday'
    return `${days} days ago`
  }

  const toggleSection = (section: string) => {
    setExpandedSections(prev => 
      prev.includes(section) 
        ? prev.filter(s => s !== section)
        : [...prev, section]
    )
  }

  // Calculate selected model's projections
  const modelData = useMemo(() => {
    if (!selectedModel) return null
    const mergedInputs = { ...DEFAULT_INPUTS, ...selectedModel.model_inputs }
    return calculateModel(mergedInputs, selectedModel.scenario)
  }, [selectedModel])

  const breakEvenYear = modelData?.findIndex(d => d.selfSustaining >= 0)
  const hasBreakEven = breakEvenYear !== undefined && breakEvenYear !== -1

  // Generate CSV for selected model
  const downloadCSV = () => {
    if (!selectedModel || !modelData) return
    
    const rows: string[][] = []
    rows.push([`${selectedModel.organization} - Ziggurat Financial Model`])
    rows.push([`Scenario: ${selectedModel.scenario.toUpperCase()}`])
    rows.push([`Generated: ${new Date().toISOString()}`])
    rows.push([])
    rows.push(['Metric', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'])
    
    const metrics = [
      { label: 'Total Product Revenue', key: 'totalProductRev' },
      { label: 'Philanthropic Operating', key: 'philanthropy' },
      { label: 'Total Revenue', key: 'totalRevenue' },
      { label: 'Gross Profit', key: 'grossProfit' },
      { label: 'Operating Expenses', key: 'totalOpex' },
      { label: 'Net Operating', key: 'netOperating' },
      { label: 'Self-Sustaining', key: 'selfSustaining' },
      { label: 'Daily Active Learners', key: 'dailyLearners' },
      { label: 'Affiliate Revenue', key: 'affiliateRevenue' },
    ]
    
    metrics.forEach(({ label, key }) => {
      rows.push([label, ...modelData.map(d => (d[key as keyof typeof d] as number).toString())])
    })
    
    const csv = rows.map(row => row.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${selectedModel.organization.toLowerCase().replace(/\s+/g, '-')}-model-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const copyModel = async () => {
    if (!selectedModel || !modelData) return
    const summary = `${selectedModel.organization} Model (${selectedModel.scenario})
Y5 Revenue: ${fmt(modelData[4].totalRevenue, 1)}
Y5 Self-Sustaining: ${fmt(modelData[4].selfSustaining, 1)}
Break-Even: ${hasBreakEven ? `Year ${(breakEvenYear ?? 0) + 1}` : 'Beyond Y5'}
Y5 Daily Learners: ${fmtRaw(modelData[4].dailyLearners)}`
    await navigator.clipboard.writeText(summary)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-black via-black to-zinc-950 text-white selection:bg-purple-500/30">
      {/* UNIFIED NAV - Consistent with Kelly app */}
      <header className="sticky top-0 z-50 bg-black/90 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link 
              href="/" 
              className="group flex items-center gap-2 text-white/50 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
              <span className="text-sm">Back to Kelly</span>
            </Link>
            <span className="text-white/20">|</span>
            <span className="font-serif text-white">The Daily Lesson</span>
            <span className="text-white/30">/</span>
            <span className="text-white/60">Investor Workspaces</span>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-4 text-sm text-white/50">
              <Link href="/about" className="hover:text-white transition-colors">About</Link>
              <Link href="/strategic" className="hover:text-white transition-colors">Strategic</Link>
              <Link href="/join" className="text-white font-medium">Investors</Link>
            </div>
            <span className="flex items-center gap-1.5 px-2 py-0.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-[10px] text-emerald-400 font-medium uppercase">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Beta
            </span>
            <button
              onClick={() => setShowNewModelForm(true)}
              className="flex items-center gap-2 px-4 py-2 bg-white text-black rounded-lg text-sm font-medium hover:bg-white/90 transition-colors"
            >
              <Plus className="w-4 h-4" />
              Create Model
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Bar */}
        <div className="grid grid-cols-5 gap-4 mb-8">
          {[
            { label: 'Active Models', value: models.length, icon: TrendingUp, color: 'amber' },
            { label: 'Total Views', value: models.reduce((s, m) => s + m.view_count, 0), icon: Eye, color: 'blue' },
            { label: 'Open Questions', value: models.reduce((s, m) => s + m.questions.filter(q => q.replies.length === 0).length, 0), icon: MessageSquare, color: 'red' },
            { label: 'Avg Capital', value: fmt(models.reduce((s, m) => s + (m.model_inputs.purchasePrice || DEFAULT_INPUTS.purchasePrice) + (m.model_inputs.renovationBuildout || DEFAULT_INPUTS.renovationBuildout), 0) / models.length, 0), icon: DollarSign, color: 'emerald', isText: true },
            { label: 'Y5 Affiliates', value: '1.5M', icon: GraduationCap, color: 'purple', isText: true },
          ].map(({ label, value, icon: Icon, color, isText }) => (
            <div 
              key={label} 
              className={cn(
                "group p-5 rounded-2xl border backdrop-blur-sm transition-all duration-300 hover:scale-[1.02]",
                color === 'amber' && "bg-gradient-to-br from-amber-500/10 to-amber-600/5 border-amber-500/20 hover:border-amber-500/40",
                color === 'blue' && "bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20 hover:border-blue-500/40",
                color === 'red' && "bg-gradient-to-br from-red-500/10 to-red-600/5 border-red-500/20 hover:border-red-500/40",
                color === 'emerald' && "bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20 hover:border-emerald-500/40",
                color === 'purple' && "bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20 hover:border-purple-500/40"
              )}
            >
              <div className="flex items-center gap-2 mb-3">
                <div className={cn(
                  "w-7 h-7 rounded-lg flex items-center justify-center",
                  color === 'amber' && "bg-amber-500/20",
                  color === 'blue' && "bg-blue-500/20",
                  color === 'red' && "bg-red-500/20",
                  color === 'emerald' && "bg-emerald-500/20",
                  color === 'purple' && "bg-purple-500/20"
                )}>
                  <Icon className={cn(
                    "w-3.5 h-3.5",
                    color === 'amber' && "text-amber-400",
                    color === 'blue' && "text-blue-400",
                    color === 'red' && "text-red-400",
                    color === 'emerald' && "text-emerald-400",
                    color === 'purple' && "text-purple-400"
                  )} />
                </div>
                <span className="text-[11px] uppercase tracking-wider text-white/40 font-medium">{label}</span>
              </div>
              <div className={cn(
                "text-2xl font-bold font-mono tracking-tight",
                color === 'amber' && "text-amber-100",
                color === 'blue' && "text-blue-100",
                color === 'red' && "text-red-100",
                color === 'emerald' && "text-emerald-100",
                color === 'purple' && "text-purple-100"
              )}>{isText ? value : value}</div>
            </div>
          ))}
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-purple-400/50" />
          </div>
        )}

        {/* Two Column Layout */}
        {!isLoading && (
        <div className="grid grid-cols-3 gap-6">
          {/* Models List */}
          <div className="col-span-1 space-y-3">
            <h2 className="text-[10px] font-semibold text-white/40 uppercase tracking-widest mb-4">All Workspaces</h2>
            
            {models.length === 0 ? (
              <div className="bg-white/5 rounded-xl border border-white/10 p-6 text-center">
                <AlertTriangle className="w-8 h-8 text-amber-400 mx-auto mb-2" />
                <p className="text-sm text-white/50 mb-2">No investor workspaces yet</p>
                <p className="text-xs text-white/30 mb-4">Real investors and their financial models will appear here once they create their workspaces.</p>
                <button
                  onClick={() => setShowNewModelForm(true)}
                  className="flex items-center gap-2 px-3 py-2 bg-emerald-600 hover:bg-emerald-700 rounded-lg text-sm w-full justify-center transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  Create Workspace
                </button>
              </div>
            ) : (
              models.map((model) => {
                const mergedInputs = { ...DEFAULT_INPUTS, ...model.model_inputs }
                const data = calculateModel(mergedInputs, model.scenario)
                const modelBreakEven = data.findIndex(d => d.selfSustaining >= 0)
                
                return (
                  <button
                    key={model.id}
                    onClick={() => setSelectedModel(model)}
                    className={cn(
                      "w-full text-left p-4 rounded-xl border transition-all",
                      selectedModel?.id === model.id
                        ? "bg-white/10 border-white/30"
                        : "bg-white/5 border-white/10 hover:bg-white/[0.07]"
                    )}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h3 className="font-medium">{model.investor_name}</h3>
                        <p className="text-sm text-white/50">{model.organization}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={cn(
                          "text-[10px] px-1.5 py-0.5 rounded font-mono",
                          model.scenario === 'conservative' && "bg-blue-500/20 text-blue-300",
                          model.scenario === 'baseline' && "bg-amber-500/20 text-amber-300",
                          model.scenario === 'optimistic' && "bg-emerald-500/20 text-emerald-300",
                        )}>
                          {model.scenario.toUpperCase()}
                        </span>
                        {model.is_public ? (
                          <Globe className="w-3 h-3 text-emerald-400" />
                        ) : (
                          <Lock className="w-3 h-3 text-white/30" />
                        )}
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                      <div className="bg-white/5 rounded p-2">
                        <p className="text-white/40">Y5 Revenue</p>
                        <p className="font-mono font-semibold">{fmt(data[4].totalRevenue, 1)}</p>
                      </div>
                      <div className="bg-white/5 rounded p-2">
                        <p className="text-white/40">Break-Even</p>
                        <p className={cn("font-mono font-semibold", modelBreakEven === -1 ? "text-red-400" : "text-emerald-400")}>
                          {modelBreakEven !== -1 ? `Year ${modelBreakEven + 1}` : '5+'}
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4 text-[10px] text-white/40">
                      <span className="flex items-center gap-1">
                        <Eye className="w-3 h-3" /> {model.view_count}
                      </span>
                      <span className="flex items-center gap-1">
                        <MessageSquare className="w-3 h-3" /> {model.questions.length}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" /> {formatRelativeTime(model.updated_at)}
                      </span>
                    </div>
                  </button>
                )
              })
            )}
          </div>

          {/* Model Detail */}
          <div className="col-span-2">
            {models.length === 0 ? (
              <div className="bg-white/5 rounded-xl border border-white/10 p-8 text-center h-full flex items-center justify-center">
                <div>
                  <Globe className="w-12 h-12 text-white/20 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Investor Workspaces</h3>
                  <p className="text-white/50 mb-6 max-w-sm">Real investors will create their own financial models here. Each workspace is a living document where they can model different scenarios, ask questions, and collaborate on investment decisions.</p>
                  <div className="space-y-2 text-sm text-white/40">
                    <p>✓ Transparency: All models are public by default</p>
                    <p>✓ Reality: No templates, no bullshit — just real numbers</p>
                    <p>✓ Integrity: Investors build trust through honest modeling</p>
                  </div>
                </div>
              </div>
            ) : selectedModel && modelData ? (
              <div className="space-y-4">
                {/* Header */}
                <div className="bg-white/5 rounded-xl border border-white/10 p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h2 className="text-xl font-semibold mb-1">{selectedModel.investor_name}</h2>
                      <p className="text-white/50">{selectedModel.organization}</p>
                      <div className="flex gap-2 mt-2">
                        {selectedModel.focus_areas.map(area => (
                          <span key={area} className="text-[10px] px-2 py-0.5 bg-white/10 rounded-full">{area}</span>
                        ))}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={copyModel} className="flex items-center gap-2 px-3 py-2 bg-white/10 rounded-lg text-sm hover:bg-white/15">
                        {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                        Copy
                      </button>
                      <button onClick={downloadCSV} className="flex items-center gap-2 px-3 py-2 bg-white/10 rounded-lg text-sm hover:bg-white/15">
                        <Download className="w-4 h-4" />
                        CSV
                      </button>
                      <Link
                        href={`/ziggurat/spreadsheet`}
                        className="flex items-center gap-2 px-3 py-2 bg-amber-500 text-black rounded-lg text-sm font-medium hover:bg-amber-400"
                      >
                        <FileSpreadsheet className="w-4 h-4" />
                        Open Full Model
                      </Link>
                    </div>
                  </div>
                  
                  {selectedModel.notes && (
                    <p className="text-sm text-white/70 bg-white/5 rounded-lg p-3">
                      {selectedModel.notes}
                    </p>
                  )}
                </div>

                {/* Quick Summary */}
                <div className="bg-white/5 rounded-xl border border-white/10 overflow-hidden">
                  <button
                    onClick={() => toggleSection('summary')}
                    className="w-full flex items-center justify-between p-4 hover:bg-white/5"
                  >
                    <div className="flex items-center gap-3">
                      <Calculator className="w-5 h-5 text-amber-400" />
                      <span className="font-semibold">5-Year Summary</span>
                      <span className={cn(
                        "text-xs px-2 py-0.5 rounded font-mono",
                        selectedModel.scenario === 'conservative' && "bg-blue-500/20 text-blue-300",
                        selectedModel.scenario === 'baseline' && "bg-amber-500/20 text-amber-300",
                        selectedModel.scenario === 'optimistic' && "bg-emerald-500/20 text-emerald-300",
                      )}>
                        {selectedModel.scenario} ({SCENARIOS[selectedModel.scenario].mult}x)
                      </span>
                    </div>
                    {expandedSections.includes('summary') ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                  
                  {expandedSections.includes('summary') && (
                    <div className="p-4 border-t border-white/10 overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-white/40 text-xs uppercase tracking-wider">
                            <th className="text-left py-2 px-3">Metric</th>
                            {[1, 2, 3, 4, 5].map(y => (
                              <th key={y} className="text-right py-2 px-3 w-24">Y{y}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                          <tr className="hover:bg-white/5">
                            <td className="py-2 px-3">Product Revenue</td>
                            {modelData.map(d => (
                              <td key={d.year} className="py-2 px-3 text-right font-mono text-emerald-400/80">{fmt(d.totalProductRev, 1)}</td>
                            ))}
                          </tr>
                          <tr className="hover:bg-white/5 text-white/60">
                            <td className="py-2 px-3">Philanthropy</td>
                            {modelData.map(d => (
                              <td key={d.year} className="py-2 px-3 text-right font-mono">{fmt(d.philanthropy, 1)}</td>
                            ))}
                          </tr>
                          <tr className="bg-amber-500/5 font-semibold">
                            <td className="py-2 px-3">Total Revenue</td>
                            {modelData.map(d => (
                              <td key={d.year} className="py-2 px-3 text-right font-mono">{fmt(d.totalRevenue, 1)}</td>
                            ))}
                          </tr>
                          <tr className="hover:bg-white/5 text-red-400/80">
                            <td className="py-2 px-3">Operating Expenses</td>
                            {modelData.map(d => (
                              <td key={d.year} className="py-2 px-3 text-right font-mono">({fmt(d.totalOpex, 1)})</td>
                            ))}
                          </tr>
                          <tr className={cn("font-bold", hasBreakEven ? "bg-emerald-500/10" : "bg-red-500/10")}>
                            <td className="py-2 px-3">Self-Sustaining</td>
                            {modelData.map(d => (
                              <td key={d.year} className={cn("py-2 px-3 text-right font-mono", d.selfSustaining >= 0 ? "text-emerald-400" : "text-red-400")}>
                                {fmt(d.selfSustaining, 1)}
                              </td>
                            ))}
                          </tr>
                          <tr className="hover:bg-white/5">
                            <td className="py-2 px-3">Daily Learners</td>
                            {modelData.map(d => (
                              <td key={d.year} className="py-2 px-3 text-right font-mono text-purple-400">{fmtRaw(d.dailyLearners)}</td>
                            ))}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                {/* Break-Even Analysis */}
                <div className={cn(
                  "rounded-xl border overflow-hidden",
                  hasBreakEven ? "bg-emerald-500/5 border-emerald-500/30" : "bg-red-500/5 border-red-500/30"
                )}>
                  <button
                    onClick={() => toggleSection('breakeven')}
                    className="w-full flex items-center justify-between p-4 hover:bg-white/5"
                  >
                    <div className="flex items-center gap-3">
                      <Target className={cn("w-5 h-5", hasBreakEven ? "text-emerald-400" : "text-red-400")} />
                      <span className="font-semibold">Break-Even Analysis</span>
                      <span className={cn("text-xs px-2 py-0.5 rounded font-mono", hasBreakEven ? "bg-emerald-500/20 text-emerald-300" : "bg-red-500/20 text-red-300")}>
                        {hasBreakEven ? `Year ${(breakEvenYear ?? 0) + 1}` : 'Beyond Y5'}
                      </span>
                    </div>
                    {expandedSections.includes('breakeven') ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                  
                  {expandedSections.includes('breakeven') && (
                    <div className="p-4 border-t border-white/10">
                      <div className="grid grid-cols-3 gap-4 mb-4">
                        <div className="bg-white/5 rounded-lg p-4">
                          <p className="text-xs text-white/40 mb-1">Self-Sustaining Year 5</p>
                          <p className={cn("text-2xl font-bold font-mono", modelData[4].selfSustaining >= 0 ? "text-emerald-400" : "text-red-400")}>
                            {fmt(modelData[4].selfSustaining, 1)}
                          </p>
                        </div>
                        <div className="bg-white/5 rounded-lg p-4">
                          <p className="text-xs text-white/40 mb-1">Y5 Philanthropy Dependency</p>
                          <p className={cn("text-2xl font-bold font-mono", modelData[4].philanthroDep < 0.3 ? "text-emerald-400" : "text-amber-400")}>
                            {fmtPct(modelData[4].philanthroDep)}
                          </p>
                        </div>
                        <div className="bg-white/5 rounded-lg p-4">
                          <p className="text-xs text-white/40 mb-1">5-Year Total Product Rev</p>
                          <p className="text-2xl font-bold font-mono text-emerald-400">
                            {fmt(modelData.reduce((s, d) => s + d.totalProductRev, 0), 0)}
                          </p>
                        </div>
                      </div>
                      
                      {!hasBreakEven && (
                        <div className="flex items-start gap-3 p-3 bg-red-500/10 rounded-lg">
                          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                          <div className="text-sm">
                            <p className="font-medium text-red-300 mb-1">Break-even not achieved by Year 5</p>
                            <p className="text-white/60">Consider adjusting capital structure, increasing revenue assumptions, or extending the philanthropic runway.</p>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Learner-Affiliate Impact */}
                <div className="bg-purple-500/5 rounded-xl border border-purple-500/30 overflow-hidden">
                  <button
                    onClick={() => toggleSection('affiliate')}
                    className="w-full flex items-center justify-between p-4 hover:bg-white/5"
                  >
                    <div className="flex items-center gap-3">
                      <Zap className="w-5 h-5 text-purple-400" />
                      <span className="font-semibold">Learner-Affiliate Economic Model</span>
                    </div>
                    {expandedSections.includes('affiliate') ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                  
                  {expandedSections.includes('affiliate') && (
                    <div className="p-4 border-t border-white/10">
                      <p className="text-sm text-white/60 mb-4">
                        {AFFILIATE_MODEL.description}
                      </p>
                      
                      <div className="grid grid-cols-4 gap-3 mb-4">
                        <div className="bg-white/5 rounded-lg p-3 text-center">
                          <p className="text-xs text-white/40 mb-1">Learner Share</p>
                          <p className="text-lg font-bold text-purple-400">{fmtPct(AFFILIATE_MODEL.learnerShare)}</p>
                        </div>
                        <div className="bg-white/5 rounded-lg p-3 text-center">
                          <p className="text-xs text-white/40 mb-1">Avg Commission</p>
                          <p className="text-lg font-bold font-mono">${AFFILIATE_MODEL.avgCommissionPerConvert}</p>
                        </div>
                        <div className="bg-white/5 rounded-lg p-3 text-center">
                          <p className="text-xs text-white/40 mb-1">Y5 Daily Learners</p>
                          <p className="text-lg font-bold font-mono">{fmtRaw(modelData[4].dailyLearners)}</p>
                        </div>
                        <div className="bg-white/5 rounded-lg p-3 text-center">
                          <p className="text-xs text-white/40 mb-1">Y5 Affiliate Rev</p>
                          <p className="text-lg font-bold text-emerald-400">{fmt(modelData[4].affiliateRevenue, 1)}</p>
                        </div>
                      </div>
                      
                      <div className="flex gap-2">
                        {AFFILIATE_MODEL.tiers.map(tier => (
                          <div key={tier.name} className="flex-1 bg-white/5 rounded-lg p-2 text-center">
                            <p className="text-xs font-medium">{tier.name}</p>
                            <p className="text-[10px] text-white/40">{tier.minShares}+ shares</p>
                            <p className="text-xs text-purple-400">{tier.bonus}x bonus</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Questions */}
                <div className="bg-white/5 rounded-xl border border-white/10 p-4">
                  <h3 className="text-sm font-medium text-white/50 uppercase tracking-wider mb-4">
                    Questions ({selectedModel.questions.length})
                  </h3>
                  
                  <div className="space-y-4 mb-4">
                    {selectedModel.questions.map((q) => (
                      <div key={q.id} className="bg-white/5 rounded-lg p-4">
                        <div className="flex items-start justify-between mb-2">
                          <p className="text-sm font-medium">{q.author}</p>
                          <p className="text-xs text-white/40">{formatRelativeTime(q.created_at)}</p>
                        </div>
                        <p className="text-white/80 mb-3">{q.content}</p>
                        
                        {q.replies.map((r) => (
                          <div key={r.id} className="ml-4 pl-4 border-l border-white/10 mt-3">
                            <div className="flex items-start justify-between mb-1">
                              <p className="text-sm font-medium text-emerald-400">{r.author}</p>
                              <p className="text-xs text-white/40">{formatRelativeTime(r.created_at)}</p>
                            </div>
                            <p className="text-sm text-white/70">{r.content}</p>
                          </div>
                        ))}
                      </div>
                    ))}
                    
                    {selectedModel.questions.length === 0 && (
                      <p className="text-sm text-white/40 text-center py-4">No questions yet</p>
                    )}
                  </div>
                  
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={newQuestion}
                      onChange={(e) => setNewQuestion(e.target.value)}
                      placeholder="Ask a question about this model..."
                      className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-sm placeholder:text-white/30 focus:outline-none focus:border-white/30"
                    />
                    <button className="px-4 py-2 bg-white text-black rounded-lg text-sm font-medium hover:bg-white/90">
                      Ask
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-white/30 min-h-[400px]">
                <div className="text-center">
                  <Building2 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Select a workspace to view details</p>
                  <p className="text-sm mt-2">Each investor can create their own model with custom assumptions</p>
                </div>
              </div>
            )}
          </div>
        </div>
        )}
        {/* Investor Dashboard - All Features Surfaced */}
        <section className="mt-12 mb-8">
          <div className="flex items-center gap-3 mb-6">
            <h2 className="text-lg font-semibold">Investor Command Center</h2>
            <span className="text-xs px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded-full">Live</span>
          </div>
          <InvestorDashboard />
        </section>

        {/* Open Coalition Invitation */}
        <section className="mt-16 mb-8">
          <div className="max-w-4xl mx-auto">
            <div className="rounded-2xl border border-white/10 bg-gradient-to-b from-white/[0.03] to-transparent overflow-hidden">
              <div className="p-6 border-b border-white/[0.06]">
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white/90">Join the Coalition</h2>
                    <p className="text-sm text-white/40 mt-1">The Ziggurat is open to all investors committed to lifelong learning</p>
                  </div>
                  <span className="px-3 py-1 bg-emerald-500/10 text-emerald-400 text-xs font-medium rounded-full border border-emerald-500/20">
                    Open Invitation
                  </span>
                </div>
              </div>
              
              <div className="p-6 space-y-4">
                <p className="text-sm text-white/60 leading-relaxed">
                  Everyone who interacts with Kelly automatically earns in exchange for their language contribution. 
                  We aren&apos;t stealing voices - we&apos;re preserving them. The Ziggurat is where voices come most alive, 
                  lifting champions and celebrating birthdays every single day.
                </p>
                
                <div className="grid grid-cols-3 gap-4 py-4">
                  <div className="text-center p-3 rounded-lg bg-white/[0.02]">
                    <div className="text-2xl font-bold text-white/90">$309M</div>
                    <div className="text-[10px] text-white/40 uppercase tracking-wider mt-1">Phase 1</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-white/[0.02]">
                    <div className="text-2xl font-bold text-emerald-400">100M</div>
                    <div className="text-[10px] text-white/40 uppercase tracking-wider mt-1">Y5 Daily Learners</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-white/[0.02]">
                    <div className="text-2xl font-bold text-amber-400">365</div>
                    <div className="text-[10px] text-white/40 uppercase tracking-wider mt-1">Birthdays Celebrated</div>
                  </div>
                </div>
              </div>
              
              {/* Claude Artifact - minimal embed with Anthropic credit */}
              <div className="border-t border-white/[0.06]">
                <div className="p-4 bg-black/20">
                  <iframe 
                    src="https://claude.site/public/artifacts/2398c844-8df3-4085-8dc6-b26624ac9af7/embed" 
                    title="Coalition Investment Model" 
                    className="w-full rounded-lg"
                    style={{ height: '400px' }}
                    allow="clipboard-write"
                  />
                  <p className="text-[10px] text-white/30 text-center mt-3">
                    Interactive model built with Claude by Anthropic
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* New Model Form Modal */}
      {showNewModelForm && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-zinc-900 rounded-2xl border border-white/10 w-full max-w-lg p-6">
            <h2 className="text-xl font-semibold mb-6">Create Your Model</h2>
            
            <form className="space-y-4">
              <div>
                <label className="block text-sm text-white/50 mb-2">Your Name</label>
                <input
                  type="text"
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 focus:outline-none focus:border-white/30"
                  placeholder="Full name"
                />
              </div>
              
              <div>
                <label className="block text-sm text-white/50 mb-2">Email</label>
                <input
                  type="email"
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 focus:outline-none focus:border-white/30"
                  placeholder="you@organization.com"
                />
              </div>
              
              <div>
                <label className="block text-sm text-white/50 mb-2">Organization</label>
                <input
                  type="text"
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 focus:outline-none focus:border-white/30"
                  placeholder="Your foundation or firm"
                />
              </div>
              
              <div>
                <label className="block text-sm text-white/50 mb-2">Starting Scenario</label>
                <select className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 focus:outline-none focus:border-white/30">
                  <option value="baseline">Baseline (1.0x)</option>
                  <option value="conservative">Conservative (0.7x)</option>
                  <option value="optimistic">Optimistic (1.5x)</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm text-white/50 mb-2">Focus Areas</label>
                <textarea
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 focus:outline-none focus:border-white/30 h-24 resize-none"
                  placeholder="What aspects of the Ziggurat are you most interested in? (e.g., Break-even timeline, EdTech scalability, Historic preservation)"
                />
              </div>
              
              <div className="flex items-center gap-3 pt-2">
                <input type="checkbox" id="public" className="w-4 h-4" defaultChecked />
                <label htmlFor="public" className="text-sm text-white/70">
                  Make my workspace public (others can learn from your analysis)
                </label>
              </div>
              
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowNewModelForm(false)}
                  className="flex-1 px-4 py-3 border border-white/20 rounded-lg text-sm hover:bg-white/5"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-3 bg-white text-black rounded-lg text-sm font-medium hover:bg-white/90"
                >
                  Create Workspace
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
