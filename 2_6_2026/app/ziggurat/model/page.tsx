// ziggurat-component.tsx
// Main React component for Ziggurat Financial Model
// Ready for v0.app enhancement

"use client";

import React, { useState, useMemo, useEffect, useRef } from 'react';
import { 
  Building2, Users, DollarSign, TrendingUp, Package, Monitor, 
  FileText, MessageSquare, ChevronDown, Info, Target, Layers, 
  BarChart3, CheckCircle2, AlertTriangle, Zap, Globe, GraduationCap, 
  Heart, Building, Landmark, X, Sparkles, Clock, ExternalLink, 
  Film, Cpu, Share2, Copy, Check
} from 'lucide-react';

import { 
  BUILDING, 
  KELLY_VISION, 
  DEFAULT_INPUTS, 
  GROWTH, 
  SCENARIOS, 
  SDG_GOALS,
  SLIDER_CONFIG,
  COPY,
  COLORS 
} from '@/lib/ziggurat/data';

import type { 
  Scenario, 
  FinancialInputs, 
  YearCalculation, 
  FullCalculation,
  FeedbackForm 
} from '@/lib/ziggurat/types';

// ============================================================
// UTILITIES
// ============================================================

const fmt = (n: number | null | undefined, decimals = 1): string => {
  if (n === null || n === undefined || isNaN(n)) return '—';
  const neg = n < 0;
  const abs = Math.abs(n);
  let result: string;
  if (abs >= 1e9) result = `$${(abs / 1e9).toFixed(decimals)}B`;
  else if (abs >= 1e6) result = `$${(abs / 1e6).toFixed(decimals)}M`;
  else if (abs >= 1e3) result = `$${(abs / 1e3).toFixed(0)}K`;
  else result = `$${abs.toLocaleString()}`;
  return neg ? `(${result})` : result;
};

const fmtNum = (n: number): string => Math.round(n ?? 0).toLocaleString();
const fmtPct = (n: number): string => `${((n ?? 0) * 100).toFixed(1)}%`;

// Animated value hook
function useAnimatedValue(target: number, duration = 800): number {
  const [value, setValue] = useState(0);
  const prev = useRef(0);
  
  useEffect(() => {
    const start = prev.current;
    const startTime = Date.now();
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(start + (target - start) * eased);
      if (progress < 1) requestAnimationFrame(animate);
      else prev.current = target;
    };
    
    requestAnimationFrame(animate);
  }, [target, duration]);
  
  return value;
}

// ============================================================
// CALCULATION ENGINE
// ============================================================

function calculateFinancials(
  inputs: FinancialInputs, 
  viewYear: number, 
  scenario: Scenario
): FullCalculation {
  const mult = SCENARIOS[scenario].mult;
  const i = inputs;
  const g = GROWTH;
  
  const years: YearCalculation[] = [1, 2, 3, 4, 5].map(year => {
    const y = year - 1;
    
    const headcount = Math.round(i.headcountY1 + (i.headcountY5 - i.headcountY1) * (y / 4));
    const salaryGrowth = Math.pow(1.03, y);
    const payroll = headcount * i.avgSalary * salaryGrowth;
    const contractors = 3000000 * g.contractors[y];

    // Revenue
    const monthlyRev = i.monthlySubsY1 * g.subs[y] * mult * i.monthlyPrice * 12;
    const annualRev = i.annualSubsY1 * g.subs[y] * mult * i.annualPrice;
    const cumDistricts = i.districtDealsY1 * g.district[y] * mult;
    const districtRev = cumDistricts * i.studentsPerDistrict * i.pricePerStudent;
    const onlineTotal = monthlyRev + annualRev + districtRev;
    const onlineCosts = onlineTotal * 0.18;

    const digitalRev = (i.monthlyPackY1 * 9 + i.completeYearY1 * 29 + i.premiumYearY1 * 49) * g.digital[y] * mult;
    const digitalCosts = digitalRev * 0.05;

    const printUnits = (i.essentialsY1 + i.standardY1 + i.collectorsY1) * g.print[y] * mult;
    const printRev = (i.essentialsY1 * 79 + i.standardY1 * 99 + i.collectorsY1 * 149) * g.print[y] * mult;
    const printCOGS = printUnits * (45 * g.cogsScale[y] + 8) + printRev * 0.08;

    const liveRev = [5, 15, 30, 50, 75][y] * 52 * [50, 75, 100, 150, 200][y] * [0.3, 0.25, 0.2, 0.15, 0.1][y] * 5 * mult;
    const liveCosts = [5, 15, 30, 50, 75][y] * 52 * 100 + liveRev * 0.1;

    const hwRev = year >= 3 ? (i.iLearnMiniY3 * 149 + i.iLearnStdY3 * 299 + i.iLearnProY3 * 499) * g.hardware[y] * mult : 0;
    const hwCOGS = hwRev * (1 - g.hwMargin[y]);

    const licenseRev = i.enterpriseDealsY1 * g.licensing[y] * mult * i.enterpriseDealSize * Math.pow(1.1, y) +
      (year >= 2 ? i.govDealsY3 * [0, 0.33, 1, 1.67, 2.67][y] * mult * i.govDealSize * Math.pow(1.2, Math.max(0, y - 1)) : 0);
    const licenseCosts = licenseRev * 0.2;

    const totalProductRev = onlineTotal + digitalRev + printRev + liveRev + hwRev + licenseRev;
    const philanthroOperating = i.philanthroYearlyY1 * g.philanthropy[y];
    const totalRevenue = totalProductRev + philanthroOperating;

    const totalCOR = onlineCosts + digitalCosts + printCOGS + liveCosts + hwCOGS + licenseCosts;
    const grossProfit = totalRevenue - totalCOR;

    const totalOpex = payroll + contractors +
      i.buildingOps * g.building[y] +
      i.techInfra * g.tech[y] +
      i.contentProd * g.content[y] +
      i.marketing * g.marketing[y] +
      i.gaLegal * g.ga[y];

    const netOperating = grossProfit - totalOpex;
    const selfSustaining = totalProductRev - totalCOR - totalOpex;

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
    };
  });

  // Capital needed - aligned with pitch ($700M-$925M)
  const acquisitionTotal = i.purchasePrice * 1.085; // Including closing costs
  const renovationTotal = (i.renovationBuildout || 0) + (i.ledDisplay || 0) + (i.techInfrastructure || 0) + (i.softCostsContingency || 0);
  const totalCapitalNeeded = i.purchasePrice + renovationTotal + i.operatingReserve;
  
  // Capital sources - aligned with pitch capital structure
  const equityTotal = (i.strategicPartnerEquity || 0) + (i.institutionalEquity || 0);
  const totalCapitalRaised = equityTotal + (i.historicTaxCredits || 0) + (i.debtFinancing || 0);
  const fundingGap = totalCapitalNeeded - totalCapitalRaised;
  const fundingPercent = totalCapitalNeeded > 0 ? (totalCapitalRaised / totalCapitalNeeded) * 100 : 0;

  const current = viewYear === 0 ? years[4] : years[viewYear - 1];
  const breakEvenYear = years.findIndex(y => y.selfSustaining >= 0) + 1 || null;

  return {
    years,
    current,
    philanthropyTotal: equityTotal, // Use equity for display
    acquisitionTotal,
    totalCapitalNeeded,
    totalCapitalRaised,
    fundingGap,
    fundingPercent,
    totalRenovation: renovationTotal,
    breakEvenYear,
    fiveYearProductRev: years.reduce((s, y) => s + y.totalProductRev, 0),
    fiveYearPhilanthropy: years.reduce((s, y) => s + y.philanthroOperating, 0),
  };
}

// ============================================================
// COMPONENTS
// ============================================================

// Glass Card
function GlassCard({ 
  children, 
  className = "", 
  onClick 
}: { 
  children: React.ReactNode; 
  className?: string;
  onClick?: () => void;
}) {
  return (
    <div 
      onClick={onClick}
      className={`
        relative overflow-hidden rounded-2xl
        bg-white/5 backdrop-blur-xl
        border border-white/10
        hover:bg-white/10 hover:border-white/20 
        transition-all duration-300
        ${onClick ? 'cursor-pointer' : ''}
        ${className}
      `}
    >
      {children}
    </div>
  );
}

// Animated Metric
function AnimatedMetric({ 
  label, 
  value, 
  format = 'currency', 
  icon: Icon, 
  color = 'amber',
  subtitle 
}: {
  label: string;
  value: number | string;
  format?: 'currency' | 'number' | 'percent' | 'text';
  icon?: React.ComponentType<{ size?: number; className?: string }>;
  color?: 'amber' | 'emerald' | 'red' | 'blue' | 'purple';
  subtitle?: string;
}) {
  const animated = useAnimatedValue(typeof value === 'number' ? value : 0);
  const display = format === 'currency' ? fmt(animated) : 
                  format === 'percent' ? fmtPct(animated) : 
                  format === 'text' ? value :
                  fmtNum(animated);
  const isNeg = typeof value === 'number' && value < 0;
  
  const colorClasses = {
    amber: 'from-amber-500/20 to-amber-600/5 border-amber-500/30 text-amber-400',
    emerald: 'from-emerald-500/20 to-emerald-600/5 border-emerald-500/30 text-emerald-400',
    red: 'from-red-500/20 to-red-600/5 border-red-500/30 text-red-400',
    blue: 'from-blue-500/20 to-blue-600/5 border-blue-500/30 text-blue-400',
    purple: 'from-purple-500/20 to-purple-600/5 border-purple-500/30 text-purple-400',
  };
  
  return (
    <GlassCard className={`p-5 bg-gradient-to-br ${colorClasses[color]}`}>
      <div className="flex items-start justify-between mb-3">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</span>
        {Icon && <Icon size={20} className={colorClasses[color].split(' ').pop()} />}
      </div>
      <div className={`text-3xl font-bold font-mono tracking-tight ${isNeg ? 'text-red-400' : colorClasses[color].split(' ').pop()}`}>
        {display}
      </div>
      {subtitle && <div className="text-xs text-slate-500 mt-2">{subtitle}</div>}
    </GlassCard>
  );
}

// Premium Slider
function PremiumSlider({ 
  label, 
  value, 
  onChange, 
  min, 
  max, 
  step = 1, 
  format = 'currency',
  info 
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  format?: 'currency' | 'number' | 'percent';
  info?: string;
}) {
  const display = format === 'currency' ? fmt(value) : format === 'percent' ? fmtPct(value) : fmtNum(value);
  const pct = ((value - min) / (max - min)) * 100;
  
  return (
    <div className="group mb-6">
      <div className="flex justify-between items-center mb-3">
        <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
          {label}
          {info && (
            <span className="relative group/tip">
              <Info size={14} className="text-slate-500 cursor-help hover:text-amber-400" />
              <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 text-xs bg-slate-800 rounded-lg opacity-0 group-hover/tip:opacity-100 transition-opacity whitespace-nowrap z-30 border border-slate-700">
                {info}
              </span>
            </span>
          )}
        </label>
        <span className="text-sm font-mono text-amber-400 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700/50">
          {display}
        </span>
      </div>
      <div className="relative h-3">
        <div className="absolute inset-0 bg-slate-800 rounded-full overflow-hidden shadow-inner">
          <div 
            className="h-full bg-gradient-to-r from-amber-600 via-amber-500 to-amber-400 rounded-full transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full opacity-0 cursor-pointer"
        />
        <div 
          className="absolute top-1/2 -translate-y-1/2 w-5 h-5 bg-white rounded-full shadow-lg pointer-events-none ring-4 ring-amber-500/30 group-hover:ring-amber-500/50 transition-all"
          style={{ left: `calc(${pct}% - 10px)` }}
        />
      </div>
    </div>
  );
}

// Section Accordion
function Section({ 
  title, 
  icon: Icon, 
  children, 
  defaultOpen = true, 
  badge 
}: {
  title: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badge?: string;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  return (
    <GlassCard className="mb-4">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-5 py-4 flex items-center justify-between hover:bg-white/5 transition-colors rounded-t-2xl"
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
            <Icon size={20} className="text-amber-400" />
          </div>
          <span className="font-semibold text-white text-lg">{title}</span>
          {badge && (
            <span className="px-3 py-1 text-xs font-mono bg-amber-500/20 text-amber-400 rounded-full">
              {badge}
            </span>
          )}
        </div>
        <ChevronDown 
          size={20} 
          className={`text-slate-400 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} 
        />
      </button>
      <div className={`overflow-hidden transition-all duration-500 ${isOpen ? 'max-h-[2000px] opacity-100' : 'max-h-0 opacity-0'}`}>
        <div className="p-5 pt-0 border-t border-white/5">{children}</div>
      </div>
    </GlassCard>
  );
}

// Tab Button
function TabButton({ 
  active, 
  onClick, 
  icon: Icon, 
  label 
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-2 px-5 py-3 rounded-xl font-medium text-sm whitespace-nowrap
        transition-all duration-300
        ${active 
          ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-slate-900 shadow-lg shadow-amber-500/25' 
          : 'text-slate-400 hover:text-white hover:bg-white/5'
        }
      `}
    >
      <Icon size={18} />
      <span>{label}</span>
    </button>
  );
}

// Year Selector
function YearSelector({ year, setYear }: { year: number; setYear: (y: number) => void }) {
  return (
    <div className="inline-flex bg-slate-800/60 backdrop-blur rounded-xl p-1.5 gap-1 border border-white/10">
      {[1, 2, 3, 4, 5].map((y) => (
        <button
          key={y}
          onClick={() => setYear(y)}
          className={`
            w-12 h-10 rounded-lg font-medium text-sm transition-all duration-300
            ${year === y 
              ? 'bg-amber-500 text-slate-900 shadow-lg shadow-amber-500/30' 
              : 'text-slate-400 hover:text-white hover:bg-white/10'
            }
          `}
        >
          Y{y}
        </button>
      ))}
      <button
        onClick={() => setYear(0)}
        className={`
          px-4 h-10 rounded-lg font-medium text-sm transition-all
          ${year === 0 
            ? 'bg-amber-500 text-slate-900 shadow-lg' 
            : 'text-slate-400 hover:text-white hover:bg-white/10'
          }
        `}
      >
        All
      </button>
    </div>
  );
}

// Progress Ring
function ProgressRing({ 
  percent, 
  size = 100, 
  color = 'amber',
  label 
}: {
  percent: number;
  size?: number;
  color?: 'amber' | 'emerald' | 'red' | 'blue';
  label?: string;
}) {
  const radius = (size - 10) / 2;
  const circumference = radius * 2 * Math.PI;
  const animated = useAnimatedValue(Math.min(percent, 100), 1000);
  const offset = circumference - (animated / 100) * circumference;
  
  const colors = { 
    amber: '#f59e0b', 
    emerald: '#10b981', 
    red: '#ef4444', 
    blue: '#3b82f6' 
  };
  
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#1e293b"
          strokeWidth={10}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={colors[color]}
          strokeWidth={10}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-1000"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-white">{Math.round(animated)}%</span>
        {label && <span className="text-xs text-slate-400 mt-1">{label}</span>}
      </div>
    </div>
  );
}

// Stacked Bar
function StackedBar({ 
  segments, 
  total 
}: {
  segments: { label: string; value: number; color: string }[];
  total: number;
}) {
  const valid = segments.filter(s => s.value > 0);
  
  return (
    <div className="space-y-3">
      <div className="h-8 rounded-full overflow-hidden flex bg-slate-800 shadow-inner">
        {valid.map((seg, i) => (
          <div
            key={i}
            className={`${seg.color} first:rounded-l-full last:rounded-r-full transition-all duration-700`}
            style={{ width: `${(seg.value / total) * 100}%` }}
            title={`${seg.label}: ${fmt(seg.value)}`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-2">
        {valid.map((seg, i) => (
          <div key={i} className="flex items-center gap-2 text-sm">
            <div className={`w-3 h-3 rounded-full ${seg.color}`} />
            <span className="text-slate-400">{seg.label}</span>
            <span className="font-mono text-slate-300">{fmtPct(seg.value / total)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Timeline
function Timeline({ events }: { events: typeof BUILDING.history }) {
  const typeColors = {
    build: 'bg-blue-500',
    sale: 'bg-amber-500',
    legal: 'bg-red-500',
    culture: 'bg-purple-500',
    opportunity: 'bg-emerald-500',
  };
  
  return (
    <div className="relative">
      <div className="absolute left-6 top-0 bottom-0 w-px bg-gradient-to-b from-amber-500 via-slate-600 to-slate-700" />
      <div className="space-y-4">
        {events.map((event, i) => (
          <div key={i} className="flex gap-4 items-start">
            <div className={`relative z-10 w-12 h-12 rounded-xl ${typeColors[event.type]} flex items-center justify-center shadow-lg`}>
              <span className="text-xs font-bold text-white">{event.year}</span>
            </div>
            <div className={`flex-1 p-4 rounded-xl ${event.type === 'opportunity' ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-white/5'}`}>
              <p className={`text-sm ${event.type === 'opportunity' ? 'text-emerald-300 font-medium' : 'text-slate-300'}`}>
                {event.event}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Floor Visualization
function FloorVisualization({ floors }: { floors: typeof BUILDING.floorPlan }) {
  const colorMap: Record<string, string> = {
    amber: 'from-amber-500/30 to-amber-600/20 border-amber-500/40',
    pink: 'from-pink-500/30 to-pink-600/20 border-pink-500/40',
    blue: 'from-blue-500/30 to-blue-600/20 border-blue-500/40',
    purple: 'from-purple-500/30 to-purple-600/20 border-purple-500/40',
    emerald: 'from-emerald-500/30 to-emerald-600/20 border-emerald-500/40',
    cyan: 'from-cyan-500/30 to-cyan-600/20 border-cyan-500/40',
    violet: 'from-violet-500/30 to-violet-600/20 border-violet-500/40',
    orange: 'from-orange-500/30 to-orange-600/20 border-orange-500/40',
  };
  
  return (
    <div className="flex flex-col items-center gap-1">
      {[...floors].reverse().map((floor, i) => {
        const widthPct = 40 + (floors.length - 1 - i) * 7;
        return (
          <div
            key={i}
            className={`h-8 rounded-lg bg-gradient-to-r ${colorMap[floor.color]} border backdrop-blur-sm flex items-center justify-center gap-2 hover:scale-105 transition-transform cursor-pointer group`}
            style={{ width: `${widthPct}%` }}
            title={floor.vision}
          >
            <span className="text-xs font-mono text-white/80">{floor.level}</span>
            <span className="text-xs font-medium text-white truncate max-w-[120px] hidden md:block">{floor.name}</span>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function ZigguratVisionApp() {
  // State
  const [tab, setTab] = useState<'vision' | 'building' | 'model' | 'product' | 'impact'>('vision');
  const [year, setYear] = useState(5);
  const [scenario, setScenario] = useState<Scenario>('baseline');
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [inputs, setInputs] = useState<FinancialInputs>(DEFAULT_INPUTS);
  const [feedback, setFeedback] = useState<FeedbackForm>({
    name: '', email: '', org: '', type: 'question', message: '', submitted: false
  });

  const updateInput = (key: keyof FinancialInputs, value: number) => {
    setInputs(prev => ({ ...prev, [key]: value }));
  };

  // Calculations
  const calc = useMemo(
    () => calculateFinancials(inputs, year, scenario),
    [inputs, year, scenario]
  );

  const tabs = [
    { id: 'vision' as const, label: 'Vision', icon: Sparkles },
    { id: 'building' as const, label: 'Building', icon: Building2 },
    { id: 'model' as const, label: 'Model', icon: BarChart3 },
    { id: 'product' as const, label: 'Product', icon: Package },
    { id: 'impact' as const, label: 'Impact', icon: Heart },
  ];

  return (
    <div className="min-h-screen bg-[#0a0f1a] text-white antialiased">
      {/* Background */}
      <div className="fixed inset-0 bg-gradient-to-br from-slate-900 via-[#0a0f1a] to-slate-900 -z-10" />
      
      {/* Header */}
      <header className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="w-14 h-14 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-xl shadow-blue-500/30">
                  <Building2 size={28} className="text-white" />
                </div>
                <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-amber-500 rounded-full flex items-center justify-center border-2 border-slate-900">
                  <Sparkles size={12} className="text-slate-900" />
                </div>
              </div>
              <div>
                <h1 className="text-xl md:text-2xl font-bold tracking-tight">
                  <span className="text-blue-400">The Ziggurat</span>
                  <span className="text-slate-600 mx-2">×</span>
                  <span className="text-amber-400">Curious Kelly</span>
                </h1>
                <p className="text-sm text-slate-500">Global Education HQ | Open Financial Model</p>
              </div>
            </div>
            
            <div className="flex flex-wrap items-center gap-3">
              <a 
                href="https://curiouskelly.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 rounded-xl text-sm text-blue-400 transition-colors"
              >
                <Globe size={16} />
                curiouskelly.com
                <ExternalLink size={14} />
              </a>
              
              <select
                value={scenario}
                onChange={(e) => setScenario(e.target.value as Scenario)}
                className="bg-slate-800/80 border border-white/10 rounded-xl px-4 py-2 text-sm focus:outline-none cursor-pointer"
              >
                {Object.entries(SCENARIOS).map(([key, { name }]) => (
                  <option key={key} value={key}>{name}</option>
                ))}
              </select>
              
              <button
                onClick={() => setFeedbackOpen(true)}
                className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-amber-500 to-amber-600 text-slate-900 font-semibold rounded-xl shadow-lg shadow-amber-500/25"
              >
                <MessageSquare size={16} />
                Feedback
              </button>
            </div>
          </div>

          <div className="flex gap-2 mt-4 overflow-x-auto pb-1">
            {tabs.map(t => (
              <TabButton
                key={t.id}
                active={tab === t.id}
                onClick={() => setTab(t.id)}
                icon={t.icon}
                label={t.label}
              />
            ))}
          </div>
        </div>
      </header>

      {/* Metrics Bar */}
      <div className="border-b border-white/5 bg-slate-900/50 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-4 py-5">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <AnimatedMetric label="Capital Required" value={calc.totalCapitalNeeded} icon={Building2} color="amber" />
            <AnimatedMetric label="Funding Gap" value={calc.fundingGap} icon={calc.fundingGap <= 0 ? CheckCircle2 : AlertTriangle} color={calc.fundingGap <= 0 ? 'emerald' : 'red'} />
            <AnimatedMetric label={`Y${year || 5} Revenue`} value={calc.current.totalRevenue} icon={TrendingUp} color="emerald" />
            <AnimatedMetric label={`Y${year || 5} Net`} value={calc.current.netOperating} icon={calc.current.netOperating >= 0 ? Zap : AlertTriangle} color={calc.current.netOperating >= 0 ? 'emerald' : 'red'} />
            <AnimatedMetric label="Team Size" value={calc.current.headcount} format="number" icon={Users} color="blue" />
            <AnimatedMetric label="Break-Even" value={calc.breakEvenYear ? `Year ${calc.breakEvenYear}` : 'N/A'} format="text" icon={Target} color={calc.breakEvenYear ? 'emerald' : 'amber'} />
          </div>
        </div>
      </div>

      {/* Main Content - Add your tab content here */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Vision Tab */}
        {tab === 'vision' && (
          <div className="space-y-8">
            {/* Hero */}
            <div className="relative overflow-hidden rounded-3xl">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 via-slate-900/50 to-amber-600/10" />
              <div className="relative p-8 md:p-12 lg:p-16">
                <div className="flex items-center gap-2 text-blue-400 text-sm font-semibold tracking-wider mb-6">
                  <Sparkles size={16} />
                  {COPY.hero.tag}
                </div>
                <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 leading-tight">
                  <span className="text-white">{COPY.hero.title}</span><br />
                  <span className="bg-gradient-to-r from-blue-400 to-amber-400 bg-clip-text text-transparent">
                    {COPY.hero.titleAccent}
                  </span>
                </h2>
                <p className="text-lg md:text-xl text-slate-300 max-w-3xl mb-8 leading-relaxed">
                  {COPY.hero.body}
                </p>
              </div>
            </div>

            {/* Before/After */}
            <div className="grid md:grid-cols-2 gap-6">
              <GlassCard className="p-6">
                <div className="flex items-center gap-2 text-red-400 text-sm font-semibold mb-4">
                  <Clock size={16} />
                  {COPY.before.subtitle}
                </div>
                <h3 className="text-xl font-bold mb-4">{COPY.before.title}</h3>
                <ul className="space-y-3">
                  {COPY.before.items.map((item, i) => (
                    <li key={i} className="flex items-start gap-3 text-slate-400">
                      <X size={18} className="text-red-400 mt-0.5 flex-shrink-0" />
                      {item}
                    </li>
                  ))}
                </ul>
              </GlassCard>
              
              <GlassCard className="p-6 bg-gradient-to-br from-emerald-500/10 to-blue-500/5 border-emerald-500/20">
                <div className="flex items-center gap-2 text-emerald-400 text-sm font-semibold mb-4">
                  <Sparkles size={16} />
                  {COPY.after.subtitle}
                </div>
                <h3 className="text-xl font-bold mb-4">{COPY.after.title}</h3>
                <ul className="space-y-3">
                  {COPY.after.items.map((item, i) => (
                    <li key={i} className="flex items-start gap-3 text-slate-300">
                      <CheckCircle2 size={18} className="text-emerald-400 mt-0.5 flex-shrink-0" />
                      {item}
                    </li>
                  ))}
                </ul>
              </GlassCard>
            </div>

            {/* Floor Visualization */}
            <GlassCard className="p-8">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center">
                  <Layers size={24} className="text-blue-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">Floor-by-Floor Vision</h3>
                  <p className="text-sm text-slate-400">13 levels, 1 million square feet</p>
                </div>
              </div>
              <FloorVisualization floors={BUILDING.floorPlan} />
            </GlassCard>
          </div>
        )}

        {/* Building Tab */}
        {tab === 'building' && (
          <div className="space-y-8">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <GlassCard className="p-5 text-center">
                <div className="text-3xl font-bold text-amber-400">{fmtNum(BUILDING.sqft)}</div>
                <div className="text-sm text-slate-400 mt-1">Square Feet</div>
              </GlassCard>
              <GlassCard className="p-5 text-center">
                <div className="text-3xl font-bold text-blue-400">{BUILDING.acres}</div>
                <div className="text-sm text-slate-400 mt-1">Acres</div>
              </GlassCard>
              <GlassCard className="p-5 text-center">
                <div className="text-3xl font-bold text-emerald-400">{BUILDING.floors}</div>
                <div className="text-sm text-slate-400 mt-1">Floors</div>
              </GlassCard>
              <GlassCard className="p-5 text-center">
                <div className="text-3xl font-bold text-purple-400">{BUILDING.built}</div>
                <div className="text-sm text-slate-400 mt-1">Built</div>
              </GlassCard>
            </div>

            <GlassCard className="p-8">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-amber-500/20 flex items-center justify-center">
                  <Clock size={24} className="text-amber-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">Building History</h3>
                  <p className="text-sm text-slate-400">From construction to opportunity</p>
                </div>
              </div>
              <Timeline events={BUILDING.history} />
            </GlassCard>
          </div>
        )}

        {/* Model Tab */}
        {tab === 'model' && (
          <div className="space-y-8">
            <YearSelector year={year} setYear={setYear} />
            
            <div className="grid lg:grid-cols-2 gap-6">
              <GlassCard className="p-6">
                <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                  <Landmark size={20} className="text-amber-400" />
                  Capital Stack
                </h3>
                <div className="flex items-center gap-8 mb-6">
                  <ProgressRing percent={calc.fundingPercent} size={120} color={calc.fundingGap <= 0 ? 'emerald' : 'amber'} label="Funded" />
                  <div>
                    <div className="text-sm text-slate-400">Total Raised</div>
                    <div className="text-3xl font-bold">{fmt(calc.totalCapitalRaised)}</div>
                    <div className="text-sm text-slate-500">of {fmt(calc.totalCapitalNeeded)} needed</div>
                  </div>
                </div>
              </GlassCard>

              <GlassCard className="p-6">
                <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                  <BarChart3 size={20} className="text-emerald-400" />
                  P&L Summary (Y{year || 5})
                </h3>
                <div className="space-y-2">
                  <div className="flex justify-between py-2">
                    <span className="text-slate-400">Total Revenue</span>
                    <span className="font-mono">{fmt(calc.current.totalRevenue)}</span>
                  </div>
                  <div className="flex justify-between py-2 text-red-400">
                    <span>Total Expenses</span>
                    <span className="font-mono">({fmt(calc.current.totalCOR + calc.current.totalOpex)})</span>
                  </div>
                  <div className={`flex justify-between py-4 border-t-2 ${calc.current.netOperating >= 0 ? 'border-emerald-500' : 'border-red-500'}`}>
                    <span className="font-bold text-lg">Net Operating</span>
                    <span className={`font-mono font-bold text-lg ${calc.current.netOperating >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {fmt(calc.current.netOperating)}
                    </span>
                  </div>
                </div>
              </GlassCard>
            </div>

            <div className="grid lg:grid-cols-2 gap-6">
              <Section title="Infrastructure" icon={Building2} badge={fmt(calc.totalCapitalNeeded)}>
                <PremiumSlider label="Purchase Price" value={inputs.purchasePrice} onChange={v => updateInput('purchasePrice', v)} {...SLIDER_CONFIG.purchasePrice} />
                <PremiumSlider label="Phase 1 Renovation" value={inputs.phase1Reno} onChange={v => updateInput('phase1Reno', v)} {...SLIDER_CONFIG.phase1Reno} />
                <PremiumSlider label="Operating Reserve" value={inputs.operatingReserve} onChange={v => updateInput('operatingReserve', v)} {...SLIDER_CONFIG.operatingReserve} />
              </Section>

              <Section title="Capital Sources" icon={Landmark} badge={fmt(calc.totalCapitalRaised)}>
                <PremiumSlider label="Emerson Collective" value={inputs.emersonCollective} onChange={v => updateInput('emersonCollective', v)} {...SLIDER_CONFIG.emersonCollective} />
                <PremiumSlider label="Chan Zuckerberg Initiative" value={inputs.czi} onChange={v => updateInput('czi', v)} {...SLIDER_CONFIG.czi} />
                <PremiumSlider label="Historic Tax Credits" value={inputs.historicTaxCredits} onChange={v => updateInput('historicTaxCredits', v)} {...SLIDER_CONFIG.historicTaxCredits} />
              </Section>
            </div>
          </div>
        )}

{/* Product Tab */}
  {tab === 'product' && (
  <div className="space-y-8">
  <YearSelector year={year} setYear={setYear} />
  
  {/* Product Audit Summary */}
  <GlassCard className="p-6 border-amber-500/30">
  <div className="flex items-center gap-3 mb-4">
  <Package size={24} className="text-amber-400" />
  <h3 className="text-lg font-bold">Product Portfolio Audit</h3>
  </div>
  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
  <div className="bg-emerald-500/10 rounded-xl p-3 text-center border border-emerald-500/30">
  <p className="text-2xl font-bold text-emerald-400">4</p>
  <p className="text-xs text-slate-400">Live Products</p>
  </div>
  <div className="bg-blue-500/10 rounded-xl p-3 text-center border border-blue-500/30">
  <p className="text-2xl font-bold text-blue-400">6</p>
  <p className="text-xs text-slate-400">In Development</p>
  </div>
  <div className="bg-purple-500/10 rounded-xl p-3 text-center border border-purple-500/30">
  <p className="text-2xl font-bold text-purple-400">12</p>
  <p className="text-xs text-slate-400">Concept Stage</p>
  </div>
  <div className="bg-amber-500/10 rounded-xl p-3 text-center border border-amber-500/30">
  <p className="text-2xl font-bold text-amber-400">22</p>
  <p className="text-xs text-slate-400">Total Products</p>
  </div>
  </div>
  </GlassCard>
  
  <GlassCard className="p-6">
  <h3 className="text-lg font-bold mb-6">Revenue Mix (Y{year || 5})</h3>
  <StackedBar
  total={calc.current.totalRevenue}
  segments={[
  { label: 'Online Subs', value: calc.current.onlineTotal, color: 'bg-blue-500' },
  { label: 'Digital', value: calc.current.digitalRev, color: 'bg-emerald-500' },
  { label: 'Print', value: calc.current.printRev, color: 'bg-orange-500' },
  { label: 'Hardware', value: calc.current.hwRev, color: 'bg-pink-500' },
  { label: 'Licensing', value: calc.current.licenseRev, color: 'bg-cyan-500' },
  { label: 'Live/Services', value: calc.current.liveRev, color: 'bg-violet-500' },
  { label: 'Philanthropy', value: calc.current.philanthroOperating, color: 'bg-amber-500' },
  ]}
  />
  </GlassCard>
  
  {/* Current Products - LIVE */}
  <div className="grid lg:grid-cols-2 gap-6">
  <Section title="Online Subscriptions" icon={Monitor} badge={fmt(calc.current.onlineTotal)}>
  <div className="space-y-2 mb-4">
  <div className="flex items-center gap-2 text-xs">
  <span className="w-2 h-2 bg-emerald-400 rounded-full"></span>
  <span className="text-emerald-400">LIVE</span>
  <span className="text-slate-400">Explorer (Free) | Learner ($9.99-$199) | Family ($14.99-$349)</span>
  </div>
  </div>
  <PremiumSlider label="Monthly Subscribers (Y1)" value={inputs.monthlySubsY1} onChange={v => updateInput('monthlySubsY1', v)} {...SLIDER_CONFIG.monthlySubsY1} format="number" />
  <PremiumSlider label="Monthly Price" value={inputs.monthlyPrice} onChange={v => updateInput('monthlyPrice', v)} {...SLIDER_CONFIG.monthlyPrice} />
  <PremiumSlider label="School Districts (Y1)" value={inputs.districtDealsY1} onChange={v => updateInput('districtDealsY1', v)} {...SLIDER_CONFIG.districtDealsY1} format="number" />
  </Section>
  
  <Section title="Digital Downloads" icon={FileText} badge={fmt(calc.current.digitalRev)}>
  <div className="space-y-2 mb-4">
  <div className="flex items-center gap-2 text-xs">
  <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
  <span className="text-blue-400">DEV</span>
  <span className="text-slate-400">Monthly Pack ($9) | Complete Year ($29) | Premium ($49)</span>
  </div>
  </div>
  <PremiumSlider label="Monthly Pack (Y1)" value={inputs.monthlyPackY1} onChange={v => updateInput('monthlyPackY1', v)} {...SLIDER_CONFIG.monthlyPackY1} format="number" />
  <PremiumSlider label="Complete Year (Y1)" value={inputs.completeYearY1} onChange={v => updateInput('completeYearY1', v)} {...SLIDER_CONFIG.completeYearY1} format="number" />
  </Section>
  </div>
  
  {/* Product Derivatives - DAYDREAM */}
  <GlassCard className="p-6 border-purple-500/30">
  <div className="flex items-center gap-3 mb-4">
  <Sparkles size={20} className="text-purple-400" />
  <h3 className="text-lg font-bold">Product Derivatives (Roadmap)</h3>
  </div>
  
  <div className="grid md:grid-cols-3 gap-4">
  {/* Print Edition */}
  <div className="bg-orange-500/10 rounded-xl p-4 border border-orange-500/20">
  <h4 className="font-semibold text-orange-400 mb-2">Print Edition</h4>
  <ul className="text-xs text-slate-400 space-y-1">
  <li>Essentials Kit ($39/quarter)</li>
  <li>Kelly Annual Book ($69)</li>
  <li>Collectors Edition ($149)</li>
  </ul>
  <p className="text-sm font-bold text-orange-300 mt-2">{fmt(calc.current.printRev)}</p>
  </div>
  
  {/* iLearn Hardware */}
  <div className="bg-pink-500/10 rounded-xl p-4 border border-pink-500/20">
  <h4 className="font-semibold text-pink-400 mb-2">iLearn Hardware (Y3+)</h4>
  <ul className="text-xs text-slate-400 space-y-1">
  <li>iLearn Mini ($79) - Solar e-ink</li>
  <li>iLearn Standard ($199) - Color WiFi</li>
  <li>iLearn Pro ($399) - AR tablet</li>
  </ul>
  <p className="text-sm font-bold text-pink-300 mt-2">{fmt(calc.current.hwRev)}</p>
  </div>
  
  {/* Licensing */}
  <div className="bg-cyan-500/10 rounded-xl p-4 border border-cyan-500/20">
  <h4 className="font-semibold text-cyan-400 mb-2">Licensing</h4>
  <ul className="text-xs text-slate-400 space-y-1">
  <li>Enterprise ($10K-$200K/yr)</li>
  <li>Government ($50K-$500K)</li>
  <li>Broadcast ($25K-$150K/territory)</li>
  </ul>
  <p className="text-sm font-bold text-cyan-300 mt-2">{fmt(calc.current.licenseRev)}</p>
  </div>
  
  {/* Live Experiences */}
  <div className="bg-violet-500/10 rounded-xl p-4 border border-violet-500/20">
  <h4 className="font-semibold text-violet-400 mb-2">Live Experiences</h4>
  <ul className="text-xs text-slate-400 space-y-1">
  <li>Live Classes ($29/mo add-on)</li>
  <li>Ziggurat Field Trips ($25/person)</li>
  <li>Curiosity Camps ($499/week)</li>
  </ul>
  <p className="text-sm font-bold text-violet-300 mt-2">{fmt(calc.current.liveRev)}</p>
  </div>
  
  {/* Merchandise */}
  <div className="bg-rose-500/10 rounded-xl p-4 border border-rose-500/20">
  <h4 className="font-semibold text-rose-400 mb-2">Merchandise</h4>
  <ul className="text-xs text-slate-400 space-y-1">
  <li>Kelly Plush Collection ($19-$49)</li>
  <li>Curious Apparel ($25-$65)</li>
  <li>Kelly Learning Toys ($29-$99)</li>
  </ul>
  <p className="text-sm font-bold text-rose-300 mt-2">~$350K Y1</p>
  </div>
  
  {/* Advanced */}
  <div className="bg-indigo-500/10 rounded-xl p-4 border border-indigo-500/20">
  <h4 className="font-semibold text-indigo-400 mb-2">Advanced</h4>
  <ul className="text-xs text-slate-400 space-y-1">
  <li>Kelly API ($99-$999/mo)</li>
  <li>AI Tutor Add-on ($19.99/mo)</li>
  <li>Certified Educator ($299)</li>
  <li>Podcast Sponsorship</li>
  </ul>
  <p className="text-sm font-bold text-indigo-300 mt-2">~$400K Y1</p>
  </div>
  </div>
  </GlassCard>
  
  {/* Affiliate Program */}
  <GlassCard className="p-6 border-green-500/30">
  <div className="flex items-center gap-3 mb-4">
  <Share2 size={20} className="text-green-400" />
  <h3 className="text-lg font-bold">Affiliate Program (Distribution)</h3>
  </div>
  <div className="grid md:grid-cols-5 gap-3">
  {[
  { tier: 'Starter', refs: '0+', rate: '20%', color: 'slate' },
  { tier: 'Advocate', refs: '3+', rate: '25%', color: 'amber' },
  { tier: 'Ambassador', refs: '10+', rate: '30%', color: 'orange' },
  { tier: 'Champion', refs: '25+', rate: '35%', color: 'rose' },
  { tier: 'Legend', refs: '50+', rate: '40%+10%', color: 'violet' },
  ].map((t, i) => (
  <div key={i} className={`bg-${t.color}-500/10 rounded-xl p-3 text-center border border-${t.color}-500/20`}>
  <p className="font-bold text-sm">{t.tier}</p>
  <p className="text-xs text-slate-400">{t.refs} refs</p>
  <p className={`text-${t.color}-400 font-bold`}>{t.rate}</p>
  </div>
  ))}
  </div>
  <p className="text-xs text-slate-500 mt-3">+ Language Guardian Program: 50% commission for language evangelists in underserved markets</p>
  </GlassCard>
  </div>
  )}

        {/* Impact Tab */}
        {tab === 'impact' && (
          <div className="space-y-8">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <AnimatedMetric label="Daily Learners" value={calc.current.dailyActiveLearners} format="number" icon={GraduationCap} color="emerald" />
              <AnimatedMetric label="Employees" value={calc.current.headcount} format="number" icon={Users} color="blue" />
              <AnimatedMetric label="Languages" value={calc.current.languages} format="number" icon={Globe} color="purple" />
              <AnimatedMetric label="Commons Visitors" value={calc.current.commonsVisitors} format="number" icon={Building} color="amber" />
            </div>

            <GlassCard className="p-6">
              <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                <Target size={20} className="text-emerald-400" />
                UN Sustainable Development Goals
              </h3>
              {SDG_GOALS.map((item, i) => (
                <div key={i} className="p-4 bg-white/5 rounded-xl mb-3">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 text-xs font-medium rounded">{item.goal}</span>
                    <span className="font-semibold">{item.title}</span>
                  </div>
                  <p className="text-sm text-slate-400 mb-3">{item.desc}</p>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500" style={{ width: `${item.progress}%` }} />
                    </div>
                    <span className="text-sm font-mono text-emerald-400">{item.progress}%</span>
                  </div>
                </div>
              ))}
            </GlassCard>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 bg-slate-900/50 py-12 mt-16">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-sm text-slate-400">{COPY.footer.company}</p>
          <p className="text-sm text-slate-400">Founded by {COPY.footer.founder}</p>
          <p className="text-sm text-slate-500">{COPY.footer.tagline}</p>
          <p className="text-xs text-slate-600 mt-2">{COPY.footer.legal}</p>
        </div>
      </footer>

      {/* Feedback Modal */}
      {feedbackOpen && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={() => setFeedbackOpen(false)}>
          <GlassCard className="max-w-md w-full" onClick={e => e.stopPropagation()}>
            <div className="p-6 border-b border-white/10 flex items-center justify-between">
              <h3 className="text-xl font-bold">Submit Feedback</h3>
              <button onClick={() => setFeedbackOpen(false)} className="p-2 hover:bg-white/10 rounded-lg">
                <X size={20} />
              </button>
            </div>
            <div className="p-6">
              {feedback.submitted ? (
                <div className="text-center py-8">
                  <CheckCircle2 size={48} className="mx-auto text-emerald-400 mb-4" />
                  <div className="text-xl font-bold mb-2">Thank You!</div>
                  <button onClick={() => { setFeedbackOpen(false); setFeedback({ ...feedback, submitted: false }); }} className="px-6 py-2 bg-white/10 rounded-xl">
                    Close
                  </button>
                </div>
              ) : (
                <form onSubmit={(e) => { e.preventDefault(); setFeedback({ ...feedback, submitted: true }); }}>
                  <div className="space-y-4">
                    <input type="text" placeholder="Name *" required value={feedback.name} onChange={(e) => setFeedback({ ...feedback, name: e.target.value })} className="w-full px-4 py-3 bg-white/5 rounded-xl border border-white/10 focus:border-amber-500/50 focus:outline-none" />
                    <input type="email" placeholder="Email *" required value={feedback.email} onChange={(e) => setFeedback({ ...feedback, email: e.target.value })} className="w-full px-4 py-3 bg-white/5 rounded-xl border border-white/10 focus:border-amber-500/50 focus:outline-none" />
                    <textarea placeholder="Your message *" required rows={4} value={feedback.message} onChange={(e) => setFeedback({ ...feedback, message: e.target.value })} className="w-full px-4 py-3 bg-white/5 rounded-xl border border-white/10 focus:border-amber-500/50 focus:outline-none resize-none" />
                  </div>
                  <div className="flex gap-3 mt-6">
                    <button type="button" onClick={() => setFeedbackOpen(false)} className="flex-1 px-4 py-3 bg-white/5 rounded-xl font-medium">Cancel</button>
                    <button type="submit" className="flex-1 px-4 py-3 bg-gradient-to-r from-amber-500 to-amber-600 text-slate-900 rounded-xl font-semibold">Submit</button>
                  </div>
                </form>
              )}
            </div>
          </GlassCard>
        </div>
      )}
    </div>
  );
}
