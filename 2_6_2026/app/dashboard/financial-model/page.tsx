// app/dashboard/financial-model/page.tsx
// LOTD Financial Model — Old Money Style
// Black & white institutional aesthetic
'use client';

import { useState } from 'react';
import {
  AreaChart, Area, BarChart, Bar, ComposedChart, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine
} from 'recharts';

// ============================================
// DATA
// ============================================

const LEARNER_ADOPTION = [
  { year: 2026, total: 0.1, t1: 0.05, t2: 0.03, t3: 0.015, t4: 0.005, t5: 0 },
  { year: 2027, total: 1, t1: 0.4, t2: 0.3, t3: 0.2, t4: 0.08, t5: 0.02 },
  { year: 2028, total: 10, t1: 3, t2: 3.5, t3: 2.5, t4: 0.8, t5: 0.2 },
  { year: 2029, total: 50, t1: 12, t2: 18, t3: 14, t4: 5, t5: 1 },
  { year: 2030, total: 150, t1: 30, t2: 50, t3: 45, t4: 20, t5: 5 },
  { year: 2031, total: 300, t1: 50, t2: 90, t3: 100, t4: 45, t5: 15 },
  { year: 2032, total: 500, t1: 70, t2: 140, t3: 170, t4: 90, t5: 30 },
];

const HARDWARE_UNITS = [
  { year: 2026, nano: 0, mini: 2, tiny: 2.5, pro: 0.5, ultra: 0.05 },
  { year: 2027, nano: 50, mini: 10, tiny: 10, pro: 2, ultra: 0.2 },
  { year: 2028, nano: 500, mini: 50, tiny: 40, pro: 8, ultra: 0.5 },
  { year: 2029, nano: 2000, mini: 150, tiny: 100, pro: 20, ultra: 1.5 },
  { year: 2030, nano: 5000, mini: 300, tiny: 200, pro: 50, ultra: 5 },
  { year: 2031, nano: 10000, mini: 500, tiny: 350, pro: 80, ultra: 10 },
  { year: 2032, nano: 20000, mini: 800, tiny: 500, pro: 120, ultra: 20 },
];

const HARDWARE_REVENUE = [
  { year: 2026, nano: 0, mini: 0.2, tiny: 0.6, pro: 0.4, ultra: 0.25, total: 1.4 },
  { year: 2027, nano: 0.5, mini: 1, tiny: 2.5, pro: 1.5, ultra: 1, total: 6.5 },
  { year: 2028, nano: 5, mini: 5, tiny: 10, pro: 6, ultra: 2.5, total: 28.5 },
  { year: 2029, nano: 20, mini: 15, tiny: 25, pro: 15, ultra: 7.5, total: 82.5 },
  { year: 2030, nano: 50, mini: 30, tiny: 50, pro: 37.5, ultra: 25, total: 192.5 },
  { year: 2031, nano: 100, mini: 50, tiny: 87, pro: 60, ultra: 50, total: 347 },
  { year: 2032, nano: 200, mini: 79, tiny: 125, pro: 90, ultra: 100, total: 594 },
];

const REVENUE_MIX = [
  { year: 2026, broadcast: 2, hardware: 1.4, print: 0.5, infra: 0, total: 3.9 },
  { year: 2027, broadcast: 8, hardware: 6.5, print: 2, infra: 1, total: 17.5 },
  { year: 2028, broadcast: 25, hardware: 28.5, print: 5, infra: 7, total: 65.5 },
  { year: 2029, broadcast: 60, hardware: 82.5, print: 12, infra: 18, total: 172.5 },
  { year: 2030, broadcast: 120, hardware: 192.5, print: 25, infra: 40, total: 377.5 },
  { year: 2031, broadcast: 200, hardware: 347, print: 40, infra: 75, total: 662 },
  { year: 2032, broadcast: 300, hardware: 594, print: 60, infra: 140, total: 1094 },
];

const LTV_DATA = [
  { age: 2, t1: 2847, t2: 769, t3: 285 },
  { age: 6, t1: 3251, t2: 878, t3: 325 },
  { age: 12, t1: 2912, t2: 786, t3: 291 },
  { age: 18, t1: 2398, t2: 647, t3: 240 },
  { age: 25, t1: 2567, t2: 693, t3: 257 },
  { age: 35, t1: 2891, t2: 781, t3: 289 },
  { age: 50, t1: 2634, t2: 711, t3: 263 },
  { age: 65, t1: 1842, t2: 497, t3: 184 },
];

const CROSS_SUBSIDY = [
  { year: 2026, ultras: 50, nanos: 8 },
  { year: 2027, ultras: 200, nanos: 32 },
  { year: 2028, ultras: 500, nanos: 80 },
  { year: 2029, ultras: 1500, nanos: 240 },
  { year: 2030, ultras: 5000, nanos: 800 },
  { year: 2031, ultras: 10000, nanos: 1590 },
  { year: 2032, ultras: 20000, nanos: 3180 },
];

const PROFITABILITY = [
  { year: 2026, revenue: 3.9, ebitda: -1.1 },
  { year: 2027, revenue: 17.5, ebitda: -5.5 },
  { year: 2028, revenue: 65.5, ebitda: 6.5 },
  { year: 2029, revenue: 172.5, ebitda: 43 },
  { year: 2030, revenue: 377.5, ebitda: 94 },
  { year: 2031, revenue: 662, ebitda: 166 },
  { year: 2032, revenue: 1094, ebitda: 274 },
];

// ============================================
// GRAYSCALE PALETTE
// ============================================

const GRAY = {
  900: '#111111',
  800: '#1f1f1f',
  700: '#333333',
  600: '#4a4a4a',
  500: '#666666',
  400: '#888888',
  300: '#aaaaaa',
  200: '#cccccc',
  100: '#e8e8e8',
  50: '#f5f5f5',
};

// ============================================
// FORMATTERS
// ============================================

const formatNum = (n: number) => {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}B`;
  if (n >= 1) return `${n.toFixed(0)}M`;
  return `${(n * 1000).toFixed(0)}K`;
};

// ============================================
// COMPONENT
// ============================================

export default function FinancialDashboard() {
  const [activeTab, setActiveTab] = useState<'overview' | 'adoption' | 'hardware' | 'revenue' | 'ltv'>('overview');

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b-2 border-black">
        <div className="max-w-6xl mx-auto px-8 py-6">
          <div className="flex justify-between items-baseline">
            <div>
              <p className="text-xs tracking-[0.3em] uppercase text-gray-500">Confidential Memorandum</p>
              <h1 className="mt-1 text-3xl font-serif font-normal tracking-tight">
                Lesson of the Day, PBC
              </h1>
            </div>
            <div className="text-right">
              <p className="text-xs tracking-[0.2em] uppercase text-gray-500">February 2026</p>
              <p className="mt-1 font-serif text-lg">Financial Model</p>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-8">
          <div className="flex gap-8">
            {[
              { id: 'overview', label: 'Executive Summary' },
              { id: 'adoption', label: 'Learner Adoption' },
              { id: 'hardware', label: 'Hardware Economics' },
              { id: 'revenue', label: 'Revenue & Profitability' },
              { id: 'ltv', label: 'Unit Economics' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={`py-4 text-sm tracking-wide border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-black text-black font-medium'
                    : 'border-transparent text-gray-500 hover:text-black'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-8 py-12">
        
        {/* ==================== OVERVIEW ==================== */}
        {activeTab === 'overview' && (
          <div className="space-y-12">
            {/* Hero Numbers */}
            <div className="grid grid-cols-4 gap-8">
              {[
                { value: '$1.094B', label: '2032 Revenue', sub: '25% EBITDA margin' },
                { value: '500M', label: '2032 Learners', sub: '6% of humanity' },
                { value: '21.4M', label: 'Hardware Units', sub: '93% mission-deployed' },
                { value: '8B', label: '2075 Target', sub: 'SDG 4 achieved' },
              ].map((stat) => (
                <div key={stat.label} className="border-l-2 border-black pl-4">
                  <p className="text-4xl font-serif">{stat.value}</p>
                  <p className="mt-1 text-sm font-medium tracking-wide uppercase">{stat.label}</p>
                  <p className="text-xs text-gray-500">{stat.sub}</p>
                </div>
              ))}
            </div>

            {/* Revenue Chart */}
            <div>
              <h2 className="text-lg font-serif border-b border-gray-200 pb-2 mb-6">
                Seven-Year Revenue Trajectory
              </h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={PROFITABILITY}>
                    <CartesianGrid stroke={GRAY[200]} strokeDasharray="3 3" />
                    <XAxis dataKey="year" tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <YAxis tickFormatter={(v) => `$${v}M`} tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <Tooltip 
                      contentStyle={{ border: '1px solid #000', borderRadius: 0, fontSize: 12 }}
                      formatter={(v: number) => [`$${v}M`, '']}
                    />
                    <Bar dataKey="revenue" fill={GRAY[800]} name="Revenue" />
                    <Line type="monotone" dataKey="ebitda" stroke={GRAY[400]} strokeWidth={2} strokeDasharray="5 5" name="EBITDA" dot={{ fill: GRAY[400], r: 3 }} />
                    <ReferenceLine y={0} stroke={GRAY[900]} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-4 text-xs text-gray-500 text-center">
                Breakeven in 2028. Sustained 25% EBITDA margin from 2029.
              </p>
            </div>

            {/* Hardware Lineup */}
            <div>
              <h2 className="text-lg font-serif border-b border-gray-200 pb-2 mb-6">
                iLearn Hardware Portfolio
              </h2>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-black">
                    <th className="py-3 text-left font-medium">Device</th>
                    <th className="py-3 text-right font-medium">Price</th>
                    <th className="py-3 text-right font-medium">COGS</th>
                    <th className="py-3 text-right font-medium">Margin</th>
                    <th className="py-3 text-left font-medium pl-8">Target Segment</th>
                  </tr>
                </thead>
                <tbody className="font-serif">
                  {[
                    { name: 'Nano', price: 10, cogs: 8, margin: '20%', segment: 'Mission deployment (T4/T5 markets)' },
                    { name: 'Mini', price: 99, cogs: 45, margin: '55%', segment: 'Young children, ages 2-7' },
                    { name: 'Tiny', price: 249, cogs: 95, margin: '62%', segment: 'Students, ages 8-18' },
                    { name: 'Pro', price: 749, cogs: 280, margin: '63%', segment: 'Professionals, continuing education' },
                    { name: 'Ultra', price: 4999, cogs: 1800, margin: '64%', segment: 'Institutions, enterprise deployment' },
                  ].map((d) => (
                    <tr key={d.name} className="border-b border-gray-100">
                      <td className="py-3 font-medium">{d.name}</td>
                      <td className="py-3 text-right font-mono">${d.price.toLocaleString()}</td>
                      <td className="py-3 text-right font-mono text-gray-500">${d.cogs}</td>
                      <td className="py-3 text-right">{d.margin}</td>
                      <td className="py-3 text-left pl-8 text-gray-600 font-sans text-xs">{d.segment}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Cross-Subsidy */}
            <div className="border border-black p-6">
              <h3 className="text-sm font-medium tracking-wide uppercase mb-4">Cross-Subsidy Model</h3>
              <div className="flex items-center justify-center gap-8">
                <div className="text-center">
                  <p className="text-3xl font-serif">1</p>
                  <p className="text-xs text-gray-500 mt-1">Ultra sold</p>
                  <p className="text-xs text-gray-400">($4,999)</p>
                </div>
                <div className="text-2xl text-gray-300">=</div>
                <div className="text-center">
                  <p className="text-3xl font-serif">159</p>
                  <p className="text-xs text-gray-500 mt-1">Nanos deployed</p>
                  <p className="text-xs text-gray-400">(at $20 each)</p>
                </div>
              </div>
              <p className="mt-6 text-xs text-gray-500 text-center">
                By 2032: 20,000 Ultra units enable 3.2 million Nano deployments to underserved learners.
              </p>
            </div>
          </div>
        )}

        {/* ==================== ADOPTION ==================== */}
        {activeTab === 'adoption' && (
          <div className="space-y-12">
            <div>
              <h2 className="text-lg font-serif border-b border-gray-200 pb-2 mb-6">
                Learner Adoption by Market Tier
              </h2>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={LEARNER_ADOPTION}>
                    <CartesianGrid stroke={GRAY[200]} strokeDasharray="3 3" />
                    <XAxis dataKey="year" tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <YAxis tickFormatter={formatNum} tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <Tooltip 
                      contentStyle={{ border: '1px solid #000', borderRadius: 0, fontSize: 12 }}
                      formatter={(v: number) => [`${formatNum(v)} learners`, '']}
                    />
                    <Area type="monotone" dataKey="t5" stackId="1" stroke={GRAY[100]} fill={GRAY[100]} name="Tier 5" />
                    <Area type="monotone" dataKey="t4" stackId="1" stroke={GRAY[200]} fill={GRAY[200]} name="Tier 4" />
                    <Area type="monotone" dataKey="t3" stackId="1" stroke={GRAY[400]} fill={GRAY[400]} name="Tier 3" />
                    <Area type="monotone" dataKey="t2" stackId="1" stroke={GRAY[600]} fill={GRAY[600]} name="Tier 2" />
                    <Area type="monotone" dataKey="t1" stackId="1" stroke={GRAY[900]} fill={GRAY[900]} name="Tier 1" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid grid-cols-5 gap-4">
              {[
                { tier: 'Tier 1', gdp: '>$40K', pop: '1.2B', desc: 'Premium markets', pct: '14%' },
                { tier: 'Tier 2', gdp: '$15-40K', pop: '2.5B', desc: 'Developing economies', pct: '28%' },
                { tier: 'Tier 3', gdp: '$5-15K', pop: '2.8B', desc: 'Growth markets', pct: '34%' },
                { tier: 'Tier 4', gdp: '$1.5-5K', pop: '1.5B', desc: 'Emerging markets', pct: '18%' },
                { tier: 'Tier 5', gdp: '<$1.5K', pop: '1.0B', desc: 'Subsidized', pct: '6%' },
              ].map((t) => (
                <div key={t.tier} className="border-t border-black pt-3">
                  <p className="text-sm font-medium">{t.tier}</p>
                  <p className="text-xs text-gray-500">{t.gdp} GDP/capita</p>
                  <p className="text-xs text-gray-400">{t.pop} addressable</p>
                  <p className="mt-2 text-lg font-serif">{t.pct}</p>
                  <p className="text-xs text-gray-500">of 2032 mix</p>
                </div>
              ))}
            </div>

            <div>
              <h3 className="text-sm font-medium tracking-wide uppercase mb-4">Growth Trajectory</h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-black">
                    <th className="py-2 text-left">Period</th>
                    <th className="py-2 text-right">Growth Rate</th>
                    <th className="py-2 text-left pl-8">Primary Driver</th>
                  </tr>
                </thead>
                <tbody className="font-serif">
                  <tr className="border-b border-gray-100">
                    <td className="py-2">2026–2028</td>
                    <td className="py-2 text-right font-mono">10×/year</td>
                    <td className="py-2 pl-8 text-gray-600 font-sans text-xs">Launch momentum, LA2028 Olympics exposure</td>
                  </tr>
                  <tr className="border-b border-gray-100">
                    <td className="py-2">2028–2030</td>
                    <td className="py-2 text-right font-mono">3-5×/year</td>
                    <td className="py-2 pl-8 text-gray-600 font-sans text-xs">T2/T3 market expansion</td>
                  </tr>
                  <tr className="border-b border-gray-100">
                    <td className="py-2">2030–2032</td>
                    <td className="py-2 text-right font-mono">1.7-2×/year</td>
                    <td className="py-2 pl-8 text-gray-600 font-sans text-xs">Mature scaling, T4/T5 deployment</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ==================== HARDWARE ==================== */}
        {activeTab === 'hardware' && (
          <div className="space-y-12">
            <div>
              <h2 className="text-lg font-serif border-b border-gray-200 pb-2 mb-6">
                Hardware Unit Volume (Thousands)
              </h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={HARDWARE_UNITS}>
                    <CartesianGrid stroke={GRAY[200]} strokeDasharray="3 3" />
                    <XAxis dataKey="year" tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <YAxis tickFormatter={(v) => `${(v/1000).toFixed(0)}M`} tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <Tooltip 
                      contentStyle={{ border: '1px solid #000', borderRadius: 0, fontSize: 12 }}
                      formatter={(v: number) => [`${v.toLocaleString()}K`, '']}
                    />
                    <Area type="monotone" dataKey="nano" stackId="1" stroke={GRAY[900]} fill={GRAY[900]} name="Nano" />
                    <Area type="monotone" dataKey="mini" stackId="1" stroke={GRAY[700]} fill={GRAY[700]} name="Mini" />
                    <Area type="monotone" dataKey="tiny" stackId="1" stroke={GRAY[500]} fill={GRAY[500]} name="Tiny" />
                    <Area type="monotone" dataKey="pro" stackId="1" stroke={GRAY[300]} fill={GRAY[300]} name="Pro" />
                    <Area type="monotone" dataKey="ultra" stackId="1" stroke={GRAY[100]} fill={GRAY[100]} name="Ultra" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div>
              <h2 className="text-lg font-serif border-b border-gray-200 pb-2 mb-6">
                Hardware Revenue by Device ($M)
              </h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={HARDWARE_REVENUE}>
                    <CartesianGrid stroke={GRAY[200]} strokeDasharray="3 3" />
                    <XAxis dataKey="year" tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <YAxis tickFormatter={(v) => `$${v}M`} tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <Tooltip 
                      contentStyle={{ border: '1px solid #000', borderRadius: 0, fontSize: 12 }}
                      formatter={(v: number) => [`$${v}M`, '']}
                    />
                    <Bar dataKey="nano" stackId="a" fill={GRAY[900]} name="Nano" />
                    <Bar dataKey="mini" stackId="a" fill={GRAY[700]} name="Mini" />
                    <Bar dataKey="tiny" stackId="a" fill={GRAY[500]} name="Tiny" />
                    <Bar dataKey="pro" stackId="a" fill={GRAY[300]} name="Pro" />
                    <Bar dataKey="ultra" stackId="a" fill={GRAY[100]} name="Ultra" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium tracking-wide uppercase mb-4">2032 Hardware Mix Analysis</h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-black">
                    <th className="py-2 text-left">Device</th>
                    <th className="py-2 text-right">Units</th>
                    <th className="py-2 text-right">% of Units</th>
                    <th className="py-2 text-right">Revenue</th>
                    <th className="py-2 text-right">% of Revenue</th>
                  </tr>
                </thead>
                <tbody className="font-serif">
                  {[
                    { name: 'Nano', units: '20.0M', pctUnits: '93.3%', rev: '$200M', pctRev: '33.7%' },
                    { name: 'Mini', units: '0.8M', pctUnits: '3.7%', rev: '$79M', pctRev: '13.3%' },
                    { name: 'Tiny', units: '0.5M', pctUnits: '2.3%', rev: '$125M', pctRev: '21.0%' },
                    { name: 'Pro', units: '0.12M', pctUnits: '0.6%', rev: '$90M', pctRev: '15.1%' },
                    { name: 'Ultra', units: '0.02M', pctUnits: '0.1%', rev: '$100M', pctRev: '16.8%' },
                  ].map((d) => (
                    <tr key={d.name} className="border-b border-gray-100">
                      <td className="py-2">{d.name}</td>
                      <td className="py-2 text-right font-mono">{d.units}</td>
                      <td className="py-2 text-right">{d.pctUnits}</td>
                      <td className="py-2 text-right font-mono">{d.rev}</td>
                      <td className="py-2 text-right">{d.pctRev}</td>
                    </tr>
                  ))}
                  <tr className="border-t border-black font-medium">
                    <td className="py-2">Total</td>
                    <td className="py-2 text-right font-mono">21.4M</td>
                    <td className="py-2 text-right">100%</td>
                    <td className="py-2 text-right font-mono">$594M</td>
                    <td className="py-2 text-right">100%</td>
                  </tr>
                </tbody>
              </table>
              <p className="mt-4 text-xs text-gray-500">
                Note: Nano represents 93% of units but only 34% of revenue. Ultra represents 0.1% of units but 17% of revenue. 
                This inverse relationship enables the cross-subsidy model.
              </p>
            </div>
          </div>
        )}

        {/* ==================== REVENUE ==================== */}
        {activeTab === 'revenue' && (
          <div className="space-y-12">
            <div>
              <h2 className="text-lg font-serif border-b border-gray-200 pb-2 mb-6">
                Revenue by Product Line ($M)
              </h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={REVENUE_MIX}>
                    <CartesianGrid stroke={GRAY[200]} strokeDasharray="3 3" />
                    <XAxis dataKey="year" tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <YAxis tickFormatter={(v) => `$${v}M`} tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <Tooltip 
                      contentStyle={{ border: '1px solid #000', borderRadius: 0, fontSize: 12 }}
                      formatter={(v: number) => [`$${v}M`, '']}
                    />
                    <Area type="monotone" dataKey="broadcast" stackId="1" stroke={GRAY[900]} fill={GRAY[900]} name="Broadcast" />
                    <Area type="monotone" dataKey="hardware" stackId="1" stroke={GRAY[600]} fill={GRAY[600]} name="Hardware" />
                    <Area type="monotone" dataKey="print" stackId="1" stroke={GRAY[400]} fill={GRAY[400]} name="Print" />
                    <Area type="monotone" dataKey="infra" stackId="1" stroke={GRAY[200]} fill={GRAY[200]} name="Infrastructure" />
                    <ReferenceLine y={1000} stroke={GRAY[400]} strokeDasharray="5 5" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-6">
              {[
                { line: 'Broadcast', rev: '$300M', pct: '27%', desc: 'Kelly subscriptions' },
                { line: 'Hardware', rev: '$594M', pct: '54%', desc: 'iLearn devices' },
                { line: 'Print', rev: '$60M', pct: '5%', desc: 'Lesson eggs, books' },
                { line: 'Infrastructure', rev: '$140M', pct: '13%', desc: 'B2B licensing' },
              ].map((l) => (
                <div key={l.line} className="border-t border-black pt-3">
                  <p className="text-sm font-medium">{l.line}</p>
                  <p className="text-xs text-gray-500">{l.desc}</p>
                  <p className="mt-2 text-2xl font-serif">{l.rev}</p>
                  <p className="text-sm text-gray-500">{l.pct} of 2032 total</p>
                </div>
              ))}
            </div>

            <div>
              <h3 className="text-sm font-medium tracking-wide uppercase mb-4">Profitability Trajectory</h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-black">
                    <th className="py-2 text-left">Year</th>
                    <th className="py-2 text-right">Revenue</th>
                    <th className="py-2 text-right">EBITDA</th>
                    <th className="py-2 text-right">Margin</th>
                    <th className="py-2 text-left pl-8">Status</th>
                  </tr>
                </thead>
                <tbody className="font-serif">
                  {PROFITABILITY.map((row) => {
                    const margin = ((row.ebitda / row.revenue) * 100).toFixed(0);
                    return (
                      <tr key={row.year} className="border-b border-gray-100">
                        <td className="py-2">{row.year}</td>
                        <td className="py-2 text-right font-mono">${row.revenue}M</td>
                        <td className={`py-2 text-right font-mono ${row.ebitda < 0 ? 'text-gray-400' : ''}`}>
                          ${row.ebitda}M
                        </td>
                        <td className="py-2 text-right">{margin}%</td>
                        <td className="py-2 pl-8 text-gray-600 font-sans text-xs">
                          {row.ebitda < 0 ? 'Investment phase' : row.year === 2028 ? 'Breakeven' : 'Profitable'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ==================== LTV ==================== */}
        {activeTab === 'ltv' && (
          <div className="space-y-12">
            <div>
              <h2 className="text-lg font-serif border-b border-gray-200 pb-2 mb-6">
                Lifetime Value by Acquisition Age
              </h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={LTV_DATA}>
                    <CartesianGrid stroke={GRAY[200]} strokeDasharray="3 3" />
                    <XAxis dataKey="age" tick={{ fontSize: 12 }} stroke={GRAY[400]} label={{ value: 'Acquisition Age', position: 'bottom', offset: -5, fontSize: 11 }} />
                    <YAxis tickFormatter={(v) => `$${v}`} tick={{ fontSize: 12 }} stroke={GRAY[400]} />
                    <Tooltip 
                      contentStyle={{ border: '1px solid #000', borderRadius: 0, fontSize: 12 }}
                      formatter={(v: number) => [`$${v.toLocaleString()}`, '']}
                    />
                    <Line type="monotone" dataKey="t1" stroke={GRAY[900]} strokeWidth={2} name="Tier 1" dot={{ fill: GRAY[900], r: 3 }} />
                    <Line type="monotone" dataKey="t2" stroke={GRAY[500]} strokeWidth={2} name="Tier 2" dot={{ fill: GRAY[500], r: 3 }} />
                    <Line type="monotone" dataKey="t3" stroke={GRAY[300]} strokeWidth={2} name="Tier 3" dot={{ fill: GRAY[300], r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-8">
              <div className="border border-black p-6">
                <p className="text-xs tracking-wide uppercase text-gray-500">Peak LTV</p>
                <p className="mt-2 text-4xl font-serif">$3,251</p>
                <p className="mt-1 text-sm text-gray-600">Tier 1, acquisition age 6</p>
                <p className="mt-4 text-xs text-gray-500">
                  96-year runway × high retention during school years × family influence
                </p>
              </div>
              <div className="border border-black p-6">
                <p className="text-xs tracking-wide uppercase text-gray-500">Best LTV:CAC</p>
                <p className="mt-2 text-4xl font-serif">406:1</p>
                <p className="mt-1 text-sm text-gray-600">Tier 1, age 6</p>
                <p className="mt-4 text-xs text-gray-500">
                  $3,251 LTV ÷ $8 CAC = exceptional unit economics
                </p>
              </div>
              <div className="border border-black p-6">
                <p className="text-xs tracking-wide uppercase text-gray-500">Average Payback</p>
                <p className="mt-2 text-4xl font-serif">1-2 mo</p>
                <p className="mt-1 text-sm text-gray-600">All segments</p>
                <p className="mt-4 text-xs text-gray-500">
                  First subscription payment recovers acquisition cost
                </p>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium tracking-wide uppercase mb-4">Unit Economics by Segment</h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-black">
                    <th className="py-2 text-left">Segment</th>
                    <th className="py-2 text-right">LTV</th>
                    <th className="py-2 text-right">CAC</th>
                    <th className="py-2 text-right">LTV:CAC</th>
                    <th className="py-2 text-right">Payback</th>
                  </tr>
                </thead>
                <tbody className="font-serif">
                  {[
                    { segment: 'T1 Student (age 6)', ltv: 3251, cac: 8, ratio: '406:1', payback: '1 mo' },
                    { segment: 'T1 Child (age 2)', ltv: 2847, cac: 8, ratio: '356:1', payback: '1 mo' },
                    { segment: 'T1 Professional (age 35)', ltv: 2891, cac: 50, ratio: '58:1', payback: '3 mo' },
                    { segment: 'T2 Student', ltv: 878, cac: 4, ratio: '220:1', payback: '1 mo' },
                    { segment: 'T3 Student', ltv: 325, cac: 2, ratio: '163:1', payback: '1 mo' },
                    { segment: 'T4 (Subsidized)', ltv: 98, cac: 1, ratio: '98:1', payback: '1 mo' },
                  ].map((s) => (
                    <tr key={s.segment} className="border-b border-gray-100">
                      <td className="py-2 font-sans text-xs">{s.segment}</td>
                      <td className="py-2 text-right font-mono">${s.ltv.toLocaleString()}</td>
                      <td className="py-2 text-right font-mono text-gray-500">${s.cac}</td>
                      <td className="py-2 text-right">{s.ratio}</td>
                      <td className="py-2 text-right text-gray-500">{s.payback}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 mt-12">
        <div className="max-w-6xl mx-auto px-8 py-6 flex justify-between text-xs text-gray-400">
          <p>Lesson of the Day, PBC — Confidential</p>
          <p>February 2026</p>
        </div>
      </footer>
    </div>
  );
}
