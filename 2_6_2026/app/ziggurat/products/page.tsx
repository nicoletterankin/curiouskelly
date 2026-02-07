'use client'

import React from "react"

import { useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { ArrowLeft, Smartphone, Monitor, Printer, Building2, Calendar, Trophy, Globe, Users } from 'lucide-react'
import { PRODUCT_LINES, GROWTH_EVENTS, PRODUCT_REVENUE_PROJECTIONS } from '@/lib/ziggurat/data'
import { cn } from '@/lib/utils'

export default function ProductsPage() {
  const [activeTab, setActiveTab] = useState<'products' | 'events'>('products')
  
  const formatCurrency = (n: number) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n)
  
  const formatNumber = (n: number) =>
    new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(n)

  const productIcons: Record<string, React.ReactNode> = {
    software: <Smartphone className="w-5 h-5" />,
    hardware: <Monitor className="w-5 h-5" />,
    print: <Printer className="w-5 h-5" />,
    licensing: <Building2 className="w-5 h-5" />,
  }

  const productColors: Record<string, string> = {
    software: 'from-blue-500/20 to-cyan-500/20 border-blue-500/30',
    hardware: 'from-purple-500/20 to-pink-500/20 border-purple-500/30',
    print: 'from-amber-500/20 to-orange-500/20 border-amber-500/30',
    licensing: 'from-emerald-500/20 to-teal-500/20 border-emerald-500/30',
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-black via-black to-zinc-950 text-white selection:bg-cyan-500/30">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-black/90 backdrop-blur-xl border-b border-white/[0.06]">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-5">
            <Link 
              href="/ziggurat" 
              className="group p-2.5 rounded-xl hover:bg-white/[0.04] transition-all duration-200"
            >
              <ArrowLeft className="w-5 h-5 text-white/50 group-hover:text-white group-hover:-translate-x-0.5 transition-all" />
            </Link>
            <div>
              <h1 className="text-xl font-semibold tracking-tight">Product Roadmap</h1>
              <p className="text-[11px] text-white/40 uppercase tracking-wider">Square is the DNA</p>
            </div>
          </div>
          
          {/* Tab Toggle */}
          <div className="flex gap-1 p-1 bg-white/[0.04] rounded-xl border border-white/[0.06]">
            <button
              onClick={() => setActiveTab('products')}
              className={cn(
                "px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
                activeTab === 'products' 
                  ? "bg-white text-black shadow-lg shadow-white/10" 
                  : "text-white/50 hover:text-white hover:bg-white/[0.04]"
              )}
            >
              Products
            </button>
            <button
              onClick={() => setActiveTab('events')}
              className={cn(
                "px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
                activeTab === 'events' 
                  ? "bg-white text-black shadow-lg shadow-white/10" 
                  : "text-white/50 hover:text-white hover:bg-white/[0.04]"
              )}
            >
              Growth Events
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10">
        {activeTab === 'products' ? (
          <>
            {/* Square DNA Banner */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-10 p-8 rounded-3xl bg-gradient-to-br from-white/[0.04] via-white/[0.02] to-transparent border border-white/[0.06] backdrop-blur-sm"
            >
              <div className="flex items-start gap-6">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/20 flex items-center justify-center shadow-lg shadow-cyan-500/10">
                  {/* Square Ziggurat icon */}
                  <svg viewBox="0 0 24 24" className="w-12 h-12 text-cyan-400">
                    <rect x="10" y="3" width="4" height="2" fill="currentColor" />
                    <rect x="8" y="5" width="8" height="2" fill="currentColor" />
                    <rect x="6" y="7" width="12" height="2" fill="currentColor" />
                    <rect x="5" y="9" width="14" height="2" fill="currentColor" />
                    <rect x="4" y="11" width="16" height="2" fill="currentColor" />
                    <rect x="3" y="13" width="18" height="2" fill="currentColor" />
                    <rect x="2" y="15" width="20" height="6" fill="currentColor" />
                  </svg>
                </div>
                <div className="flex-1">
                  <h2 className="text-xl font-semibold mb-2 tracking-tight">Square is the New Round</h2>
                  <p className="text-white/50 text-sm leading-relaxed max-w-2xl">
                    Every product in the Lesson of the Day ecosystem shares the same DNA: <span className="text-white/70">square form factor</span>.
                    Kelly's player is square. The Ziggurat building is a 1M sqft 7-level square pyramid.
                    iLearn devices are square tablets. Even our print cards are square. This isn't arbitrary - 
                    it's <span className="text-cyan-400">ZigTok style</span>, built for the way humans naturally frame and focus on learning.
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Product Grid */}
            <div className="grid md:grid-cols-2 gap-5 mb-12">
              {Object.entries(PRODUCT_LINES).map(([key, product], i) => (
                <motion.div
                  key={key}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className={cn(
                    "group p-6 rounded-2xl bg-gradient-to-br border backdrop-blur-sm transition-all duration-300 hover:scale-[1.01]",
                    productColors[product.type]
                  )}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-xl bg-white/10 flex items-center justify-center group-hover:scale-105 transition-transform">
                        {productIcons[product.type]}
                      </div>
                      <div>
                        <h3 className="font-semibold tracking-tight">{product.name}</h3>
                        <p className="text-[11px] text-white/40 uppercase tracking-wider">{product.type}</p>
                      </div>
                    </div>
                    {product.formFactor === 'square' && (
                      <span className="text-[9px] font-semibold bg-cyan-500/20 text-cyan-400 px-2.5 py-1 rounded-full tracking-wider">SQUARE</span>
                    )}
                  </div>
                  
                  <p className="text-sm text-white/50 mb-5 leading-relaxed">{product.description}</p>
                  
                  <div className="flex items-center justify-between text-sm">
                    <div className="font-medium">
                      {product.price.free && <span className="text-emerald-400">Free</span>}
                      {product.price.premium && <span className="text-white/70">${product.price.premium}/mo premium</span>}
                      {product.price.retail && <span className="text-white/70">{formatCurrency(product.price.retail)}</span>}
                      {product.price.monthly && <span className="text-white/70">${product.price.monthly}/mo</span>}
                      {product.price.perSeat && <span className="text-white/70">${product.price.perSeat}/seat/mo</span>}
                    </div>
                    <span className="text-white/30 text-xs font-mono">{product.launchDate}</span>
                  </div>
                  
                  {product.margin && (
                    <div className="mt-4 pt-4 border-t border-white/[0.06]">
                      <div className="flex justify-between text-xs">
                        <span className="text-white/30">Gross Margin</span>
                        <span className="text-emerald-400 font-semibold">{(product.margin * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            {/* Revenue by Product */}
            <div className="mb-8">
              <h2 className="text-lg font-semibold mb-4">5-Year Revenue by Product Line</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left py-3 px-4 text-white/60 font-medium">Year</th>
                      <th className="text-right py-3 px-4 text-white/60 font-medium">Kelly App</th>
                      <th className="text-right py-3 px-4 text-white/60 font-medium">Print</th>
                      <th className="text-right py-3 px-4 text-white/60 font-medium">iLearn</th>
                      <th className="text-right py-3 px-4 text-white/60 font-medium">Enterprise</th>
                      <th className="text-right py-3 px-4 text-white/60 font-medium">Events</th>
                      <th className="text-right py-3 px-4 text-white font-semibold">Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(PRODUCT_REVENUE_PROJECTIONS).map(([year, data]) => (
                      <tr key={year} className="border-b border-white/5 hover:bg-white/5">
                        <td className="py-3 px-4 font-medium">{year.replace('year', 'Year ')}</td>
                        <td className="py-3 px-4 text-right text-blue-400">
                          {formatCurrency(data.kellyApp?.revenue || 0)}
                        </td>
                        <td className="py-3 px-4 text-right text-amber-400">
                          {formatCurrency((data.dailyLessonPrint?.revenue || 0) + (data.annualBook?.revenue || 0))}
                        </td>
                        <td className="py-3 px-4 text-right text-purple-400">
                          {formatCurrency(
                            (data.iLearnMini?.revenue || 0) + 
                            (data.iLearnPro?.revenue || 0) + 
                            (data.iLearnHub?.revenue || 0)
                          )}
                        </td>
                        <td className="py-3 px-4 text-right text-emerald-400">
                          {formatCurrency(data.kellyEnterprise?.revenue || 0)}
                        </td>
                        <td className="py-3 px-4 text-right text-pink-400">
                          {formatCurrency((data.worldCup2026?.bonus || 0) + (data.olympics2028?.bonus || 0))}
                        </td>
                        <td className="py-3 px-4 text-right font-semibold text-white">
                          {formatCurrency(data.total)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        ) : (
          <>
            {/* Growth Events */}
            <div className="space-y-6">
              {Object.entries(GROWTH_EVENTS).map(([key, event], i) => (
                <motion.div
                  key={key}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="p-6 rounded-2xl bg-gradient-to-br from-white/5 to-white/10 border border-white/10"
                >
                  <div className="flex items-start gap-4 mb-4">
                    <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-amber-500/30 to-orange-500/30 flex items-center justify-center">
                      <Trophy className="w-7 h-7 text-amber-400" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h3 className="text-xl font-semibold">{event.name}</h3>
                        <span className="text-sm text-white/50">{event.date}</span>
                      </div>
                      <div className="flex items-center gap-2 mt-1 text-sm text-white/60">
                        <Globe className="w-4 h-4" />
                        {event.location}
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-white/70 mb-4">{event.opportunity}</p>
                  
                  <div className="flex items-center gap-2 mb-4 text-sm">
                    <Users className="w-4 h-4 text-white/40" />
                    <span className="text-white/60">Projected Reach:</span>
                    <span className="font-semibold">{formatNumber(event.projectedReach)} viewers</span>
                  </div>
                  
                  {/* Kelly Integration */}
                  <div className="mb-4">
                    <h4 className="text-sm font-medium text-white/60 mb-2">Kelly Integration</h4>
                    <ul className="space-y-1">
                      {event.kellyIntegration.map((item, j) => (
                        <li key={j} className="flex items-center gap-2 text-sm text-white/80">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  {/* Revenue Impact */}
                  <div className="pt-4 border-t border-white/10">
                    <h4 className="text-sm font-medium text-white/60 mb-3">Projected Revenue Impact</h4>
                    <div className="grid grid-cols-3 gap-4">
                      {event.revenueImpact.subscriptions && (
                        <div className="text-center p-3 rounded-lg bg-blue-500/10">
                          <div className="text-lg font-semibold text-blue-400">
                            +{formatNumber(event.revenueImpact.subscriptions)}
                          </div>
                          <div className="text-xs text-white/50">New Subscribers</div>
                        </div>
                      )}
                      {event.revenueImpact.hardware && (
                        <div className="text-center p-3 rounded-lg bg-purple-500/10">
                          <div className="text-lg font-semibold text-purple-400">
                            +{formatNumber(event.revenueImpact.hardware)}
                          </div>
                          <div className="text-xs text-white/50">iLearn Units</div>
                        </div>
                      )}
                      {event.revenueImpact.licensing && (
                        <div className="text-center p-3 rounded-lg bg-emerald-500/10">
                          <div className="text-lg font-semibold text-emerald-400">
                            {formatCurrency(event.revenueImpact.licensing)}
                          </div>
                          <div className="text-xs text-white/50">Partnership Value</div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}
      </main>
    </div>
  )
}
