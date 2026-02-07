"use client"

import { useState } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import { ChevronDown } from "lucide-react"
import { 
  offerStatus, 
  commitmentStatus, 
  coalitionPartners, 
  timeline, 
  disclaimer,
  TIERS,
  RAISE_TARGET,
  PHASE_1_TARGET,
  formatMoney,
  COALITION_DATA,
} from "@/data/coalition-data"
import { Mail, Download, Users, Building2, TrendingUp, Target } from "lucide-react"
import { InvestmentCalendar } from "@/components/coalition/investment-calendar"
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, ReferenceLine 
} from 'recharts'

// Ziggurat property details - TRUE NUMBERS
const ZIGGURAT = {
  offer: 200_000_000,
  closing: 4_000_000,
  renovation: 75_000_000,
  reserve: 30_000_000,
  phase1Total: 309_000_000,
  sqft: 1_003_041,
  floors: 7,
  acres: 92,
  built: 1971,
  address: "24000 Avila Road, Laguna Niguel, CA 92677",
}

// Four revenue pillars with 2030 numbers
const PILLARS = [
  { name: "BROADCAST", description: "Subscriptions + Ads + Enterprise", reach: "5B connected humans", revenue2030: 120_000_000 },
  { name: "HARDWARE", description: "iLearn (Direct + Apple + B2B)", reach: "410K units/year by 2030", revenue2030: 190_000_000 },
  { name: "PRINT", description: "Eggs, Books, Kits", reach: "3B offline humans", revenue2030: 25_000_000 },
  { name: "INFRASTRUCTURE", description: "Ziggurat + Licensing", reach: "100-year asset", revenue2030: 40_000_000 },
]

// Building history timeline
const BUILDING_TIMELINE = [
  { year: "1971", event: "Ziggurat built by Chet Holifield", highlight: false },
  { year: "2015", event: "National Register eligible", highlight: false },
  { year: "2023", event: "First GSA auction - no bids", highlight: false },
  { year: "2024", event: "Second auction - $177M bid cancelled", highlight: false },
  { year: "2025", event: "IRS vacates, building now empty", highlight: false },
  { year: "Feb 2026", event: "Third auction - Coalition offer", highlight: true },
]

// Learner growth milestones
const MILESTONES = [
  { year: 2026, learners: 100_000, label: "Kelly launches from the Ziggurat" },
  { year: 2027, learners: 1_000_000, label: "Growth year" },
  { year: 2028, learners: 10_000_000, label: "LA Olympics - global visibility" },
  { year: 2029, learners: 50_000_000, label: "International expansion" },
  { year: 2030, learners: 150_000_000, label: "Scale - $375M revenue" },
  { year: 2032, learners: 500_000_000, label: "Self-sustaining - $1.05B revenue" },
  { year: 2035, learners: 1_000_000_000, label: "First billion learners" },
  { year: 2040, learners: 2_000_000_000, label: "Two billion" },
  { year: 2050, learners: 5_000_000_000, label: "Majority of humanity" },
  { year: 2075, learners: 8_000_000_000, label: "UN SDG 4 achieved" },
]

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

function formatCurrency(amount: number, compact = false): string {
  if (compact) {
    if (amount >= 1_000_000_000) return `$${(amount / 1_000_000_000).toFixed(1)}B`
    if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(0)}M`
    if (amount >= 1_000) return `$${(amount / 1_000).toFixed(0)}K`
  }
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(amount)
}

function formatNumber(num: number): string {
  if (num >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1)}B`
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(0)}M`
  if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`
  return num.toLocaleString()
}

// ═══════════════════════════════════════════════════════════════════════════
// ANIMATIONS
// ═══════════════════════════════════════════════════════════════════════════

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08, delayChildren: 0.1 } }
}

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: 'easeOut' } }
}

// ═══════════════════════════════════════════════════════════════════════════
// STATUS BADGE
// ═══════════════════════════════════════════════════════════════════════════

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    "Outreach pending": "border-amber-600/50 text-amber-400",
    "To be filed": "border-purple-600/50 text-purple-400",
    "Research phase": "border-cyan-600/50 text-cyan-400",
    "Committed": "border-green-600/50 text-green-400",
  }
  return (
    <span className={`px-2 py-0.5 text-xs border rounded ${styles[status] || styles["Outreach pending"]}`}>
      {status}
    </span>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════

const NAV_ITEMS = [
  { id: "mission", label: "Mission" },
  { id: "ziggurat", label: "Ziggurat" },
  { id: "coalition", label: "Coalition" },
  { id: "financials", label: "Financials" },
  { id: "governance", label: "Governance" },
  { id: "timeline", label: "Timeline" },
  { id: "calendar", label: "Calendar" },
]

function Navigation() {
  const [isOpen, setIsOpen] = useState(false)
  
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0a0a0a]/90 backdrop-blur-md border-b border-white/5">
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="text-white/80 hover:text-white text-sm tracking-widest">
          LESSON OF THE DAY, PBC
        </Link>
        
        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-8">
          {NAV_ITEMS.map(({ id, label }) => (
            <a 
              key={id} 
              href={`#${id}`}
              className="text-white/50 hover:text-white text-xs uppercase tracking-wider transition-colors"
            >
              {label}
            </a>
          ))}
        </div>
        
        {/* Mobile toggle */}
        <button 
          onClick={() => setIsOpen(!isOpen)} 
          className="md:hidden text-white/70"
        >
          <ChevronDown className={`w-5 h-5 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </button>
      </div>
      
      {/* Mobile menu */}
      {isOpen && (
        <div className="md:hidden border-t border-white/5 bg-[#0a0a0a]">
          {NAV_ITEMS.map(({ id, label }) => (
            <a 
              key={id}
              href={`#${id}`}
              onClick={() => setIsOpen(false)}
              className="block px-6 py-3 text-white/60 hover:text-white hover:bg-white/5 text-sm"
            >
              {label}
            </a>
          ))}
        </div>
      )}
    </nav>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// PAGE COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

export default function CoalitionPage() {
  const coalitionTotal = COALITION_DATA.coalition.reduce((sum, p) => sum + p.amount, 0)
  const gap = COALITION_DATA.ziggurat.phase1Total - coalitionTotal
  
  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white">
      <Navigation />
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* HERO */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="min-h-screen flex flex-col justify-center px-6 pt-24 pb-16">
        <div className="max-w-4xl mx-auto w-full">
          <motion.div
            variants={container}
            initial="hidden"
            animate="show"
            className="space-y-8"
          >
            <motion.div variants={item}>
              <Link href="/strategic" className="text-zinc-500 text-xs tracking-[0.2em] uppercase hover:text-white transition-colors inline-flex items-center gap-2">
                <span>←</span> Strategic Overview
              </Link>
            </motion.div>
            
            <motion.p variants={item} className="text-zinc-500 text-xs tracking-[0.4em] uppercase">
              Lesson of the Day, PBC
            </motion.p>
            
            <motion.h1 
              variants={item}
              className="text-4xl md:text-6xl lg:text-7xl font-light text-white leading-tight"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
            >
              Coalition Model
            </motion.h1>
            
            <motion.p 
              variants={item}
              className="text-2xl md:text-3xl text-blue-400 font-light"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
            >
              Victory of the People
            </motion.p>
            
            <motion.div variants={item} className="pt-8 space-y-2">
              <p className="text-zinc-400">{commitmentStatus.asOfDate}</p>
              <p className="text-zinc-500 text-sm">Coalition building in progress</p>
            </motion.div>
            
            <motion.div variants={item} className="pt-4">
              <p className="text-amber-400 text-sm font-medium">
                {offerStatus.statusLabel}
              </p>
            </motion.div>
          </motion.div>
        </div>
        
        {/* Scroll indicator */}
        <div className="absolute bottom-12 left-1/2 -translate-x-1/2">
          <motion.div 
            animate={{ y: [0, 8, 0] }} 
            transition={{ duration: 2, repeat: Infinity }}
            className="text-zinc-600"
          >
            <ChevronDown className="w-6 h-6" />
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* DISCLAIMER BANNER */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-24 px-6 bg-amber-950/20 border-b border-amber-800/50">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="space-y-4"
          >
            <motion.div variants={item} className="flex items-start gap-4">
              <span className="text-amber-500 text-2xl flex-shrink-0 mt-1">⚠</span>
              <div>
                <p className="text-amber-200 font-medium mb-2">{disclaimer}</p>
                <p className="text-amber-300/70 text-sm">
                  This page shows target partners and planned offer terms. All amounts are goals, not confirmed. As of {commitmentStatus.asOfDate}, no commitments have been received.
                </p>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      <section id="mission" className="py-32 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              The Mission
            </motion.p>
            
            <motion.div variants={item} className="space-y-6">
              <h2 
                className="text-3xl md:text-5xl font-light text-white leading-tight"
                style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
              >
                Kelly teaches 8 billion humans.
              </h2>
              <p 
                className="text-2xl md:text-3xl text-zinc-400 font-light"
                style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
              >
                Every person. Every day. For life.
              </p>
            </motion.div>
            
            <motion.p variants={item} className="text-xl text-blue-400 font-light italic">
              Loving saving grace for curious minds.
            </motion.p>
            
            <motion.div variants={item} className="pt-8 space-y-4 text-zinc-400 text-lg leading-relaxed">
              <p>Not a company to be sold.</p>
              <p>Infrastructure for humanity to be shared.</p>
            </motion.div>
            
            <motion.div variants={item} className="pt-12 border-t border-white/5 pt-8">
              <p className="text-zinc-600 text-sm leading-relaxed max-w-2xl">
                Alexa became the voice of the home.<br />
                Siri became the voice of your device.<br />
                Kelly becomes the voice of learning.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THE ZIGGURAT */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section id="ziggurat" className="py-32 px-6 bg-[#141414]">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              The Ziggurat
            </motion.p>
            
            {/* Stats Grid */}
            <motion.div variants={item} className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { value: COALITION_DATA.ziggurat.sqft.toLocaleString(), label: "Square Feet" },
                { value: COALITION_DATA.ziggurat.floors, label: "Floors" },
                { value: COALITION_DATA.ziggurat.acres, label: "Acres" },
                { value: COALITION_DATA.ziggurat.built, label: "Built" },
              ].map(({ value, label }) => (
                <div key={label} className="bg-[#0a0a0a] border border-white/5 p-6">
                  <p className="text-2xl md:text-3xl text-white font-light tabular-nums">{value}</p>
                  <p className="text-xs text-zinc-500 uppercase tracking-wider mt-1">{label}</p>
                </div>
              ))}
            </motion.div>
            
            <motion.div variants={item} className="space-y-2">
              <p className="text-white">{COALITION_DATA.ziggurat.address}</p>
              <p className="text-zinc-500 text-sm">
                A brutalist masterpiece.<br />
                Kelly's home for the next hundred years.
              </p>
            </motion.div>
            
            {/* Building Timeline */}
            <motion.div variants={item} className="pt-8">
              <p className="text-zinc-600 text-xs tracking-[0.4em] uppercase mb-6">Building Timeline</p>
              <div className="space-y-3">
                {COALITION_DATA.buildingTimeline.map(({ year, event, highlight }) => (
                  <div key={year + event} className="flex gap-6 items-baseline">
                    <span className={`text-sm font-mono ${highlight ? 'text-blue-400' : 'text-zinc-500'}`}>
                      {year}
                    </span>
                    <span className={`text-sm ${highlight ? 'text-blue-400' : 'text-zinc-400'}`}>
                      {event}
                    </span>
                  </div>
                ))}
                <div className="flex gap-6 items-baseline">
                  <span className="text-sm font-mono text-emerald-400">2026</span>
                  <span className="text-sm text-emerald-400">Our goal — Close by February 7</span>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THE INVITATION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section id="coalition" className="py-32 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              The Invitation
            </motion.p>
            
            <motion.h2 
              variants={item}
              className="text-3xl md:text-4xl font-light text-white"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
            >
              You're Invited to the Table
            </motion.h2>
            
            <motion.div variants={item} className="space-y-4 text-zinc-400 leading-relaxed">
              <p>
                We're assembling a coalition of mission-aligned partners 
                to acquire the Ziggurat and build Kelly's home.
              </p>
              <p className="text-zinc-500">
                No one has committed yet.<br />
                Everyone sees this together for the first time.
              </p>
            </motion.div>
            
            {/* Coalition Table */}
            <motion.div variants={item} className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10 text-left">
                    <th className="pb-3 text-xs text-zinc-500 uppercase tracking-wider font-normal">Partner</th>
                    <th className="pb-3 text-xs text-zinc-500 uppercase tracking-wider font-normal text-right">Target Amount</th>
                    <th className="pb-3 text-xs text-zinc-500 uppercase tracking-wider font-normal text-center">Type</th>
                    <th className="pb-3 text-xs text-zinc-500 uppercase tracking-wider font-normal text-right">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {coalitionPartners.map((p) => (
                    <tr key={p.id} className="border-b border-white/5">
                      <td className="py-4 text-white">
                        <div>
                          <p className="font-medium">{p.name}</p>
                          <p className="text-xs text-zinc-600">{p.category}</p>
                        </div>
                      </td>
                      <td className="py-4 text-zinc-300 text-right tabular-nums">{formatCurrency(p.targetAmount, true)}</td>
                      <td className="py-4 text-zinc-500 text-center text-sm">{p.tier === 'founding' ? 'PRI' : p.tier === 'anchor' ? 'PRI/Grant' : 'Revenue Share'}</td>
                      <td className="py-4 text-right"><StatusBadge status={p.stage} /></td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr className="border-t border-white/20">
                    <td className="pt-4 text-white font-medium">Coalition Total</td>
                    <td className="pt-4 text-blue-400 text-right tabular-nums font-medium">{formatCurrency(coalitionTotal, true)}</td>
                    <td></td>
                    <td></td>
                  </tr>
                </tfoot>
              </table>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* WHAT WE NEED */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-32 px-6 bg-[#141414]">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              Phase 1 Capital
            </motion.p>
            
            <motion.div variants={item} className="space-y-4 font-mono text-sm">
              {[
                { label: "Acquisition", amount: COALITION_DATA.ziggurat.offer },
                { label: "Closing & Due Diligence", amount: COALITION_DATA.ziggurat.closing },
                { label: "Renovation (Kelly HQ)", amount: COALITION_DATA.ziggurat.renovation },
                { label: "Operating Reserve", amount: COALITION_DATA.ziggurat.reserve },
              ].map(({ label, amount }) => (
                <div key={label} className="flex justify-between text-zinc-400">
                  <span>{label}</span>
                  <span className="tabular-nums">{formatCurrency(amount)}</span>
                </div>
              ))}
              
              <div className="border-t border-white/20 pt-4 flex justify-between text-white">
                <span>Total</span>
                <span className="tabular-nums">{formatCurrency(COALITION_DATA.ziggurat.phase1Total)}</span>
              </div>
            </motion.div>
            
            <motion.div variants={item} className="pt-8 space-y-3 font-mono text-sm">
              <div className="flex justify-between text-zinc-500">
                <span>Coalition Committed</span>
                <span className="tabular-nums">$0</span>
              </div>
              <div className="flex justify-between text-blue-400">
                <span>Coalition Invited</span>
                <span className="tabular-nums">{formatCurrency(coalitionTotal)}</span>
              </div>
              <div className="flex justify-between text-amber-400">
                <span>Gap to Close</span>
                <span className="tabular-nums">{formatCurrency(gap)}</span>
              </div>
            </motion.div>
            
            <motion.p variants={item} className="text-zinc-600 text-sm">
              The gap represents an opportunity for additional partners.
            </motion.p>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* FIVE-YEAR VIEW */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section id="financials" className="py-32 px-6 border-t border-white/5">
        <div className="max-w-5xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              Five-Year View (2026-2030)
            </motion.p>
            
            <motion.div variants={item} className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="pb-3 text-left text-xs text-zinc-500 uppercase tracking-wider font-normal"></th>
                    {Object.keys(COALITION_DATA.financials).map(year => (
                      <th key={year} className="pb-3 text-right text-xs text-zinc-500 uppercase tracking-wider font-normal">{year}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-white/5">
                    <td className="py-3 text-white font-medium">Revenue</td>
                    {Object.values(COALITION_DATA.financials).map((data, i) => (
                      <td key={i} className="py-3 text-right text-white tabular-nums">{formatCurrency(data.revenue, true)}</td>
                    ))}
                  </tr>
                  <tr className="border-b border-white/5">
                    <td className="py-3 text-zinc-500 pl-4">Broadcast</td>
                    {Object.values(COALITION_DATA.financials).map((data, i) => (
                      <td key={i} className="py-3 text-right text-zinc-500 tabular-nums">{formatCurrency(data.broadcast, true)}</td>
                    ))}
                  </tr>
                  <tr className="border-b border-white/5">
                    <td className="py-3 text-zinc-500 pl-4">Print</td>
                    {Object.values(COALITION_DATA.financials).map((data, i) => (
                      <td key={i} className="py-3 text-right text-zinc-500 tabular-nums">{data.print ? formatCurrency(data.print, true) : '—'}</td>
                    ))}
                  </tr>
                  <tr className="border-b border-white/5">
                    <td className="py-3 text-zinc-500 pl-4">iLearn</td>
                    {Object.values(COALITION_DATA.financials).map((data, i) => (
                      <td key={i} className="py-3 text-right text-zinc-500 tabular-nums">{data.ilearn ? formatCurrency(data.ilearn, true) : '—'}</td>
                    ))}
                  </tr>
                  <tr className="border-b border-white/5">
                    <td className="py-3 text-zinc-500 pl-4">Ziggurat</td>
                    {Object.values(COALITION_DATA.financials).map((data, i) => (
                      <td key={i} className="py-3 text-right text-zinc-500 tabular-nums">{data.ziggurat ? formatCurrency(data.ziggurat, true) : '—'}</td>
                    ))}
                  </tr>
                  <tr className="border-b border-white/10">
                    <td className="py-3 text-white font-medium">EBITDA</td>
                    {Object.values(COALITION_DATA.financials).map((data, i) => (
                      <td key={i} className={`py-3 text-right tabular-nums font-medium ${data.ebitda < 0 ? 'text-red-400/70' : 'text-blue-400'}`}>
                        {data.ebitda < 0 ? `(${formatCurrency(Math.abs(data.ebitda), true)})` : formatCurrency(data.ebitda, true)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-3 text-white font-medium">Learners</td>
                    {Object.values(COALITION_DATA.financials).map((data, i) => (
                      <td key={i} className="py-3 text-right text-blue-400 tabular-nums font-medium">{formatNumber(data.learners)}</td>
                    ))}
                  </tr>
                </tbody>
                <tfoot>
                  <tr className="bg-zinc-900 border-t border-zinc-700">
                    <td className="py-4 px-0 font-medium text-white">TOTAL TARGETED</td>
                    <td className="py-4 text-right font-mono text-white font-medium">{formatCurrency(commitmentStatus.totalTargeted, true)}</td>
                    <td colSpan={2} className="py-4 px-0 text-center text-zinc-500 text-sm">
                      Phase 1 need: {formatCurrency(commitmentStatus.phase1Need, true)}
                    </td>
                  </tr>
                </tfoot>
              </table>
            </motion.div>
            
            {/* Revenue & Learners Chart */}
            <motion.div variants={item} className="mt-12">
              <p className="text-xs text-zinc-600 uppercase tracking-wider mb-6">Revenue Growth Trajectory</p>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={Object.entries(COALITION_DATA.financials).map(([year, data]) => ({
                      year,
                      revenue: data.revenue / 1_000_000_000,
                      ebitda: data.ebitda / 1_000_000_000,
                      learners: data.learners / 1_000_000_000,
                    }))}
                    margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="ebitdaGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                    <XAxis 
                      dataKey="year" 
                      stroke="#52525b" 
                      tick={{ fill: '#71717a', fontSize: 11 }}
                      axisLine={{ stroke: '#27272a' }}
                    />
                    <YAxis 
                      stroke="#52525b" 
                      tick={{ fill: '#71717a', fontSize: 11 }}
                      axisLine={{ stroke: '#27272a' }}
                      tickFormatter={(v) => `$${v}B`}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#18181b', 
                        border: '1px solid #27272a',
                        borderRadius: '8px',
                        fontSize: '12px'
                      }}
                      labelStyle={{ color: '#a1a1aa' }}
                      formatter={(value: number, name: string) => [
                        `$${value.toFixed(1)}B`, 
                        name === 'revenue' ? 'Revenue' : 'EBITDA'
                      ]}
                    />
                    <ReferenceLine y={0} stroke="#27272a" />
                    <Area 
                      type="monotone" 
                      dataKey="revenue" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      fill="url(#revenueGradient)" 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="ebitda" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      fill="url(#ebitdaGradient)" 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center gap-8 mt-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500" />
                  <span className="text-xs text-zinc-500">Revenue</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-emerald-500" />
                  <span className="text-xs text-zinc-500">EBITDA</span>
                </div>
              </div>
            </motion.div>
            
            {/* Commitment Summary Cards */}
            <motion.div variants={item} className="grid grid-cols-3 gap-4">
              <div className="bg-[#0a0a0a] border border-white/10 p-6">
                <p className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Currently Committed</p>
                <p className="text-2xl text-emerald-400 tabular-nums">{formatCurrency(commitmentStatus.totalCommitted, true)}</p>
              </div>
              <div className="bg-[#0a0a0a] border border-white/10 p-6">
                <p className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Phase 1 Need</p>
                <p className="text-2xl text-white tabular-nums">{formatCurrency(commitmentStatus.phase1Need, true)}</p>
              </div>
              <div className="bg-[#0a0a0a] border border-white/10 p-6">
                <p className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Gap</p>
                <p className="text-2xl text-amber-400 tabular-nums">{formatCurrency(commitmentStatus.gap, true)}</p>
              </div>
            </motion.div>
            
            {/* Pillar descriptions */}
            <motion.div variants={item} className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-8">
              {[
                { name: "Broadcast", desc: "Kelly teaches via web, mobile, and TV" },
                { name: "Print", desc: "Physical materials — no internet required" },
                { name: "iLearn", desc: "Hardware through Apple Retail (LOTD owns 100%)" },
                { name: "Ziggurat", desc: "Events, training, tours, leasing" },
              ].map(({ name, desc }) => (
                <div key={name}>
                  <p className="text-white text-sm font-medium">{name}</p>
                  <p className="text-zinc-500 text-xs mt-1">{desc}</p>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THE FOUR PILLARS */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-32 px-6 bg-[#141414]">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              The Four Pillars
            </motion.p>
            
            <motion.div variants={item} className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {COALITION_DATA.pillars.map((pillar) => (
                <div 
                  key={pillar.name}
                  className="bg-[#0a0a0a] border border-white/5 p-6 hover:border-blue-500/30 transition-colors"
                >
                  <p className="text-blue-400 text-xs tracking-wider uppercase">{pillar.name}</p>
                  <p className="text-white text-lg mt-4">{pillar.description}</p>
                  <p className="text-zinc-500 text-sm mt-1">{pillar.reach}</p>
                  <p className="text-zinc-400 text-sm mt-4 pt-4 border-t border-white/5">
                    2030: <span className="text-white">{formatCurrency(pillar.revenue2030, true)}</span>
                  </p>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* PATH TO 8 BILLION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-32 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              The Journey
            </motion.p>
            
            <motion.div variants={item} className="space-y-8">
              {COALITION_DATA.milestones.map((milestone, i) => (
                <div key={milestone.year} className="flex gap-8 items-baseline">
                  <span className="text-zinc-500 text-lg font-mono w-16">{milestone.year}</span>
                  <div>
                    <p className="text-blue-400 text-xl tabular-nums">{formatNumber(milestone.learners)} learners</p>
                    <p className="text-zinc-500 text-sm mt-1">{milestone.label}</p>
                  </div>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* GOVERNANCE */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section id="governance" className="py-32 px-6 bg-[#141414]">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              Governance
            </motion.p>
            
            <motion.div variants={item} className="grid md:grid-cols-2 gap-8">
              {/* Structure */}
              <div className="space-y-6">
                <p className="text-zinc-500 text-xs tracking-widest uppercase">Structure</p>
                <div className="space-y-4 text-sm">
                  <div>
                    <p className="text-zinc-500">Entity</p>
                    <p className="text-white">Lesson of the Day, PBC</p>
                    <p className="text-zinc-500 text-xs">California Public Benefit Corporation</p>
                  </div>
                  <div>
                    <p className="text-zinc-500">Ownership</p>
                    <p className="text-white">Nicolette Rankin — 100%</p>
                    <p className="text-zinc-500 text-xs">(10,000,000 shares)</p>
                  </div>
                  <div>
                    <p className="text-zinc-500">Operations</p>
                    <p className="text-white">Dallas Short — Principal</p>
                    <p className="text-zinc-500 text-xs">Salary only, no equity</p>
                  </div>
                  <div>
                    <p className="text-zinc-500">Succession</p>
                    <p className="text-white">Designated successor</p>
                    <p className="text-zinc-500 text-xs">Inherits 100% ownership</p>
                  </div>
                </div>
              </div>
              
              {/* Commitment */}
              <div className="space-y-6">
                <p className="text-zinc-500 text-xs tracking-widest uppercase">Commitment</p>
                <div className="space-y-4 text-zinc-400 leading-relaxed">
                  <p>This company will never be sold.</p>
                  <p>This company will never go public.</p>
                  <p>This company exists to teach.</p>
                  <p className="text-zinc-500 text-sm pt-4">
                    Coalition partners are stewards of a mission,
                    not shareholders expecting returns.
                  </p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* VICTORY OF THE PEOPLE */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-32 px-6 border-t border-white/5">
        <div className="max-w-3xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-8"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              Why We Call It
            </motion.p>
            
            <motion.h2 
              variants={item}
              className="text-3xl md:text-4xl font-light text-blue-400"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
            >
              Victory of the People
            </motion.h2>
            
            <motion.div variants={item} className="space-y-6 text-zinc-400 leading-relaxed">
              <p>
                Kelly doesn't belong to investors.<br />
                Kelly belongs to the 8 billion people she teaches.
              </p>
              <p>
                The Ziggurat doesn't exist to generate returns.<br />
                It exists for a hundred years as education's home.
              </p>
              <p>
                This is how we solve SDG 4.<br />
                Not with a company. With infrastructure.
              </p>
              <p className="text-zinc-500">
                Like libraries. Like public schools. Like the internet itself.<br />
                Kelly becomes the foundation everyone builds upon.
              </p>
              <p className="text-white pt-4">
                That's the victory.<br />
                Not an exit. Not a multiple.<br />
                A world where every human has a great teacher.<br />
                Every day. For life.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* KEY DATES */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section id="timeline" className="py-32 px-6 bg-[#141414]">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              Timeline
            </motion.p>
            
            <motion.div variants={item} className="space-y-6">
              {COALITION_DATA.timeline.map((t, i) => (
                <div key={i} className="flex gap-8 items-start">
                  <span className="text-zinc-500 text-sm font-mono w-40 shrink-0">{t.date}</span>
                  <div>
                    <p className="text-white">{t.label}</p>
                    <p className="text-zinc-500 text-sm">{t.sublabel}</p>
                  </div>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* INVESTMENT CALENDAR */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <div id="calendar">
        <InvestmentCalendar />
      </div>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* YOUR INPUT */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-32 px-6 border-t border-white/5">
        <div className="max-w-3xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              Your Input
            </motion.p>
            
            <motion.h2 
              variants={item}
              className="text-2xl md:text-3xl font-light text-white"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
            >
              We'd Love to Hear From You
            </motion.h2>
            
            <motion.div variants={item} className="space-y-4 text-zinc-400 leading-relaxed">
              <p>This model is a starting point, not a final answer.</p>
              <p className="text-zinc-500">
                If you see something that doesn't work, tell us.<br />
                If you want to participate differently, tell us.<br />
                If you know someone who should be here, tell us.
              </p>
              <p className="text-white">Coalition partners help shape the coalition.</p>
            </motion.div>
            
            {/* Contact */}
            <motion.div variants={item} className="pt-8 border-t border-white/5">
              <p className="text-white">Nicolette Rankin</p>
              <p className="text-zinc-500 text-sm">Founder & CEO</p>
              <a 
                href="mailto:nicolette@thedailylesson.com" 
                className="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-2 mt-2"
              >
                <Mail className="w-4 h-4" />
                nicolette@thedailylesson.com
              </a>
            </motion.div>
            
            {/* Questions */}
            <motion.div variants={item} className="pt-8 space-y-3 text-sm text-zinc-500">
              <p className="text-zinc-600 text-xs tracking-widest uppercase mb-4">Questions to Consider</p>
              <p>Does the $200M acquisition price feel right?</p>
              <p>What would you need to say yes?</p>
              <p>Who else belongs at this table?</p>
              <p>What governance would give you confidence?</p>
              <p>What are we missing?</p>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* DOWNLOADS */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-32 px-6 bg-[#141414]">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            <motion.p variants={item} className="text-zinc-600 text-xs tracking-[0.4em] uppercase">
              Full Model
            </motion.p>
            
            <motion.p variants={item} className="text-zinc-400 leading-relaxed">
              Download the complete financial model with detailed projections,
              capital stack breakdown, and space for your notes.
            </motion.p>
            
            <motion.div variants={item} className="flex flex-col sm:flex-row gap-4">
              <a 
                href="/downloads/LOTD_COALITION_MODEL.xlsx"
                className="inline-flex items-center justify-center gap-3 px-6 py-4 bg-blue-500 hover:bg-blue-600 text-white text-sm transition-colors"
              >
                <Download className="w-4 h-4" />
                Download Excel Model
              </a>
              <a 
                href="/downloads/LOTD_Coalition_Package.zip"
                className="inline-flex items-center justify-center gap-3 px-6 py-4 border border-white/20 hover:bg-white/5 text-white text-sm transition-colors"
              >
                <Download className="w-4 h-4" />
                Download Full Package
              </a>
            </motion.div>
            
            <motion.div variants={item} className="text-zinc-500 text-sm space-y-1">
              <p className="text-zinc-600 text-xs tracking-widest uppercase mb-3">Package Includes</p>
              <p>Coalition Model (Excel)</p>
              <p>Executive Summary</p>
              <p>The Kelly Layer — Vision document</p>
              <p>iLearn + Apple — Partnership structure</p>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* FOOTER */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <footer className="py-24 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          <div className="space-y-8">
            <div>
              <p className="text-white text-sm tracking-widest">LESSON OF THE DAY, PBC</p>
              <p className="text-zinc-500 text-sm mt-1">California Public Benefit Corporation</p>
              <p className="text-zinc-600 text-sm">Founded 2022</p>
            </div>
            
            <div className="flex gap-6 text-sm">
              <a href="https://thedailylesson.com" className="text-zinc-500 hover:text-white transition-colors">thedailylesson.com</a>
              <a href="https://curiouskelly.com" className="text-zinc-500 hover:text-white transition-colors">curiouskelly.com</a>
            </div>
            
            <div className="pt-8 border-t border-white/5 text-zinc-600 text-sm leading-relaxed">
              <p>Kelly teaches.</p>
              <p>The Ziggurat stands.</p>
              <p>The people learn.</p>
            </div>
          </div>
        </div>
      </footer>
    </main>
  )
}
