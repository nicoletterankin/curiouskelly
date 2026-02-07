'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { ChevronDown, CheckCircle2, Circle, ExternalLink } from 'lucide-react'
import {
  RAISE_TARGET,
  DEADLINE,
  TIERS,
  COALITION_PARTNERS,
  DAILY_PRIORITIES,
  formatMoney,
  getStageBadge,
  getStageColor,
  computeTotalPipeline,
  computeTierTotal,
  computeStageTotal,
  daysUntilDeadline,
} from '@/data/coalition-data'

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } },
}

const item = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0 } }

export default function DallasPage() {
  const [timeRemaining, setTimeRemaining] = useState('')
  const [checklist, setChecklist] = useState<Record<string, boolean>>({})
  const [expandedTier, setExpandedTier] = useState<string>('founding')

  // Countdown timer
  useEffect(() => {
    const updateCountdown = () => {
      const now = new Date()
      const diff = DEADLINE.getTime() - now.getTime()

      if (diff <= 0) {
        setTimeRemaining('DEADLINE PASSED')
        return
      }

      const days = Math.floor(diff / (1000 * 60 * 60 * 24))
      const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60))
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60))
      const seconds = Math.floor((diff % (1000 * 60)) / 1000)

      setTimeRemaining(`${days}d ${hours}h ${minutes}m ${seconds}s`)
    }

    updateCountdown()
    const interval = setInterval(updateCountdown, 1000)
    return () => clearInterval(interval)
  }, [])

  // Load checklist from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('dallas-checklist')
    if (saved) {
      setChecklist(JSON.parse(saved))
    }
  }, [])

  // Save checklist to localStorage
  const toggleCheckbox = (key: string) => {
    const updated = { ...checklist, [key]: !checklist[key] }
    setChecklist(updated)
    localStorage.setItem('dallas-checklist', JSON.stringify(updated))
  }

  const totalPipeline = computeTotalPipeline()
  const gap = RAISE_TARGET - totalPipeline
  const percentComplete = Math.round((totalPipeline / RAISE_TARGET) * 100)

  // Determine countdown color
  const isUrgent = daysUntilDeadline() < 1
  const countdownColor = isUrgent ? 'text-red-500' : 'text-amber-400'

  return (
    <main className="min-h-screen bg-black text-white">
      {/* HEADER */}
      <section className="border-b border-white/10 bg-gradient-to-b from-black to-black/50 px-6 py-12">
        <div className="mx-auto max-w-7xl">
          <motion.div
            variants={container}
            initial="hidden"
            animate="show"
            className="space-y-6"
          >
            <motion.div variants={item}>
              <p className="mb-2 text-xs uppercase tracking-widest text-amber-500">
                War Room
              </p>
              <h1 className="text-5xl font-light text-white" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>
                DALLAS COMMAND CENTER
              </h1>
              <p className="mt-2 text-zinc-400">
                $3B Coalition Raise — 4-Day Sprint to Feb 4 5pm PT
              </p>
            </motion.div>

            <motion.div variants={item} className="pt-4">
              <div className={`inline-block rounded-lg border border-amber-500/30 bg-amber-500/10 px-6 py-4 ${countdownColor}`}>
                <p className="text-xs uppercase tracking-widest mb-1 text-amber-400/70">Time Until Deadline</p>
                <p className="font-mono text-2xl font-bold">{timeRemaining}</p>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* STATS ROW */}
      <section className="border-b border-white/10 px-6 py-12">
        <div className="mx-auto max-w-7xl">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
          >
            <motion.div variants={item} className="rounded-lg border border-white/10 bg-white/[0.02] p-6">
              <p className="text-xs uppercase tracking-widest text-zinc-500 mb-2">Pipeline</p>
              <p className="text-3xl font-bold text-blue-400 font-mono">{formatMoney(totalPipeline)}</p>
              <p className="text-xs text-zinc-600 mt-2">{percentComplete}% of goal</p>
            </motion.div>

            <motion.div variants={item} className="rounded-lg border border-white/10 bg-white/[0.02] p-6">
              <p className="text-xs uppercase tracking-widest text-zinc-500 mb-2">Target</p>
              <p className="text-3xl font-bold text-emerald-400 font-mono">{formatMoney(RAISE_TARGET)}</p>
              <p className="text-xs text-zinc-600 mt-2">Full infrastructure</p>
            </motion.div>

            <motion.div variants={item} className="rounded-lg border border-white/10 bg-white/[0.02] p-6">
              <p className="text-xs uppercase tracking-widest text-zinc-500 mb-2">Gap</p>
              <p className={`text-3xl font-bold font-mono ${gap > 0 ? 'text-amber-400' : 'text-green-400'}`}>
                {formatMoney(gap)}
              </p>
              <p className="text-xs text-zinc-600 mt-2">To raise</p>
            </motion.div>

            <motion.div variants={item} className="rounded-lg border border-white/10 bg-white/[0.02] p-6">
              <p className="text-xs uppercase tracking-widest text-zinc-500 mb-2">Partners</p>
              <p className="text-3xl font-bold text-purple-400 font-mono">{COALITION_PARTNERS.length}</p>
              <p className="text-xs text-zinc-600 mt-2">In pipeline</p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* PROGRESS BAR */}
      <section className="border-b border-white/10 px-6 py-8">
        <div className="mx-auto max-w-7xl">
          <motion.div variants={item} initial="hidden" whileInView="show" viewport={{ once: true }}>
            <div className="h-3 w-full rounded-full bg-zinc-900">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-amber-500 transition-all duration-500"
                style={{ width: `${Math.min(percentComplete, 100)}%` }}
              />
            </div>
            <p className="mt-2 text-right text-xs text-zinc-500">
              {formatMoney(totalPipeline)} / {formatMoney(RAISE_TARGET)}
            </p>
          </motion.div>
        </div>
      </section>

      {/* TWO COLUMN LAYOUT */}
      <section className="px-6 py-12">
        <div className="mx-auto max-w-7xl">
          <div className="grid gap-12 lg:grid-cols-3">
            {/* LEFT: Pipeline by Tier */}
            <div className="lg:col-span-2 space-y-4">
              <h2 className="text-2xl font-light mb-6" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>
                Pipeline by Tier
              </h2>

              {TIERS.map((tier) => {
                const tierPartners = COALITION_PARTNERS.filter((p) => p.tier === tier.id)
                const tierTotal = computeTierTotal(tier.id)
                const isExpanded = expandedTier === tier.id

                return (
                  <motion.div
                    key={tier.id}
                    initial="hidden"
                    whileInView="show"
                    viewport={{ once: true }}
                    variants={item}
                    className="rounded-lg border border-white/10 bg-white/[0.02] overflow-hidden"
                  >
                    <button
                      onClick={() => setExpandedTier(isExpanded ? '' : tier.id)}
                      className="w-full flex items-center justify-between p-4 hover:bg-white/5 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">{tier.emoji}</span>
                        <div className="text-left">
                          <p className="font-medium">{tier.name}</p>
                          <p className="text-xs text-zinc-500">{tierPartners.length} partners • {formatMoney(tierTotal)}</p>
                        </div>
                      </div>
                      <ChevronDown
                        className={`w-5 h-5 text-zinc-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                      />
                    </button>

                    {isExpanded && (
                      <div className="border-t border-white/5 divide-y divide-white/5">
                        {tierPartners.map((partner) => {
                          const badge = getStageBadge(partner.stage)
                          return (
                            <div key={partner.id} className="p-4 flex items-center justify-between hover:bg-white/5">
                              <div>
                                <p className="font-medium text-sm">{partner.name}</p>
                                <p className="text-xs text-zinc-600 mt-0.5">{partner.category}</p>
                              </div>
                              <div className="flex items-center gap-3">
                                <span className="font-mono text-sm text-zinc-400">{formatMoney(partner.targetAmount)}</span>
                                <span className={`px-2 py-1 text-xs rounded border ${badge.bg} ${badge.text} ${badge.border}`}>
                                  {partner.stage}
                                </span>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </motion.div>
                )
              })}
            </div>

            {/* RIGHT: Today's Checklist */}
            <div>
              <h2 className="text-2xl font-light mb-6" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>
                Action Items
              </h2>

              {DAILY_PRIORITIES.map((day) => {
                const dayKey = day.day.split(' ')[0]

                return (
                  <motion.div
                    key={day.date}
                    initial="hidden"
                    whileInView="show"
                    viewport={{ once: true }}
                    variants={item}
                    className="mb-6 rounded-lg border border-white/10 bg-white/[0.02] p-4"
                  >
                    <div className="mb-3">
                      <p className="font-medium text-sm">{day.day}</p>
                      <p className={`text-xs mt-1 px-2 py-1 rounded w-fit ${
                        day.priority === 'CRITICAL'
                          ? 'bg-red-500/20 text-red-400'
                          : 'bg-amber-500/20 text-amber-400'
                      }`}>
                        {day.priority}
                      </p>
                    </div>

                    <div className="space-y-2">
                      {day.tasks.map((task, idx) => {
                        const taskKey = `${dayKey}-${idx}`
                        const isChecked = checklist[taskKey]

                        return (
                          <button
                            key={taskKey}
                            onClick={() => toggleCheckbox(taskKey)}
                            className="flex items-start gap-2 text-left text-xs text-zinc-400 hover:text-zinc-300 transition-colors w-full"
                          >
                            {isChecked ? (
                              <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
                            ) : (
                              <Circle className="w-4 h-4 text-zinc-600 flex-shrink-0 mt-0.5" />
                            )}
                            <span className={isChecked ? 'line-through text-zinc-600' : ''}>
                              {task}
                            </span>
                          </button>
                        )
                      })}
                    </div>
                  </motion.div>
                )
              })}
            </div>
          </div>
        </div>
      </section>

      {/* TIER SUMMARY */}
      <section className="border-t border-white/10 px-6 py-12">
        <div className="mx-auto max-w-7xl">
          <h2 className="text-2xl font-light mb-6" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>
            Investment Tiers
          </h2>
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
          >
            {TIERS.map((tier) => (
              <motion.div key={tier.id} variants={item} className="rounded-lg border border-white/10 bg-white/[0.02] p-6">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-3xl">{tier.emoji}</span>
                  <p className="font-medium">{tier.name}</p>
                </div>
                <div className="space-y-2 text-xs text-zinc-500 mt-4">
                  <div>
                    <p className="text-zinc-600">Minimum</p>
                    <p className="text-zinc-300 font-mono">{formatMoney(tier.minAmount)}</p>
                  </div>
                  <div>
                    <p className="text-zinc-600">Target</p>
                    <p className="text-zinc-300 font-mono">{formatMoney(tier.targetAmount)}</p>
                  </div>
                  <div>
                    <p className="text-zinc-600">Slots</p>
                    <p className="text-zinc-300">{tier.targetPartners || '∞'}</p>
                  </div>
                  <div>
                    <p className="text-zinc-600">Terms</p>
                    <p className="text-zinc-300">{tier.terms.type}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* QUICK LINKS */}
      <section className="border-t border-white/10 px-6 py-12 bg-gradient-to-b from-black to-black/50">
        <div className="mx-auto max-w-7xl">
          <h2 className="text-2xl font-light mb-6" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>
            Quick Links
          </h2>
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="grid grid-cols-1 gap-4 sm:grid-cols-3"
          >
            <motion.div variants={item}>
              <Link
                href="/coalition"
                className="flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] p-6 hover:bg-white/5 transition-colors group"
              >
                <div>
                  <p className="font-medium text-sm">Coalition</p>
                  <p className="text-xs text-zinc-600">Public raise page</p>
                </div>
                <ExternalLink className="w-4 h-4 text-zinc-600 group-hover:text-white" />
              </Link>
            </motion.div>

            <motion.div variants={item}>
              <Link
                href="/strategic"
                className="flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] p-6 hover:bg-white/5 transition-colors group"
              >
                <div>
                  <p className="font-medium text-sm">Strategic</p>
                  <p className="text-xs text-zinc-600">Long-term plan</p>
                </div>
                <ExternalLink className="w-4 h-4 text-zinc-600 group-hover:text-white" />
              </Link>
            </motion.div>

            <motion.div variants={item}>
              <Link
                href="/ziggurat"
                className="flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] p-6 hover:bg-white/5 transition-colors group"
              >
                <div>
                  <p className="font-medium text-sm">Ziggurat</p>
                  <p className="text-xs text-zinc-600">Project details</p>
                </div>
                <ExternalLink className="w-4 h-4 text-zinc-600 group-hover:text-white" />
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </main>
  )
}
