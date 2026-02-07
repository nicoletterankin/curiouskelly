'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { ArrowRight, Smartphone, Monitor, Printer, Building2, Users, Globe, Sparkles } from 'lucide-react'

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
}

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
}

const PRODUCT_CATEGORIES = [
  {
    id: 'subscriptions',
    name: 'Kelly Subscriptions',
    tagline: 'Your AI teacher for life',
    description: 'Daily lessons for ages 2 to 102. Start free, upgrade anytime.',
    icon: Users,
    href: '/pricing',
    color: 'from-blue-600 to-indigo-600',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/30',
    stats: [
      { label: 'Free tier', value: '$0' },
      { label: 'Plus', value: '$9.99/mo' },
      { label: 'Family', value: '$29.99/mo' },
    ],
  },
  {
    id: 'ilearn',
    name: 'iLearn Devices',
    tagline: 'Built for learning. Nothing else.',
    description: 'No distractions. No social media. No ads. Just Kelly and 365 days of wonder.',
    icon: Monitor,
    href: '/products/ilearn',
    color: 'from-purple-600 to-pink-600',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/30',
    stats: [
      { label: 'Mini', value: '$149' },
      { label: 'Standard', value: '$299' },
      { label: 'Pro', value: '$499' },
    ],
  },
  {
    id: 'print',
    name: 'Print Products',
    tagline: 'Learning in your hands',
    description: 'For the 3 billion without internet. Each egg contains one day\'s lesson.',
    icon: Printer,
    href: '/products/print',
    color: 'from-amber-500 to-orange-500',
    bgColor: 'bg-amber-500/10',
    borderColor: 'border-amber-500/30',
    stats: [
      { label: 'Daily Egg', value: '$2.50' },
      { label: 'Month Box', value: '$50' },
      { label: 'Year Set', value: '$500' },
    ],
  },
  {
    id: 'experiences',
    name: 'Ziggurat Experiences',
    tagline: 'Where education meets architecture',
    description: '1 million square feet. 92 acres. Come visit Kelly\'s home.',
    icon: Building2,
    href: '/products/experiences',
    color: 'from-cyan-600 to-teal-600',
    bgColor: 'bg-cyan-500/10',
    borderColor: 'border-cyan-500/30',
    stats: [
      { label: 'Day Pass', value: '$25' },
      { label: 'Guided Tour', value: '$50' },
      { label: 'Training', value: '$500' },
    ],
  },
]

export default function ProductsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-blue-600 via-blue-500 to-indigo-600 text-white">
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />
        <div className="relative max-w-7xl mx-auto px-8 py-24 lg:py-32">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-3xl"
          >
            <p className="text-blue-200 font-semibold flex items-center gap-2">
              <Sparkles className="w-4 h-4" />
              The Daily Lesson
            </p>
            <h1 className="mt-4 text-5xl lg:text-6xl font-bold tracking-tight">
              Products & Experiences
            </h1>
            <p className="mt-6 text-xl text-blue-100 leading-relaxed">
              From free daily lessons to dedicated hardware, from print editions 
              to in-person experiences at the Ziggurat. Kelly reaches every learner.
            </p>
            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                href="/pricing"
                className="px-8 py-4 bg-white text-blue-600 font-semibold rounded-xl hover:bg-blue-50 transition"
              >
                Start Learning Free
              </Link>
              <Link
                href="/coalition"
                className="px-8 py-4 bg-blue-700/50 text-white font-semibold rounded-xl hover:bg-blue-700 transition flex items-center gap-2"
              >
                Join the Coalition
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-8 py-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { value: '365', label: 'Daily Lessons' },
              { value: '12+', label: 'Languages' },
              { value: '2-102', label: 'Ages Served' },
              { value: '8B', label: 'Goal: Every Human' },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <p className="text-3xl font-bold text-blue-600 font-mono tabular-nums">{stat.value}</p>
                <p className="text-sm text-gray-600 mt-1">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Product Categories Grid */}
      <section className="py-24 px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="grid md:grid-cols-2 gap-8"
          >
            {PRODUCT_CATEGORIES.map((category) => {
              const Icon = category.icon
              return (
                <motion.div key={category.id} variants={item}>
                  <Link
                    href={category.href}
                    className={`block group relative overflow-hidden rounded-3xl border ${category.borderColor} ${category.bgColor} p-8 hover:shadow-xl transition-all duration-300`}
                  >
                    {/* Gradient accent */}
                    <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${category.color}`} />
                    
                    <div className="flex items-start justify-between">
                      <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${category.color} flex items-center justify-center shadow-lg`}>
                        <Icon className="w-7 h-7 text-white" />
                      </div>
                      <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-gray-600 group-hover:translate-x-1 transition-all" />
                    </div>

                    <h3 className="mt-6 text-2xl font-bold text-gray-900">{category.name}</h3>
                    <p className="text-gray-600 font-medium">{category.tagline}</p>
                    <p className="mt-3 text-gray-500 text-sm leading-relaxed">{category.description}</p>

                    {/* Price stats */}
                    <div className="mt-6 pt-6 border-t border-gray-200 flex gap-6">
                      {category.stats.map((stat) => (
                        <div key={stat.label}>
                          <p className="text-lg font-bold text-gray-900 font-mono">{stat.value}</p>
                          <p className="text-xs text-gray-500">{stat.label}</p>
                        </div>
                      ))}
                    </div>
                  </Link>
                </motion.div>
              )
            })}
          </motion.div>
        </div>
      </section>

      {/* Global Mission Banner */}
      <section className="bg-gradient-to-br from-gray-900 to-gray-800 text-white py-24 px-8">
        <div className="max-w-4xl mx-auto text-center">
          <Globe className="w-12 h-12 mx-auto text-blue-400" />
          <h2 className="mt-6 text-4xl font-bold">8 Billion Learners by 2075</h2>
          <p className="mt-4 text-xl text-gray-300">
            Every product funds the mission. Every learner matters.
            UN Sustainable Development Goal 4 — Quality Education for All.
          </p>
          <div className="mt-8 flex items-center justify-center gap-4">
            <div className="flex-1 max-w-md h-3 bg-white/20 rounded-full overflow-hidden">
              <div className="w-[2%] h-full bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full" />
            </div>
            <span className="text-blue-300 text-sm font-mono">2% of goal</span>
          </div>
          <Link
            href="/coalition"
            className="mt-10 inline-flex items-center gap-2 px-8 py-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-500 transition"
          >
            Join the $3B Coalition
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>

      {/* Footer CTA */}
      <section className="py-16 px-8 bg-blue-50">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-gray-900">Ready to start learning?</h2>
          <p className="mt-4 text-gray-600">
            Kelly is waiting. Your first lesson is free, forever.
          </p>
          <Link
            href="/"
            className="mt-8 inline-flex items-center gap-2 px-8 py-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-500 transition"
          >
            Meet Kelly
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>
    </div>
  )
}
