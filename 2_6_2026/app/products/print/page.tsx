'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Globe, Sun, Leaf, Heart } from 'lucide-react'

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

const PRINT_PRODUCTS = [
  {
    id: 'daily',
    name: 'Daily Egg',
    description: 'One lesson, one egg',
    price: 2.50,
    icon: '🥚',
    color: 'bg-amber-50 border-amber-200',
  },
  {
    id: 'week',
    name: 'Week Pack',
    description: '7 lessons bundled',
    price: 15,
    icon: '🥚🥚🥚',
    color: 'bg-orange-50 border-orange-200',
  },
  {
    id: 'month',
    name: 'Month Box',
    description: '30 lessons, beautifully boxed',
    price: 50,
    icon: '📦',
    color: 'bg-rose-50 border-rose-200',
  },
  {
    id: 'year',
    name: 'Year Set',
    description: 'All 365 lessons, collector edition',
    price: 500,
    icon: '🎁',
    color: 'bg-amber-100 border-amber-400',
    featured: true,
  },
]

const IMPACT_STATS = [
  { value: '3B', label: 'People without internet', icon: Globe },
  { value: '0', label: 'Electricity required', icon: Sun },
  { value: '100%', label: 'Biodegradable materials', icon: Leaf },
  { value: '365', label: 'Days of wonder', icon: Heart },
]

export default function PrintProductsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-amber-50 to-white">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-amber-50/90 backdrop-blur-xl border-b border-amber-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link 
              href="/products" 
              className="p-2 rounded-lg hover:bg-amber-100 transition"
            >
              <ArrowLeft className="w-5 h-5 text-amber-700" />
            </Link>
            <div>
              <h1 className="text-xl font-semibold text-amber-900">Print Products</h1>
              <p className="text-xs text-amber-600">Learning in your hands</p>
            </div>
          </div>
          <Link
            href="/coalition"
            className="px-4 py-2 bg-amber-600 text-white text-sm font-medium rounded-lg hover:bg-amber-500 transition"
          >
            Fund Print Distribution
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-24 lg:py-32 px-8">
        <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
          
          {/* Egg Visualization */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative"
          >
            <div className="relative w-80 h-80 mx-auto">
              {/* Main open egg */}
              <motion.div 
                className="absolute inset-0 flex items-center justify-center"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              >
                <div className="w-48 h-56 bg-gradient-to-b from-amber-200 to-amber-300 rounded-[50%] shadow-xl relative">
                  {/* Cracked top */}
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-32 h-16 bg-gradient-to-b from-amber-100 to-amber-200 rounded-t-full border-b-4 border-amber-300 border-dashed" />
                  {/* Lesson inside */}
                  <div className="absolute top-8 left-1/2 -translate-x-1/2 w-20 h-28 bg-white rounded shadow-inner flex items-center justify-center">
                    <span className="text-xs text-amber-800 font-medium">Day 1</span>
                  </div>
                </div>
              </motion.div>
              
              {/* Floating closed eggs */}
              <motion.div 
                className="absolute top-0 left-0 w-12 h-14 bg-gradient-to-b from-blue-200 to-blue-300 rounded-[50%] shadow-lg"
                animate={{ y: [0, -15, 0], x: [0, 5, 0] }}
                transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
              />
              <motion.div 
                className="absolute top-8 right-4 w-10 h-12 bg-gradient-to-b from-pink-200 to-pink-300 rounded-[50%] shadow-lg"
                animate={{ y: [0, -12, 0], x: [0, -5, 0] }}
                transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
              />
              <motion.div 
                className="absolute bottom-8 left-8 w-11 h-13 bg-gradient-to-b from-green-200 to-green-300 rounded-[50%] shadow-lg"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 5.5, repeat: Infinity, ease: "easeInOut", delay: 0.3 }}
              />
              <motion.div 
                className="absolute bottom-0 right-12 w-9 h-11 bg-gradient-to-b from-yellow-200 to-yellow-300 rounded-[50%] shadow-lg"
                animate={{ y: [0, -8, 0], x: [0, 3, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 0.8 }}
              />
            </div>
          </motion.div>
          
          {/* Content */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <p className="text-amber-600 font-semibold">For the 3 billion without internet</p>
            <h1 className="mt-4 text-5xl font-bold text-amber-900 tracking-tight">
              Learning in your hands
            </h1>
            <p className="mt-6 text-xl text-amber-700 leading-relaxed">
              Each egg contains one day&apos;s lesson. Collect all 365. 
              No electricity required. No internet needed. Just wonder.
            </p>
            
            {/* Price grid */}
            <motion.div
              variants={container}
              initial="hidden"
              animate="show"
              className="mt-8 grid grid-cols-2 gap-4"
            >
              {PRINT_PRODUCTS.map((product) => (
                <motion.div
                  key={product.id}
                  variants={item}
                  className={`rounded-xl p-4 border-2 ${product.color} ${product.featured ? 'ring-2 ring-amber-400' : ''}`}
                >
                  <span className="text-3xl">{product.icon}</span>
                  <p className="mt-2 font-semibold text-amber-900">{product.name}</p>
                  <p className="text-2xl font-bold text-amber-800">${product.price}</p>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Impact Stats */}
      <section className="py-16 px-8 bg-amber-100">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {IMPACT_STATS.map((stat) => {
              const Icon = stat.icon
              return (
                <div key={stat.label} className="text-center">
                  <Icon className="w-8 h-8 mx-auto text-amber-600" />
                  <p className="mt-4 text-4xl font-bold text-amber-900 font-mono">{stat.value}</p>
                  <p className="mt-1 text-sm text-amber-700">{stat.label}</p>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-amber-900">How It Works</h2>
          <p className="mt-4 text-xl text-amber-700">Education without boundaries</p>
          
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="mt-16 grid md:grid-cols-3 gap-8"
          >
            {[
              {
                step: '1',
                title: 'Open Your Egg',
                description: 'Each biodegradable egg contains a single lesson card with today\'s knowledge.',
              },
              {
                step: '2',
                title: 'Read & Learn',
                description: 'The lesson is designed for all ages, with pictures and simple explanations.',
              },
              {
                step: '3',
                title: 'Share & Discuss',
                description: 'Pass the card to a friend, discuss with family, or add to your collection.',
              },
            ].map((step) => (
              <motion.div key={step.step} variants={item} className="text-center">
                <div className="w-12 h-12 mx-auto bg-amber-200 rounded-full flex items-center justify-center text-amber-800 font-bold text-xl">
                  {step.step}
                </div>
                <h3 className="mt-4 text-xl font-bold text-amber-900">{step.title}</h3>
                <p className="mt-2 text-amber-700">{step.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Distribution Partners */}
      <section className="py-24 px-8 bg-amber-50">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-amber-900">Distribution Partners</h2>
          <p className="mt-4 text-xl text-amber-700">
            We partner with NGOs, schools, and community organizations to reach learners everywhere.
          </p>
          
          <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-6">
            {['Schools', 'Libraries', 'Community Centers', 'NGOs'].map((partner) => (
              <div key={partner} className="bg-white rounded-xl p-6 border border-amber-200">
                <p className="font-semibold text-amber-800">{partner}</p>
              </div>
            ))}
          </div>
          
          <Link
            href="/coalition"
            className="mt-12 inline-flex items-center gap-2 px-8 py-4 bg-amber-600 text-white font-semibold rounded-xl hover:bg-amber-500 transition"
          >
            Become a Distribution Partner
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 px-8 bg-gradient-to-br from-amber-600 to-orange-600 text-white">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold">Education for Everyone</h2>
          <p className="mt-4 text-xl text-amber-100">
            Help us reach the 3 billion people without internet access. Every egg is a spark of curiosity.
          </p>
          <Link
            href="/coalition"
            className="mt-10 inline-flex items-center gap-2 px-8 py-4 bg-white text-amber-600 font-semibold rounded-xl hover:bg-amber-50 transition"
          >
            Join the $3B Coalition
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>
    </div>
  )
}
