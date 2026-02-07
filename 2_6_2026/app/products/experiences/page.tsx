'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, MapPin, Calendar, Users, Clock, Check, Building2 } from 'lucide-react'

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

const EXPERIENCES = [
  { 
    id: 'day-pass',
    name: 'Day Pass', 
    price: '$25', 
    icon: '🎟️',
    description: 'Self-guided tour',
    features: ['Building access', 'Kelly demo stations', 'Gift shop'],
    color: 'from-blue-500 to-blue-600',
  },
  { 
    id: 'guided-tour',
    name: 'Guided Tour', 
    price: '$50', 
    icon: '🗺️',
    description: '2-hour docent experience',
    features: ['Expert guide', 'Behind-the-scenes', 'Q&A session'],
    color: 'from-indigo-500 to-indigo-600',
  },
  { 
    id: 'teacher-training',
    name: 'Teacher Training', 
    price: '$500', 
    icon: '🎓',
    description: 'Full-day certification',
    features: ['Kelly certification', 'Curriculum training', 'Materials included'],
    color: 'from-purple-500 to-purple-600',
    featured: true,
  },
  { 
    id: 'corporate-event',
    name: 'Corporate Event', 
    price: '$15,000+', 
    icon: '🏢',
    description: 'Private venue rental',
    features: ['Exclusive access', 'Custom catering', 'A/V included'],
    color: 'from-gray-700 to-gray-800',
  },
]

const BUILDING_STATS = [
  { value: '92', label: 'Acres' },
  { value: '1M', label: 'Square Feet' },
  { value: '7', label: 'Floors' },
  { value: '1971', label: 'Built' },
]

export default function ExperiencesPage() {
  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gray-900/90 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link 
              href="/products" 
              className="p-2 rounded-lg hover:bg-white/10 transition"
            >
              <ArrowLeft className="w-5 h-5 text-white/60" />
            </Link>
            <div>
              <h1 className="text-xl font-semibold text-white">Ziggurat Experiences</h1>
              <p className="text-xs text-white/40">Where education meets architecture</p>
            </div>
          </div>
          <Link
            href="/ziggurat"
            className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-500 transition"
          >
            Learn About the Building
          </Link>
        </div>
      </header>

      {/* Hero Section with Building Background */}
      <section className="relative min-h-[80vh] text-white overflow-hidden">
        {/* Background gradient (simulating building image) */}
        <div className="absolute inset-0 bg-gradient-to-b from-gray-800 via-gray-900 to-gray-900" />
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-5" />
        
        {/* Ziggurat silhouette visualization */}
        <div className="absolute bottom-0 left-0 right-0 h-64 flex items-end justify-center opacity-20">
          <svg viewBox="0 0 400 150" className="w-full max-w-2xl">
            {/* Stepped pyramid shape */}
            <polygon points="200,10 300,50 300,50 320,50 320,70 340,70 340,90 360,90 360,110 380,110 380,150 20,150 20,110 40,110 40,90 60,90 60,70 80,70 80,50 100,50 100,50" fill="currentColor" className="text-blue-500" />
          </svg>
        </div>
        
        {/* Gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-gray-900 via-gray-900/50 to-transparent" />
        
        {/* Content */}
        <div className="relative z-10 max-w-6xl mx-auto px-8 py-32">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <p className="text-blue-400 font-semibold flex items-center gap-2">
              <Building2 className="w-4 h-4" />
              Kelly&apos;s Home
            </p>
            <h1 className="mt-4 text-6xl lg:text-7xl font-bold tracking-tight">
              The Ziggurat
            </h1>
            <p className="mt-4 text-2xl text-gray-300">
              1 million square feet. 92 acres. Laguna Niguel, California.
            </p>
            <p className="mt-2 text-xl text-gray-400">
              Where education meets architecture.
            </p>
            
            {/* Stats */}
            <div className="mt-12 grid grid-cols-4 gap-8">
              {BUILDING_STATS.map((stat) => (
                <div key={stat.label}>
                  <p className="text-4xl font-bold text-blue-400 font-mono tabular-nums">{stat.value}</p>
                  <p className="text-gray-400">{stat.label}</p>
                </div>
              ))}
            </div>
            
            <div className="mt-12 flex gap-4">
              <a 
                href="#experiences"
                className="px-8 py-4 bg-blue-600 rounded-xl font-semibold hover:bg-blue-500 transition"
              >
                Plan Your Visit
              </a>
              <Link
                href="/ziggurat"
                className="px-8 py-4 bg-white/10 rounded-xl font-semibold hover:bg-white/20 transition"
              >
                Virtual Tour
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Location Info */}
      <section className="py-16 px-8 bg-gray-800 border-y border-white/10">
        <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-center gap-8 text-gray-300">
          <div className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-blue-400" />
            <span>24000 Avila Road, Laguna Niguel, CA 92677</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-400" />
            <span>Open Daily 9am - 6pm</span>
          </div>
          <div className="flex items-center gap-2">
            <Calendar className="w-5 h-5 text-blue-400" />
            <span>Opening December 2026</span>
          </div>
        </div>
      </section>

      {/* Experience Cards */}
      <section id="experiences" className="py-24 px-8 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900">Book Your Experience</h2>
            <p className="mt-4 text-xl text-gray-600">From day passes to corporate events</p>
          </div>

          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="grid md:grid-cols-2 lg:grid-cols-4 gap-6"
          >
            {EXPERIENCES.map((exp) => (
              <motion.div
                key={exp.id}
                variants={item}
                className={`bg-white rounded-2xl overflow-hidden border border-gray-200 hover:shadow-xl transition ${exp.featured ? 'ring-2 ring-blue-500' : ''}`}
              >
                {/* Ticket stub top */}
                <div className={`bg-gradient-to-br ${exp.color} p-6 text-white`}>
                  <span className="text-4xl">{exp.icon}</span>
                  <h3 className="mt-4 text-xl font-bold">{exp.name}</h3>
                  <p className="text-white/80">{exp.description}</p>
                </div>
                
                {/* Perforated edge effect */}
                <div className="flex justify-between px-2 bg-gray-50">
                  {[...Array(8)].map((_, i) => (
                    <div key={i} className="w-3 h-3 bg-white rounded-full -mt-1.5 border border-gray-200" />
                  ))}
                </div>
                
                {/* Details */}
                <div className="p-6 bg-gray-50">
                  <ul className="space-y-2">
                    {exp.features.map((f) => (
                      <li key={f} className="flex items-center gap-2 text-sm text-gray-600">
                        <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                        {f}
                      </li>
                    ))}
                  </ul>
                  
                  <div className="mt-6 flex items-center justify-between">
                    <span className="text-2xl font-bold text-gray-900">{exp.price}</span>
                    <button className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-500 transition">
                      Book
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Group Bookings */}
      <section className="py-24 px-8 bg-gray-50">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900">Group Bookings</h2>
            <p className="mt-4 text-xl text-gray-600">Special rates for schools, organizations, and large groups</p>
          </div>

          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="grid md:grid-cols-3 gap-8"
          >
            {[
              { title: 'School Field Trips', description: '20+ students, includes curriculum materials', price: '$15/student' },
              { title: 'Corporate Retreats', description: 'Team building with Kelly, catering available', price: 'From $5,000' },
              { title: 'Special Events', description: 'Weddings, galas, product launches', price: 'Custom quote' },
            ].map((group) => (
              <motion.div
                key={group.title}
                variants={item}
                className="bg-white rounded-2xl p-8 border border-gray-200"
              >
                <Users className="w-8 h-8 text-blue-600" />
                <h3 className="mt-4 text-xl font-bold text-gray-900">{group.title}</h3>
                <p className="mt-2 text-gray-600 text-sm">{group.description}</p>
                <p className="mt-4 text-lg font-bold text-blue-600">{group.price}</p>
                <button className="mt-4 w-full py-2 border-2 border-blue-600 text-blue-600 rounded-lg font-semibold hover:bg-blue-50 transition">
                  Inquire
                </button>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 px-8 bg-gradient-to-br from-blue-600 to-indigo-700 text-white">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold">Opening December 2026</h2>
          <p className="mt-4 text-xl text-blue-100">
            Join the coalition to help transform this historic building into the world&apos;s largest learning center.
          </p>
          <div className="mt-10 flex flex-wrap justify-center gap-4">
            <Link
              href="/coalition"
              className="inline-flex items-center gap-2 px-8 py-4 bg-white text-blue-600 font-semibold rounded-xl hover:bg-blue-50 transition"
            >
              Join the $3B Coalition
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/ziggurat"
              className="inline-flex items-center gap-2 px-8 py-4 bg-blue-700/50 text-white font-semibold rounded-xl hover:bg-blue-700 transition"
            >
              Explore the Vision
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
