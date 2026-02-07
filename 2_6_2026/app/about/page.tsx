'use client'

/**
 * /about - Premium about page with strategic overview
 * Old money branding - black/white/gold aesthetic
 */

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { ArrowRight, BookOpen, Globe, Users, TrendingUp, Building2 } from 'lucide-react'

export default function AboutPage() {
  const router = useRouter()
  
  // Optional: Redirect to main app with about overlay
  // useEffect(() => {
  //   router.replace('/?view=about')
  // }, [router])
  
  const fadeInUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  }

  return (
    <div className="bg-white text-black min-h-screen">
      {/* UNIFIED NAV - Consistent with Kelly app (light mode) */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-md border-b border-black/5">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3 group">
            <span className="text-black/50 group-hover:text-black transition-colors text-sm">
              &larr; Back to Kelly
            </span>
            <span className="text-black/20">|</span>
            <span className="font-serif text-black">The Daily Lesson</span>
          </Link>
          <div className="flex items-center gap-6 text-sm">
            <Link href="/about" className="text-black font-medium">About</Link>
            <Link href="/strategic" className="text-black/50 hover:text-black transition-colors">Strategic</Link>
            <Link href="/join" className="text-black/50 hover:text-black transition-colors">Investors</Link>
            <span className="flex items-center gap-1.5 px-2 py-0.5 bg-amber-500/10 border border-amber-500/20 rounded-full text-[10px] text-amber-600 font-medium uppercase">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
              Alpha
            </span>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-4xl mx-auto">
          <motion.div variants={fadeInUp} initial="initial" animate="animate" className="space-y-6">
            <h1 className="text-6xl md:text-7xl font-black tracking-tight leading-tight">
              About Us
            </h1>
            <p className="text-xl text-gray-600 max-w-2xl">
              Lesson of the Day, PBC is a California Public Benefit Corporation advancing UN Sustainable Development Goal 4: Quality Education for All. Currently in private alpha—launching publicly Q2 2026.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Mission */}
      <section className="py-20 px-6 bg-black text-white">
        <div className="max-w-4xl mx-auto">
          <motion.div variants={fadeInUp} initial="initial" whileInView="animate" viewport={{ once: true }} className="space-y-6">
            <div>
              <h2 className="text-4xl font-black mb-4">Our Mission</h2>
              <p className="text-2xl font-light leading-relaxed">
                To ensure inclusive and equitable quality education, promoting lifelong learning opportunities for every person on Earth.
              </p>
            </div>
            <div className="border-l-4 border-white pl-6 py-4">
              <p className="text-white/80">
                We believe technology is a force multiplier for human potential. By combining AI-powered education with strategic infrastructure, we can scale quality learning to billions of people simultaneously—regardless of geography, language, or economic status.
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* The Founder */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <motion.div variants={fadeInUp} initial="initial" whileInView="animate" viewport={{ once: true }} className="space-y-8">
            <h2 className="text-4xl font-black">Founded by Nicolette Rankin</h2>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-4">
                <p className="text-lg text-gray-700">
                  20 years of educational technology experience. Previously co-founded OpenEnglish, one of the world's largest online language learning platforms.
                </p>
                <div className="space-y-2 text-sm">
                  <p><strong>At OpenEnglish:</strong></p>
                  <p className="text-gray-600">• $171M annual revenue at scale</p>
                  <p className="text-gray-600">• 2M+ students served globally</p>
                  <p className="text-gray-600">• 18 countries with operations</p>
                </div>
              </div>
              
              <div className="bg-black/2 border border-black/10 p-8 rounded-lg">
                <h3 className="font-bold mb-4">Curious Kelly Development</h3>
                <p className="text-3xl font-black mb-2">20,000 Hours</p>
                <p className="text-sm text-gray-600">
                  Equivalent to 10 years of full-time development work. Includes curriculum design, AI avatar development, lip-sync technology, and multi-language localization.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Who is Curious Kelly */}
      <section className="py-20 px-6 bg-black/2">
        <div className="max-w-4xl mx-auto">
          <motion.div variants={fadeInUp} initial="initial" whileInView="animate" viewport={{ once: true }} className="space-y-8">
            <h2 className="text-4xl font-black">Who is Curious Kelly?</h2>
            
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div className="space-y-4">
                <p className="text-lg text-gray-700">
                  Kelly is an AI-powered educator designed to spark wonder and cultivate curiosity in learners of all ages. She adapts her teaching style, vocabulary, and examples to match each student's age and learning archetype.
                </p>
                <p className="text-gray-600">
                  Built on 20,000+ hours of educational content development, Kelly delivers consistent, high-quality instruction whether you're 4 or 94. She speaks 6 languages fluently and adjusts her approach based on how you learn best.
                </p>
                <div className="flex flex-wrap gap-2 pt-4">
                  <span className="px-3 py-1 bg-black/5 rounded-full text-sm">5 Age Groups</span>
                  <span className="px-3 py-1 bg-black/5 rounded-full text-sm">10 Archetypes</span>
                  <span className="px-3 py-1 bg-black/5 rounded-full text-sm">6 Languages</span>
                </div>
              </div>
              <div className="bg-gradient-to-br from-amber-50 to-orange-50 p-8 rounded-2xl border border-amber-200/50 text-center">
                <div className="text-6xl mb-4">🎓</div>
                <p className="text-xl font-serif text-amber-900">"Every question deserves a thoughtful answer."</p>
                <p className="text-sm text-amber-700 mt-2">— Kelly's Philosophy</p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* The 365-Day Journey */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <motion.div variants={fadeInUp} initial="initial" whileInView="animate" viewport={{ once: true }} className="space-y-8">
            <h2 className="text-4xl font-black">The 365-Day Journey</h2>
            
            <p className="text-lg text-gray-700">
              Each day brings a new lesson with 5 phases designed for deep learning and lasting retention.
            </p>
            
            <div className="grid grid-cols-5 gap-4">
              <div className="text-center p-4 border border-blue-200 bg-blue-50/50 rounded-lg">
                <div className="w-3 h-3 rounded-full bg-blue-500 mx-auto mb-3"></div>
                <h4 className="font-bold text-sm mb-1">Hook</h4>
                <p className="text-xs text-gray-600">Hook your curiosity with a surprising fact</p>
              </div>
              <div className="text-center p-4 border border-violet-200 bg-violet-50/50 rounded-lg">
                <div className="w-3 h-3 rounded-full bg-violet-500 mx-auto mb-3"></div>
                <h4 className="font-bold text-sm mb-1">Story</h4>
                <p className="text-xs text-gray-600">Learn through narrative and context</p>
              </div>
              <div className="text-center p-4 border border-amber-200 bg-amber-50/50 rounded-lg">
                <div className="w-3 h-3 rounded-full bg-amber-500 mx-auto mb-3"></div>
                <h4 className="font-bold text-sm mb-1">Wonder</h4>
                <p className="text-xs text-gray-600">Explore deeper questions and connections</p>
              </div>
              <div className="text-center p-4 border border-emerald-200 bg-emerald-50/50 rounded-lg">
                <div className="w-3 h-3 rounded-full bg-emerald-500 mx-auto mb-3"></div>
                <h4 className="font-bold text-sm mb-1">Action</h4>
                <p className="text-xs text-gray-600">Apply what you've learned hands-on</p>
              </div>
              <div className="text-center p-4 border border-rose-200 bg-rose-50/50 rounded-lg">
                <div className="w-3 h-3 rounded-full bg-rose-500 mx-auto mb-3"></div>
                <h4 className="font-bold text-sm mb-1">Wisdom</h4>
                <p className="text-xs text-gray-600">Reflect and solidify understanding</p>
              </div>
            </div>
            
            <div className="bg-black/5 p-6 rounded-xl">
              <p className="text-sm text-gray-600">
                <strong>Why 365 days?</strong> Consistent daily learning creates lasting habits. Research shows that spaced repetition and daily engagement lead to 4x better retention than intensive cramming sessions.
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* The Science of Learning */}
      <section className="py-20 px-6 bg-black text-white">
        <div className="max-w-4xl mx-auto">
          <motion.div variants={fadeInUp} initial="initial" whileInView="animate" viewport={{ once: true }} className="space-y-8">
            <h2 className="text-4xl font-black">The Science of Learning</h2>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div>
                  <h3 className="font-bold text-lg mb-2">Spaced Repetition</h3>
                  <p className="text-white/70 text-sm">Daily lessons leverage the spacing effect—information reviewed at increasing intervals is retained longer than material studied all at once.</p>
                </div>
                <div>
                  <h3 className="font-bold text-lg mb-2">Active Recall</h3>
                  <p className="text-white/70 text-sm">Our True/False hooks and reflection prompts force active retrieval, strengthening neural pathways associated with each concept.</p>
                </div>
                <div>
                  <h3 className="font-bold text-lg mb-2">Multimodal Learning</h3>
                  <p className="text-white/70 text-sm">Video, audio, and interactive elements engage multiple senses, creating richer memory encoding.</p>
                </div>
              </div>
              <div className="space-y-6">
                <div>
                  <h3 className="font-bold text-lg mb-2">Age-Appropriate Scaffolding</h3>
                  <p className="text-white/70 text-sm">Kelly adjusts complexity, vocabulary, and examples to match developmental stages—from concrete thinking in young learners to abstract reasoning in adults.</p>
                </div>
                <div>
                  <h3 className="font-bold text-lg mb-2">Curiosity-Driven Engagement</h3>
                  <p className="text-white/70 text-sm">Beginning with hooks that challenge assumptions triggers dopamine release, making learning intrinsically rewarding.</p>
                </div>
                <div>
                  <h3 className="font-bold text-lg mb-2">Social Learning</h3>
                  <p className="text-white/70 text-sm">Community voting and shared results tap into social learning theory—we learn by observing how others think.</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Core Values */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <motion.div variants={fadeInUp} initial="initial" whileInView="animate" viewport={{ once: true }} className="space-y-8">
            <h2 className="text-4xl font-black mb-8">Core Values</h2>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="p-6 border border-black/10 rounded-lg hover:border-black/30 transition-all">
                <h3 className="text-xl font-bold mb-3">Accessibility</h3>
                <p className="text-gray-600 text-sm">
                  Quality education should never depend on geography, wealth, or circumstance. Technology makes this possible at scale.
                </p>
              </div>
              <div className="p-6 border border-black/10 rounded-lg hover:border-black/30 transition-all">
                <h3 className="text-xl font-bold mb-3">Impact</h3>
                <p className="text-gray-600 text-sm">
                  As a Public Benefit Corporation, we measure success by social impact alongside financial returns. Education changes lives.
                </p>
              </div>
              <div className="p-6 border border-black/10 rounded-lg hover:border-black/30 transition-all">
                <h3 className="text-xl font-bold mb-3">Quality</h3>
                <p className="text-gray-600 text-sm">
                  Every learner deserves the same quality instruction. Our AI delivers consistent, universal education at any scale.
                </p>
              </div>
              <div className="p-6 border border-black/10 rounded-lg hover:border-black/30 transition-all">
                <h3 className="text-xl font-bold mb-3">Innovation</h3>
                <p className="text-gray-600 text-sm">
                  We combine cutting-edge AI with strategic infrastructure to create unprecedented opportunities for human learning.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Products & Services */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <motion.div variants={fadeInUp} initial="initial" whileInView="animate" viewport={{ once: true }} className="space-y-8">
            <h2 className="text-4xl font-black mb-8">What We Offer</h2>
            
            <div className="grid md:grid-cols-3 gap-6">
              <Link href="/" className="group p-6 border border-black/10 rounded-lg hover:border-black/30 transition-all hover:shadow-lg">
                <BookOpen className="w-8 h-8 mb-4 group-hover:translate-y-1 transition-transform" />
                <h3 className="font-bold mb-2">Curious Kelly</h3>
                <p className="text-sm text-gray-600 mb-4">
                  AI-powered daily lessons for learners ages 2-102
                </p>
                <p className="text-xs font-semibold text-black/60">curiouskelly.com</p>
              </Link>
              
              <Link href="/pricing" className="group p-6 border border-black/10 rounded-lg hover:border-black/30 transition-all hover:shadow-lg">
                <Users className="w-8 h-8 mb-4 group-hover:translate-y-1 transition-transform" />
                <h3 className="font-bold mb-2">Plans</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Individual subscriptions and family plans
                </p>
                <p className="text-xs font-semibold text-black/60">Flexible pricing</p>
              </Link>
              
              <Link href="/ziggurat/investors" className="group p-6 border border-black/10 rounded-lg hover:border-black/30 transition-all hover:shadow-lg">
                <Building2 className="w-8 h-8 mb-4 group-hover:translate-y-1 transition-transform" />
                <h3 className="font-bold mb-2">The Ziggurat Vision</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Exploring community-driven adaptive reuse with intentional partnerships
                </p>
                <p className="text-xs font-semibold text-black/60">Early exploration phase</p>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats */}
      <section className="py-20 px-6 bg-black text-white">
        <div className="max-w-4xl mx-auto">
          <motion.div variants={fadeInUp} initial="initial" whileInView="animate" viewport={{ once: true }} className="grid md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-4xl font-black mb-2">365</div>
              <p className="text-sm text-white/60">Daily Lessons</p>
            </div>
            <div className="text-center">
              <div className="text-4xl font-black mb-2">5,110</div>
              <p className="text-sm text-white/60">Unique Moments/Year</p>
            </div>
            <div className="text-center">
              <div className="text-4xl font-black mb-2">100</div>
              <p className="text-sm text-white/60">Age Range (2-102)</p>
            </div>
            <div className="text-center">
              <div className="text-4xl font-black mb-2">6+</div>
              <p className="text-sm text-white/60">Languages</p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div variants={fadeInUp} initial="initial" whileInView="animate" viewport={{ once: true }} className="space-y-6">
            <h2 className="text-4xl font-black">Ready to Transform Education?</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Join millions of learners experiencing the future of quality education.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Link href="/" className="px-8 py-3 bg-black text-white font-semibold rounded hover:bg-black/80 transition-colors inline-flex items-center justify-center gap-2">
                Experience Kelly <ArrowRight className="w-4 h-4" />
              </Link>
              <Link href="/strategic" className="px-8 py-3 border-2 border-black text-black font-semibold rounded hover:bg-black/5 transition-colors inline-flex items-center justify-center gap-2">
                Learn More <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-black/10 bg-black/2 py-12 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-gray-600">© 2026 Lesson of the Day, PBC</p>
          <p className="text-sm text-gray-500 mt-2">Building the future of education through technology and impact.</p>
        </div>
      </footer>
    </div>
  )
}
