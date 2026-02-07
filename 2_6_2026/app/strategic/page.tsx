// ═══════════════════════════════════════════════════════════════════════════
// STRATEGIC & INVESTORS — THE DAILY LESSON
// Unified investor relations and strategic vision page
// ═══════════════════════════════════════════════════════════════════════════

"use client"

import { useState, useRef } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import { ArrowUpRight } from "lucide-react"

const CONFIG = {
  videos: {
    // Ziggurat building hero - until GSA remix video is uploaded
    // TODO: Replace with actual GSA remix video when available
    desktop: "https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-storyteller.mp4",
    mobile: "https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-storyteller.mp4",
    poster: "/ziggurat/heroes/ocean-beacon-night.jpg"
  },
  contacts: [
    {
      name: "Nicolette Rankin",
      title: "Founder",
      company: "LESSON OF THE DAY, PBC",
      phone: "805-233-5084",
      emails: ["nicoletterankin@gmail.com", "nicolette@thedailylesson.com"],
    },
    {
      name: "Dallas Short",
      title: "Principal",
      company: "LESSON OF THE DAY, PBC",
      phone: "310-383-9041",
      emails: ["dallasrshort@gmail.com", "dallas@thedailylesson.com"],
      license: "CalDRE# 01906262"
    },
    {
      name: "Curious Kelly",
      title: "AI Learning Companion",
      company: "Say hello",
      emails: ["hello@curiouskelly.com"],
    }
  ],
  address: {
    street: "24000 Avila Road",
    city: "Laguna Niguel, California 92677"
  }
}

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.2 }
  }
}

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.8, ease: 'easeOut' } }
}

export default function StrategicPage() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(true)
  const [showOverlay, setShowOverlay] = useState(true)
  const videoRef = useRef<HTMLVideoElement>(null)

  const handlePlay = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0
      videoRef.current.muted = false
      videoRef.current.play()
      setIsPlaying(true)
      setIsMuted(false)
      setShowOverlay(false)
    }
  }

  const handleVideoEnd = () => {
    setShowOverlay(true)
    setIsPlaying(false)
    if (videoRef.current) {
      videoRef.current.muted = true
      setIsMuted(true)
    }
  }

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !videoRef.current.muted
      setIsMuted(!isMuted)
    }
  }

  return (
    <main className="bg-black text-white antialiased">
      
      {/* UNIFIED NAV - Consistent with Kelly app */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <a href="/" className="flex items-center gap-3 group">
            <span className="text-white/60 group-hover:text-white transition-colors text-sm">
              &larr; Back to Kelly
            </span>
            <span className="text-white/20">|</span>
            <span className="font-serif text-white">The Daily Lesson</span>
          </a>
          <div className="flex items-center gap-6 text-sm">
            <a href="/about" className="text-white/50 hover:text-white transition-colors">About</a>
            <a href="/strategic" className="text-white font-medium">Strategic</a>
            <a href="/join" className="text-white/50 hover:text-white transition-colors">Investors</a>
            <span className="flex items-center gap-1.5 px-2 py-0.5 bg-amber-500/10 border border-amber-500/20 rounded-full text-[10px] text-amber-400 font-medium uppercase">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
              Alpha
            </span>
          </div>
        </div>
      </nav>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* HERO SECTION - Video Background */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="relative h-screen flex items-center justify-center overflow-hidden">
        
        {/* Video Background */}
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover"
          style={{ opacity: 0.65 }}
          poster={CONFIG.videos.poster}
          playsInline
          muted
          loop
          autoPlay
          onEnded={handleVideoEnd}
        >
          <source src={CONFIG.videos.desktop} type="video/mp4" media="(min-width: 768px)" />
          <source src={CONFIG.videos.mobile} type="video/mp4" />
        </video>

        {/* Gradient Overlay */}
        <div 
          className="absolute inset-0 transition-opacity duration-700"
          style={{
            background: "linear-gradient(to bottom, rgba(0,0,0,0.35) 0%, rgba(0,0,0,0.1) 40%, rgba(0,0,0,0.1) 60%, rgba(0,0,0,0.95) 100%)",
            opacity: showOverlay ? 1 : 0.25
          }}
        />

        {/* Content Overlay */}
        <div 
          className={`relative z-10 text-center px-6 max-w-4xl transition-opacity duration-500 ${
            showOverlay ? "opacity-100" : "opacity-0 pointer-events-none"
          }`}
        >
          {/* Eyebrow */}
          <p 
            className="text-zinc-500 mb-6"
            style={{ 
              fontSize: "11px", 
              letterSpacing: "0.45em", 
              textTransform: "uppercase" 
            }}
          >
            Lesson of the Day, PBC
          </p>

          {/* Title */}
          <h1 
            className="text-white mb-4"
            style={{ 
              fontFamily: "Georgia, 'Times New Roman', serif",
              fontSize: "clamp(3rem, 10vw, 7rem)",
              fontWeight: 300,
              letterSpacing: "0.1em",
              lineHeight: 1.1
            }}
          >
            THE DAILY LESSON
          </h1>

          {/* Subtitle */}
          <p 
            className="text-zinc-400 mb-16"
            style={{ 
              fontFamily: "Georgia, 'Times New Roman', serif",
              fontSize: "clamp(1rem, 3vw, 1.35rem)",
              fontWeight: 300,
              letterSpacing: "0.05em"
            }}
          >
            California's First AI-Powered Teaching Campus
          </p>

          {/* Play Button */}
          <button
            onClick={handlePlay}
            className="group relative mx-auto flex items-center justify-center transition-all duration-300 hover:scale-[1.03]"
            style={{
              width: "clamp(100px, 15vw, 140px)",
              height: "clamp(100px, 15vw, 140px)",
              borderRadius: "50%",
              background: "transparent",
              border: "1px solid rgba(255,255,255,0.15)"
            }}
            aria-label="Play video"
          >
            <svg 
              className="text-white/70 group-hover:text-white transition-colors"
              style={{ 
                width: "clamp(40px, 6vw, 56px)", 
                height: "clamp(40px, 6vw, 56px)",
                marginLeft: "6px"
              }}
              fill="currentColor" 
              viewBox="0 0 24 24"
            >
              <path d="M8 5v14l11-7z" />
            </svg>
            
            {/* Pulse Ring */}
            <span 
              className="absolute inset-0 rounded-full border border-white/20 animate-pulse"
            />
          </button>

          {/* Tagline */}
          <p 
            className="text-zinc-600 mt-12"
            style={{ 
              fontSize: "clamp(0.8rem, 2vw, 1rem)", 
              letterSpacing: "0.2em" 
            }}
          >
            92 acres | Seven levels | One mission
          </p>

          {/* Three Foundations */}
          <div className="flex justify-center gap-16 md:gap-24 mt-16">
            {[
              { icon: "HARDWARE", label: "DEVICES" },
              { icon: "BROADCAST", label: "STREAMING" },
              { icon: "PRINT", label: "PHYSICAL" }
            ].map((pillar) => (
              <div key={pillar.label} className="text-center">
                <p 
                  className="text-zinc-600"
                  style={{ fontSize: "10px", letterSpacing: "0.25em" }}
                >
                  {pillar.icon}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Video Controls */}
        {isPlaying && (
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex gap-3 z-20">
            <button
              onClick={toggleMute}
              className="flex items-center justify-center transition-all hover:bg-white/10 w-12 h-12 rounded-full bg-black/50 backdrop-blur-md border border-white/10"
            >
              {isMuted ? (
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                </svg>
              )}
            </button>
            <button
              onClick={() => setShowOverlay(true)}
              className="flex items-center justify-center transition-all hover:bg-white/10 w-12 h-12 rounded-full bg-black/50 backdrop-blur-md border border-white/10"
            >
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
              </svg>
            </button>
          </div>
        )}

        {/* Scroll Indicator */}
        {showOverlay && (
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-bounce">
            <span className="text-zinc-700" style={{ fontSize: "10px", letterSpacing: "0.3em" }}>
              SCROLL
            </span>
            <svg className="w-4 h-4 text-zinc-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>
        )}
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* BUSINESS UNITS SECTION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="space-y-16"
          >
            <motion.div variants={item}>
              <p className="text-zinc-600 mb-4" style={{ fontSize: "10px", letterSpacing: "0.4em", textTransform: "uppercase" }}>
                Three Operating Divisions
              </p>
              <h2 
                className="text-white"
                style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: "clamp(2rem, 5vw, 3rem)", fontWeight: 300 }}
              >
                How We Teach
              </h2>
            </motion.div>

            {/* Business Units Grid - Floating Bubbles */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                {
                  title: "Hardware",
                  subtitle: "Devices & Infrastructure",
                  description: "Offline-capable devices powered by our AI learning engine. Tablets, learning pods, and edge computing infrastructure for the 3.7B without reliable internet.",
                  metrics: ["Low-bandwidth design", "Biometric assessment", "8-hour battery"]
                },
                {
                  title: "Broadcast",
                  subtitle: "Streaming & Linear",
                  description: "Live and on-demand video distribution via satellite, terrestrial broadcast, and CDN. Reaching classrooms, communities, and homes simultaneously.",
                  metrics: ["5-minute lessons", "20+ languages", "Global reach"]
                },
                {
                  title: "Print",
                  subtitle: "Physical & Tangible",
                  description: "Printed curriculum, workbooks, and materials for teachers and students. Designed to integrate seamlessly with digital learning and work offline.",
                  metrics: ["Color + B&W", "Durable design", "Teacher guides"]
                }
              ].map((unit) => (
                <motion.div 
                  key={unit.title}
                  variants={item}
                  className="group p-6 rounded-xl hover:scale-105 transition-transform duration-300"
                  style={{
                    backgroundColor: "rgba(255, 255, 255, 0.03)",
                    backdropFilter: "blur(0px)"
                  }}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="text-white text-xl font-light mb-0.5" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>
                        {unit.title}
                      </h3>
                      <p className="text-zinc-500 text-xs">{unit.subtitle}</p>
                    </div>
                    <div className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-0.5 flex-shrink-0" />
                  </div>
                  
                  <p className="text-zinc-400 text-sm leading-relaxed mb-4">
                    {unit.description}
                  </p>
                  
                  <div className="space-y-1.5 pt-3 border-t border-white/5">
                    {unit.metrics.map((metric) => (
                      <div key={metric} className="flex items-center gap-2">
                        <span className="w-0.5 h-0.5 rounded-full bg-zinc-700" />
                        <span className="text-xs text-zinc-500">{metric}</span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THE BUILDING SECTION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-32 px-6">
        <div className="max-w-3xl mx-auto">
          
          <p className="text-zinc-600 mb-8" style={{ fontSize: "10px", letterSpacing: "0.4em", textTransform: "uppercase" }}>
            The Building
          </p>

          <h2 
            className="text-white mb-12"
            style={{ 
              fontFamily: "Georgia, 'Times New Roman', serif",
              fontSize: "clamp(2.5rem, 6vw, 4rem)",
              fontWeight: 300,
              letterSpacing: "0.02em",
              lineHeight: 1.15
            }}
          >
            The Chet Holifield<br />Federal Building
          </h2>

          {/* Transparency Note */}
          <div className="mb-12 p-6 border-l-2 border-zinc-700 bg-white/[0.02]">
            <p className="text-xs uppercase tracking-widest text-zinc-500 mb-2">Transparency</p>
            <p className="text-zinc-400 text-sm leading-relaxed">
              The Ziggurat project is in early exploration phase. We are NOT currently under contract, 
              and we are exploring this opportunity with community involvement and partnership. 
              This represents an aspirational vision, not a committed project.
            </p>
          </div>

          {/* Body */}
          <div className="space-y-6 text-lg leading-relaxed text-zinc-400">
            <p>
              Ninety-two acres in Laguna Niguel, California. One million square feet 
              designed by <span className="text-white">William Pereira</span> — the architect 
              of the Transamerica Pyramid and LACMA.
            </p>
            <p>
              For fifty years, it processed paperwork. Then it went dark. 
              Three failed auctions. No one could see what it was built for.
            </p>
            <p className="text-white">We see it.</p>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mt-20 pt-16 border-t border-white/10">
            {[
              { value: "92", label: "ACRES" },
              { value: "1M", label: "SQ FT" },
              { value: "7", label: "LEVELS" },
              { value: "1971", label: "BUILT" }
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <div 
                  className="text-white"
                  style={{ 
                    fontFamily: "Georgia, 'Times New Roman', serif",
                    fontSize: "clamp(3rem, 8vw, 4.5rem)",
                    fontWeight: 300,
                    letterSpacing: "-0.02em",
                    lineHeight: 1
                  }}
                >
                  {stat.value}
                </div>
                <div className="text-zinc-600 mt-3" style={{ fontSize: "10px", letterSpacing: "0.25em" }}>
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* FINANCIAL MODEL SECTION */}
      {/* ═══════════════════════════════════════════════════���═══════════════ */}
      
      <section className="py-24 px-6 bg-zinc-950 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="space-y-12"
          >
            <motion.div variants={item}>
              <p className="text-zinc-600 mb-4" style={{ fontSize: "10px", letterSpacing: "0.4em", textTransform: "uppercase" }}>
                Open Financial Model
              </p>
              <h2 
                className="text-white flex items-center gap-4"
                style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: "clamp(2rem, 5vw, 3rem)", fontWeight: 300 }}
              >
                The Ziggurat <span className="text-zinc-600">×</span> Curious Kelly
              </h2>
            </motion.div>

            {/* Key Metrics */}
            <motion.div variants={item} className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: "Square Feet", value: "1,003,041", color: "text-white" },
                { label: "Floors", value: "7", color: "text-white" },
                { label: "Jobs Created (Y5)", value: "780", color: "text-white" },
                { label: "Daily Learners (Y5)", value: "1,500,000", color: "text-amber-400" }
              ].map((metric) => (
                <div key={metric.label} className="bg-white/5 border border-white/10 p-6">
                  <p className={`text-2xl md:text-3xl font-light mb-1 ${metric.color}`} style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>
                    {metric.value}
                  </p>
                  <p className="text-xs text-zinc-500 uppercase tracking-wider">{metric.label}</p>
                </div>
              ))}
            </motion.div>

            {/* Vision Statement */}
            <motion.div variants={item} className="bg-white/[0.02] border border-white/10 p-8">
              <p className="text-xs uppercase tracking-widest text-amber-500 mb-4">The Vision</p>
              <h3 className="text-xl md:text-2xl text-white mb-4 font-light" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>
                A Temple of Learning
              </h3>
              <p className="text-zinc-400 leading-relaxed">
                Transform a 1971 brutalist masterpiece into the global headquarters for Curious Kelly, 
                serving 3.7 billion learners without reliable internet through our offline-first educational platform.
              </p>
            </motion.div>

            {/* Building Timeline */}
            <motion.div variants={item}>
              <div className="flex items-center gap-2 mb-4">
                <svg className="w-4 h-4 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-xs uppercase tracking-widest text-zinc-500">Building Timeline</span>
              </div>
              <div className="space-y-3">
                {[
                  { year: "2015", event: "Determined NRHP-eligible", highlight: false },
                  { year: "2023", event: "First auction - no bids", highlight: false },
                  { year: "2024", event: "Second auction - $177M bid cancelled", highlight: false },
                  { year: "2025", event: "February: IRS vacates, building now vacant", highlight: false },
                  { year: "2025", event: "Third auction - GSA reviewing offers", highlight: false },
                  { year: "2026", event: "Coalition offer submitted, awaiting GSA response", highlight: true }
                ].map((item, idx) => (
                  <div key={idx} className="flex gap-4 items-center">
                    <span className={`text-sm font-mono ${item.highlight ? 'text-emerald-400' : 'text-amber-500'}`}>{item.year}</span>
                    <span className={`text-sm ${item.highlight ? 'text-emerald-400' : 'text-zinc-400'}`}>{item.event}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* CTA */}
            <motion.div variants={item} className="flex flex-col sm:flex-row gap-4">
              <Link 
                href="/ziggurat/spreadsheet"
                className="inline-flex items-center justify-center gap-3 px-8 py-4 text-sm text-white bg-white/10 hover:bg-white/15 border border-white/20 transition-all"
              >
                Full Model
                <ArrowUpRight className="w-4 h-4" />
              </Link>
              <Link 
                href="/ziggurat/investors"
                className="inline-flex items-center justify-center gap-3 px-8 py-4 text-sm text-black bg-white hover:bg-white/90 transition-all"
              >
                Investor Workspaces
                <ArrowUpRight className="w-4 h-4" />
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* TIMELINE SECTION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-24 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="space-y-16"
          >
            <motion.div variants={item}>
              <p className="text-zinc-600 mb-4" style={{ fontSize: "10px", letterSpacing: "0.4em", textTransform: "uppercase" }}>
                The Timeline
              </p>
              <h2 
                className="text-white"
                style={{ 
                  fontFamily: "Georgia, 'Times New Roman', serif",
                  fontSize: "clamp(2rem, 5vw, 3rem)",
                  fontWeight: 300
                }}
              >
                Five-Year Journey
              </h2>
            </motion.div>

            <div className="space-y-8">
              {[
                { year: '2015', event: 'Foundation', desc: 'OpenEnglish exits — team reflects on scale, impact, sustainability' },
                { year: '2023', event: 'Genesis', desc: 'Curious Kelly concept: personalized video content for all learners' },
                { year: '2024', event: 'Private Alpha', desc: '365 daily lessons across 5 phases, multiple ages, languages launching' },
                { year: '2025', event: 'Ziggurat Campaign', desc: 'Building identified, coalition formed, GSA engagement begins' },
                { year: '2026', event: 'Coalition Launch', desc: 'Q1 — Offer submitted, awaiting GSA response. Platform growing daily.' }
              ].map((milestone, idx) => (
                <motion.div 
                  key={idx} 
                  variants={item} 
                  className="flex gap-8 pb-8 border-b border-white/10 last:border-b-0"
                >
                  <div className="w-24 flex-shrink-0">
                    <p 
                      className="text-white"
                      style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: "1.5rem" }}
                    >
                      {milestone.year}
                    </p>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-light mb-2 text-white">{milestone.event}</h3>
                    <p className="text-zinc-500 text-sm">{milestone.desc}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* COALITION SECTION - Victory of the People */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-24 px-6 border-t border-white/5">
        <div className="max-w-5xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="space-y-12"
          >
            {/* Header */}
            <motion.div variants={item} className="text-center">
              <p className="text-blue-400 text-sm tracking-widest mb-4">COALITION MODEL</p>
              <h2 
                className="text-white mb-4"
                style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: "clamp(2rem, 5vw, 3rem)", fontWeight: 300 }}
              >
                Victory of the People
              </h2>
              <p className="text-zinc-400 max-w-2xl mx-auto">
                No one has committed yet. Everyone sees this together for the first time.
                Not investors. Co-builders.
              </p>
              <div className="flex items-center justify-center gap-4 text-sm mt-6">
                <span className="text-zinc-500">January 29, 2026</span>
                <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded-full">
                  Offer Submitted
                </span>
              </div>
            </motion.div>

            {/* Coalition Table */}
            <motion.div variants={item} className="overflow-x-auto bg-zinc-900/30 rounded-lg p-6">
              <h3 className="text-lg text-white mb-4">You're Invited to the Table</h3>
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="py-3 pr-4 text-zinc-500 font-normal">Partner</th>
                    <th className="py-3 pr-4 text-zinc-500 font-normal text-right">Amount</th>
                    <th className="py-3 pr-4 text-zinc-500 font-normal">Type</th>
                    <th className="py-3 text-zinc-500 font-normal text-right">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { partner: "Emerson Collective", amount: "$50M", type: "PRI", status: "Invited" },
                    { partner: "Chan Zuckerberg Initiative", amount: "$50M", type: "PRI", status: "Invited" },
                    { partner: "Gates Foundation", amount: "$40M", type: "Grant", status: "Invited" },
                    { partner: "Donald Bren / Irvine Co", amount: "$30M", type: "Mission Debt", status: "Invited" },
                    { partner: "State of California", amount: "$25M", type: "Grant", status: "Invited" },
                    { partner: "Historic Tax Credits", amount: "$20M", type: "Tax Credit", status: "Eligible" },
                    { partner: "FullSail / Education Partners", amount: "$20M", type: "PRI", status: "Invited" },
                    { partner: "Legoland / Merlin", amount: "$15M", type: "Partnership", status: "Invited" },
                    { partner: "CDFI / NMTC", amount: "$25M", type: "Debt", status: "Available" },
                    { partner: "Federal Education Grants", amount: "$15M", type: "Grant", status: "To Apply" },
                  ].map((row, idx) => (
                    <tr key={idx} className="border-b border-white/5 hover:bg-white/[0.02] transition-colors">
                      <td className="py-3 pr-4 text-white">{row.partner}</td>
                      <td className="py-3 pr-4 text-zinc-300 text-right">{row.amount}</td>
                      <td className="py-3 pr-4 text-zinc-500">{row.type}</td>
                      <td className="py-3 text-right">
                        <span className={`px-2 py-0.5 text-xs border rounded-full ${
                          row.status === 'Invited' ? 'border-blue-500 text-blue-400' :
                          row.status === 'Eligible' || row.status === 'Available' ? 'border-green-500 text-green-400' :
                          'border-zinc-500 text-zinc-400'
                        }`}>
                          {row.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr className="border-t border-zinc-700">
                    <td className="py-4 text-white font-medium">Coalition Total</td>
                    <td className="py-4 text-right text-blue-400 font-medium">$290M</td>
                    <td colSpan={2}></td>
                  </tr>
                </tfoot>
              </table>
            </motion.div>

            {/* Phase 1 Capital Needs */}
            <motion.div variants={item} className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/[0.02] border border-white/10 p-6">
                <p className="text-xs uppercase tracking-widest text-blue-400 mb-4">Phase 1 Capital Required</p>
                <p className="text-3xl font-light text-white mb-4" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>$309M</p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between py-2 border-b border-zinc-800/50"><span className="text-zinc-400">Pre-Auction Purchase</span><span className="text-white">$200M</span></div>
                  <div className="flex justify-between py-2 border-b border-zinc-800/50"><span className="text-zinc-400">Closing & Due Diligence</span><span className="text-white">$4M</span></div>
                  <div className="flex justify-between py-2 border-b border-zinc-800/50"><span className="text-zinc-400">Phase 1 Renovation</span><span className="text-white">$75M</span></div>
                  <div className="flex justify-between py-2"><span className="text-zinc-400">Operating Reserve (18mo)</span><span className="text-white">$30M</span></div>
                </div>
              </div>
              <div className="bg-white/[0.02] border border-white/10 p-6">
                <p className="text-xs uppercase tracking-widest text-emerald-500 mb-4">Funding Status</p>
                <div className="grid grid-cols-3 gap-4 text-center mb-6">
                  <div className="p-4 bg-zinc-900/50 rounded-lg">
                    <div className="text-zinc-500 text-xs mb-1">Committed</div>
                    <div className="text-xl text-zinc-600">$0</div>
                  </div>
                  <div className="p-4 bg-zinc-900/50 rounded-lg">
                    <div className="text-zinc-500 text-xs mb-1">Invited</div>
                    <div className="text-xl text-blue-400">$290M</div>
                  </div>
                  <div className="p-4 bg-zinc-900/50 rounded-lg">
                    <div className="text-zinc-500 text-xs mb-1">Gap</div>
                    <div className="text-xl text-amber-400">$19M</div>
                  </div>
                </div>
                <p className="text-xs text-zinc-500 text-center">2030 Target: $202M revenue / 100M learners</p>
              </div>
            </motion.div>

            {/* Five-Year Financial View */}
            <motion.div variants={item} className="overflow-x-auto bg-zinc-900/30 rounded-lg p-6">
              <h3 className="text-lg text-white mb-4">Five-Year View</h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800">
                    <th className="text-left py-3 text-zinc-500 font-normal"></th>
                    {['2026', '2027', '2028', '2029', '2030'].map(y => (
                      <th key={y} className="text-right py-3 text-zinc-400 font-normal">{y}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-zinc-800/50">
                    <td className="py-2 text-white font-medium">Revenue</td>
                    <td className="py-2 text-right text-white">$1.7M</td>
                    <td className="py-2 text-right text-white">$7.8M</td>
                    <td className="py-2 text-right text-white">$33M</td>
                    <td className="py-2 text-right text-white">$90M</td>
                    <td className="py-2 text-right text-white">$202M</td>
                  </tr>
                  <tr className="border-b border-zinc-800/50">
                    <td className="py-2 text-white font-medium">EBITDA</td>
                    <td className="py-2 text-right text-red-400">($650K)</td>
                    <td className="py-2 text-right text-red-400">($2.4M)</td>
                    <td className="py-2 text-right text-red-400">($3M)</td>
                    <td className="py-2 text-right text-green-400">$15M</td>
                    <td className="py-2 text-right text-green-400">$56M</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-blue-400 font-medium">Learners</td>
                    <td className="py-2 text-right text-blue-400">50K</td>
                    <td className="py-2 text-right text-blue-400">500K</td>
                    <td className="py-2 text-right text-blue-400">5M</td>
                    <td className="py-2 text-right text-blue-400">25M</td>
                    <td className="py-2 text-right text-blue-400">100M</td>
                  </tr>
                </tbody>
              </table>
            </motion.div>

            {/* Links */}
            <motion.div variants={item} className="flex flex-wrap gap-6">
              <Link 
                href="/coalition"
                className="inline-flex items-center gap-2 text-sm text-white/60 hover:text-white transition-colors"
              >
                Full Coalition Document
                <ArrowUpRight className="w-4 h-4" />
              </Link>
              <Link 
                href="/ziggurat/investors"
                className="inline-flex items-center gap-2 text-sm text-white/60 hover:text-white transition-colors"
              >
                Financial Model
                <ArrowUpRight className="w-4 h-4" />
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THE VISION SECTION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-32 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          
          <p className="text-zinc-600 mb-8" style={{ fontSize: "10px", letterSpacing: "0.4em", textTransform: "uppercase" }}>
            The Vision
          </p>

          <h2 
            className="text-white mb-8"
            style={{ 
              fontFamily: "Georgia, 'Times New Roman', serif",
              fontSize: "clamp(2rem, 5vw, 3.5rem)",
              fontWeight: 300,
              letterSpacing: "0.02em",
              lineHeight: 1.2
            }}
          >
            Not a platform.<br />Infrastructure.
          </h2>

          <p className="mb-16 text-lg leading-relaxed text-zinc-400 max-w-xl">
            A physical foundation for education that reaches{" "}
            <span className="text-white">every learner, every day, in every language</span>. 
            Built to last a century.
          </p>

          {/* Three Foundation Cards */}
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                title: "HARDWARE",
                text: "Learning devices manufactured on-site. A device in every hand, built to work offline."
              },
              {
                title: "BROADCAST",
                text: "24/7 educational programming. Kelly reaches every screen, every channel, every timezone."
              },
              {
                title: "PRINT",
                text: "Physical presence in every home. Daily lessons delivered to doorsteps worldwide."
              }
            ].map((card) => (
              <div
                key={card.title}
                className="p-8 transition-all duration-300 hover:bg-white/[0.03] bg-white/[0.015] border border-white/5"
              >
                <h3 className="text-zinc-500 mb-4" style={{ fontSize: "11px", letterSpacing: "0.2em" }}>
                  {card.title}
                </h3>
                <p className="text-zinc-500" style={{ fontSize: "0.9rem", lineHeight: 1.7 }}>
                  {card.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* DOCUMENTS SECTION - We Show Our Work */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-24 px-6 bg-zinc-950/50 border-t border-white/5">
        <div className="max-w-5xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="space-y-12"
          >
            <motion.div variants={item}>
              <p className="text-zinc-600 mb-4" style={{ fontSize: "10px", letterSpacing: "0.4em", textTransform: "uppercase" }}>
                We Show Our Work
              </p>
              <h2 
                className="text-white mb-4"
                style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: "clamp(1.75rem, 4vw, 2.5rem)", fontWeight: 300 }}
              >
                Investment Documents
              </h2>
              <p className="text-zinc-500 max-w-2xl">
                Full transparency. Every number validated. Our thesis, model, and projections—available for review.
              </p>
            </motion.div>

            <motion.div variants={item} className="grid md:grid-cols-3 gap-6">
              {[
                {
                  title: "One Pager",
                  description: "Executive summary. The problem, solution, projections, and ask—on one page.",
                  size: "1 page",
                  href: "/docs/LOTD_One_Pager.pdf",
                  highlight: true
                },
                {
                  title: "Master Investment Thesis",
                  description: "Complete thesis document. Track record, building, product, economics, coalition, moat, risks.",
                  size: "45 pages",
                  href: "/docs/LOTD_Master_Investment_Thesis.pdf",
                  highlight: false
                },
                {
                  title: "Financial Model",
                  description: "Seven-year operating plan. Unit economics, sensitivity analysis, deployment schedule.",
                  size: "Interactive",
                  href: "/dashboard/financial-model",
                  highlight: false
                }
              ].map((doc) => (
                <a
                  key={doc.title}
                  href={doc.href}
                  target={doc.href.endsWith('.pdf') ? '_blank' : undefined}
                  rel={doc.href.endsWith('.pdf') ? 'noopener noreferrer' : undefined}
                  className={`group p-8 transition-all duration-300 border ${
                    doc.highlight 
                      ? 'bg-white/[0.03] border-white/20 hover:border-white/40' 
                      : 'bg-white/[0.015] border-white/10 hover:border-white/20'
                  }`}
                >
                  <div className="flex items-start justify-between mb-4">
                    <h3 
                      className="text-white"
                      style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: "1.1rem" }}
                    >
                      {doc.title}
                    </h3>
                    <ArrowUpRight className="w-4 h-4 text-zinc-600 group-hover:text-white transition-colors" />
                  </div>
                  <p className="text-zinc-500 text-sm mb-6 leading-relaxed">
                    {doc.description}
                  </p>
                  <p className="text-zinc-600 text-xs uppercase tracking-wider">
                    {doc.size}
                  </p>
                </a>
              ))}
            </motion.div>

            <motion.div variants={item} className="pt-8 border-t border-white/5">
              <p className="text-zinc-600 text-sm">
                Additional materials available upon request: Harvard Business School Case Study (N9-814-020), 
                GSA auction documentation, manufacturing specifications, partnership term sheets.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* FOUNDER SECTION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-24 px-6 bg-zinc-950 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="space-y-16"
          >
            <motion.div variants={item}>
              <p className="text-zinc-600 mb-4" style={{ fontSize: "10px", letterSpacing: "0.4em", textTransform: "uppercase" }}>
                The Founder
              </p>
              <h2 
                className="text-white"
                style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: "clamp(2rem, 5vw, 3rem)", fontWeight: 300 }}
              >
                Nicolette Rankin
              </h2>
            </motion.div>

            <motion.div variants={item} className="grid md:grid-cols-2 gap-16">
              <div className="space-y-6">
                <p className="text-zinc-400 leading-relaxed text-lg">
                  Co-founder OpenEnglish. Scaled to $171M annual revenue, 2M students across 18 countries. Pioneered live video instruction and adaptive learning before the category existed.
                </p>
                <p className="text-zinc-500 leading-relaxed">
                  Now building the infrastructure for personalized video education at global scale—moving from synchronous instruction to 365-day asynchronous learning across all ages.
                </p>
              </div>

              <div className="space-y-4">
                <div className="bg-white/5 border border-white/10 p-8">
                  <p className="text-white text-3xl mb-2" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>20+</p>
                  <p className="text-sm uppercase tracking-widest text-zinc-500">Years EdTech</p>
                </div>
                <div className="bg-white/5 border border-white/10 p-8">
                  <p className="text-white text-3xl mb-2" style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}>18</p>
                  <p className="text-sm uppercase tracking-widest text-zinc-500">Countries Served</p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* CONTACT SECTION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <section className="py-32 px-6 border-t border-white/5">
        <div className="max-w-5xl mx-auto">
          
          <h2 
            className="text-white mb-12 text-center"
            style={{ 
              fontFamily: "Georgia, 'Times New Roman', serif",
              fontSize: "2rem",
              fontWeight: 300,
              letterSpacing: "0.08em"
            }}
          >
            Contact
          </h2>

          {/* Contact Cards Grid */}
          <div className="grid md:grid-cols-3 gap-6">
            {CONFIG.contacts.map((contact, idx) => (
              <div key={idx} className="text-left p-8 bg-white/[0.015] border border-white/10 hover:border-white/20 transition-colors">
                <h3 
                  className="text-white mb-1"
                  style={{ 
                    fontFamily: "Georgia, 'Times New Roman', serif",
                    fontSize: "1.2rem",
                    letterSpacing: "0.04em"
                  }}
                >
                  {contact.name}
                </h3>
                <p className="text-zinc-500 mb-2 italic" style={{ fontSize: "0.85rem" }}>
                  {contact.title}
                </p>
                <p 
                  className="text-zinc-600 mb-4 pb-4 border-b border-white/10"
                  style={{ fontSize: "9px", letterSpacing: "0.2em" }}
                >
                  {contact.company}
                </p>

                <div className="space-y-1.5" style={{ fontSize: "0.85rem" }}>
                  {contact.phone && (
                    <p>
                      <a href={`tel:${contact.phone}`} className="text-zinc-400 hover:text-white transition-colors">
                        {contact.phone}
                      </a>
                    </p>
                  )}
                  {contact.emails.map((email, emailIdx) => (
                    <p key={emailIdx}>
                      <a href={`mailto:${email}`} className="text-zinc-400 hover:text-white transition-colors">
                        {email}
                      </a>
                    </p>
                  ))}
                  {contact.license && (
                    <p className="text-zinc-600 pt-2 text-xs">{contact.license}</p>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Address */}
          <div className="mt-8 text-center">
            <p className="text-zinc-500 text-sm">{CONFIG.address.street}</p>
            <p className="text-zinc-500 text-sm">{CONFIG.address.city}</p>
          </div>

          {/* Financial Workspace Link */}
          <div className="mt-8 text-center">
            <Link 
              href="/ziggurat/spreadsheet"
              className="inline-flex items-center gap-3 px-6 py-3 text-sm text-white bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all"
            >
              <span>Financial Models & Workspaces</span>
              <ArrowUpRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* FOOTER */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      
      <footer className="py-12 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          {/* Navigation */}
          <div className="flex justify-center gap-8 mb-8">
            <Link href="/" className="text-zinc-600 hover:text-white text-xs tracking-widest transition-colors">PLATFORM</Link>
            <Link href="/about" className="text-zinc-600 hover:text-white text-xs tracking-widest transition-colors">ABOUT</Link>
            <Link href="/ziggurat/spreadsheet" className="text-zinc-600 hover:text-white text-xs tracking-widest transition-colors">MODELS</Link>
          </div>

          {/* Alpha Status */}
          <div className="flex justify-center mb-6">
            <span className="inline-flex items-center gap-2 px-3 py-1 bg-white/5 border border-white/10 text-zinc-500 text-[10px] tracking-widest">
              <span className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-pulse" />
              PRIVATE BETA | Q2 2026 LAUNCH
            </span>
          </div>

          {/* Corporate */}
          <p className="text-zinc-700 text-center" style={{ fontSize: "11px", letterSpacing: "0.12em" }}>
            © 2026 Lesson of the Day, PBC | California Public Benefit Corporation
          </p>
        </div>
      </footer>
    </main>
  )
}
