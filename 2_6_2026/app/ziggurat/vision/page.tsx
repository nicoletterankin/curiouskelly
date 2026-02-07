"use client"

import React from "react"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import Image from "next/image"
import Link from "next/link"
import { 
  ArrowLeft, 
  Share2, 
  Heart, 
  MapPin, 
  ChevronRight, 
  Users, 
  Building2, 
  Sparkles,
  Vote,
  Mail,
  ExternalLink,
  Calendar,
  TreeDeciduous,
  Waves,
  Flag,
  Lightbulb,
  Check
} from "lucide-react"
import { cn } from "@/lib/utils"

// Vision concepts with all the new images
const VISIONS = [
  {
    id: "led-rainbow",
    name: "Rainbow LED",
    category: "LED Installation",
    description: "2,100 feet of rainbow light tracing every terrace at dusk. A beacon of hope visible for miles.",
    image: "/ziggurat/heroes/ocean-beacon-night.jpg",
    icon: Lightbulb,
    color: "from-red-500 via-yellow-500 to-purple-500",
    votes: 1247,
    tags: ["Night-only", "Low energy", "Reversible"],
  },
  {
    id: "olympic",
    name: "LA 2028 Olympics",
    category: "Special Event",
    description: "Transform the Ziggurat into a medal podium with drone light athletes for the 2028 Los Angeles Olympics.",
    image: "/ziggurat/heroes/la2028-olympics.jpg",
    icon: Flag,
    color: "from-red-600 via-white to-blue-600",
    votes: 892,
    tags: ["Temporary", "National pride", "Historic"],
  },
  {
    id: "biophilic",
    name: "Living Forest",
    category: "Sustainability",
    description: "Vertical gardens on every terrace. Glass facades revealing greenery. A breathing building.",
    image: "/ziggurat/heroes/living-building.jpg",
    icon: TreeDeciduous,
    color: "from-green-400 to-emerald-600",
    votes: 1089,
    tags: ["Permanent", "Carbon negative", "Biodiversity"],
  },
  {
    id: "beach-day",
    name: "Coastal Glass",
    category: "Architecture",
    description: "Reflective glass panels in ocean blues and aqua tones. A beach resort aesthetic for Orange County.",
    image: "/ziggurat/heroes/kelly-blue-day.jpg",
    icon: Waves,
    color: "from-cyan-400 to-blue-500",
    votes: 756,
    tags: ["Daytime", "Reflective", "Modern"],
  },
  {
    id: "beach-night",
    name: "Ocean Projection",
    category: "Media Art",
    description: "Photorealistic beach scene projected on the building facade at night. Waves lapping against a pyramid.",
    image: "/ziggurat/heroes/ocean-beacon-night.jpg",
    icon: Waves,
    color: "from-amber-400 via-cyan-500 to-blue-800",
    votes: 634,
    tags: ["Night-only", "Projection mapping", "Dynamic"],
  },
]

// Community stats
const STATS = [
  { value: "4,618", label: "Community Votes", icon: Vote },
  { value: "127", label: "Ideas Submitted", icon: Lightbulb },
  { value: "23K", label: "Newsletter Subs", icon: Mail },
  { value: "1971", label: "Year Built", icon: Building2 },
]

// Timeline milestones
const TIMELINE = [
  { date: "Jan 2026", event: "Third GSA auction completed", status: "done" },
  { date: "Jan 2026", event: "Coalition offer submitted", status: "done" },
  { date: "Mar 2026", event: "GSA response expected", status: "active" },
  { date: "Q2 2026", event: "Close escrow", status: "upcoming" },
  { date: "2027", event: "Phase 1 Construction", status: "upcoming" },
  { date: "2028", event: "LA Olympics / Public Opening", status: "upcoming" },
]

export default function ZigguratVisionPage() {
  const [selectedVision, setSelectedVision] = useState(VISIONS[0])
  const [hasVoted, setHasVoted] = useState<string | null>(null)
  const [email, setEmail] = useState("")
  const [showThankYou, setShowThankYou] = useState(false)

  const handleVote = (visionId: string) => {
    setHasVoted(visionId)
    // In production, this would hit an API
  }

  const handleSubscribe = (e: React.FormEvent) => {
    e.preventDefault()
    setShowThankYou(true)
    setTimeout(() => setShowThankYou(false), 3000)
    setEmail("")
  }

  return (
    <div className="min-h-screen bg-[#0A0A0C] text-white">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 px-4 py-3 flex items-center justify-between bg-gradient-to-b from-black/80 to-transparent">
        <Link 
          href="/ziggurat"
          className="w-10 h-10 rounded-full bg-black/40 backdrop-blur-md border border-white/20 flex items-center justify-center hover:bg-black/50 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
        </Link>
        
        <div className="text-center">
          <h1 className="text-sm font-semibold">The Ziggurat Project</h1>
          <p className="text-[10px] text-white/50">Community Vision Board</p>
        </div>
        
        <button className="w-10 h-10 rounded-full bg-black/40 backdrop-blur-md border border-white/20 flex items-center justify-center hover:bg-black/50 transition-colors">
          <Share2 className="w-5 h-5" />
        </button>
      </header>

      {/* Hero Section */}
      <section className="relative pt-20 pb-8 px-4">
        <div className="max-w-6xl mx-auto">
          {/* Location badge */}
          <div className="flex items-center justify-center gap-1 text-white/50 text-xs mb-4">
            <MapPin className="w-3 h-3" />
            <span>Laguna Niguel, California</span>
          </div>
          
          <h2 className="text-3xl md:text-5xl font-light text-center mb-3">
            <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Reimagine the Ziggurat
            </span>
          </h2>
          
          <p className="text-white/60 text-center max-w-2xl mx-auto mb-8">
            Help us transform a vacant federal building into Orange County's most iconic landmark. 
            Vote for your favorite vision. Shape our community's future.
          </p>

          {/* Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
            {STATS.map((stat) => {
              const Icon = stat.icon
              return (
                <div 
                  key={stat.label}
                  className="bg-white/5 border border-white/10 rounded-xl p-4 text-center"
                >
                  <Icon className="w-5 h-5 mx-auto mb-2 text-white/40" />
                  <div className="text-2xl font-light text-white">{stat.value}</div>
                  <div className="text-[10px] text-white/40 uppercase tracking-wider">{stat.label}</div>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Featured Vision - Large Display */}
      <section className="px-4 pb-8">
        <div className="max-w-6xl mx-auto">
          <div className="relative aspect-[16/9] md:aspect-[21/9] rounded-2xl overflow-hidden mb-4">
            <AnimatePresence mode="wait">
              <motion.div
                key={selectedVision.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="absolute inset-0"
              >
                <Image
                  src={selectedVision.image || "/placeholder.svg"}
                  alt={selectedVision.name}
                  fill
                  className="object-cover"
                  priority
                />
                {/* Gradient overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
              </motion.div>
            </AnimatePresence>
            
            {/* Vision info overlay */}
            <div className="absolute bottom-0 left-0 right-0 p-6">
              <div className="flex items-end justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className={cn(
                      "px-2 py-0.5 rounded-full text-[10px] font-medium bg-gradient-to-r text-white",
                      selectedVision.color
                    )}>
                      {selectedVision.category}
                    </span>
                    {selectedVision.tags.map(tag => (
                      <span key={tag} className="px-2 py-0.5 rounded-full text-[10px] bg-white/10 text-white/70">
                        {tag}
                      </span>
                    ))}
                  </div>
                  <h3 className="text-2xl md:text-4xl font-light text-white mb-2">
                    {selectedVision.name}
                  </h3>
                  <p className="text-white/70 max-w-xl text-sm md:text-base">
                    {selectedVision.description}
                  </p>
                </div>
                
                {/* Vote button */}
                <button
                  onClick={() => handleVote(selectedVision.id)}
                  disabled={hasVoted !== null}
                  className={cn(
                    "flex-shrink-0 flex items-center gap-2 px-6 py-3 rounded-full font-semibold transition-all",
                    hasVoted === selectedVision.id
                      ? "bg-green-500 text-white"
                      : hasVoted
                        ? "bg-white/10 text-white/40 cursor-not-allowed"
                        : "bg-white text-black hover:bg-white/90"
                  )}
                >
                  {hasVoted === selectedVision.id ? (
                    <>
                      <Check className="w-5 h-5" />
                      Voted!
                    </>
                  ) : (
                    <>
                      <Heart className={cn("w-5 h-5", hasVoted && "opacity-50")} />
                      <span>{selectedVision.votes.toLocaleString()}</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Vision Selector */}
      <section className="px-4 pb-12">
        <div className="max-w-6xl mx-auto">
          <h3 className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-4 px-1">
            All Visions
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {VISIONS.map((vision) => {
              const Icon = vision.icon
              const isSelected = selectedVision.id === vision.id
              return (
                <button
                  key={vision.id}
                  onClick={() => setSelectedVision(vision)}
                  className={cn(
                    "relative aspect-video rounded-xl overflow-hidden transition-all",
                    isSelected 
                      ? "ring-2 ring-white ring-offset-2 ring-offset-[#0A0A0C]" 
                      : "hover:ring-1 hover:ring-white/30"
                  )}
                >
                  <Image
                    src={vision.image || "/placeholder.svg"}
                    alt={vision.name}
                    fill
                    className="object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />
                  <div className="absolute bottom-0 left-0 right-0 p-3">
                    <div className="flex items-center gap-1.5 mb-1">
                      <Icon className="w-3 h-3 text-white/70" />
                      <span className="text-xs font-medium text-white truncate">{vision.name}</span>
                    </div>
                    <div className="flex items-center gap-1 text-[10px] text-white/50">
                      <Heart className="w-2.5 h-2.5" />
                      {vision.votes.toLocaleString()}
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      </section>

      {/* Timeline Section */}
      <section className="px-4 pb-12">
        <div className="max-w-6xl mx-auto">
          <h3 className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-4 px-1">
            Project Timeline
          </h3>
          
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-4 top-0 bottom-0 w-px bg-white/10" />
            
            <div className="space-y-4">
              {TIMELINE.map((item, i) => (
                <div key={i} className="flex items-start gap-4 pl-2">
                  <div className={cn(
                    "w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 z-10",
                    item.status === "done" ? "bg-green-500 border-green-500" :
                    item.status === "active" ? "bg-white border-white animate-pulse" :
                    "bg-[#0A0A0C] border-white/30"
                  )}>
                    {item.status === "done" && <Check className="w-3 h-3 text-white" />}
                  </div>
                  <div className="flex-1 pb-4">
                    <div className="flex items-center gap-2">
                      <span className={cn(
                        "text-sm font-medium",
                        item.status === "active" ? "text-white" : "text-white/70"
                      )}>
                        {item.event}
                      </span>
                      {item.status === "active" && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] bg-white/20 text-white">
                          Current
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-white/40">{item.date}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Newsletter Section */}
      <section className="px-4 pb-12">
        <div className="max-w-6xl mx-auto">
          <div className="bg-gradient-to-br from-purple-500/20 to-cyan-500/20 border border-white/10 rounded-2xl p-6 md:p-8">
            <div className="flex flex-col md:flex-row md:items-center gap-6">
              <div className="flex-1">
                <h3 className="text-xl md:text-2xl font-light text-white mb-2">
                  Stay Updated
                </h3>
                <p className="text-white/60 text-sm">
                  Get monthly updates on community votes, GSA negotiations, and construction progress.
                </p>
              </div>
              
              <form onSubmit={handleSubscribe} className="flex gap-2 w-full md:w-auto">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="your@email.com"
                  required
                  className="flex-1 md:w-64 px-4 py-3 rounded-xl bg-black/40 border border-white/20 text-white placeholder:text-white/40 focus:outline-none focus:border-white/40"
                />
                <button
                  type="submit"
                  className="px-6 py-3 rounded-xl bg-white text-black font-semibold hover:bg-white/90 transition-colors"
                >
                  {showThankYou ? "Thanks!" : "Subscribe"}
                </button>
              </form>
            </div>
          </div>
        </div>
      </section>

      {/* Community Actions */}
      <section className="px-4 pb-20">
        <div className="max-w-6xl mx-auto">
          <h3 className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-4 px-1">
            Get Involved
          </h3>
          
          <div className="grid md:grid-cols-3 gap-3">
            <Link
              href="/ziggurat"
              className="bg-white/5 border border-white/10 rounded-xl p-5 hover:bg-white/10 transition-colors group"
            >
              <Sparkles className="w-6 h-6 text-purple-400 mb-3" />
              <h4 className="text-white font-medium mb-1">Explore the Lesson</h4>
              <p className="text-white/50 text-sm mb-3">
                Learn the full story from Curious Kelly
              </p>
              <div className="flex items-center gap-1 text-white/40 text-xs group-hover:text-white/60 transition-colors">
                Start learning <ChevronRight className="w-3 h-3" />
              </div>
            </Link>
            
            <a
              href="https://www.gsa.gov/real-estate/real-estate-services/overview-redesignating-and-disposing-of-real-property"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-white/5 border border-white/10 rounded-xl p-5 hover:bg-white/10 transition-colors group"
            >
              <Building2 className="w-6 h-6 text-cyan-400 mb-3" />
              <h4 className="text-white font-medium mb-1">Contact GSA</h4>
              <p className="text-white/50 text-sm mb-3">
                Voice your support for community reuse
              </p>
              <div className="flex items-center gap-1 text-white/40 text-xs group-hover:text-white/60 transition-colors">
                Visit GSA.gov <ExternalLink className="w-3 h-3" />
              </div>
            </a>
            
            <a
              href="https://cityoflagunaniguel.org"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-white/5 border border-white/10 rounded-xl p-5 hover:bg-white/10 transition-colors group"
            >
              <Users className="w-6 h-6 text-green-400 mb-3" />
              <h4 className="text-white font-medium mb-1">Local Government</h4>
              <p className="text-white/50 text-sm mb-3">
                Attend Laguna Niguel city council meetings
              </p>
              <div className="flex items-center gap-1 text-white/40 text-xs group-hover:text-white/60 transition-colors">
                City website <ExternalLink className="w-3 h-3" />
              </div>
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-4 pb-8 text-center">
        <p className="text-white/30 text-xs">
          A community project by Lesson of the Day, PBC
        </p>
        <p className="text-white/30 text-xs mt-1">
          Founded by Nicolette Rankin
        </p>
        <p className="text-white/20 text-[10px] mt-1">
          Building imagery is conceptual. Not affiliated with GSA.
        </p>
      </footer>
    </div>
  )
}
