"use client"

import { ArrowRight, MapPin, Building2, Users, Globe, Play, Download, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import Image from "next/image"
import Link from "next/link"

const STATS = [
  { value: "92", label: "Acres", icon: MapPin },
  { value: "1M", label: "Sq Ft", icon: Building2 },
  { value: "780", label: "Jobs by 2030", icon: Users },
  { value: "47", label: "Languages", icon: Globe },
]

const INVESTMENT = {
  phase1: "$309M",
  breakdown: [
    { item: "Building Acquisition", amount: "$200M" },
    { item: "Phase 1 Renovation", amount: "$75M" },
    { item: "Operating Reserve", amount: "$30M" },
    { item: "Closing Costs", amount: "$4M" },
  ],
  returns: [
    { year: "2026", revenue: "$1.7M", learners: "100K" },
    { year: "2028", revenue: "$33M", learners: "10M" },
    { year: "2030", revenue: "$202M", learners: "150M" },
  ],
}

const FLOOR_PLAN = [
  { level: "F1", name: "The Foundation", use: "iLearn device assembly, manufacturing, fulfillment" },
  { level: "F2", name: "The Workshop", use: "Printing presses, lesson manufacturing" },
  { level: "F3", name: "The Studio", use: "Broadcast, video, 100+ language localization" },
  { level: "F4", name: "The Commons", use: "Main entry, Kelly welcome center, learning space" },
  { level: "F5", name: "The Academy", use: "Educator training, curriculum development" },
  { level: "F6", name: "The Council", use: "Leadership, partnerships, Pacific views" },
  { level: "F7", name: "The Observatory", use: "Daily broadcast origin, heliport" },
]

export default function BrenPage() {
  return (
    <div className="min-h-screen bg-[#0A0A0C] text-white">
      {/* Hero Section */}
      <section className="relative h-[70vh] min-h-[600px]">
        <div className="absolute inset-0">
          <Image
            src="/ziggurat/kelly/hero-composite.jpg"
            alt="The Ziggurat - Kelly AI Teacher Global Headquarters"
            fill
            className="object-cover object-center"
            priority
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#0A0A0C] via-[#0A0A0C]/60 to-transparent" />
        </div>

        <div className="relative z-10 flex flex-col justify-end h-full px-8 pb-16 max-w-5xl mx-auto">
          <div className="space-y-4">
            <p className="text-sm font-mono text-blue-400 tracking-widest uppercase">
              Investment Opportunity
            </p>
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight">
              THE ZIGGURAT
            </h1>
            <p className="text-xl md:text-2xl text-zinc-300 max-w-2xl">
              Global Headquarters for Universal Education
            </p>
            <p className="text-zinc-400">
              Lesson of the Day, PBC &nbsp;|&nbsp; thedailylesson.com
            </p>
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="border-y border-zinc-800 bg-zinc-900/50">
        <div className="max-w-5xl mx-auto px-8 py-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {STATS.map((stat) => (
              <div key={stat.label} className="text-center">
                <stat.icon className="w-5 h-5 mx-auto mb-2 text-zinc-500" />
                <div className="text-3xl font-bold text-white">{stat.value}</div>
                <div className="text-sm text-zinc-400">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* The Vision */}
      <section className="px-8 py-16 max-w-5xl mx-auto">
        <div className="grid md:grid-cols-2 gap-12">
          <div>
            <h2 className="text-3xl font-bold mb-6">The Vision</h2>
            <div className="space-y-4 text-zinc-300">
              <p>
                Transform the Chet Holifield Federal Building — a 1971 brutalist masterpiece
                by William Pereira — into the global headquarters for Kelly, an AI teacher
                delivering free daily lessons to every person on Earth.
              </p>
              <p>
                After three failed GSA auctions and federal vacancy since February 2025,
                this iconic landmark awaits its true purpose: education without borders.
              </p>
              <p className="text-white font-medium">
                Not an app. Not a platform. A place.
              </p>
            </div>

            {/* Video CTA */}
            <div className="mt-8 flex gap-4">
              <Link href="/ziggurat/laguna-ridge/LAGUNA_RIDGE.mp4" target="_blank">
                <Button className="bg-white text-black hover:bg-zinc-200 gap-2">
                  <Play className="w-4 h-4" />
                  Watch Video
                </Button>
              </Link>
              <Link href="/ziggurat/investors">
                <Button variant="outline" className="border-zinc-700 hover:bg-zinc-800 gap-2">
                  Full Model
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
            </div>
          </div>

          {/* Before/After */}
          <div className="space-y-4">
            <Image
              src="/ziggurat/slate/aerial-comparison.jpg"
              alt="Before and After - Slate Blue Exterior"
              width={600}
              height={400}
              className="rounded-xl w-full"
            />
            <p className="text-xs text-zinc-500 text-center">
              Elastomeric concrete coating — Slate Blue-Gray — Concept visualization
            </p>
          </div>
        </div>
      </section>

      {/* Floor Plan */}
      <section className="px-8 py-16 bg-zinc-900/30">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Seven Floors of Purpose</h2>
          <div className="space-y-3">
            {FLOOR_PLAN.map((floor) => (
              <div
                key={floor.level}
                className="flex items-center gap-4 p-4 rounded-lg bg-black/40 border border-zinc-800"
              >
                <div className="w-12 h-12 rounded-lg bg-zinc-800 flex items-center justify-center font-mono font-bold text-zinc-400">
                  {floor.level}
                </div>
                <div>
                  <div className="font-semibold">{floor.name}</div>
                  <div className="text-sm text-zinc-400">{floor.use}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Investment Summary */}
      <section className="px-8 py-16 max-w-5xl mx-auto">
        <h2 className="text-3xl font-bold mb-8">Phase 1 Investment</h2>

        <div className="grid md:grid-cols-2 gap-12">
          {/* Uses */}
          <div>
            <div className="text-5xl font-bold text-blue-400 mb-6">{INVESTMENT.phase1}</div>
            <div className="space-y-3">
              {INVESTMENT.breakdown.map((item) => (
                <div key={item.item} className="flex justify-between py-2 border-b border-zinc-800">
                  <span className="text-zinc-400">{item.item}</span>
                  <span className="font-mono">{item.amount}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Projections */}
          <div>
            <h3 className="text-xl font-semibold mb-4">Revenue Projections</h3>
            <div className="space-y-4">
              {INVESTMENT.returns.map((row) => (
                <div key={row.year} className="flex items-center gap-4">
                  <div className="w-16 font-mono text-zinc-500">{row.year}</div>
                  <div className="flex-1">
                    <div className="font-bold">{row.revenue}</div>
                    <div className="text-sm text-zinc-500">{row.learners} learners</div>
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-6 text-sm text-zinc-500">
              Revenue from: Broadcast subscriptions, iLearn hardware (Apple Retail),
              print publishing, Kelly Live Classes
            </p>
          </div>
        </div>
      </section>

      {/* Kelly Demo */}
      <section className="px-8 py-16 bg-zinc-900/30">
        <div className="max-w-5xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold mb-4">Meet Kelly</h2>
              <p className="text-zinc-300 mb-6">
                Kelly is an AI teacher who delivers one lesson every day — to anyone,
                anywhere, at any age. She speaks 47 languages, adapts to 12 learning archetypes,
                and reaches 5 age groups from toddlers to elders.
              </p>
              <p className="text-zinc-400 mb-6">
                Every day at sunrise from The Observatory, Kelly broadcasts the day's lesson
                to the world. The Ziggurat becomes not just headquarters, but origin point.
              </p>
              <Link href="/ziggurat/kelly/demo.mp4" target="_blank">
                <Button className="gap-2 bg-blue-600 hover:bg-blue-700">
                  <Play className="w-4 h-4" />
                  Kelly Demo (56s)
                </Button>
              </Link>
            </div>
            <div>
              <Image
                src="/ziggurat/kelly/portrait.png"
                alt="Kelly AI Teacher"
                width={400}
                height={400}
                className="rounded-2xl mx-auto"
              />
            </div>
          </div>
        </div>
      </section>

      {/* The Ask */}
      <section className="px-8 py-16 max-w-5xl mx-auto">
        <div className="bg-gradient-to-br from-blue-950/50 to-zinc-900/50 rounded-2xl p-8 md:p-12 border border-blue-900/50">
          <h2 className="text-3xl font-bold mb-6">The Ask</h2>
          <div className="space-y-4 text-zinc-300">
            <p className="text-lg">
              <strong className="text-white">1. Advocacy with GSA</strong> — Your political connections
              can help navigate the federal disposition process.
            </p>
            <p className="text-lg">
              <strong className="text-white">2. Anchor Commitment</strong> — A $50-100M anchor investment
              validates the project for MacArthur and other institutional partners.
            </p>
            <p className="text-lg">
              <strong className="text-white">3. Foundation Introduction</strong> — Connect us with the
              Donald Bren Foundation board for formal partnership discussions.
            </p>
          </div>
          <div className="mt-8 pt-8 border-t border-zinc-700">
            <p className="text-zinc-400">
              The building shaped like the temples where humanity first stored its knowledge,
              dedicated to sharing that knowledge with everyone.
            </p>
            <p className="mt-4 text-white font-medium">
              This is the capstone.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-8 py-12 border-t border-zinc-800">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="text-center md:text-left">
            <div className="font-semibold">Lesson of the Day, PBC</div>
            <div className="text-sm text-zinc-500">Public Benefit Corporation • California</div>
          </div>
          <div className="flex gap-4">
            <Link href="https://thedailylesson.com" target="_blank">
              <Button variant="outline" size="sm" className="gap-2 border-zinc-700">
                <ExternalLink className="w-3 h-3" />
                thedailylesson.com
              </Button>
            </Link>
            <Link href="/ziggurat/investors">
              <Button variant="outline" size="sm" className="gap-2 border-zinc-700">
                Full Financial Model
              </Button>
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
