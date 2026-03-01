"use client"

import { useState } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Play, Download, ChevronLeft, ChevronRight } from "lucide-react"
import Link from "next/link"

const SLATE_IMAGES = [
  {
    id: "front-comparison",
    src: "/ziggurat/slate/front-comparison.jpg",
    title: "Front View — Before & After",
    description: "Elastomeric concrete coating transforms the yellow brutalist facade to slate blue-gray",
  },
  {
    id: "aerial-comparison",
    src: "/ziggurat/slate/aerial-comparison.jpg",
    title: "Aerial View — Before & After",
    description: "92-acre campus with refreshed grounds and native landscaping",
  },
  {
    id: "front-slate",
    src: "/ziggurat/slate/front-slate.jpg",
    title: "Front View — After",
    description: "Clean architectural presence, honoring Pereira's brutalist vision",
  },
  {
    id: "aerial-slate",
    src: "/ziggurat/slate/aerial-slate.jpg",
    title: "Aerial View — After",
    description: "The Ziggurat transformed, ready for its educational mission",
  },
  {
    id: "commons",
    src: "/ziggurat/slate/commons-interior.png",
    title: "The Commons — Interior",
    description: "Ground floor public learning space with Kelly welcome center",
  },
  {
    id: "levels",
    src: "/ziggurat/slate/levels-diagram.png",
    title: "Seven Floors of Purpose",
    description: "Foundation → Workshop → Studio → Commons → Academy → Council → Observatory",
  },
  {
    id: "future",
    src: "/ziggurat/slate/future-2028.png",
    title: "2028 Vision",
    description: "Grand opening celebration during LA Olympics year",
  },
]

export default function SlateVisionPage() {
  const [currentIndex, setCurrentIndex] = useState(0)
  const current = SLATE_IMAGES[currentIndex]

  const next = () => setCurrentIndex((i) => (i + 1) % SLATE_IMAGES.length)
  const prev = () => setCurrentIndex((i) => (i - 1 + SLATE_IMAGES.length) % SLATE_IMAGES.length)

  return (
    <div className="min-h-screen bg-[#0A0A0C] text-white">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Ziggurat Vision</h1>
            <p className="text-sm text-zinc-500">Slate Blue Direction — Feb 2026</p>
          </div>
          <div className="flex gap-3">
            <Link href="/bren">
              <Button variant="outline" size="sm" className="border-zinc-700">
                Investor One-Pager
              </Button>
            </Link>
            <Link href="https://pub-29446fb0037f47e49993ebd6b4ed714e.r2.dev/ziggurat/LAGUNA_RIDGE.mp4" target="_blank">
              <Button size="sm" className="gap-2 bg-white text-black hover:bg-zinc-200">
                <Play className="w-4 h-4" />
                Video
              </Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Main Gallery */}
      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Current Image */}
        <div className="relative aspect-video rounded-xl overflow-hidden bg-zinc-900 mb-6">
          <Image
            src={current.src}
            alt={current.title}
            fill
            className="object-contain"
            priority
          />

          {/* Navigation Arrows */}
          <button
            onClick={prev}
            className="absolute left-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-black/60 hover:bg-black/80 transition-colors"
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
          <button
            onClick={next}
            className="absolute right-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-black/60 hover:bg-black/80 transition-colors"
          >
            <ChevronRight className="w-6 h-6" />
          </button>

          {/* Counter */}
          <div className="absolute bottom-4 right-4 px-3 py-1 rounded-full bg-black/60 text-sm font-mono">
            {currentIndex + 1} / {SLATE_IMAGES.length}
          </div>
        </div>

        {/* Caption */}
        <div className="text-center mb-12">
          <h2 className="text-2xl font-bold mb-2">{current.title}</h2>
          <p className="text-zinc-400">{current.description}</p>
        </div>

        {/* Thumbnail Strip */}
        <div className="grid grid-cols-7 gap-3 mb-16">
          {SLATE_IMAGES.map((img, i) => (
            <button
              key={img.id}
              onClick={() => setCurrentIndex(i)}
              className={`aspect-video rounded-lg overflow-hidden border-2 transition-colors ${
                i === currentIndex ? "border-white" : "border-zinc-800 hover:border-zinc-600"
              }`}
            >
              <Image
                src={img.src}
                alt={img.title}
                width={200}
                height={112}
                className="w-full h-full object-cover"
              />
            </button>
          ))}
        </div>

        {/* Direction Statement */}
        <div className="max-w-3xl mx-auto text-center space-y-6">
          <h3 className="text-xl font-semibold">The Direction</h3>
          <blockquote className="text-zinc-400 italic border-l-2 border-zinc-700 pl-6 text-left">
            "I don't think it makes sense to spend money on LCD screens and we should just
            paint the yellow building a slate blue instead of an ugly yellow and refresh the
            grounds and spend money on learners and manufacturing and printing presses and
            global broadcasting channels."
          </blockquote>
          <p className="text-sm text-zinc-500">— Nicolette, February 15, 2026</p>

          <div className="pt-8 space-y-4 text-zinc-300 text-left max-w-2xl mx-auto">
            <p>
              <strong className="text-white">No LED screens.</strong> No spectacle budget.
              The building serves the mission, not the other way around.
            </p>
            <p>
              <strong className="text-white">Slate blue paint.</strong> Elastomeric concrete coating
              transforms the outdated yellow to a dignified slate blue-gray.
            </p>
            <p>
              <strong className="text-white">Capital to operations.</strong> Printing presses,
              manufacturing, broadcast channels, learners — that's where the money goes.
            </p>
          </div>
        </div>

        {/* Download Section */}
        <div className="mt-16 pt-12 border-t border-zinc-800">
          <h3 className="text-lg font-semibold mb-6">Download Assets</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {SLATE_IMAGES.slice(0, 4).map((img) => (
              <a
                key={img.id}
                href={img.src}
                download
                className="p-4 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-zinc-600 transition-colors group"
              >
                <Download className="w-5 h-5 mb-2 text-zinc-500 group-hover:text-white transition-colors" />
                <div className="text-sm font-medium">{img.title}</div>
                <div className="text-xs text-zinc-500">Click to download</div>
              </a>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800 px-6 py-8 mt-12">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-sm text-zinc-500">
            Lesson of the Day, PBC — thedailylesson.com
          </div>
          <div className="flex gap-4">
            <Link href="/bren" className="text-sm text-zinc-400 hover:text-white">
              /bren
            </Link>
            <Link href="/ziggurat/investors" className="text-sm text-zinc-400 hover:text-white">
              /ziggurat/investors
            </Link>
            <Link href="/coalition" className="text-sm text-zinc-400 hover:text-white">
              /coalition
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
