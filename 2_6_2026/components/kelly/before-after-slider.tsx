"use client"

import React from "react"

import { useState, useRef, useCallback } from "react"
import Image from "next/image"
import { cn } from "@/lib/utils"

interface BeforeAfterSliderProps {
  beforeSrc: string
  afterSrc: string
  beforeLabel?: string
  afterLabel?: string
  initialPosition?: number
  className?: string
}

export function BeforeAfterSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "BEFORE",
  afterLabel = "AFTER",
  initialPosition = 50,
  className,
}: BeforeAfterSliderProps) {
  const [position, setPosition] = useState(initialPosition)
  const containerRef = useRef<HTMLDivElement>(null)
  const isDragging = useRef(false)

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = clientX - rect.left
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setPosition(pct)
  }, [])

  const handleStart = () => {
    isDragging.current = true
  }
  
  const handleEnd = () => {
    isDragging.current = false
  }
  
  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging.current) handleMove(e.clientX)
  }
  
  const handleTouchMove = (e: React.TouchEvent) => {
    handleMove(e.touches[0].clientX)
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative overflow-hidden rounded-2xl cursor-ew-resize select-none",
        "shadow-2xl border border-white/10",
        className
      )}
      onMouseDown={handleStart}
      onMouseUp={handleEnd}
      onMouseLeave={handleEnd}
      onMouseMove={handleMouseMove}
      onTouchStart={handleStart}
      onTouchEnd={handleEnd}
      onTouchMove={handleTouchMove}
    >
      {/* After image (full, background) */}
      <Image
        src={afterSrc || "/placeholder.svg"}
        alt={afterLabel}
        width={1920}
        height={1080}
        className="w-full h-auto"
        priority
      />

      {/* Before image (clipped) */}
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      >
        <Image
          src={beforeSrc || "/placeholder.svg"}
          alt={beforeLabel}
          width={1920}
          height={1080}
          className="w-full h-full object-cover"
          priority
        />
      </div>

      {/* Slider handle line */}
      <div
        className="absolute top-0 h-full w-0.5 bg-white shadow-lg"
        style={{ left: `${position}%`, transform: "translateX(-50%)" }}
      >
        {/* Handle button */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white/15 backdrop-blur-md text-white px-4 py-2.5 rounded-full text-sm font-bold whitespace-nowrap shadow-xl border border-white/20">
          <span className="opacity-60">{"<"}</span>
          <span className="mx-2">|</span>
          <span className="opacity-60">{">"}</span>
        </div>
      </div>

      {/* Labels */}
      <span className="absolute top-4 left-4 bg-white/15 backdrop-blur-sm px-4 py-2 rounded-xl text-xs font-semibold tracking-wider border border-white/10 text-white">
        {beforeLabel}
      </span>
      <span className="absolute top-4 right-4 bg-white/15 backdrop-blur-sm px-4 py-2 rounded-xl text-xs font-semibold tracking-wider border border-white/10 text-white">
        {afterLabel}
      </span>
    </div>
  )
}
