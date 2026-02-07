import React from "react"
import type { Metadata, Viewport } from 'next'

export const metadata: Metadata = {
  title: 'Kelly Studio | Director\'s Interface',
  description: 'Direct Kelly without touching code. The visual interface for building the AI teacher that changes education forever.',
}

export const viewport: Viewport = {
  themeColor: '#0f1419',
  width: 'device-width',
  initialScale: 1,
}

export default function StudioLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
