import React from "react"
import type { Metadata, Viewport } from "next"

export const metadata: Metadata = {
  title: "Nicolette Rankin - Founder, The Daily Lesson",
  description:
    "Nicolette Rankin is the founder of Lesson of the Day, PBC and creator of Curious Kelly.",
  icons: {
    icon: "/favicon-32.png",
  },
}

export const viewport: Viewport = {
  themeColor: "#faf8f5",
  width: "device-width",
  initialScale: 1,
}

export default function NicoletteLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return <>{children}</>
}
