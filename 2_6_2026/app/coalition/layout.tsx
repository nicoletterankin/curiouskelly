import React from "react"
import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Coalition Model — Lesson of the Day, PBC",
  description: "Kelly teaches 8 billion humans. Join the coalition building education infrastructure for humanity.",
  openGraph: {
    title: "Coalition Model — Victory of the People",
    description: "An invitation to build Kelly's home. First look for coalition partners.",
    images: ["/og-coalition.png"],
  },
}

export default function CoalitionLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
