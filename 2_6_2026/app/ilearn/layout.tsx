import React from "react"
import type { Metadata, Viewport } from "next"

export const metadata: Metadata = {
  title: "iLearn.how - A New Lesson Every Day",
  description:
    "365 daily lessons for every age, every language. Meet Curious Kelly, your AI teacher.",
  icons: {
    icon: "/favicon-32.png",
  },
}

export const viewport: Viewport = {
  themeColor: "#fafafa",
  width: "device-width",
  initialScale: 1,
}

export default function ILearnLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return <>{children}</>
}
