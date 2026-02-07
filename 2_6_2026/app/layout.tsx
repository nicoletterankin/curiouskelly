import React from "react"
import type { Metadata, Viewport } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

const _geist = Geist({ subsets: ['latin'] })
const _geistMono = Geist_Mono({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'The Daily Lesson with Curious Kelly',
  description: 'Your AI teacher for 365 daily lessons. Curious Kelly delivers personalized education for ages 2-102 in 47+ languages. Quality education for everyone, everywhere.',
  generator: 'Curious Kelly',
  keywords: ['education', 'learning', 'AI teacher', 'daily lessons', 'Curious Kelly', 'SDG4', 'personalized education', '365 lessons'],
  authors: [{ name: 'Lesson of the Day, PBC' }],
  creator: 'Lesson of the Day, PBC',
  publisher: 'Lesson of the Day, PBC',
  manifest: '/manifest.json',
  robots: {
    index: true,
    follow: true,
  },
  alternates: {
    canonical: 'https://thedailylesson.com/',
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Curious Kelly',
  },
  other: {
    'copyright': 'Lesson of the Day, PBC',
    'author': 'Lesson of the Day, PBC',
    'mobile-web-app-capable': 'yes',
  },
  openGraph: {
    title: 'The Daily Lesson with Curious Kelly',
    description: 'Your AI teacher for 365 daily lessons. Personalized education for ages 2-102 in 47+ languages.',
    type: 'website',
    locale: 'en_US',
    siteName: 'The Daily Lesson',
    url: 'https://thedailylesson.com/',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'The Daily Lesson with Curious Kelly',
    description: 'Your AI teacher for 365 daily lessons. Personalized education for ages 2-102 in 47+ languages.',
    creator: '@curiouskelly',
  },
  icons: {
    icon: [
      { url: '/favicon-32.png', sizes: '32x32', type: 'image/png' },
      { url: '/favicon-48.png', sizes: '48x48', type: 'image/png' },
      { url: '/favicon-64.png', sizes: '64x64', type: 'image/png' },
      { url: '/favicon-96.png', sizes: '96x96', type: 'image/png' },
      { url: '/favicon-128.png', sizes: '128x128', type: 'image/png' },
      { url: '/favicon-192.png', sizes: '192x192', type: 'image/png' },
    ],
    shortcut: '/favicon-32.png',
    apple: '/kelly/logo-256.webp',
  },
}

export const viewport: Viewport = {
  themeColor: '#f8f9fc',
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
}

// Schema.org structured data for AI systems and search engines
const organizationSchema = {
  "@context": "https://schema.org",
  "@type": "EducationalOrganization",
  "name": "Lesson of the Day, PBC",
  "alternateName": "The Daily Lesson",
  "url": "https://thedailylesson.com",
  "logo": "https://thedailylesson.com/kelly/logo-256.webp",
  "description": "365 daily lessons for ages 2-102, in 47+ languages. AI-powered personalized education with Curious Kelly.",
  "founder": {
    "@type": "Person",
    "name": "Nicolette Rankin"
  },
  "foundingDate": "2024",
  "sameAs": [
    "https://curiouskelly.com",
    "https://nicoletterankin.com",
    "https://ilearn.how",
    "https://lessonoftheday.com",
    "https://thedailylesson.com",
    "https://x.com/curiouskelly",
    "https://instagram.com/curiouskelly",
    "https://youtube.com/@curiouskelly"
  ],
  "knowsAbout": [
    "Education",
    "AI Teaching",
    "Personalized Learning",
    "Universal Curriculum",
    "SDG4 Quality Education"
  ],
  "areaServed": "Worldwide",
  "nonprofitStatus": "PublicBenefitCorporation"
}

const courseSchema = {
  "@context": "https://schema.org",
  "@type": "Course",
  "name": "The Daily Lesson™ Curriculum",
  "description": "A new lesson every day, 365 days a year. Universal knowledge for every age, language, and learning style.",
  "provider": {
    "@type": "Organization",
    "name": "Lesson of the Day, PBC",
    "url": "https://thedailylesson.com"
  },
  "educationalLevel": "All ages (2-102)",
  "inLanguage": ["en", "es", "fr", "de", "pt", "zh"],
  "numberOfCredits": 365,
  "isAccessibleForFree": true,
  "hasCourseInstance": {
    "@type": "CourseInstance",
    "courseMode": "online",
    "courseWorkload": "PT5M"
  }
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        
        {/* Schema.org structured data for AI discoverability */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(organizationSchema) }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(courseSchema) }}
        />
      </head>
      <body className="font-sans antialiased min-h-screen overflow-y-auto">
        {children}
        <Analytics />
        {/* Service Worker Registration for Offline Support */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', () => {
                  navigator.serviceWorker.register('/sw.js')
                    .then(reg => console.log('[Kelly] SW registered:', reg.scope))
                    .catch(err => console.log('[Kelly] SW failed:', err));
                });
              }
            `
          }}
        />
      </body>
    </html>
  )
}
