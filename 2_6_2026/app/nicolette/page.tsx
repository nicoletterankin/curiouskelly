import type { Metadata } from "next"
import Image from "next/image"
import Link from "next/link"

export const metadata: Metadata = {
  title: "Nicolette Rankin - Founder, The Daily Lesson",
  description:
    "Nicolette Rankin is the founder of Lesson of the Day, PBC and creator of Curious Kelly - an AI-powered daily learning platform delivering 365 lessons a year for ages 2 to 102, in 47+ languages.",
  openGraph: {
    title: "Nicolette Rankin - Founder, The Daily Lesson",
    description:
      "Building the world's first universal daily curriculum. 365 lessons for every age, every language, every learner.",
    type: "profile",
    url: "https://nicoletterankin.com",
  },
  twitter: {
    card: "summary_large_image",
    title: "Nicolette Rankin",
    description:
      "Founder of The Daily Lesson. Building universal education with AI.",
  },
  alternates: {
    canonical: "https://nicoletterankin.com",
  },
}

function NicoletteIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 2L2 7l10 5 10-5-10-5z" />
      <path d="M2 17l10 5 10-5" />
      <path d="M2 12l10 5 10-5" />
    </svg>
  )
}

function ArrowUpRight({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="7" y1="17" x2="17" y2="7" />
      <polyline points="7 7 17 7 17 17" />
    </svg>
  )
}

export default function NicolettePage() {
  return (
    <div className="nicolette-page min-h-screen bg-[#faf8f5] text-[#1a1a1a]">
      {/* Nav */}
      <nav className="flex items-center justify-between px-6 md:px-12 py-6 max-w-5xl mx-auto">
        <span className="text-sm font-medium tracking-tight text-[#1a1a1a]">
          Nicolette Rankin
        </span>
        <Link
          href="https://thedailylesson.com"
          className="text-sm text-[#8b7355] hover:text-[#1a1a1a] transition-colors"
        >
          The Daily Lesson
        </Link>
      </nav>

      {/* Hero */}
      <main className="px-6 md:px-12 max-w-5xl mx-auto">
        <div className="flex flex-col md:flex-row gap-12 md:gap-16 pt-12 md:pt-24 pb-16">
          {/* Left - Photo */}
          <div className="flex-shrink-0">
            <div className="w-48 h-48 md:w-64 md:h-64 rounded-2xl overflow-hidden bg-[#e8e2d8]">
              <Image
                src="/images/nicolette-portrait.jpg"
                alt="Nicolette Rankin"
                width={256}
                height={256}
                className="w-full h-full object-cover"
                priority
              />
            </div>
          </div>

          {/* Right - Bio */}
          <div className="flex-1 flex flex-col justify-center">
            <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-[#1a1a1a] text-balance leading-tight">
              I believe every person on Earth deserves a great lesson, every
              single day.
            </h1>
            <p className="mt-6 text-base md:text-lg leading-relaxed text-[#555]">
              I'm Nicolette Rankin, the founder of{" "}
              <Link
                href="https://thedailylesson.com"
                className="text-[#8b7355] underline underline-offset-2 decoration-[#8b7355]/30 hover:decoration-[#8b7355] transition-colors"
              >
                Lesson of the Day, PBC
              </Link>{" "}
              -- a public benefit corporation building the world's first
              universal daily curriculum. Our AI teacher,{" "}
              <Link
                href="https://thedailylesson.com"
                className="text-[#8b7355] underline underline-offset-2 decoration-[#8b7355]/30 hover:decoration-[#8b7355] transition-colors"
              >
                Curious Kelly
              </Link>
              , delivers 365 lessons a year for learners ages 2 to 102, in 47+
              languages.
            </p>
            <p className="mt-4 text-base md:text-lg leading-relaxed text-[#555]">
              Before founding The Daily Lesson, I spent years studying what makes
              learning stick -- across cultures, ages, and contexts. I started
              this company because I saw a simple gap: the best ideas in the
              world aren't reaching the people who need them most. Kelly is my
              answer to that.
            </p>
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-[#e0dbd2]" />

        {/* Mission + Links */}
        <div className="flex flex-col md:flex-row gap-12 md:gap-24 py-16">
          {/* Mission */}
          <div className="flex-1">
            <h2 className="text-xs font-semibold uppercase tracking-widest text-[#999] mb-4">
              Mission
            </h2>
            <p className="text-base leading-relaxed text-[#555]">
              Aligned with{" "}
              <span className="text-[#1a1a1a] font-medium">
                UN Sustainable Development Goal 4
              </span>{" "}
              -- Quality Education for all. We believe in open access, lifelong
              learning, and the power of a great teacher to change a life in five
              minutes.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              {[
                "365 daily lessons",
                "47+ languages",
                "Ages 2-102",
                "AI-personalized",
                "Free tier",
              ].map((tag) => (
                <span
                  key={tag}
                  className="px-3 py-1.5 text-xs font-medium text-[#8b7355] bg-[#8b7355]/8 rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Links */}
          <div className="md:w-64">
            <h2 className="text-xs font-semibold uppercase tracking-widest text-[#999] mb-4">
              Connect
            </h2>
            <div className="flex flex-col gap-3">
              {[
                {
                  label: "The Daily Lesson",
                  href: "https://thedailylesson.com",
                  description: "Start learning today",
                },
                {
                  label: "X / Twitter",
                  href: "https://x.com/curiouskelly",
                  description: "@curiouskelly",
                },
                {
                  label: "Instagram",
                  href: "https://instagram.com/curiouskelly",
                  description: "@curiouskelly",
                },
                {
                  label: "YouTube",
                  href: "https://youtube.com/@curiouskelly",
                  description: "@curiouskelly",
                },
                {
                  label: "Email",
                  href: "mailto:hello@thedailylesson.com",
                  description: "hello@thedailylesson.com",
                },
              ].map((link) => (
                <Link
                  key={link.label}
                  href={link.href}
                  target={link.href.startsWith("mailto") ? undefined : "_blank"}
                  rel={
                    link.href.startsWith("mailto")
                      ? undefined
                      : "noopener noreferrer"
                  }
                  className="group flex items-center justify-between py-2 border-b border-[#e0dbd2]/60 hover:border-[#8b7355]/40 transition-colors"
                >
                  <div>
                    <span className="text-sm font-medium text-[#1a1a1a] group-hover:text-[#8b7355] transition-colors">
                      {link.label}
                    </span>
                    <span className="block text-xs text-[#999] mt-0.5">
                      {link.description}
                    </span>
                  </div>
                  <ArrowUpRight className="w-4 h-4 text-[#ccc] group-hover:text-[#8b7355] transition-colors" />
                </Link>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-[#e0dbd2] py-8 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-[#999]">
          <span>Lesson of the Day, PBC</span>
          <Link
            href="https://thedailylesson.com"
            className="flex items-center gap-2 text-[#8b7355] hover:text-[#1a1a1a] transition-colors font-medium"
          >
            <NicoletteIcon className="w-4 h-4" />
            Start learning with Kelly
            <ArrowUpRight className="w-3 h-3" />
          </Link>
        </footer>
      </main>
    </div>
  )
}
