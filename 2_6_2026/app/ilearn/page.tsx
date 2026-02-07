import type { Metadata } from "next"
import Image from "next/image"
import Link from "next/link"

export const metadata: Metadata = {
  title: "iLearn.how - A New Lesson Every Day",
  description:
    "365 daily lessons for every age, every language. Meet Curious Kelly, your AI teacher who makes learning feel like scrolling your favorite feed. Start free.",
  openGraph: {
    title: "iLearn.how - A New Lesson Every Day",
    description:
      "365 daily lessons. 47+ languages. Ages 2-102. Free to start. Meet your AI teacher, Curious Kelly.",
    type: "website",
    url: "https://ilearn.how",
  },
  twitter: {
    card: "summary_large_image",
    title: "iLearn.how",
    description:
      "A new lesson every day. 365 topics, 47+ languages, all ages. Start free.",
  },
  alternates: {
    canonical: "https://ilearn.how",
  },
}

function ArrowRight({ className }: { className?: string }) {
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
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  )
}

function GlobeIcon({ className }: { className?: string }) {
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
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  )
}

function SparklesIcon({ className }: { className?: string }) {
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
      <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z" />
      <path d="M19 14l.75 2.25L22 17l-2.25.75L19 20l-.75-2.25L16 17l2.25-.75L19 14z" />
    </svg>
  )
}

function CalendarIcon({ className }: { className?: string }) {
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
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  )
}

function UsersIcon({ className }: { className?: string }) {
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
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  )
}

const SAMPLE_TOPICS = [
  "How Electricity Flows",
  "Why the Sky Is Blue",
  "The Secret Life of Ants",
  "How Bridges Stay Up",
  "Why Music Makes Us Feel",
  "How Your Brain Learns",
  "The Story of Zero",
  "Why Leaves Change Color",
  "How Volcanoes Erupt",
  "The History of Chocolate",
  "How Airplanes Fly",
  "Why We Dream",
]

export default function ILearnPage() {
  return (
    <div className="ilearn-page min-h-screen bg-[#fafafa] text-[#111]">
      {/* Hero */}
      <section className="relative overflow-hidden">
        {/* Subtle background pattern */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage:
              "radial-gradient(circle at 1px 1px, #111 1px, transparent 0)",
            backgroundSize: "32px 32px",
          }}
        />

        <div className="relative px-6 md:px-12 max-w-5xl mx-auto">
          {/* Nav */}
          <nav className="flex items-center justify-between py-6">
            <span className="text-lg font-bold tracking-tight text-[#111]">
              ilearn.how
            </span>
            <Link
              href="https://thedailylesson.com"
              className="flex items-center gap-2 px-5 py-2.5 bg-[#111] text-[#fafafa] rounded-full text-sm font-medium hover:bg-[#333] transition-colors"
            >
              Start learning
              <ArrowRight className="w-4 h-4" />
            </Link>
          </nav>

          {/* Hero content */}
          <div className="pt-16 md:pt-28 pb-20 md:pb-32 text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#111]/5 rounded-full text-sm text-[#555] mb-8">
              <span className="w-2 h-2 rounded-full bg-[#22c55e] animate-pulse" />
              Today's lesson is live
            </div>

            <h1 className="text-4xl sm:text-5xl md:text-7xl font-bold tracking-tight text-[#111] text-balance leading-[1.1] max-w-3xl mx-auto">
              A new lesson, every single day.
            </h1>

            <p className="mt-6 text-lg md:text-xl text-[#666] max-w-xl mx-auto leading-relaxed text-pretty">
              365 topics. 47+ languages. For ages 2 to 102. Meet{" "}
              <span className="text-[#111] font-medium">Curious Kelly</span>,
              the AI teacher who makes learning feel effortless.
            </p>

            <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="https://thedailylesson.com"
                className="flex items-center gap-3 px-8 py-4 bg-[#111] text-[#fafafa] rounded-full text-base font-semibold hover:bg-[#333] transition-colors shadow-lg shadow-[#111]/10"
              >
                Start learning -- it's free
                <ArrowRight className="w-5 h-5" />
              </Link>
              <span className="text-sm text-[#999]">
                No signup required to watch
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Kelly preview */}
      <section className="px-6 md:px-12 max-w-5xl mx-auto -mt-8 mb-16">
        <div className="relative rounded-2xl overflow-hidden bg-[#111] shadow-2xl shadow-[#111]/20 aspect-video flex items-center justify-center">
          <Image
            src="/kelly/logo-1024.webp"
            alt="Curious Kelly - Your AI teacher"
            width={200}
            height={200}
            className="opacity-80"
          />
          <div className="absolute bottom-6 left-6 right-6 flex items-end justify-between">
            <div>
              <div className="text-white/60 text-xs font-medium uppercase tracking-wider mb-1">
                Today's lesson
              </div>
              <div className="text-white text-lg font-semibold">
                How Electricity Flows
              </div>
            </div>
            <Link
              href="https://thedailylesson.com"
              className="flex items-center gap-2 px-5 py-2.5 bg-white/15 backdrop-blur-md border border-white/20 text-white rounded-full text-sm font-medium hover:bg-white/25 transition-colors"
            >
              Watch now
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Stats row */}
      <section className="px-6 md:px-12 max-w-5xl mx-auto mb-20">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            {
              icon: CalendarIcon,
              stat: "365",
              label: "lessons per year",
            },
            {
              icon: GlobeIcon,
              stat: "47+",
              label: "languages",
            },
            {
              icon: UsersIcon,
              stat: "2-102",
              label: "ages served",
            },
            {
              icon: SparklesIcon,
              stat: "5 min",
              label: "per lesson",
            },
          ].map((item) => (
            <div
              key={item.label}
              className="flex flex-col items-center text-center p-6 rounded-xl border border-[#e5e5e5] bg-[#fafafa]"
            >
              <item.icon className="w-5 h-5 text-[#999] mb-3" />
              <span className="text-2xl md:text-3xl font-bold text-[#111]">
                {item.stat}
              </span>
              <span className="text-sm text-[#777] mt-1">{item.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="px-6 md:px-12 max-w-5xl mx-auto mb-20">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-[#999] mb-8 text-center">
          How it works
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          {[
            {
              step: "01",
              title: "Open the app",
              description:
                "Visit thedailylesson.com on any device. No download, no signup, no friction.",
            },
            {
              step: "02",
              title: "Watch today's lesson",
              description:
                "Kelly teaches in 5 phases: Hook, Story, Wonder, Action, Wisdom. Five minutes total.",
            },
            {
              step: "03",
              title: "Come back tomorrow",
              description:
                "A brand new topic every day. Build a streak, explore the archive, learn forever.",
            },
          ].map((item) => (
            <div key={item.step} className="flex flex-col">
              <span className="text-4xl font-bold text-[#e5e5e5] mb-4">
                {item.step}
              </span>
              <h3 className="text-lg font-semibold text-[#111] mb-2">
                {item.title}
              </h3>
              <p className="text-sm text-[#666] leading-relaxed">
                {item.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Sample topics ticker */}
      <section className="mb-20 overflow-hidden">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-[#999] mb-6 text-center px-6">
          365 topics like these
        </h2>
        <div className="flex gap-3 animate-[scroll_30s_linear_infinite] w-max">
          {[...SAMPLE_TOPICS, ...SAMPLE_TOPICS].map((topic, i) => (
            <span
              key={`${topic}-${i}`}
              className="flex-shrink-0 px-5 py-2.5 bg-[#111] text-[#fafafa] rounded-full text-sm font-medium whitespace-nowrap"
            >
              {topic}
            </span>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 md:px-12 max-w-5xl mx-auto mb-20">
        <div className="bg-[#111] rounded-2xl p-10 md:p-16 text-center">
          <h2 className="text-2xl md:text-4xl font-bold text-[#fafafa] tracking-tight text-balance">
            Your next great idea starts with today's lesson.
          </h2>
          <p className="mt-4 text-[#999] text-base max-w-md mx-auto">
            Free to start. No account required. Just press play.
          </p>
          <Link
            href="https://thedailylesson.com"
            className="inline-flex items-center gap-3 mt-8 px-8 py-4 bg-[#fafafa] text-[#111] rounded-full text-base font-semibold hover:bg-[#e5e5e5] transition-colors"
          >
            Go to The Daily Lesson
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 md:px-12 max-w-5xl mx-auto border-t border-[#e5e5e5] py-8 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-[#999]">
        <span>Lesson of the Day, PBC</span>
        <div className="flex items-center gap-6">
          <Link
            href="https://thedailylesson.com"
            className="hover:text-[#111] transition-colors"
          >
            thedailylesson.com
          </Link>
          <Link
            href="https://nicoletterankin.com"
            className="hover:text-[#111] transition-colors"
          >
            nicoletterankin.com
          </Link>
        </div>
      </footer>

      {/* Scroll animation keyframe */}
      <style
        dangerouslySetInnerHTML={{
          __html: `
        @keyframes scroll {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
      `,
        }}
      />
    </div>
  )
}
