// ═══════════════════════════════════════════════════════════════════════════
// DOCS INDEX — THE DAILY LESSON
// Public document library - We Show Our Work
// ═══════════════════════════════════════════════════════════════════════════

import Link from "next/link"
import { ArrowUpRight, FileText, BarChart3, Building2 } from "lucide-react"

export const metadata = {
  title: "Documents | Lesson of the Day",
  description: "Investment documents, curriculum thesis, financial models, and strategic materials for Lesson of the Day coalition partners.",
}

const DOCUMENTS = [
  {
    id: "one-pager",
    title: "One Pager",
    description: "Executive summary. The problem, solution, projections, and ask—on one page.",
    category: "Executive",
    size: "1 page",
    updated: "February 2026",
    href: "/docs/LOTD_One_Pager.pdf",
    icon: FileText,
    preview: {
      type: "stats",
      items: [
        { label: "TAM", value: "$8T" },
        { label: "Ask", value: "$3B" },
        { label: "Goal", value: "8B learners" },
      ]
    },
    highlight: true,
  },
  {
    id: "thesis",
    title: "Master Investment Thesis",
    description: "Complete thesis document covering track record, building, product, economics, coalition structure, competitive moat, and risk factors.",
    category: "Investment",
    size: "45 pages",
    updated: "February 2026",
    href: "/docs/LOTD_Master_Investment_Thesis.pdf",
    icon: FileText,
    preview: {
      type: "toc",
      items: [
        "Executive Summary",
        "Track Record",
        "The Building",
        "The Product",
        "Unit Economics",
        "Coalition Structure",
        "Risk Factors",
      ]
    },
  },


  {
    id: "financial-model",
    title: "Financial Model",
    description: "Interactive seven-year operating plan with unit economics, sensitivity analysis, and deployment schedule.",
    category: "Financial",
    size: "Interactive",
    updated: "February 2026",
    href: "/dashboard/financial-model",
    icon: BarChart3,
    isInternal: true,
    preview: {
      type: "chart",
      items: [
        { year: "2026", value: 22 },
        { year: "2028", value: 430 },
        { year: "2030", value: 3300 },
        { year: "2032", value: 15300 },
      ]
    },
  },
  {
    id: "ziggurat",
    title: "The Ziggurat",
    description: "Chet Holifield Federal Building. Floor-by-floor transformation plan for 1M SF manufacturing campus.",
    category: "Real Estate",
    size: "Interactive",
    updated: "February 2026",
    href: "/ziggurat",
    icon: Building2,
    isInternal: true,
    preview: {
      type: "stats",
      items: [
        { label: "Sq Ft", value: "1M" },
        { label: "Floors", value: "7" },
        { label: "Acres", value: "92" },
      ]
    },
  },
]

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3 group">
            <span className="text-white/60 group-hover:text-white transition-colors text-sm">
              &larr; Back to Kelly
            </span>
            <span className="text-white/20">|</span>
            <span className="font-serif text-white">The Daily Lesson</span>
          </Link>
          <div className="flex items-center gap-6 text-sm">
            <Link href="/about" className="text-white/50 hover:text-white transition-colors">About</Link>
            <Link href="/strategic" className="text-white/50 hover:text-white transition-colors">Strategic</Link>
            <Link href="/docs" className="text-white font-medium">Documents</Link>
          </div>
        </div>
      </nav>

      {/* Header */}
      <header className="pt-32 pb-16 px-6">
        <div className="max-w-4xl mx-auto">
          <p 
            className="text-zinc-600 mb-4"
            style={{ fontSize: "10px", letterSpacing: "0.4em", textTransform: "uppercase" }}
          >
            We Show Our Work
          </p>
          <h1 
            className="text-white mb-6"
            style={{ 
              fontFamily: "Georgia, 'Times New Roman', serif",
              fontSize: "clamp(2.5rem, 6vw, 4rem)",
              fontWeight: 300,
              letterSpacing: "0.02em"
            }}
          >
            Documents
          </h1>
          <p className="text-zinc-400 text-lg max-w-2xl leading-relaxed">
            Full transparency. Every number validated. Our thesis, model, and projections—available for review by coalition partners and stakeholders.
          </p>
        </div>
      </header>

      {/* Documents Grid */}
      <section className="px-6 pb-32">
        <div className="max-w-4xl mx-auto">
          <div className="grid gap-6">
            {DOCUMENTS.map((doc) => {
              const Icon = doc.icon
              const isExternal = !doc.isInternal
              
              return (
                <Link
                  key={doc.id}
                  href={doc.href}
                  target={isExternal ? "_blank" : undefined}
                  rel={isExternal ? "noopener noreferrer" : undefined}
                  className={`group flex flex-col md:flex-row items-start gap-6 p-8 border transition-all duration-300 ${
                    doc.highlight 
                      ? 'bg-white/[0.03] border-white/20 hover:border-white/30' 
                      : 'bg-white/[0.015] border-white/10 hover:border-white/20 hover:bg-white/[0.03]'
                  }`}
                >
                  {/* Preview Panel */}
                  <div className="w-full md:w-48 flex-shrink-0">
                    <div className="aspect-[4/3] bg-zinc-900 border border-white/5 rounded-lg p-4 flex flex-col justify-center">
                      {doc.preview?.type === 'stats' && (
                        <div className="grid grid-cols-3 gap-2">
                          {doc.preview.items.map((item: { label: string; value: string }, i: number) => (
                            <div key={i} className="text-center">
                              <p className="text-white font-mono text-sm">{item.value}</p>
                              <p className="text-zinc-600 text-[9px] uppercase tracking-wider mt-0.5">{item.label}</p>
                            </div>
                          ))}
                        </div>
                      )}
                      {doc.preview?.type === 'toc' && (
                        <div className="space-y-1">
                          {doc.preview.items.slice(0, 5).map((item: string, i: number) => (
                            <div key={i} className="flex items-center gap-2">
                              <span className="text-zinc-700 text-[9px]">{i + 1}.</span>
                              <span className="text-zinc-500 text-[10px] truncate">{item}</span>
                            </div>
                          ))}
                          {doc.preview.items.length > 5 && (
                            <p className="text-zinc-700 text-[9px] pl-4">+{doc.preview.items.length - 5} more</p>
                          )}
                        </div>
                      )}
                      {doc.preview?.type === 'chart' && (
                        <div className="flex items-end justify-between h-16 px-1">
                          {doc.preview.items.map((item: { year: string; value: number }, i: number) => (
                            <div key={i} className="flex flex-col items-center gap-1">
                              <div 
                                className="w-6 bg-blue-500/30 rounded-sm"
                                style={{ height: `${Math.max(4, (item.value / 15300) * 48)}px` }}
                              />
                              <span className="text-[8px] text-zinc-600">{item.year.slice(-2)}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-4 mb-2">
                      <h2 
                        className="text-white"
                        style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: "1.25rem" }}
                      >
                        {doc.title}
                      </h2>
                      <ArrowUpRight className="w-5 h-5 text-zinc-600 group-hover:text-white transition-colors flex-shrink-0" />
                    </div>
                    
                    <p className="text-zinc-500 mb-4 leading-relaxed">
                      {doc.description}
                    </p>
                    
                    <div className="flex items-center gap-4 text-xs text-zinc-600 uppercase tracking-wider">
                      <span>{doc.category}</span>
                      <span className="text-zinc-700">•</span>
                      <span>{doc.size}</span>
                      <span className="text-zinc-700">•</span>
                      <span>{doc.updated}</span>
                    </div>
                  </div>
                </Link>
              )
            })}
          </div>

          {/* Additional Materials */}
          <div className="mt-16 p-8 border border-white/5 bg-white/[0.01]">
            <h3 
              className="text-white mb-4"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: "1.1rem" }}
            >
              Additional Materials
            </h3>
            <p className="text-zinc-500 text-sm leading-relaxed mb-4">
              The following materials are available upon request for qualified coalition partners:
            </p>
            <ul className="text-zinc-600 text-sm space-y-2">
              <li>• Harvard Business School Case Study (N9-814-020) — Open English</li>
              <li>• GSA auction documentation and bid history</li>
              <li>• Manufacturing specifications and capacity planning</li>
              <li>• Partnership term sheets and governance frameworks</li>
              <li>• Kelly technical architecture and content pipeline</li>
            </ul>
            <div className="mt-6 pt-6 border-t border-white/5">
              <p className="text-zinc-600 text-sm">
                Contact{" "}
                <a href="mailto:nicolette@thedailylesson.com" className="text-white hover:underline">
                  nicolette@thedailylesson.com
                </a>
                {" "}to request access.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-12 px-6">
        <div className="max-w-4xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6 text-sm text-zinc-600">
          <p>Lesson of the Day, PBC · California Entity No. 5774402</p>
          <p>24000 Avila Road, Laguna Niguel, California 92677</p>
        </div>
      </footer>
    </div>
  )
}
