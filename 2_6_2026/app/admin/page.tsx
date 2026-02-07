'use client'

import Link from 'next/link'
import { 
  Sparkles, 
  Users, 
  Video, 
  BarChart3, 
  TestTube,
  Building2,
  ArrowRight,
  Clapperboard,
  LayoutDashboard,
} from 'lucide-react'

const adminTools = [
  {
    title: 'Executive Command Center',
    description: 'Full pipeline KPIs, gap analysis, cost estimates, and one-click production',
    href: '/admin/command-center',
    icon: LayoutDashboard,
    color: 'bg-rose-500/20 text-rose-400',
    highlight: true,
  },
  {
    title: 'Video Production',
    description: 'Central command for 365-day video generation pipeline',
    href: '/admin/video-production',
    icon: Clapperboard,
    color: 'bg-emerald-500/20 text-emerald-400',
  },
  {
    title: 'Dallas',
    description: 'Strategic initiative dashboard',
    href: '/dallas',
    icon: Building2,
    color: 'bg-blue-500/20 text-blue-400',
  },
  {
    title: 'Video Engine Comparison',
    description: 'Compare HeyGen, Fal.ai, Sync.so, and MuseTalk side-by-side',
    href: '/admin/compare',
    icon: Sparkles,
    color: 'bg-amber-500/20 text-amber-400',
  },
  {
    title: 'HeyGen Dashboard',
    description: 'Manage HeyGen API, generate videos, check credits',
    href: '/admin/heygen',
    icon: Video,
    color: 'bg-sky-500/20 text-sky-400',
  },
  {
    title: 'HeyGen API Test',
    description: 'Test HeyGen API connection and video generation',
    href: '/admin/heygen-test',
    icon: TestTube,
    color: 'bg-purple-500/20 text-purple-400',
  },
  {
    title: 'Learners',
    description: 'View and manage learner accounts and progress',
    href: '/admin/learners',
    icon: Users,
    color: 'bg-emerald-500/20 text-emerald-400',
  },
  {
    title: 'Analytics',
    description: 'View engagement metrics and usage statistics',
    href: '/admin/analytics',
    icon: BarChart3,
    color: 'bg-rose-500/20 text-rose-400',
  },
]

export default function AdminPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/40 backdrop-blur-xl">
        <div className="max-w-5xl mx-auto px-6 py-6">
          <h1 className="text-3xl font-bold text-white">Admin Dashboard</h1>
          <p className="text-white/50 mt-1">Curious Kelly management tools</p>
        </div>
      </header>

      {/* Tools grid */}
      <main className="max-w-5xl mx-auto px-6 py-8">
        <div className="grid md:grid-cols-2 gap-4">
          {adminTools.map(tool => (
            <Link
              key={tool.href}
              href={tool.href}
              className="group p-6 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all"
            >
              <div className="flex items-start gap-4">
                <div className={`w-12 h-12 rounded-xl ${tool.color} flex items-center justify-center`}>
                  <tool.icon className="w-6 h-6" />
                </div>
                <div className="flex-1">
                  <h2 className="text-lg font-semibold text-white group-hover:text-white transition-colors">
                    {tool.title}
                  </h2>
                  <p className="text-sm text-white/50 mt-1">{tool.description}</p>
                </div>
                <ArrowRight className="w-5 h-5 text-white/30 group-hover:text-white/60 group-hover:translate-x-1 transition-all" />
              </div>
            </Link>
          ))}
        </div>

        {/* Quick stats placeholder */}
        <div className="mt-8 p-6 rounded-2xl bg-white/5 border border-white/10">
          <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">Content Status</h3>
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: 'Total Lessons', value: '365' },
              { label: 'Videos Ready', value: '6' },
              { label: 'Audio Ready', value: '1,825' },
              { label: 'Coverage', value: '0.5%' },
            ].map(stat => (
              <div key={stat.label} className="text-center">
                <div className="text-2xl font-bold text-white">{stat.value}</div>
                <div className="text-xs text-white/40 mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}
