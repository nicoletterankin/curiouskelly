'use client'

import React from "react"

import { useState, useEffect } from 'react'
import { ChevronDown, ChevronUp, Lock, Eye, EyeOff } from 'lucide-react'
import Link from 'next/link'

// Admin credentials
const ADMIN_USERS = [
  "dallas_short@thedailylesson.com",
  "nicolette_rankin@thedailylesson.com"
]
const ADMIN_PASSWORD = "HOLIFIELD"

const DEADLINES = {
  verbalCommitments: "2026-01-30T17:00:00-08:00",
  dallasStartsCalling: "2026-01-28T08:30:00-08:00",
  appleIntroduction: "2026-06-01T00:00:00-08:00",
}

type Status = "done" | "in_progress" | "blocked" | "not_started" | "cancelled"

interface Task {
  id: string
  title: string
  status: Status
  assignee: "nicolette" | "dallas" | "cursor" | "claude" | "v0"
  priority: "critical" | "high" | "medium" | "low"
  notes?: string
  blockedBy?: string
}

interface Project {
  id: string
  name: string
  category: string
  tasks: Task[]
}

const PROJECTS: Project[] = [
  {
    id: "infra",
    name: "Infrastructure",
    category: "DEPLOY",
    tasks: [
      { id: "infra-1", title: "Deploy invest.thedailylesson.com", status: "not_started", assignee: "v0", priority: "critical", notes: "V0_INVEST_LANDING.tsx ready" },
      { id: "infra-2", title: "Deploy dallas.thedailylesson.com", status: "not_started", assignee: "v0", priority: "critical", notes: "V0_DALLAS_WELCOME.tsx ready" },
      { id: "infra-3", title: "Upload videos to CDN", status: "not_started", assignee: "nicolette", priority: "critical" },
      { id: "infra-4", title: "DNS Configuration", status: "not_started", assignee: "nicolette", priority: "high" },
      { id: "infra-5", title: "SSL Certificate Verification", status: "not_started", assignee: "nicolette", priority: "medium" }
    ]
  },
  {
    id: "email",
    name: "Email Configuration",
    category: "SETUP",
    tasks: [
      { id: "email-1", title: "Email Routing Setup", status: "not_started", assignee: "cursor", priority: "high" },
      { id: "email-2", title: "Gmail Integration", status: "not_started", assignee: "nicolette", priority: "medium" },
      { id: "email-3", title: "Email Verification", status: "not_started", assignee: "nicolette", priority: "high" }
    ]
  },
  {
    id: "dallas",
    name: "Team Onboarding",
    category: "TEAM",
    tasks: [
      { id: "dallas-1", title: "Welcome Package Sent", status: "not_started", assignee: "nicolette", priority: "critical" },
      { id: "dallas-2", title: "Onboarding Complete", status: "not_started", assignee: "dallas", priority: "critical" },
      { id: "dallas-3", title: "Initial Briefing", status: "not_started", assignee: "dallas", priority: "critical" },
      { id: "dallas-4", title: "Call List Generated", status: "done", assignee: "claude", priority: "high" },
      { id: "dallas-5", title: "Outreach Begins", status: "not_started", assignee: "dallas", priority: "critical" }
    ]
  },
  {
    id: "investors-t1",
    name: "Tier 1 Investors",
    category: "FUNDRAISE",
    tasks: [
      { id: "inv-1", title: "Strategic Partner A — Outreach", status: "in_progress", assignee: "dallas", priority: "critical" },
      { id: "inv-2", title: "Strategic Partner B — Contact", status: "in_progress", assignee: "nicolette", priority: "critical" },
      { id: "inv-3", title: "Strategic Partner C — Education Aligned", status: "in_progress", assignee: "nicolette", priority: "critical" },
      { id: "inv-4", title: "Strategic Partner D — Education Aligned", status: "in_progress", assignee: "nicolette", priority: "critical" },
      { id: "inv-5", title: "Strategic Partner E — Contact Made", status: "in_progress", assignee: "dallas", priority: "critical" },
      { id: "inv-6", title: "Government Relations — Outreach", status: "in_progress", assignee: "nicolette", priority: "high" }
    ]
  },
  {
    id: "investors-t2",
    name: "Tier 2 Investors",
    category: "FUNDRAISE",
    tasks: [
      { id: "inv-t2-1", title: "Technology Partner — Demo Prep", status: "in_progress", assignee: "nicolette", priority: "high" },
      { id: "inv-t2-2", title: "Foundation Partner A — Outreach", status: "in_progress", assignee: "dallas", priority: "high" },
      { id: "inv-t2-3", title: "Foundation Partner B — Email Sent", status: "in_progress", assignee: "nicolette", priority: "high" },
      { id: "inv-t2-4", title: "VC Partner A — LinkedIn", status: "in_progress", assignee: "dallas", priority: "high" },
      { id: "inv-t2-5", title: "VC Partner B — Introduced", status: "in_progress", assignee: "dallas", priority: "high" },
      { id: "inv-t2-6", title: "Capital Partner — Materials Sent", status: "in_progress", assignee: "nicolette", priority: "high" }
    ]
  },
  {
    id: "content",
    name: "Content & Files",
    category: "ASSETS",
    tasks: [
      { id: "content-1", title: "Investor Package Ready", status: "done", assignee: "claude", priority: "high" },
      { id: "content-2", title: "Welcome Package Ready", status: "done", assignee: "claude", priority: "high" },
      { id: "content-3", title: "Deploy Package Ready", status: "done", assignee: "claude", priority: "medium" },
      { id: "content-4", title: "Landing Pages Ready", status: "done", assignee: "claude", priority: "critical" },
      { id: "content-5", title: "Video Assets Ready", status: "done", assignee: "claude", priority: "critical" }
    ]
  },
  {
    id: "kelly",
    name: "Product Development",
    category: "PRODUCT",
    tasks: [
      { id: "kelly-1", title: "Core Platform — Stable", status: "done", assignee: "nicolette", priority: "high" },
      { id: "kelly-2", title: "Video Content — In Production", status: "in_progress", assignee: "nicolette", priority: "critical", notes: "365 days × 7 phases × 5 ages" },
      { id: "kelly-3", title: "Lip-sync Enhancement", status: "blocked", assignee: "cursor", priority: "high", blockedBy: "Focus on fundraise" },
      { id: "kelly-4", title: "Full Launch", status: "not_started", assignee: "nicolette", priority: "critical", notes: "Q2 2026" }
    ]
  }
]

const STATUS_STYLES: Record<Status, string> = {
  done: "bg-black text-white",
  in_progress: "bg-black/80 text-white",
  blocked: "bg-black/40 text-white",
  not_started: "bg-black/10 text-black/60",
  cancelled: "bg-black/5 text-black/30 line-through"
}

const ASSIGNEE_LABELS: Record<string, string> = {
  nicolette: "NR",
  dallas: "DS",
  cursor: "⌘",
  claude: "◈",
  v0: "v0"
}

function getTimeRemaining(deadline: string) {
  const diff = new Date(deadline).getTime() - new Date().getTime()
  if (diff < 0) return "PASSED"
  const hours = Math.floor(diff / (1000 * 60 * 60))
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60))
  if (hours > 24) {
    const days = Math.floor(hours / 24)
    return `${days}d ${hours % 24}h`
  }
  return `${hours}h ${minutes}m`
}

// ============================================================
// INVESTOR VIEW (PUBLIC)
// ============================================================
function InvestorView() {
  const stats = {
    total: PROJECTS.flatMap(p => p.tasks).length,
    done: PROJECTS.flatMap(p => p.tasks).filter(t => t.status === "done").length,
    inProgress: PROJECTS.flatMap(p => p.tasks).filter(t => t.status === "in_progress").length,
  }

  const progressPercent = Math.round((stats.done / stats.total) * 100)
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })

  return (
    <main className="min-h-screen bg-white text-black pb-24 overflow-y-auto">
      {/* Header */}
      <header className="border-b border-black/10">
        <div className="max-w-4xl mx-auto px-6 py-12">
          <p className="text-xs tracking-widest text-black/50 mb-3 uppercase font-semibold">Lesson of the Day, PBC</p>
          <h1 className="text-5xl md:text-6xl font-black tracking-tight mb-4">Project Status</h1>
          <p className="text-lg text-black/60">{today}</p>
        </div>
      </header>

      {/* Progress Overview */}
      <section className="border-b border-black/10">
        <div className="max-w-4xl mx-auto px-6 py-12">
          <div className="flex items-end justify-between mb-6">
            <div>
              <p className="text-6xl font-black">{progressPercent}%</p>
              <p className="text-sm text-black/50 mt-2 uppercase tracking-widest font-semibold">Overall Progress</p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold">{stats.done} / {stats.total}</p>
              <p className="text-sm text-black/50 mt-1">Tasks Complete</p>
            </div>
          </div>
          
          {/* Progress Bar */}
          <div className="h-3 bg-black/10 rounded-full overflow-hidden">
            <div 
              className="h-full bg-black transition-all duration-500" 
              style={{ width: `${progressPercent}%` }} 
            />
          </div>
        </div>
      </section>

      {/* Key Milestones */}
      <section className="border-b border-black/10">
        <div className="max-w-4xl mx-auto px-6 py-12">
          <h2 className="text-2xl font-black mb-8 uppercase tracking-tight">Key Milestones</h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="p-6 border border-black/10 rounded-lg">
              <div className="w-10 h-10 bg-black rounded-full flex items-center justify-center mb-4">
                <span className="text-white text-sm font-bold">1</span>
              </div>
              <h3 className="font-bold mb-2">Infrastructure</h3>
              <p className="text-sm text-black/60">Investor and team portals deployment</p>
              <p className="text-xs text-black/40 mt-3 uppercase tracking-wider">In Progress</p>
            </div>
            
            <div className="p-6 border border-black/10 rounded-lg">
              <div className="w-10 h-10 bg-black rounded-full flex items-center justify-center mb-4">
                <span className="text-white text-sm font-bold">2</span>
              </div>
              <h3 className="font-bold mb-2">Investor Outreach</h3>
              <p className="text-sm text-black/60">Strategic partner engagement active</p>
              <p className="text-xs text-black/40 mt-3 uppercase tracking-wider">Active</p>
            </div>
            
            <div className="p-6 border border-black/10 rounded-lg">
              <div className="w-10 h-10 bg-black/20 rounded-full flex items-center justify-center mb-4">
                <span className="text-black/60 text-sm font-bold">3</span>
              </div>
              <h3 className="font-bold mb-2">Product Launch</h3>
              <p className="text-sm text-black/60">Full video content for 365 days</p>
              <p className="text-xs text-black/40 mt-3 uppercase tracking-wider">Q2 2026</p>
            </div>
          </div>
        </div>
      </section>

      {/* Category Progress */}
      <section className="border-b border-black/10">
        <div className="max-w-4xl mx-auto px-6 py-12">
          <h2 className="text-2xl font-black mb-8 uppercase tracking-tight">Progress by Category</h2>
          
          <div className="space-y-6">
            {['DEPLOY', 'SETUP', 'TEAM', 'FUNDRAISE', 'ASSETS', 'PRODUCT'].map(category => {
              const categoryProjects = PROJECTS.filter(p => p.category === category)
              const categoryTasks = categoryProjects.flatMap(p => p.tasks)
              const done = categoryTasks.filter(t => t.status === "done").length
              const total = categoryTasks.length
              const percent = total > 0 ? Math.round((done / total) * 100) : 0
              
              return (
                <div key={category} className="flex items-center gap-6">
                  <div className="w-28 flex-shrink-0">
                    <p className="text-xs font-bold uppercase tracking-widest text-black/60">{category}</p>
                  </div>
                  <div className="flex-1">
                    <div className="h-2 bg-black/10 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-black transition-all duration-500" 
                        style={{ width: `${percent}%` }} 
                      />
                    </div>
                  </div>
                  <div className="w-16 text-right">
                    <p className="text-sm font-bold">{percent}%</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Timeline */}
      <section className="border-b border-black/10">
        <div className="max-w-4xl mx-auto px-6 py-12">
          <h2 className="text-2xl font-black mb-8 uppercase tracking-tight">Timeline</h2>
          
          <div className="space-y-4">
            <div className="flex items-center gap-4 p-4 border border-black/10 rounded-lg">
              <div className="w-3 h-3 bg-black rounded-full flex-shrink-0" />
              <div className="flex-1">
                <p className="font-bold">Investor Outreach Active</p>
                <p className="text-sm text-black/60">Strategic partnerships in progress</p>
              </div>
              <p className="text-xs text-black/40 uppercase tracking-wider">Now</p>
            </div>
            
            <div className="flex items-center gap-4 p-4 border border-black/10 rounded-lg">
              <div className="w-3 h-3 bg-black/40 rounded-full flex-shrink-0" />
              <div className="flex-1">
                <p className="font-bold">Verbal Commitments Target</p>
                <p className="text-sm text-black/60">Initial funding round</p>
              </div>
              <p className="text-xs text-black/40 uppercase tracking-wider">Thu Jan 30</p>
            </div>
            
            <div className="flex items-center gap-4 p-4 border border-black/10 rounded-lg">
              <div className="w-3 h-3 bg-black/20 rounded-full flex-shrink-0" />
              <div className="flex-1">
                <p className="font-bold">Full Product Launch</p>
                <p className="text-sm text-black/60">365 days of video content live</p>
              </div>
              <p className="text-xs text-black/40 uppercase tracking-wider">Q2 2026</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-black/10 bg-black/2">
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-center sm:text-left">
              <p className="font-semibold">Lesson of the Day, PBC</p>
              <p className="text-sm text-black/50">California Public Benefit Corporation</p>
            </div>
            <div className="flex items-center gap-6 text-sm text-black/50">
              <Link href="/about" className="hover:text-black transition-colors">About</Link>
              <Link href="/strategic" className="hover:text-black transition-colors">Strategic</Link>
              <span>© 2026</span>
            </div>
          </div>
        </div>
      </footer>
    </main>
  )
}

// ============================================================
// ADMIN VIEW (PASSWORD PROTECTED)
// ============================================================
function AdminView({ onLogout }: { onLogout: () => void }) {
  const [projects, setProjects] = useState(PROJECTS)
  const [filter, setFilter] = useState<string>("all")
  const [expandedProjects, setExpandedProjects] = useState<string[]>(PROJECTS.map(p => p.id))

  const toggleProject = (id: string) => {
    setExpandedProjects(prev => prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id])
  }

  const toggleTaskStatus = (projectId: string, taskId: string) => {
    setProjects(prev => prev.map(p => {
      if (p.id !== projectId) return p
      return {
        ...p,
        tasks: p.tasks.map(t => {
          if (t.id !== taskId) return t
          const statusOrder: Status[] = ["not_started", "in_progress", "blocked", "done"]
          const currentIndex = statusOrder.indexOf(t.status)
          const nextStatus = statusOrder[(currentIndex + 1) % statusOrder.length]
          return { ...t, status: nextStatus }
        })
      }
    }))
  }

  const stats = {
    total: projects.flatMap(p => p.tasks).length,
    done: projects.flatMap(p => p.tasks).filter(t => t.status === "done").length,
    inProgress: projects.flatMap(p => p.tasks).filter(t => t.status === "in_progress").length,
    critical: projects.flatMap(p => p.tasks).filter(t => t.priority === "critical" && t.status !== "done").length
  }

  const categories = [...new Set(projects.map(p => p.category))]
  const filteredProjects = filter === "all" ? projects : projects.filter(p => p.category === filter)
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })

  return (
    <main className="min-h-screen bg-white text-black pb-24 overflow-y-auto">
      {/* Header */}
      <header className="border-b border-black/10">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-start justify-between mb-8">
            <div>
              <p className="text-xs tracking-widest text-black/60 mb-2 uppercase font-semibold">Lesson of the Day, PBC</p>
              <h1 className="text-5xl font-black tracking-tight mb-3">COMMAND CENTER</h1>
              <p className="text-sm text-black/60">{today}</p>
              <button 
                onClick={onLogout}
                className="mt-3 text-xs text-black/40 hover:text-black transition-colors underline"
              >
                Sign Out (Admin)
              </button>
            </div>

            {/* Deadlines */}
            <div className="text-right space-y-4">
              <div>
                <p className="text-xs tracking-widest text-black/60 uppercase font-semibold">Verbal Commitments</p>
                <p className="text-lg font-mono font-bold">Thu 5pm PT</p>
                <p className="text-xs text-black/50">{getTimeRemaining(DEADLINES.verbalCommitments)}</p>
              </div>
              <div>
                <p className="text-xs tracking-widest text-black/60 uppercase font-semibold">Dallas Calls</p>
                <p className="text-sm font-mono font-bold">Tue 8:30am PT</p>
                <p className="text-xs text-black/50">{getTimeRemaining(DEADLINES.dallasStartsCalling)}</p>
              </div>
            </div>
          </div>

          {/* Stats Bar */}
          <div className="flex gap-8 pt-6 border-t border-black/10">
            <div>
              <p className="text-3xl font-black">{stats.done}/{stats.total}</p>
              <p className="text-xs tracking-widest text-black/60 mt-1 uppercase font-semibold">Completed</p>
            </div>
            <div>
              <p className="text-3xl font-black">{stats.inProgress}</p>
              <p className="text-xs tracking-widest text-black/60 mt-1 uppercase font-semibold">In Progress</p>
            </div>
            <div>
              <p className="text-3xl font-black">{stats.critical}</p>
              <p className="text-xs tracking-widest text-black/60 mt-1 uppercase font-semibold">Critical</p>
            </div>
          </div>
        </div>
      </header>

      {/* Filters */}
      <div className="border-b border-black/10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex gap-3 flex-wrap">
          <button
            onClick={() => setFilter("all")}
            className={`px-4 py-2 text-xs font-bold uppercase tracking-widest rounded transition-colors ${filter === "all" ? "bg-black text-white" : "bg-black/5 text-black/60 hover:bg-black/10"}`}
          >
            All
          </button>
          {categories.map(cat => (
            <button
              key={cat}
              onClick={() => setFilter(cat)}
              className={`px-4 py-2 text-xs font-bold uppercase tracking-widest rounded transition-colors ${filter === cat ? "bg-black text-white" : "bg-black/5 text-black/60 hover:bg-black/10"}`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Projects */}
      <div className="max-w-7xl mx-auto px-6 py-8 pb-32 space-y-6">
        {filteredProjects.map(project => {
          const isExpanded = expandedProjects.includes(project.id)
          const projectDone = project.tasks.filter(t => t.status === "done").length
          const projectTotal = project.tasks.length
          
          return (
            <div key={project.id} className="border border-black/10 rounded-lg overflow-hidden">
              <button
                onClick={() => toggleProject(project.id)}
                className="w-full flex items-center justify-between p-4 hover:bg-black/2 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <span className="px-2 py-1 text-[10px] font-bold uppercase tracking-widest bg-black/5 rounded">
                    {project.category}
                  </span>
                  <h3 className="font-bold">{project.name}</h3>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-black/50">{projectDone}/{projectTotal}</span>
                  {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </div>
              </button>
              
              {isExpanded && (
                <div className="border-t border-black/10">
                  {project.tasks.map(task => (
                    <div 
                      key={task.id} 
                      className="flex items-center gap-4 p-4 border-b border-black/5 last:border-b-0 hover:bg-black/2 transition-colors cursor-pointer"
                      onClick={() => toggleTaskStatus(project.id, task.id)}
                    >
                      <div className={`px-2 py-1 text-[10px] font-bold uppercase tracking-wider rounded ${STATUS_STYLES[task.status]}`}>
                        {task.status.replace("_", " ")}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className={`font-medium ${task.status === "done" ? "text-black/40" : ""}`}>{task.title}</p>
                        {task.notes && (
                          <p className="text-xs text-black/50 mt-1 truncate">{task.notes}</p>
                        )}
                        {task.blockedBy && (
                          <p className="text-xs text-black/40 mt-1 italic">Blocked: {task.blockedBy}</p>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 text-[10px] font-bold rounded ${task.priority === "critical" ? "bg-black text-white" : "bg-black/10 text-black/60"}`}>
                          {task.priority.toUpperCase()}
                        </span>
                        <span className="w-8 h-8 flex items-center justify-center bg-black/5 rounded-full text-xs font-bold">
                          {ASSIGNEE_LABELS[task.assignee]}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </main>
  )
}

// ============================================================
// LOGIN VIEW
// ============================================================
function LoginView({ onLogin }: { onLogin: () => void }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    
    if (ADMIN_USERS.includes(email.toLowerCase()) && password === ADMIN_PASSWORD) {
      onLogin()
    } else {
      setError('Invalid credentials')
    }
  }

  return (
    <main className="min-h-screen bg-white text-black flex items-center justify-center">
      <div className="w-full max-w-md px-6">
        <div className="text-center mb-12">
          <p className="text-xs tracking-widest text-black/50 mb-4 uppercase font-semibold">Lesson of the Day, PBC</p>
          <h1 className="text-4xl font-black tracking-tight mb-2">Admin Access</h1>
          <p className="text-black/50">Command Center</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-xs font-bold uppercase tracking-widest text-black/60 mb-2">
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-3 border border-black/20 rounded-lg focus:outline-none focus:border-black transition-colors"
              placeholder="email@thedailylesson.com"
            />
          </div>

          <div>
            <label className="block text-xs font-bold uppercase tracking-widest text-black/60 mb-2">
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 border border-black/20 rounded-lg focus:outline-none focus:border-black transition-colors pr-12"
                placeholder="••••••••"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-black/40 hover:text-black transition-colors"
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {error && (
            <p className="text-sm text-red-600 text-center">{error}</p>
          )}

          <button
            type="submit"
            className="w-full py-3 bg-black text-white font-bold rounded-lg hover:bg-black/80 transition-colors flex items-center justify-center gap-2"
          >
            <Lock className="w-4 h-4" />
            Sign In
          </button>
        </form>

        <div className="mt-12 text-center">
          <Link href="/command" className="text-sm text-black/40 hover:text-black transition-colors">
            View Public Status →
          </Link>
        </div>
      </div>
    </main>
  )
}

// ============================================================
// MAIN COMPONENT
// ============================================================
export default function CommandCenter() {
  const [mode, setMode] = useState<'investor' | 'admin' | 'login'>('investor')

  // Check for admin access from URL or localStorage
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search)
    if (urlParams.get('admin') === 'true') {
      setMode('login')
    }
    // Check if already authenticated
    if (typeof window !== 'undefined' && sessionStorage.getItem('admin_auth') === 'true') {
      setMode('admin')
    }
  }, [])

  const handleLogin = () => {
    sessionStorage.setItem('admin_auth', 'true')
    setMode('admin')
  }

  const handleLogout = () => {
    sessionStorage.removeItem('admin_auth')
    setMode('investor')
  }

  if (mode === 'login') {
    return <LoginView onLogin={handleLogin} />
  }

  if (mode === 'admin') {
    return <AdminView onLogout={handleLogout} />
  }

  return <InvestorView />
}
