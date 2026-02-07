"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

interface ScriptResult {
  success: boolean
  day: number
  title: string
  scripts?: {
    hook_script: string
    story_script: string
    wonder_script: string
    action_script: string
    wisdom_script: string
    hook_fact: string
    hook_correct_answer: boolean
    subtitle: string
    quote: string
    quote_author: string
  }
  message?: string
  error?: string
}

interface ScanResult {
  summary: {
    total_lessons: number
    aligned: number
    misaligned: number
    contaminated: number
    no_scripts: number
    likely_misaligned: number
  }
  misaligned_days: Array<{
    day: number
    title: string
    topic: string
    curriculum_topic: string
    issue: string
    hook_preview: string
  }>
}

interface StatusResult {
  day: number
  title: string
  topic: string
  curriculum_topic: string
  curriculum_objective: string
  scripts_aligned: boolean
  current_scripts: {
    hook_script: string
    story_script: string
    wonder_script: string
    action_script: string
    wisdom_script: string
  }
  metadata: {
    hook_fact: string
    subtitle: string
    quote: string
    quote_author: string
    content_version: number
    updated_at: string
  }
  kelly_assets: {
    total: string
    with_audio: string
    with_video: string
    pending: string
  }
}

// Variant pipeline types
interface VariantPipelineStatus {
  pipeline: {
    days_with_scripts: number
    total_scripts: number
    audio_done: number
    video_done: number
    percent_complete: string
  }
  fully_complete_days: number[]
  recent_status: Array<{
    day_number: number
    variant: string
    phases_complete: number
    fully_scripted: boolean
    audio_complete: boolean
    video_complete: boolean
  }>
}

interface VariantScriptsResult {
  day: number
  title: string
  topic: string
  variants: string[]
  scripts: Record<string, Record<string, { text: string; word_count: number }>>
  pipeline_status: Record<string, { generated_at: string; model: string; audio: boolean; video: boolean }>
  total_scripts: number
}

interface VariantGenerateResult {
  success: boolean
  mode: string
  day?: number
  variants: string[]
  total_scripts: number
  results: Array<{
    day: number
    variant: string
    title: string
    success: boolean
    error?: string
  }>
  message?: string
}

export default function RegenerateScriptsPage() {
  const [day, setDay] = useState(37)
  const [startDay, setStartDay] = useState(37)
  const [endDay, setEndDay] = useState(45)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ScriptResult | null>(null)
  const [scanResult, setScanResult] = useState<ScanResult | null>(null)
  const [statusResult, setStatusResult] = useState<StatusResult | null>(null)
  const [batchResults, setBatchResults] = useState<unknown>(null)
  // Variant state
  const [variantPipeline, setVariantPipeline] = useState<VariantPipelineStatus | null>(null)
  const [variantScripts, setVariantScripts] = useState<VariantScriptsResult | null>(null)
  const [variantGenResult, setVariantGenResult] = useState<VariantGenerateResult | null>(null)
  const [variantDay, setVariantDay] = useState(37)
  const [variantBatchStart, setVariantBatchStart] = useState(37)
  const [variantBatchEnd, setVariantBatchEnd] = useState(42)
  const [selectedVariant, setSelectedVariant] = useState<string>("all")
  const [activeTab, setActiveTab] = useState<
    "single" | "batch" | "scan" | "status" | "variants"
  >("variants")

  // --- Variant functions ---
  const loadVariantPipeline = async () => {
    setLoading(true)
    try {
      const res = await fetch("/api/admin/kelly-scripts/generate")
      const data = await res.json()
      setVariantPipeline(data)
    } catch (e) {
      alert(`Error loading pipeline: ${e}`)
    }
    setLoading(false)
  }

  const viewVariantScripts = async () => {
    setLoading(true)
    setVariantScripts(null)
    try {
      const url =
        selectedVariant === "all"
          ? `/api/kelly/scripts/${variantDay}`
          : `/api/kelly/scripts/${variantDay}?variant=${selectedVariant}`
      const res = await fetch(url)
      const data = await res.json()
      if (res.ok) {
        setVariantScripts(data)
      } else {
        alert(data.error || "Not found")
      }
    } catch (e) {
      alert(`Error: ${e}`)
    }
    setLoading(false)
  }

  const generateVariantScripts = async (mode: "single" | "batch") => {
    setLoading(true)
    setVariantGenResult(null)
    try {
      const body =
        mode === "single"
          ? {
              day: variantDay,
              ...(selectedVariant !== "all" && { variant: selectedVariant }),
            }
          : {
              startDay: variantBatchStart,
              endDay: variantBatchEnd,
              ...(selectedVariant !== "all" && { variant: selectedVariant }),
            }
      const res = await fetch("/api/admin/kelly-scripts/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      setVariantGenResult(data)
      // Refresh pipeline status
      loadVariantPipeline()
    } catch (e) {
      alert(`Error: ${e}`)
    }
    setLoading(false)
  }

  const checkStatus = async () => {
    setLoading(true)
    try {
      const res = await fetch(
        `/api/admin/regenerate-scripts?day=${day}`,
      )
      const data = await res.json()
      setStatusResult(data)
    } catch (e) {
      setStatusResult(null)
      alert(`Error: ${e}`)
    }
    setLoading(false)
  }

  const regenerateSingle = async () => {
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch("/api/admin/regenerate-scripts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ day }),
      })
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setResult({
        success: false,
        day,
        title: "",
        error: String(e),
      })
    }
    setLoading(false)
  }

  const regenerateBatch = async () => {
    setLoading(true)
    setBatchResults(null)
    try {
      const res = await fetch("/api/admin/regenerate-scripts", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ startDay, endDay }),
      })
      const data = await res.json()
      setBatchResults(data)
    } catch (e) {
      setBatchResults({ error: String(e) })
    }
    setLoading(false)
  }

  const runScan = async () => {
    setLoading(true)
    setScanResult(null)
    try {
      const res = await fetch("/api/admin/regenerate-scripts/scan")
      const data = await res.json()
      setScanResult(data)
    } catch (e) {
      alert(`Error: ${e}`)
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-4xl space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground">
            Script Regeneration Pipeline
          </h1>
          <p className="mt-1 text-muted-foreground">
            Regenerate phase scripts for corrected curriculum topics
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex flex-wrap gap-2">
          {(
            ["variants", "status", "single", "batch", "scan"] as const
          ).map((tab) => (
            <Button
              key={tab}
              variant={activeTab === tab ? "default" : "outline"}
              onClick={() => setActiveTab(tab)}
              size="sm"
            >
              {tab === "variants"
                ? "Variant Scripts"
                : tab === "status"
                  ? "Check Status"
                  : tab === "single"
                    ? "Single Day"
                    : tab === "batch"
                      ? "Batch Range"
                      : "Scan All"}
            </Button>
          ))}
        </div>

        {/* Variant Scripts */}
        {activeTab === "variants" && (
          <div className="space-y-4">
            {/* Pipeline Status Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Kelly Variant Pipeline</span>
                  <Button onClick={loadVariantPipeline} disabled={loading} size="sm">
                    {loading ? "Loading..." : "Refresh Pipeline"}
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {variantPipeline ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
                      <div className="rounded-lg bg-muted p-3 text-center">
                        <div className="text-2xl font-bold text-foreground">
                          {variantPipeline.pipeline.days_with_scripts}
                        </div>
                        <div className="text-xs text-muted-foreground">Days Scripted</div>
                      </div>
                      <div className="rounded-lg bg-muted p-3 text-center">
                        <div className="text-2xl font-bold text-foreground">
                          {variantPipeline.pipeline.total_scripts}
                        </div>
                        <div className="text-xs text-muted-foreground">Total Scripts</div>
                      </div>
                      <div className="rounded-lg bg-muted p-3 text-center">
                        <div className="text-2xl font-bold text-foreground">
                          {variantPipeline.pipeline.audio_done}
                        </div>
                        <div className="text-xs text-muted-foreground">Audio Done</div>
                      </div>
                      <div className="rounded-lg bg-muted p-3 text-center">
                        <div className="text-2xl font-bold text-foreground">
                          {variantPipeline.pipeline.video_done}
                        </div>
                        <div className="text-xs text-muted-foreground">Video Done</div>
                      </div>
                      <div className="rounded-lg bg-muted p-3 text-center">
                        <div className="text-2xl font-bold text-foreground">
                          {variantPipeline.pipeline.percent_complete}%
                        </div>
                        <div className="text-xs text-muted-foreground">Complete</div>
                      </div>
                    </div>
                    {variantPipeline.fully_complete_days.length > 0 && (
                      <div className="text-sm text-muted-foreground">
                        <span className="font-semibold text-foreground">Fully scripted days:</span>{" "}
                        {variantPipeline.fully_complete_days.join(", ")}
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Click &quot;Refresh Pipeline&quot; to load status.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* View / Generate Controls */}
            <Card>
              <CardHeader>
                <CardTitle>View & Generate Variant Scripts</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Controls Row */}
                <div className="flex flex-wrap items-end gap-3">
                  <div>
                    <label className="mb-1 block text-xs text-muted-foreground">Day</label>
                    <Input
                      type="number"
                      min={1}
                      max={365}
                      value={variantDay}
                      onChange={(e) => setVariantDay(Number(e.target.value))}
                      className="w-20"
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-xs text-muted-foreground">Variant</label>
                    <select
                      value={selectedVariant}
                      onChange={(e) => setSelectedVariant(e.target.value)}
                      className="h-9 rounded-md border border-input bg-background px-3 text-sm text-foreground"
                    >
                      <option value="all">All Variants</option>
                      <option value="kid">Kid (2-10)</option>
                      <option value="adult">Adult (18-64)</option>
                      <option value="elder">Elder (65-102)</option>
                    </select>
                  </div>
                  <Button onClick={viewVariantScripts} disabled={loading} size="sm">
                    {loading ? "Loading..." : "View Scripts"}
                  </Button>
                  <Button
                    onClick={() => generateVariantScripts("single")}
                    disabled={loading}
                    variant="destructive"
                    size="sm"
                  >
                    {loading ? "Generating..." : `Generate Day ${variantDay}`}
                  </Button>
                </div>

                {/* Batch Generate */}
                <div className="flex flex-wrap items-end gap-3 border-t border-border pt-3">
                  <div>
                    <label className="mb-1 block text-xs text-muted-foreground">From</label>
                    <Input
                      type="number"
                      min={1}
                      max={365}
                      value={variantBatchStart}
                      onChange={(e) => setVariantBatchStart(Number(e.target.value))}
                      className="w-20"
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-xs text-muted-foreground">To</label>
                    <Input
                      type="number"
                      min={1}
                      max={365}
                      value={variantBatchEnd}
                      onChange={(e) => setVariantBatchEnd(Number(e.target.value))}
                      className="w-20"
                    />
                  </div>
                  <Button
                    onClick={() => generateVariantScripts("batch")}
                    disabled={loading}
                    variant="destructive"
                    size="sm"
                  >
                    {loading
                      ? "Generating..."
                      : `Batch Generate Days ${variantBatchStart}-${variantBatchEnd}`}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Generation Results */}
            {variantGenResult && (
              <Card>
                <CardHeader>
                  <CardTitle
                    className={variantGenResult.success ? "text-green-600" : "text-red-600"}
                  >
                    {variantGenResult.success ? "Generation Complete" : "Generation Had Errors"}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <p>
                      Mode: {variantGenResult.mode} | Variants: {variantGenResult.variants.join(", ")} |
                      Total scripts: {variantGenResult.total_scripts}
                    </p>
                    {variantGenResult.message && (
                      <p className="text-muted-foreground">{variantGenResult.message}</p>
                    )}
                    <div className="max-h-48 overflow-auto rounded bg-muted p-2 font-mono text-xs">
                      {variantGenResult.results.map((r, i) => (
                        <div
                          key={i}
                          className={r.success ? "text-green-600" : "text-red-600"}
                        >
                          Day {r.day} / {r.variant}: {r.success ? "OK" : r.error}
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* View Scripts Result */}
            {variantScripts && (
              <Card>
                <CardHeader>
                  <CardTitle>
                    Day {variantScripts.day}: {variantScripts.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Object.entries(variantScripts.scripts).map(([variant, phases]) => (
                      <div key={variant} className="rounded-lg border border-border p-3">
                        <h3 className="mb-2 font-semibold capitalize text-foreground">
                          {variant} variant
                          {variantScripts.pipeline_status[variant] && (
                            <span className="ml-2 text-xs font-normal text-muted-foreground">
                              ({variantScripts.pipeline_status[variant].model} |{" "}
                              {new Date(
                                variantScripts.pipeline_status[variant].generated_at,
                              ).toLocaleDateString()})
                            </span>
                          )}
                        </h3>
                        <div className="space-y-2">
                          {Object.entries(phases).map(([phase, data]) => (
                            <div key={phase}>
                              <span className="font-semibold uppercase text-muted-foreground text-xs">
                                {phase}
                              </span>
                              <span className="ml-2 text-xs text-muted-foreground">
                                ({data.word_count} words)
                              </span>
                              <p className="mt-0.5 text-sm text-foreground">{data.text}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {/* Status Check */}
        {activeTab === "status" && (
          <Card>
            <CardHeader>
              <CardTitle>Check Day Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  type="number"
                  min={1}
                  max={365}
                  value={day}
                  onChange={(e) => setDay(Number(e.target.value))}
                  className="w-24"
                />
                <Button onClick={checkStatus} disabled={loading}>
                  {loading ? "Checking..." : "Check"}
                </Button>
              </div>
              {statusResult && (
                <div className="space-y-3 rounded-lg bg-muted p-4 font-mono text-sm">
                  <div>
                    <span className="font-bold text-foreground">
                      Day {statusResult.day}:
                    </span>{" "}
                    {statusResult.title}
                  </div>
                  <div>
                    <span className="font-bold text-foreground">Topic:</span>{" "}
                    {statusResult.topic}
                  </div>
                  <div>
                    <span className="font-bold text-foreground">
                      Curriculum Topic:
                    </span>{" "}
                    {statusResult.curriculum_topic}
                  </div>
                  <div>
                    <span className="font-bold text-foreground">
                      Scripts Aligned:
                    </span>{" "}
                    <span
                      className={
                        statusResult.scripts_aligned
                          ? "text-green-600"
                          : "text-red-600"
                      }
                    >
                      {statusResult.scripts_aligned ? "YES" : "NO - NEEDS REGENERATION"}
                    </span>
                  </div>
                  <div className="border-t border-border pt-2">
                    <div className="font-bold text-foreground">
                      Current Scripts:
                    </div>
                    {Object.entries(statusResult.current_scripts).map(
                      ([key, val]) => (
                        <div key={key} className="mt-1">
                          <span className="font-semibold text-muted-foreground">
                            {key}:
                          </span>{" "}
                          {val || "(empty)"}
                        </div>
                      ),
                    )}
                  </div>
                  <div className="border-t border-border pt-2">
                    <div className="font-bold text-foreground">
                      Kelly Assets:
                    </div>
                    <div>
                      Total: {statusResult.kelly_assets.total} | Audio:{" "}
                      {statusResult.kelly_assets.with_audio} | Video:{" "}
                      {statusResult.kelly_assets.with_video} | Pending:{" "}
                      {statusResult.kelly_assets.pending}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Single Day Regeneration */}
        {activeTab === "single" && (
          <Card>
            <CardHeader>
              <CardTitle>Regenerate Single Day</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  type="number"
                  min={1}
                  max={365}
                  value={day}
                  onChange={(e) => setDay(Number(e.target.value))}
                  className="w-24"
                />
                <Button
                  onClick={regenerateSingle}
                  disabled={loading}
                  variant="destructive"
                >
                  {loading
                    ? "Generating..."
                    : `Regenerate Day ${day} Scripts`}
                </Button>
              </div>
              {result && (
                <div className="space-y-2 rounded-lg bg-muted p-4 font-mono text-sm">
                  <div
                    className={
                      result.success ? "text-green-600" : "text-red-600"
                    }
                  >
                    {result.success ? "SUCCESS" : "FAILED"}:{" "}
                    {result.title}
                  </div>
                  {result.error && <div className="text-red-600">{result.error}</div>}
                  {result.scripts && (
                    <div className="space-y-2 border-t border-border pt-2">
                      {Object.entries(result.scripts).map(
                        ([key, val]) => (
                          <div key={key}>
                            <span className="font-semibold text-foreground">
                              {key}:
                            </span>{" "}
                            {String(val)}
                          </div>
                        ),
                      )}
                    </div>
                  )}
                  {result.message && (
                    <div className="text-muted-foreground">
                      {result.message}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Batch Regeneration */}
        {activeTab === "batch" && (
          <Card>
            <CardHeader>
              <CardTitle>Batch Regenerate Range</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">From:</span>
                <Input
                  type="number"
                  min={1}
                  max={365}
                  value={startDay}
                  onChange={(e) =>
                    setStartDay(Number(e.target.value))
                  }
                  className="w-24"
                />
                <span className="text-sm text-muted-foreground">To:</span>
                <Input
                  type="number"
                  min={1}
                  max={365}
                  value={endDay}
                  onChange={(e) =>
                    setEndDay(Number(e.target.value))
                  }
                  className="w-24"
                />
                <Button
                  onClick={regenerateBatch}
                  disabled={loading}
                  variant="destructive"
                >
                  {loading
                    ? "Regenerating..."
                    : `Regenerate Days ${startDay}-${endDay}`}
                </Button>
              </div>
              {batchResults && (
                <pre className="overflow-auto rounded-lg bg-muted p-4 font-mono text-xs">
                  {JSON.stringify(batchResults, null, 2)}
                </pre>
              )}
            </CardContent>
          </Card>
        )}

        {/* Scan All */}
        {activeTab === "scan" && (
          <Card>
            <CardHeader>
              <CardTitle>Scan All Lessons for Misalignment</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={runScan}
                disabled={loading}
              >
                {loading ? "Scanning..." : "Scan All 365 Days"}
              </Button>
              {scanResult && (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3">
                    <div className="rounded-lg bg-green-50 p-3 text-center dark:bg-green-950">
                      <div className="text-2xl font-bold text-green-700 dark:text-green-300">
                        {scanResult.summary.aligned}
                      </div>
                      <div className="text-xs text-green-600 dark:text-green-400">
                        Aligned
                      </div>
                    </div>
                    <div className="rounded-lg bg-red-50 p-3 text-center dark:bg-red-950">
                      <div className="text-2xl font-bold text-red-700 dark:text-red-300">
                        {scanResult.summary.contaminated}
                      </div>
                      <div className="text-xs text-red-600 dark:text-red-400">
                        Contaminated
                      </div>
                    </div>
                    <div className="rounded-lg bg-amber-50 p-3 text-center dark:bg-amber-950">
                      <div className="text-2xl font-bold text-amber-700 dark:text-amber-300">
                        {scanResult.summary.likely_misaligned}
                      </div>
                      <div className="text-xs text-amber-600 dark:text-amber-400">
                        Likely Misaligned
                      </div>
                    </div>
                  </div>
                  <div className="max-h-96 overflow-auto rounded-lg bg-muted p-3">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-border text-left">
                          <th className="p-1">Day</th>
                          <th className="p-1">Issue</th>
                          <th className="p-1">Title</th>
                          <th className="p-1">Hook Preview</th>
                        </tr>
                      </thead>
                      <tbody>
                        {scanResult.misaligned_days.map((m) => (
                          <tr
                            key={m.day}
                            className="border-b border-border/50"
                          >
                            <td className="p-1 font-mono font-bold">
                              {m.day}
                            </td>
                            <td className="p-1">
                              <span
                                className={`rounded px-1.5 py-0.5 text-xs ${
                                  m.issue === "contaminated"
                                    ? "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                                    : m.issue === "no_scripts"
                                      ? "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200"
                                      : "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200"
                                }`}
                              >
                                {m.issue}
                              </span>
                            </td>
                            <td className="max-w-48 truncate p-1">
                              {m.curriculum_topic}
                            </td>
                            <td className="max-w-48 truncate p-1 text-muted-foreground">
                              {m.hook_preview || "(empty)"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
