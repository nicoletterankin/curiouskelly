'use client'

import React from "react"

import { useState, useCallback } from 'react'
import { Upload, CheckCircle, XCircle, Loader2, FolderUp } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface UploadResult {
  filename: string
  success: boolean
  url?: string
  error?: string
}

export default function UploadVideosPage() {
  const [results, setResults] = useState<UploadResult[]>([])
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState({ current: 0, total: 0 })

  // Parse filename like: day21_de_senior_macgyver_action.mp4
  const parseFilename = (filename: string) => {
    // Pattern: day{day}_{language}_{age}_{archetype}_{phase}.mp4
    const match = filename.match(/day(\d+)_([a-z]{2})_([a-z]+)_([a-z]+)_([a-z]+)\.mp4/i)
    if (!match) return null
    
    return {
      day: parseInt(match[1]),
      language: match[2],
      age: match[3],
      archetype: match[4],
      phase: match[5],
    }
  }

  const uploadFiles = useCallback(async (files: FileList) => {
    setUploading(true)
    setProgress({ current: 0, total: files.length })
    const newResults: UploadResult[] = []

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      setProgress({ current: i + 1, total: files.length })

      const parsed = parseFilename(file.name)
      if (!parsed) {
        newResults.push({ 
          filename: file.name, 
          success: false, 
          error: 'Invalid filename format. Expected: day{N}_{lang}_{age}_{archetype}_{phase}.mp4' 
        })
        continue
      }

      try {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('day', String(parsed.day))
        formData.append('phase', parsed.phase)
        formData.append('language', parsed.language)
        formData.append('age', parsed.age)
        formData.append('archetype', parsed.archetype)

        const res = await fetch('/api/admin/upload-heygen-video', {
          method: 'POST',
          body: formData,
        })

        if (res.ok) {
          const data = await res.json()
          newResults.push({ filename: file.name, success: true, url: data.url })
        } else {
          const err = await res.json()
          newResults.push({ filename: file.name, success: false, error: err.error || 'Upload failed' })
        }
      } catch (err) {
        newResults.push({ filename: file.name, success: false, error: String(err) })
      }

      setResults([...newResults])
    }

    setUploading(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files.length > 0) {
      uploadFiles(e.dataTransfer.files)
    }
  }, [uploadFiles])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadFiles(e.target.files)
    }
  }, [uploadFiles])

  const successCount = results.filter(r => r.success).length
  const failCount = results.filter(r => !r.success).length

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">Upload HeyGen Videos</h1>
        <p className="text-white/60 mb-8">
          Drag and drop MP4 files or select a folder. Files must be named like:<br/>
          <code className="text-sm bg-white/10 px-2 py-1 rounded">day21_de_senior_macgyver_action.mp4</code>
        </p>

        {/* Drop Zone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className="border-2 border-dashed border-white/20 rounded-2xl p-12 text-center hover:border-white/40 transition-colors mb-8"
        >
          {uploading ? (
            <div className="flex flex-col items-center gap-4">
              <Loader2 className="w-12 h-12 animate-spin text-white/60" />
              <p className="text-lg">Uploading {progress.current} of {progress.total}...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <FolderUp className="w-12 h-12 text-white/40" />
              <p className="text-lg text-white/60">Drop video files here</p>
              <label className="cursor-pointer">
                <input
                  type="file"
                  multiple
                  accept="video/mp4"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <Button variant="outline" className="gap-2 bg-transparent">
                  <Upload className="w-4 h-4" />
                  Select Files
                </Button>
              </label>
            </div>
          )}
        </div>

        {/* Results Summary */}
        {results.length > 0 && (
          <div className="mb-4 flex gap-4">
            <div className="flex items-center gap-2 text-green-400">
              <CheckCircle className="w-5 h-5" />
              {successCount} uploaded
            </div>
            {failCount > 0 && (
              <div className="flex items-center gap-2 text-red-400">
                <XCircle className="w-5 h-5" />
                {failCount} failed
              </div>
            )}
          </div>
        )}

        {/* Results List */}
        {results.length > 0 && (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {results.map((result, i) => (
              <div
                key={i}
                className={`flex items-center gap-3 p-3 rounded-lg ${
                  result.success ? 'bg-green-500/10' : 'bg-red-500/10'
                }`}
              >
                {result.success ? (
                  <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                )}
                <span className="font-mono text-sm truncate flex-1">{result.filename}</span>
                {result.error && (
                  <span className="text-xs text-red-400">{result.error}</span>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Instructions */}
        <div className="mt-12 p-6 bg-white/5 rounded-xl">
          <h2 className="font-semibold mb-4">Filename Format</h2>
          <p className="text-white/60 text-sm mb-4">
            Videos must follow this naming convention:
          </p>
          <code className="block bg-black/50 p-4 rounded-lg text-sm mb-4">
            day{'{number}'}_{'{language}'}_{'{age}'}_{'{archetype}'}_{'{phase}'}.mp4
          </code>
          <div className="grid grid-cols-2 gap-4 text-sm text-white/60">
            <div>
              <strong className="text-white">Languages:</strong> en, es, fr, de, pt, zh, hi
            </div>
            <div>
              <strong className="text-white">Phases:</strong> hook, story, wonder, action, wisdom
            </div>
            <div>
              <strong className="text-white">Ages:</strong> kid, teen, adult, senior
            </div>
            <div>
              <strong className="text-white">Archetypes:</strong> scientist, storyteller, explorer, etc.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
