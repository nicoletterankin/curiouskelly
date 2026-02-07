import useSWR from 'swr'
import type { LessonPhase } from '@/lib/types'

interface Visual {
  id: string
  day: number
  phase: LessonPhase
  type: string
  url: string
  prompt?: string
  approved: boolean
  source: string
  title?: string
  ageGroup?: string
  archetype?: string
  variants?: Record<string, unknown>
  data?: Record<string, unknown>
}

interface VisualAidsResponse {
  day: number
  phase: string
  visuals: Visual[]
  count: number
  hasVisuals: boolean
}

const fetcher = async (url: string): Promise<VisualAidsResponse> => {
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error('Failed to fetch visual aids')
  }
  return res.json()
}

/**
 * Hook to fetch visual aids for a specific day and phase
 */
export function useVisualAids(dayOfYear: number, phase?: LessonPhase) {
  const params = new URLSearchParams()
  params.set('day', dayOfYear.toString())
  if (phase) params.set('phase', phase)

  const { data, error, isLoading, mutate } = useSWR<VisualAidsResponse>(
    dayOfYear > 0 ? `/api/visual?${params.toString()}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 60000, // Cache for 1 minute
    }
  )

  // Get the primary visual for the requested phase
  const primaryVisual = data?.visuals?.find(v => v.phase === phase) || data?.visuals?.[0]

  return {
    visuals: data?.visuals || [],
    primaryVisual,
    visualUrl: primaryVisual?.url || null,
    hasVisuals: data?.hasVisuals || false,
    count: data?.count || 0,
    isLoading,
    error,
    refresh: mutate,
  }
}

/**
 * Hook to get all visuals for a day (for preview/gallery)
 */
export function useDayVisuals(dayOfYear: number) {
  const { data, error, isLoading } = useSWR<VisualAidsResponse>(
    dayOfYear > 0 ? `/api/visual?day=${dayOfYear}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 60000,
    }
  )

  // Group by phase
  const byPhase: Record<LessonPhase, Visual | null> = {
    hook: data?.visuals?.find(v => v.phase === 'hook') || null,
    story: data?.visuals?.find(v => v.phase === 'story') || null,
    wonder: data?.visuals?.find(v => v.phase === 'wonder') || null,
    action: data?.visuals?.find(v => v.phase === 'action') || null,
    wisdom: data?.visuals?.find(v => v.phase === 'wisdom') || null,
  }

  return {
    visuals: data?.visuals || [],
    byPhase,
    hasVisuals: data?.hasVisuals || false,
    isLoading,
    error,
  }
}
