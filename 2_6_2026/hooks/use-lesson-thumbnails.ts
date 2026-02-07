'use client'

import useSWR from 'swr'

interface LessonThumbnail {
  dayOfYear: number
  title: string
  thumbnailUrl: string | null
}

const fetcher = async (url: string) => {
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch')
  return res.json()
}

/**
 * Fetches lesson thumbnails for a range of days
 * Uses kelly_lesson_assets.main_visual_url or falls back to teaser_images
 */
export function useLessonThumbnails(startDay: number, endDay: number) {
  const { data, error, isLoading } = useSWR<{ thumbnails: LessonThumbnail[] }>(
    `/api/lessons/thumbnails?start=${startDay}&end=${endDay}`,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 60000, // Cache for 1 minute
    }
  )

  // Create a map for O(1) lookups
  const thumbnailMap = new Map<number, LessonThumbnail>()
  if (data?.thumbnails) {
    for (const thumb of data.thumbnails) {
      thumbnailMap.set(thumb.dayOfYear, thumb)
    }
  }

  return {
    thumbnails: thumbnailMap,
    isLoading,
    error,
  }
}
