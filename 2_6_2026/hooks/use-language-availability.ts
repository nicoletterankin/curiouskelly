import useSWR from 'swr'

const fetcher = (url: string) => fetch(url).then(r => r.json())

/**
 * Hook to check which languages have audio available for a given day.
 * Auto-refreshes every 60 seconds.
 * 
 * Returns: { en: true, es: false, fr: false, ... }
 */
export function useLanguageAvailability(day: number) {
  const { data, error, isLoading } = useSWR(
    `/api/antigravity/webhook?day=${day}`,
    fetcher,
    {
      refreshInterval: 60000, // Check every minute for new assets
      revalidateOnFocus: false,
      dedupingInterval: 30000,
      fallbackData: {
        day,
        status: {
          en: { audio: 5, video: 0 },
          es: { audio: 0, video: 0 },
          fr: { audio: 0, video: 0 },
          de: { audio: 0, video: 0 },
          pt: { audio: 0, video: 0 },
          zh: { audio: 0, video: 0 },
        }
      }
    }
  )

  // Language is available if it has at least 1 audio file
  // (ideally 5 for all phases, but 1 is enough to enable)
  const availability: Record<string, boolean> = {
    en: true, // Always true as fallback
    es: false,
    fr: false,
    de: false,
    pt: false,
    zh: false,
  }

  if (data?.status) {
    for (const [lang, counts] of Object.entries(data.status)) {
      const c = counts as { audio: number; video: number }
      availability[lang] = c.audio >= 1
    }
  }

  return {
    availability,
    isLoading,
    error,
    // Detailed counts for UI if needed
    counts: data?.status || {}
  }
}
