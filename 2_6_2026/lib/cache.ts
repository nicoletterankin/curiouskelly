// Simple in-memory cache for lessons
// FULLY DISABLED - was causing stale data issues with "Why Everything Changes" 
// showing instead of correct "Why the Sun Sets Red" for Day 30

// Empty Map that never gets populated
export const lessonCache = new Map<number, { data: unknown; timestamp: number }>()
export const CACHE_TTL = 0 // Never cache

// Clear any stale data on module load
lessonCache.clear()
