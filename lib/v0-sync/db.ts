import { neon, neonConfig, NeonQueryFunction } from '@neondatabase/serverless'

// Disable browser warning since we're using API routes (server-side)
neonConfig.disableWarningInBrowsers = true

// Create base Neon client — soft-block (NEON_DATABASE_URL) first, wispy-resonance (DATABASE_URL) last
const baseSql = neon(process.env.NEON_DATABASE_URL || process.env.POSTGRES_URL || process.env.DATABASE_URL!)

// Create a wrapped sql function that handles rate limiting
// This returns empty array on rate limit instead of throwing
type SqlResult = Record<string, unknown>[]

const createRateLimitedSql = (base: NeonQueryFunction<false, false>) => {
  const handler = {
    apply: async (target: NeonQueryFunction<false, false>, thisArg: unknown, args: unknown[]) => {
      try {
        // @ts-expect-error - spread args to template literal function
        return await target.apply(thisArg, args)
      } catch (error) {
        if (error instanceof Error) {
          const msg = error.message.toLowerCase()
          if (msg.includes('too many') || msg.includes('rate') || msg.includes('429')) {
            console.warn('[db] Rate limited, returning empty result')
            return [] as SqlResult
          }
        }
        throw error
      }
    }
  }
  return new Proxy(base, handler) as typeof base
}

// Export rate-limit-safe sql client
export const sql = createRateLimitedSql(baseSql)

// Type-safe query helper (legacy support)
export async function query<T>(
  queryText: string,
  params?: unknown[]
): Promise<T[]> {
  try {
    const result = await baseSql(queryText, params)
    return result as T[]
  } catch (error) {
    if (error instanceof Error && error.message.includes('Too Many')) {
      console.warn('[db] Rate limited, returning empty result')
      return []
    }
    throw error
  }
}
