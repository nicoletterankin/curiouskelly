import { neon, neonConfig, NeonQueryFunction } from '@neondatabase/serverless'

// Disable browser warning since we're using API routes (server-side)
neonConfig.disableWarningInBrowsers = true

// Lazy-initialized Neon client - only connects when first used (not at build time)
let _baseSql: NeonQueryFunction<false, false> | null = null

function getBaseSql(): NeonQueryFunction<false, false> {
  if (!_baseSql) {
    // PRIORITY ORDER: soft-block (NEON_DATABASE_URL) first, then POSTGRES_URL, then wispy-resonance (DATABASE_URL) last
    const connectionString = process.env.NEON_DATABASE_URL || process.env.POSTGRES_URL || process.env.DATABASE_URL
    if (!connectionString) {
      throw new Error('[db] Missing NEON_DATABASE_URL, POSTGRES_URL, or DATABASE_URL environment variable')
    }
    _baseSql = neon(connectionString)
  }
  return _baseSql
}

// Create a wrapped sql function that handles rate limiting
// This returns empty array on rate limit instead of throwing
type SqlResult = Record<string, unknown>[]

const createRateLimitedSql = () => {
  const handler = {
    apply: async (_target: unknown, _thisArg: unknown, args: unknown[]) => {
      try {
        const baseSql = getBaseSql()
        // @ts-expect-error - spread args to template literal function
        return await baseSql.apply(null, args)
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
  // Create a proxy that looks like a template literal function
  const proxyTarget = (strings: TemplateStringsArray, ...values: unknown[]) => {
    const baseSql = getBaseSql()
    return baseSql(strings, ...values)
  }
  return new Proxy(proxyTarget, handler) as NeonQueryFunction<false, false>
}

// Export rate-limit-safe sql client (lazy-initialized)
export const sql = createRateLimitedSql()

// Export getBaseSql for direct access when needed
export { getBaseSql }

// Export a function to get a fresh sql connection (for API routes that need their own)
export function getSql() {
  return getBaseSql()
}

// Type-safe query helper (legacy support)
export async function query<T>(
  queryText: string,
  params?: unknown[]
): Promise<T[]> {
  try {
    const baseSql = getBaseSql()
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
