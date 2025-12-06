/**
 * Simple In-Memory Rate Limiter
 * 
 * For serverless functions. Not perfect (resets on cold start) but helps.
 * For production scale, use Redis or Vercel's built-in rate limiting.
 */

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

const store = new Map<string, RateLimitEntry>();

export interface RateLimitConfig {
  windowMs?: number;  // Time window in milliseconds
  maxRequests?: number;  // Max requests per window
}

const DEFAULT_CONFIG: Required<RateLimitConfig> = {
  windowMs: 60 * 1000,  // 1 minute
  maxRequests: 60,       // 60 requests per minute
};

/**
 * Check if a request should be rate limited
 * @returns { allowed: boolean, remaining: number, resetIn: number }
 */
export function checkRateLimit(
  identifier: string,
  config: RateLimitConfig = {}
): { allowed: boolean; remaining: number; resetIn: number } {
  const { windowMs, maxRequests } = { ...DEFAULT_CONFIG, ...config };
  const now = Date.now();
  
  // Clean up expired entries
  for (const [key, entry] of store.entries()) {
    if (entry.resetAt < now) {
      store.delete(key);
    }
  }
  
  const entry = store.get(identifier);
  
  if (!entry || entry.resetAt < now) {
    // New window
    store.set(identifier, {
      count: 1,
      resetAt: now + windowMs,
    });
    return { allowed: true, remaining: maxRequests - 1, resetIn: windowMs };
  }
  
  if (entry.count >= maxRequests) {
    return { 
      allowed: false, 
      remaining: 0, 
      resetIn: entry.resetAt - now 
    };
  }
  
  entry.count++;
  return { 
    allowed: true, 
    remaining: maxRequests - entry.count, 
    resetIn: entry.resetAt - now 
  };
}

/**
 * Get client identifier from request
 */
export function getClientId(req: any): string {
  return (
    req.headers['x-forwarded-for']?.split(',')[0]?.trim() ||
    req.headers['x-real-ip'] ||
    req.socket?.remoteAddress ||
    'unknown'
  );
}

/**
 * Create rate limit headers for response
 */
export function rateLimitHeaders(
  remaining: number,
  resetIn: number,
  limit: number = DEFAULT_CONFIG.maxRequests
): Record<string, string> {
  return {
    'X-RateLimit-Limit': String(limit),
    'X-RateLimit-Remaining': String(Math.max(0, remaining)),
    'X-RateLimit-Reset': String(Math.ceil((Date.now() + resetIn) / 1000)),
  };
}

