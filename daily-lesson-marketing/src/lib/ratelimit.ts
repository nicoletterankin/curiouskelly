/**
 * Rate Limiting for Curious Kelly API
 * Designed for massive traffic (think AnySphere/Cursor scale)
 * 
 * Uses in-memory store with LRU eviction
 * For production at scale, swap to Redis/Upstash
 */

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

// In-memory store with max size to prevent memory leaks
const MAX_ENTRIES = 100000;
const store = new Map<string, RateLimitEntry>();

// LRU eviction when store gets too large
function evictOldest() {
  if (store.size <= MAX_ENTRIES) return;
  
  // Delete oldest 10% of entries
  const entriesToDelete = Math.floor(MAX_ENTRIES * 0.1);
  const iterator = store.keys();
  for (let i = 0; i < entriesToDelete; i++) {
    const key = iterator.next().value;
    if (key) store.delete(key);
  }
}

// Clean up expired entries periodically
function cleanupExpired() {
  const now = Date.now();
  for (const [key, entry] of store.entries()) {
    if (entry.resetAt < now) {
      store.delete(key);
    }
  }
}

// Run cleanup every minute
if (typeof setInterval !== 'undefined') {
  setInterval(cleanupExpired, 60000);
}

export interface RateLimitConfig {
  maxRequests: number;  // Max requests per window
  windowMs: number;     // Window size in milliseconds
  keyPrefix?: string;   // Prefix for the rate limit key
}

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: number;
  retryAfter?: number;  // Seconds until retry (if blocked)
}

/**
 * Check rate limit for a given identifier
 */
export function checkRateLimit(
  identifier: string,
  config: RateLimitConfig
): RateLimitResult {
  const now = Date.now();
  const key = config.keyPrefix ? `${config.keyPrefix}:${identifier}` : identifier;
  
  let entry = store.get(key);
  
  // If no entry or window expired, create new
  if (!entry || entry.resetAt < now) {
    entry = {
      count: 1,
      resetAt: now + config.windowMs,
    };
    store.set(key, entry);
    evictOldest();
    
    return {
      allowed: true,
      remaining: config.maxRequests - 1,
      resetAt: entry.resetAt,
    };
  }
  
  // Increment count
  entry.count++;
  
  if (entry.count > config.maxRequests) {
    return {
      allowed: false,
      remaining: 0,
      resetAt: entry.resetAt,
      retryAfter: Math.ceil((entry.resetAt - now) / 1000),
    };
  }
  
  return {
    allowed: true,
    remaining: config.maxRequests - entry.count,
    resetAt: entry.resetAt,
  };
}

/**
 * Get client identifier from request
 * Uses X-Forwarded-For for proxied requests, falls back to IP
 */
export function getClientIdentifier(request: Request): string {
  // Try to get real IP from proxy headers
  const forwarded = request.headers.get('x-forwarded-for');
  if (forwarded) {
    // X-Forwarded-For can contain multiple IPs, take the first (original client)
    return forwarded.split(',')[0].trim();
  }
  
  const realIp = request.headers.get('x-real-ip');
  if (realIp) {
    return realIp;
  }
  
  // Cloudflare
  const cfConnectingIp = request.headers.get('cf-connecting-ip');
  if (cfConnectingIp) {
    return cfConnectingIp;
  }
  
  // Vercel
  const vercelIp = request.headers.get('x-vercel-forwarded-for');
  if (vercelIp) {
    return vercelIp.split(',')[0].trim();
  }
  
  // Fallback to a hash of user agent + accept headers (fingerprint)
  const ua = request.headers.get('user-agent') || 'unknown';
  const accept = request.headers.get('accept') || '';
  return `fp:${simpleHash(ua + accept)}`;
}

/**
 * Simple hash function for fingerprinting
 */
function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash).toString(36);
}

/**
 * Rate limit middleware result as Response (if blocked)
 */
export function rateLimitResponse(result: RateLimitResult): Response | null {
  if (result.allowed) {
    return null;
  }
  
  return new Response(
    JSON.stringify({
      error: 'Too many requests',
      message: 'Please slow down. Try again in a moment.',
      retryAfter: result.retryAfter,
    }),
    {
      status: 429,
      headers: {
        'Content-Type': 'application/json',
        'Retry-After': String(result.retryAfter || 60),
        'X-RateLimit-Remaining': '0',
        'X-RateLimit-Reset': String(Math.floor(result.resetAt / 1000)),
      },
    }
  );
}

/**
 * Add rate limit headers to response
 */
export function addRateLimitHeaders(
  response: Response,
  result: RateLimitResult,
  limit: number
): Response {
  const headers = new Headers(response.headers);
  headers.set('X-RateLimit-Limit', String(limit));
  headers.set('X-RateLimit-Remaining', String(result.remaining));
  headers.set('X-RateLimit-Reset', String(Math.floor(result.resetAt / 1000)));
  
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

// ============================================
// PRESET CONFIGURATIONS
// ============================================

export const RATE_LIMITS = {
  // Auth endpoints: 10 requests per minute
  auth: {
    maxRequests: 10,
    windowMs: 60 * 1000,
    keyPrefix: 'auth',
  },
  
  // Checkout: 5 requests per minute (prevent spam)
  checkout: {
    maxRequests: 5,
    windowMs: 60 * 1000,
    keyPrefix: 'checkout',
  },
  
  // Webhook (from Stripe): 100 per minute (high volume)
  webhook: {
    maxRequests: 100,
    windowMs: 60 * 1000,
    keyPrefix: 'webhook',
  },
  
  // API general: 60 requests per minute
  api: {
    maxRequests: 60,
    windowMs: 60 * 1000,
    keyPrefix: 'api',
  },
  
  // Lesson content: 120 per minute (users browsing)
  lessons: {
    maxRequests: 120,
    windowMs: 60 * 1000,
    keyPrefix: 'lessons',
  },
} as const;

