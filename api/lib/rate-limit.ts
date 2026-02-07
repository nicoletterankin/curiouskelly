/**
 * Simple Rate Limiter for Checkout Endpoints
 * 
 * Uses in-memory storage. For production scale, upgrade to Redis.
 * This prevents abuse like:
 * - Brute force price/plan enumeration
 * - Spam checkout attempts
 * - Card testing attacks
 */

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

// In-memory store (resets on serverless cold start, which is acceptable)
const store = new Map<string, RateLimitEntry>();

// Cleanup old entries every 5 minutes
const CLEANUP_INTERVAL = 5 * 60 * 1000;
let lastCleanup = Date.now();

function cleanup() {
  const now = Date.now();
  if (now - lastCleanup < CLEANUP_INTERVAL) return;
  
  lastCleanup = now;
  for (const [key, entry] of store.entries()) {
    if (entry.resetAt < now) {
      store.delete(key);
    }
  }
}

export interface RateLimitConfig {
  /** Max requests allowed in the window */
  limit: number;
  /** Window size in seconds */
  windowSecs: number;
  /** Identifier for this limiter (e.g., 'checkout', 'gift') */
  prefix?: string;
}

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: Date;
  retryAfterSecs?: number;
}

/**
 * Check and consume rate limit for a given identifier
 * 
 * @param identifier - Unique key (e.g., IP address, email, or composite)
 * @param config - Rate limit configuration
 * @returns Whether the request is allowed
 */
export function checkRateLimit(
  identifier: string,
  config: RateLimitConfig
): RateLimitResult {
  cleanup();
  
  const { limit, windowSecs, prefix = 'rl' } = config;
  const key = `${prefix}:${identifier}`;
  const now = Date.now();
  const windowMs = windowSecs * 1000;
  
  let entry = store.get(key);
  
  // If no entry or window expired, create new entry
  if (!entry || entry.resetAt < now) {
    entry = {
      count: 1,
      resetAt: now + windowMs,
    };
    store.set(key, entry);
    
    return {
      allowed: true,
      remaining: limit - 1,
      resetAt: new Date(entry.resetAt),
    };
  }
  
  // Increment count
  entry.count += 1;
  
  if (entry.count > limit) {
    const retryAfterSecs = Math.ceil((entry.resetAt - now) / 1000);
    return {
      allowed: false,
      remaining: 0,
      resetAt: new Date(entry.resetAt),
      retryAfterSecs,
    };
  }
  
  return {
    allowed: true,
    remaining: limit - entry.count,
    resetAt: new Date(entry.resetAt),
  };
}

/**
 * Get rate limit headers for response
 */
export function getRateLimitHeaders(result: RateLimitResult, limit: number): Record<string, string> {
  return {
    'X-RateLimit-Limit': String(limit),
    'X-RateLimit-Remaining': String(result.remaining),
    'X-RateLimit-Reset': String(Math.floor(result.resetAt.getTime() / 1000)),
    ...(result.retryAfterSecs ? { 'Retry-After': String(result.retryAfterSecs) } : {}),
  };
}

/**
 * Default rate limit configurations
 */
export const RATE_LIMITS = {
  // Checkout: 10 attempts per email per 15 minutes
  checkout: { limit: 10, windowSecs: 15 * 60, prefix: 'checkout' } as RateLimitConfig,
  
  // Gift checkout: 5 attempts per sender email per 15 minutes
  giftCheckout: { limit: 5, windowSecs: 15 * 60, prefix: 'gift' } as RateLimitConfig,
  
  // Portal session: 20 attempts per user per 15 minutes
  portal: { limit: 20, windowSecs: 15 * 60, prefix: 'portal' } as RateLimitConfig,
  
  // Cancel: 5 attempts per user per hour
  cancel: { limit: 5, windowSecs: 60 * 60, prefix: 'cancel' } as RateLimitConfig,
  
  // Gift redeem: 3 attempts per IP per 15 minutes
  giftRedeem: { limit: 3, windowSecs: 15 * 60, prefix: 'redeem' } as RateLimitConfig,
  
  // Referral tracking: 100 per IP per minute (high volume expected)
  referralTrack: { limit: 100, windowSecs: 60, prefix: 'ref' } as RateLimitConfig,
  
  // Contact form: 3 per IP per hour
  contact: { limit: 3, windowSecs: 60 * 60, prefix: 'contact' } as RateLimitConfig,
  
  // Feedback: 10 per user per minute
  feedback: { limit: 10, windowSecs: 60, prefix: 'feedback' } as RateLimitConfig,
  
  // Lesson complete: 20 per user per minute
  lessonComplete: { limit: 20, windowSecs: 60, prefix: 'lesson' } as RateLimitConfig,
  
  // Email subscribe: 5 per IP per hour
  emailSubscribe: { limit: 5, windowSecs: 60 * 60, prefix: 'subscribe' } as RateLimitConfig,
  
  // API general: 100 per IP per minute
  apiGeneral: { limit: 100, windowSecs: 60, prefix: 'api' } as RateLimitConfig,
};

/**
 * Extract client identifier from request
 * Prioritizes: email > user ID > IP address
 */
export function getClientIdentifier(
  req: { headers: Record<string, string | string[] | undefined> },
  email?: string,
  userId?: string
): string {
  if (email) return email.toLowerCase().trim();
  if (userId) return userId;
  
  // Try to get real IP from various headers
  const forwarded = req.headers['x-forwarded-for'];
  if (forwarded) {
    const ips = Array.isArray(forwarded) ? forwarded[0] : forwarded.split(',')[0];
    return ips.trim();
  }
  
  const realIp = req.headers['x-real-ip'];
  if (realIp) {
    return Array.isArray(realIp) ? realIp[0] : realIp;
  }
  
  const cfIp = req.headers['cf-connecting-ip'];
  if (cfIp) {
    return Array.isArray(cfIp) ? cfIp[0] : cfIp;
  }
  
  return 'unknown';
}
