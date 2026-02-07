/**
 * CORS Utility for API Endpoints
 * 
 * Provides consistent CORS handling across all API routes.
 * Restricts origins in production, allows localhost in development.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

// Production allowed origins
const PRODUCTION_ORIGINS = [
  'https://curiouskelly.com',
  'https://www.curiouskelly.com',
  'https://app.curiouskelly.com',
  'https://daily-lesson.vercel.app',
];

// Development origins (only allowed when NODE_ENV !== 'production')
const DEVELOPMENT_ORIGINS = [
  'http://localhost:3000',
  'http://localhost:4321',
  'http://localhost:5173',
  'http://127.0.0.1:3000',
  'http://127.0.0.1:4321',
  'http://127.0.0.1:5173',
];

// Vercel preview URL pattern
const VERCEL_PREVIEW_PATTERN = /^https:\/\/.*\.vercel\.app$/;

export interface CorsConfig {
  allowCredentials?: boolean;
  allowedMethods?: string[];
  allowedHeaders?: string[];
  maxAge?: number;
  allowAllOrigins?: boolean; // Override for public endpoints
}

const DEFAULT_CONFIG: CorsConfig = {
  allowCredentials: true,
  allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  maxAge: 86400, // 24 hours
  allowAllOrigins: false,
};

/**
 * Check if an origin is allowed
 */
export function isAllowedOrigin(origin: string | undefined, config: CorsConfig = {}): boolean {
  if (!origin) return false;
  
  // Allow all origins if configured (for truly public endpoints)
  if (config.allowAllOrigins) return true;
  
  // Always allow production origins
  if (PRODUCTION_ORIGINS.includes(origin)) return true;
  
  // Allow Vercel preview deployments
  if (VERCEL_PREVIEW_PATTERN.test(origin)) return true;
  
  // Allow development origins only in non-production
  if (process.env.NODE_ENV !== 'production') {
    if (DEVELOPMENT_ORIGINS.includes(origin)) return true;
  }
  
  return false;
}

/**
 * Get the allowed origin header value
 */
export function getAllowedOrigin(req: VercelRequest, config: CorsConfig = {}): string {
  const origin = req.headers.origin;
  
  if (config.allowAllOrigins) {
    return '*';
  }
  
  if (origin && isAllowedOrigin(origin, config)) {
    return origin;
  }
  
  // Default to primary production origin
  return PRODUCTION_ORIGINS[0];
}

/**
 * Set CORS headers on response
 */
export function setCorsHeaders(
  req: VercelRequest,
  res: VercelResponse,
  config: CorsConfig = {}
): void {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  
  const allowedOrigin = getAllowedOrigin(req, cfg);
  
  res.setHeader('Access-Control-Allow-Origin', allowedOrigin);
  
  if (cfg.allowCredentials && allowedOrigin !== '*') {
    res.setHeader('Access-Control-Allow-Credentials', 'true');
  }
  
  res.setHeader(
    'Access-Control-Allow-Methods',
    cfg.allowedMethods!.join(', ')
  );
  
  res.setHeader(
    'Access-Control-Allow-Headers',
    cfg.allowedHeaders!.join(', ')
  );
  
  if (cfg.maxAge) {
    res.setHeader('Access-Control-Max-Age', cfg.maxAge.toString());
  }
  
  // Vary header for proper caching
  res.setHeader('Vary', 'Origin');
}

/**
 * Handle CORS preflight (OPTIONS) request
 * 
 * Returns true if this was a preflight request (caller should return early)
 */
export function handlePreflight(
  req: VercelRequest,
  res: VercelResponse,
  config: CorsConfig = {}
): boolean {
  setCorsHeaders(req, res, config);
  
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return true;
  }
  
  return false;
}

/**
 * Check if request origin is allowed, return 403 if not
 * 
 * Returns true if request should proceed, false if blocked
 */
export function enforceOrigin(
  req: VercelRequest,
  res: VercelResponse,
  config: CorsConfig = {}
): boolean {
  const origin = req.headers.origin;
  
  // No origin header (same-origin request or non-browser) - allow
  if (!origin) return true;
  
  // Allow all origins if configured
  if (config.allowAllOrigins) return true;
  
  if (!isAllowedOrigin(origin, config)) {
    res.status(403).json({
      error: 'forbidden',
      message: 'Origin not allowed',
      origin,
    });
    return false;
  }
  
  return true;
}

/**
 * Complete CORS middleware
 * 
 * Usage:
 * ```ts
 * export default async function handler(req, res) {
 *   if (!cors(req, res)) return;
 *   // ... rest of handler
 * }
 * ```
 */
export function cors(
  req: VercelRequest,
  res: VercelResponse,
  config: CorsConfig = {}
): boolean {
  // Set CORS headers
  setCorsHeaders(req, res, config);
  
  // Handle preflight
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return false;
  }
  
  // Enforce origin (skip for allowAllOrigins)
  if (!config.allowAllOrigins) {
    return enforceOrigin(req, res, config);
  }
  
  return true;
}

/**
 * Public CORS - allows all origins (use sparingly)
 */
export function publicCors(req: VercelRequest, res: VercelResponse): boolean {
  return cors(req, res, { allowAllOrigins: true });
}

/**
 * Strict CORS - production origins only, no credentials
 */
export function strictCors(req: VercelRequest, res: VercelResponse): boolean {
  return cors(req, res, { 
    allowCredentials: false,
    allowAllOrigins: false,
  });
}
