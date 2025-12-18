/**
 * ZERO TRUST SECURITY LAYER
 * 
 * Every request is untrusted until verified.
 * Every action is logged.
 * Every failure is reported.
 */

import { createClient } from '@supabase/supabase-js';

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════

export const ALLOWED_RECIPIENTS = ['nicoletterankin@gmail.com'] as const;
export const ALLOWED_SENDERS = ['hello@curiouskelly.com'] as const;
export const MAX_EMAILS_PER_HOUR = 10;
export const MAX_DB_WRITES_PER_MINUTE = 100;

// ═══════════════════════════════════════════════════════════════════
// CRON AUTHENTICATION
// ═══════════════════════════════════════════════════════════════════

export interface CronAuthResult {
  authorized: boolean;
  reason?: string;
}

export function verifyCronAuth(req: any): CronAuthResult {
  const authHeader = req.headers.authorization;
  const cronSecret = process.env.CRON_SECRET;
  
  // If no CRON_SECRET configured, allow (development mode)
  if (!cronSecret) {
    console.warn('⚠️ CRON_SECRET not configured - allowing request (dev mode)');
    return { authorized: true, reason: 'dev_mode' };
  }
  
  // Verify bearer token
  if (!authHeader) {
    return { authorized: false, reason: 'missing_auth_header' };
  }
  
  if (!authHeader.startsWith('Bearer ')) {
    return { authorized: false, reason: 'invalid_auth_format' };
  }
  
  const token = authHeader.slice(7);
  if (token !== cronSecret) {
    return { authorized: false, reason: 'invalid_token' };
  }
  
  return { authorized: true };
}

// ═══════════════════════════════════════════════════════════════════
// RATE LIMITING
// ═══════════════════════════════════════════════════════════════════

const rateLimitStore: Map<string, { count: number; resetAt: number }> = new Map();

export function checkRateLimit(key: string, maxPerWindow: number, windowMs: number): boolean {
  const now = Date.now();
  const entry = rateLimitStore.get(key);
  
  if (!entry || entry.resetAt < now) {
    rateLimitStore.set(key, { count: 1, resetAt: now + windowMs });
    return true;
  }
  
  if (entry.count >= maxPerWindow) {
    return false;
  }
  
  entry.count++;
  return true;
}

export function checkEmailRateLimit(): boolean {
  return checkRateLimit('emails', MAX_EMAILS_PER_HOUR, 60 * 60 * 1000);
}

export function checkDbRateLimit(): boolean {
  return checkRateLimit('db_writes', MAX_DB_WRITES_PER_MINUTE, 60 * 1000);
}

// ═══════════════════════════════════════════════════════════════════
// INPUT VALIDATION
// ═══════════════════════════════════════════════════════════════════

export function sanitizeString(input: string, maxLength: number = 2000): string {
  if (typeof input !== 'string') return '';
  
  return input
    .slice(0, maxLength)
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<[^>]*>/g, '')
    .trim();
}

export function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email) && email.length <= 254;
}

export function validateLessonDay(day: number): boolean {
  return Number.isInteger(day) && day >= 1 && day <= 365;
}

export function validateLessonYear(year: number): boolean {
  return Number.isInteger(year) && (year === 1 || year === 2);
}

export function validatePhase(phase: string): boolean {
  const validPhases = ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro'];
  return validPhases.includes(phase);
}

// ═══════════════════════════════════════════════════════════════════
// AUDIT LOGGING
// ═══════════════════════════════════════════════════════════════════

export interface AuditEntry {
  action: string;
  actor: string; // 'system', 'cron', or user_id
  resource: string;
  details?: Record<string, any>;
  success: boolean;
  error?: string;
  ip?: string;
  timestamp: string;
}

export async function logAudit(
  supabase: ReturnType<typeof createClient>,
  entry: Omit<AuditEntry, 'timestamp'>
): Promise<void> {
  try {
    await supabase.from('audit_log').insert({
      ...entry,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    // Audit logging should never throw
    console.error('Audit log failed:', err);
  }
}

// ═══════════════════════════════════════════════════════════════════
// ENVIRONMENT VERIFICATION
// ═══════════════════════════════════════════════════════════════════

export interface EnvCheckResult {
  valid: boolean;
  missing: string[];
  warnings: string[];
}

export function verifyEnvironment(): EnvCheckResult {
  const required = [
    'PUBLIC_SUPABASE_URL',
    'SUPABASE_SERVICE_ROLE_KEY',
    'SENDGRID_API_KEY'
  ];
  
  const recommended = [
    'CRON_SECRET',
    'HEYGEN_API_KEY'
  ];
  
  const missing: string[] = [];
  const warnings: string[] = [];
  
  for (const key of required) {
    if (!process.env[key]) {
      missing.push(key);
    }
  }
  
  for (const key of recommended) {
    if (!process.env[key]) {
      warnings.push(`${key} not set - some features may be limited`);
    }
  }
  
  return {
    valid: missing.length === 0,
    missing,
    warnings
  };
}

// ═══════════════════════════════════════════════════════════════════
// CIRCUIT BREAKER
// ═══════════════════════════════════════════════════════════════════

interface CircuitState {
  failures: number;
  lastFailure: number;
  state: 'closed' | 'open' | 'half-open';
}

const circuits: Map<string, CircuitState> = new Map();
const FAILURE_THRESHOLD = 5;
const RECOVERY_TIME_MS = 60 * 1000; // 1 minute

export function checkCircuit(name: string): boolean {
  const circuit = circuits.get(name);
  
  if (!circuit) {
    circuits.set(name, { failures: 0, lastFailure: 0, state: 'closed' });
    return true;
  }
  
  if (circuit.state === 'open') {
    // Check if recovery time has passed
    if (Date.now() - circuit.lastFailure > RECOVERY_TIME_MS) {
      circuit.state = 'half-open';
      return true;
    }
    return false;
  }
  
  return true;
}

export function recordSuccess(name: string): void {
  const circuit = circuits.get(name);
  if (circuit) {
    circuit.failures = 0;
    circuit.state = 'closed';
  }
}

export function recordFailure(name: string): void {
  const circuit = circuits.get(name) || { failures: 0, lastFailure: 0, state: 'closed' as const };
  circuit.failures++;
  circuit.lastFailure = Date.now();
  
  if (circuit.failures >= FAILURE_THRESHOLD) {
    circuit.state = 'open';
  }
  
  circuits.set(name, circuit);
}

// ═══════════════════════════════════════════════════════════════════
// REQUEST FINGERPRINTING
// ═══════════════════════════════════════════════════════════════════

export function getRequestFingerprint(req: any): string {
  const ip = req.headers['x-forwarded-for'] || req.headers['x-real-ip'] || 'unknown';
  const userAgent = req.headers['user-agent'] || 'unknown';
  const timestamp = Math.floor(Date.now() / 1000 / 60); // minute precision
  
  return `${ip}-${userAgent.slice(0, 50)}-${timestamp}`;
}

// ═══════════════════════════════════════════════════════════════════
// SAFE EXECUTION WRAPPER
// ═══════════════════════════════════════════════════════════════════

export async function safeExecute<T>(
  name: string,
  fn: () => Promise<T>,
  fallback: T
): Promise<{ result: T; success: boolean; error?: string }> {
  if (!checkCircuit(name)) {
    return { result: fallback, success: false, error: 'circuit_open' };
  }
  
  try {
    const result = await fn();
    recordSuccess(name);
    return { result, success: true };
  } catch (err) {
    recordFailure(name);
    return { result: fallback, success: false, error: String(err) };
  }
}
