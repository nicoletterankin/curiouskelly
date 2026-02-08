/**
 * Database connection for KellyOS
 * Uses @neondatabase/serverless for edge compatibility
 */
import { Pool, neon } from '@neondatabase/serverless';

let pool: Pool | null = null;

export function getPool(): Pool {
  if (!pool) {
    // PRIORITY ORDER: soft-block (NEON_DATABASE_URL) first, then POSTGRES_URL, then wispy-resonance (DATABASE_URL) last
    const connectionString = process.env.NEON_DATABASE_URL || process.env.POSTGRES_URL || process.env.DATABASE_URL;
    if (!connectionString) {
      throw new Error('NEON_DATABASE_URL, POSTGRES_URL, or DATABASE_URL environment variable is required');
    }
    pool = new Pool({ connectionString });
  }
  return pool;
}

export function getSQL() {
  // PRIORITY ORDER: soft-block (NEON_DATABASE_URL) first, then POSTGRES_URL, then wispy-resonance (DATABASE_URL) last
  const connectionString = process.env.NEON_DATABASE_URL || process.env.POSTGRES_URL || process.env.DATABASE_URL;
  if (!connectionString) {
    throw new Error('NEON_DATABASE_URL, POSTGRES_URL, or DATABASE_URL environment variable is required');
  }
  return neon(connectionString);
}

export type { Pool };
