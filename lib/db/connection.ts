/**
 * Database connection for KellyOS
 * Uses @neondatabase/serverless for edge compatibility
 */
import { Pool, neon } from '@neondatabase/serverless';

let pool: Pool | null = null;

export function getPool(): Pool {
  if (!pool) {
    const connectionString = process.env.DATABASE_URL;
    if (!connectionString) {
      throw new Error('DATABASE_URL environment variable is required');
    }
    pool = new Pool({ connectionString });
  }
  return pool;
}

export function getSQL() {
  const connectionString = process.env.DATABASE_URL;
  if (!connectionString) {
    throw new Error('DATABASE_URL environment variable is required');
  }
  return neon(connectionString);
}

export type { Pool };
