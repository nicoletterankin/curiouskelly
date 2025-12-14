/**
 * /api/motion-progress.ts - Kelly Motion Library Progress
 *
 * Server-side progress endpoint for the motion generation dashboard.
 * Uses Supabase service role (no client-side keys required).
 *
 * NOTE: This repo's Vercel config runs api/*.ts as Node functions.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin, isSupabaseConfigured } from './lib/supabase';

type MotionStatus = 'completed' | 'generating' | 'pending' | 'failed' | string;

type MotionRow = {
  avatar_key: string | null;
  persona: string | null;
  age_bucket: string | null;
  phase: string | null;
  status: MotionStatus | null;
  video_url: string | null;
  completed_at: string | null;
  updated_at?: string | null;
  created_at?: string | null;
};

const EXPECTED_TOTAL = 420; // 12 personas × 5 ages × 7 phases
const DEFAULT_BUCKET_TOTAL = 84; // 12 personas × 7 phases

function normalizeStatus(status: MotionStatus | null | undefined): 'completed' | 'generating' | 'pending' | 'failed' {
  if (!status) return 'pending';
  if (status === 'completed' || status === 'generating' || status === 'pending' || status === 'failed') return status;
  return 'pending';
}

function isoToMillis(iso?: string | null): number {
  if (!iso) return 0;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : 0;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  if (!isSupabaseConfigured()) {
    return res.status(500).json({
      error: 'Supabase is not configured on the server',
      required: ['PUBLIC_SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)', 'SUPABASE_SERVICE_ROLE_KEY'],
    });
  }

  try {
    const supabase = getSupabaseAdmin();

    const { data: clips, error } = await supabase
      .from('kelly_motion_library')
      .select('avatar_key, persona, age_bucket, phase, status, video_url, completed_at, updated_at, created_at');

    if (error) {
      return res.status(500).json({ error: 'Failed to fetch motion library rows', details: error.message });
    }

    const rows: MotionRow[] = clips || [];

    const stats = {
      completed: 0,
      generating: 0,
      pending: 0,
      failed: 0,
    };

    const buckets: Record<string, { completed: number; total: number }> = {};

    let generating: MotionRow | null = null;
    const completedRows: MotionRow[] = [];

    for (const row of rows) {
      const s = normalizeStatus(row.status);
      stats[s]++;

      const bucketKey = row.age_bucket || 'unknown';
      if (!buckets[bucketKey]) buckets[bucketKey] = { completed: 0, total: 0 };
      buckets[bucketKey].total++;

      if (s === 'completed') {
        buckets[bucketKey].completed++;
        completedRows.push(row);
      }

      if (s === 'generating') {
        const rowTime = Math.max(isoToMillis(row.updated_at), isoToMillis(row.created_at));
        const curTime = generating ? Math.max(isoToMillis(generating.updated_at), isoToMillis(generating.created_at)) : -1;
        if (!generating || rowTime > curTime) generating = row;
      }
    }

    for (const key of ['kid', 'teen', 'adult', 'elder', 'super_elder']) {
      if (!buckets[key]) buckets[key] = { completed: 0, total: DEFAULT_BUCKET_TOTAL };
      if (!buckets[key].total) buckets[key].total = DEFAULT_BUCKET_TOTAL;
    }

    completedRows.sort((a, b) => isoToMillis(b.completed_at) - isoToMillis(a.completed_at));
    const recent = completedRows.slice(0, 6);

    return res.status(200).json({
      stats,
      buckets,
      generating,
      recent,
      total: EXPECTED_TOTAL,
      totalInTable: rows.length,
      generatedAt: new Date().toISOString(),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return res.status(500).json({ error: 'Failed to fetch progress', details: message });
  }
}
