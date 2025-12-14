/**
 * /api/motion-progress.ts - Kelly Motion Library Progress
 *
 * Server-side progress endpoint for the motion generation dashboard.
 * Uses Supabase service role (no client-side keys required).
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin, isSupabaseConfigured } from './lib/supabase';

const EXPECTED_TOTAL = 420; // 12 personas × 5 ages × 7 phases
const DEFAULT_BUCKET_TOTAL = 84; // 12 personas × 7 phases

type StatusRow = { status: string | null; age_bucket: string | null };
type GeneratingRow = { avatar_key: string | null; persona: string | null; age_bucket: string | null; phase: string | null; created_at?: string | null };
type RecentRow = { avatar_key: string | null; persona: string | null; age_bucket: string | null; phase: string | null; video_url: string | null; completed_at: string | null };

const CACHE_TTL_SECONDS = 30;

function normalizeStatus(status: string | null | undefined): 'completed' | 'generating' | 'pending' | 'failed' {
  if (!status) return 'pending';
  if (status === 'completed' || status === 'generating' || status === 'pending' || status === 'failed') return status;
  // Treat unknown statuses as pending (e.g. queued)
  return 'pending';
}

function setCors(res: VercelResponse) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const start = Date.now();

  setCors(res);

  if (req.method === 'OPTIONS') {
    res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=0');
    return res.status(204).send('');
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

    const [statusResult, recentResult, generatingResult] = await Promise.all([
      supabase.from('kelly_motion_library').select('status, age_bucket') as any,
      supabase
        .from('kelly_motion_library')
        .select('avatar_key, persona, age_bucket, phase, video_url, completed_at')
        .eq('status', 'completed')
        .order('completed_at', { ascending: false })
        .limit(6) as any,
      supabase
        .from('kelly_motion_library')
        .select('avatar_key, persona, age_bucket, phase, created_at')
        .eq('status', 'generating')
        .limit(1)
        .maybeSingle() as any,
    ]);

    const statusError = statusResult?.error || recentResult?.error || generatingResult?.error;
    if (statusError) {
      const msg = statusError?.message || 'Unknown Supabase error';
      return res.status(500).json({
        error: 'Failed to fetch progress',
        message: msg,
        hint: 'Check if kelly_motion_library exists and env vars are correct',
      });
    }

    const rows: StatusRow[] = (statusResult.data || []) as StatusRow[];
    const recent: RecentRow[] = (recentResult.data || []) as RecentRow[];
    const generating: GeneratingRow | null = (generatingResult.data || null) as GeneratingRow | null;

    const stats = { completed: 0, generating: 0, pending: 0, failed: 0 };
    const buckets: Record<string, { completed: number; total: number }> = {
      kid: { completed: 0, total: DEFAULT_BUCKET_TOTAL },
      teen: { completed: 0, total: DEFAULT_BUCKET_TOTAL },
      adult: { completed: 0, total: DEFAULT_BUCKET_TOTAL },
      elder: { completed: 0, total: DEFAULT_BUCKET_TOTAL },
      super_elder: { completed: 0, total: DEFAULT_BUCKET_TOTAL },
    };

    for (const r of rows) {
      const s = normalizeStatus(r.status);
      stats[s]++;
      if (s === 'completed' && r.age_bucket && buckets[r.age_bucket]) {
        buckets[r.age_bucket].completed++;
      }
    }

    // If generator hasn't pre-seeded all 420 rows, treat missing rows as pending.
    const counted = stats.completed + stats.generating + stats.pending + stats.failed;
    if (counted < EXPECTED_TOTAL) stats.pending += EXPECTED_TOTAL - counted;

    const percentComplete = Math.round((stats.completed / EXPECTED_TOTAL) * 100);
    const remaining = Math.max(0, EXPECTED_TOTAL - stats.completed);
    const etaMinutesTotal = Math.round(remaining * 2.5);
    const etaHours = Math.floor(etaMinutesTotal / 60);
    const etaMinutes = etaMinutesTotal % 60;

    const queryTime = Date.now() - start;

    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Cache-Control', `public, s-maxage=${CACHE_TTL_SECONDS}, stale-while-revalidate=60`);
    res.setHeader('X-Response-Time', `${queryTime}ms`);

    return res.status(200).json({
      stats,
      buckets,
      generating,
      recent,
      total: EXPECTED_TOTAL,
      percentComplete,
      eta: remaining === 0 ? 'Complete!' : `${etaHours}h ${etaMinutes}m`,
      queryTime,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return res.status(500).json({
      error: 'Failed to fetch progress',
      message,
    });
  }
}
