/**
 * /api/motion-progress.ts - Kelly Motion Library Progress
 *
 * Server-side progress endpoint for the motion generation dashboard.
 * Uses Supabase service role (no client-side keys required).
 */

import { createClient } from '@supabase/supabase-js';

const EXPECTED_TOTAL = 420; // 12 personas × 5 ages × 7 phases
const DEFAULT_BUCKET_TOTAL = 84; // 12 personas × 7 phases

export const config = {
  runtime: 'edge',
};

type StatusRow = { status: string | null; age_bucket: string | null };
type GeneratingRow = { avatar_key: string | null; persona: string | null; age_bucket: string | null; phase: string | null; created_at?: string | null };
type RecentRow = { avatar_key: string | null; persona: string | null; age_bucket: string | null; phase: string | null; video_url: string | null; completed_at: string | null };

const CACHE_TTL_SECONDS = 30;

function json(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      ...(init?.headers || {}),
    },
  });
}

function normalizeStatus(status: string | null | undefined): 'completed' | 'generating' | 'pending' | 'failed' {
  if (!status) return 'pending';
  if (status === 'completed' || status === 'generating' || status === 'pending' || status === 'failed') return status;
  // Treat unknown statuses as pending (e.g. queued)
  return 'pending';
}

export default async function handler(request: Request): Promise<Response> {
  const start = Date.now();

  if (request.method === 'OPTIONS') {
    return json(null, {
      status: 204,
      headers: {
        'Cache-Control': 'public, max-age=0, s-maxage=0',
      },
    });
  }

  if (request.method !== 'GET') {
    return json({ error: 'Method not allowed' }, { status: 405 });
  }

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseServiceKey) {
    return json(
      {
        error: 'Supabase is not configured on the server',
        required: ['PUBLIC_SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)', 'SUPABASE_SERVICE_ROLE_KEY'],
      },
      { status: 500 }
    );
  }

  try {
    const supabase = createClient(supabaseUrl, supabaseServiceKey, {
      auth: { autoRefreshToken: false, persistSession: false },
      db: { schema: 'public' },
    });

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
      return json(
        {
          error: 'Failed to fetch progress',
          message: msg,
          hint: 'Check if kelly_motion_library exists and env vars are correct',
        },
        { status: 500 }
      );
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

    return json(
      {
        stats,
        buckets,
        generating,
        recent,
        total: EXPECTED_TOTAL,
        percentComplete,
        eta: remaining === 0 ? 'Complete!' : `${etaHours}h ${etaMinutes}m`,
        queryTime,
      },
      {
        status: 200,
        headers: {
          'Cache-Control': `public, s-maxage=${CACHE_TTL_SECONDS}, stale-while-revalidate=60`,
          'X-Response-Time': `${queryTime}ms`,
        },
      }
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return json(
      {
        error: 'Failed to fetch progress',
        message,
      },
      { status: 500 }
    );
  }
}
