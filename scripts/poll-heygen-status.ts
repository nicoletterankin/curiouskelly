#!/usr/bin/env npx tsx
/**
 * CURIOUS KELLY - POLL HEYGEN STATUS
 *
 * Finds kelly_video_assets rows in generating/pending state where the stored
 * storage path looks like `heygen/<videoId>`, polls HeyGen, and on completion:
 * - downloads the video
 * - uploads it to Supabase Storage
 * - updates the DB record with a permanent public URL
 *
 * Usage:
 *   npx tsx scripts/poll-heygen-status.ts
 *   npx tsx scripts/poll-heygen-status.ts --day=1
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY =
  process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

const VIDEO_BUCKET = process.env.KELLY_VIDEO_BUCKET || 'kelly-videos';

if (!HEYGEN_API_KEY) {
  console.error('❌ Missing HEYGEN_API_KEY');
  process.exit(1);
}
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error('❌ Missing Supabase env vars. Required one of:');
  console.error('   - SUPABASE_URL or PUBLIC_SUPABASE_URL');
  console.error('   - SUPABASE_SERVICE_KEY or SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

function getArg(name: string) {
  const arg = process.argv.slice(2).find(a => a.startsWith(`--${name}=`));
  return arg ? arg.split('=')[1] : undefined;
}

async function heygenStatus(videoId: string): Promise<{ status: string; videoUrl?: string; duration?: number }> {
  // Try v2 first
  {
    const r = await fetch(`https://api.heygen.com/v2/video_status.get?video_id=${encodeURIComponent(videoId)}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY! },
    });
    const txt = await r.text();
    try {
      const j: any = JSON.parse(txt);
      if (j?.data?.status) {
        return { status: j.data.status, videoUrl: j.data.video_url, duration: j.data.duration };
      }
    } catch {
      // ignore
    }
  }

  // Fallback to v1
  {
    const r = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${encodeURIComponent(videoId)}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY! },
    });
    const txt = await r.text();
    const j: any = JSON.parse(txt);
    return { status: j?.data?.status, videoUrl: j?.data?.video_url, duration: j?.data?.duration };
  }
}

function extractHeygenVideoId(row: any): string | null {
  const candidates: Array<unknown> = [
    row?.storage_path,
    row?.video_storage_path,
    row?.elevenlabs_generation_id,
  ];

  for (const c of candidates) {
    if (typeof c !== 'string') continue;
    if (c.startsWith('heygen/')) return c.replace(/^heygen\//, '').trim();
  }

  return null;
}

async function updateRowCompleted(row: any, params: { publicUrl: string; storagePath: string; durationSeconds?: number }) {
  const now = new Date().toISOString();

  // Attempt A: day_number schema
  {
    const payload: any = {
      public_url: params.publicUrl,
      storage_bucket: VIDEO_BUCKET,
      storage_path: params.storagePath,
      status: 'validated',
      duration_seconds: params.durationSeconds ?? null,
      updated_at: now,
    };

    const { error } = await supabase
      .from('kelly_video_assets')
      .update(payload)
      .eq('id', row.id);

    if (!error) return;
  }

  // Attempt B: migration 004 schema
  {
    const payload: any = {
      video_public_url: params.publicUrl,
      video_storage_path: params.storagePath,
      video_duration_ms: params.durationSeconds ? Math.round(params.durationSeconds * 1000) : null,
      status: 'completed',
      generation_completed_at: now,
      updated_at: now,
    };

    const { error } = await supabase
      .from('kelly_video_assets')
      .update(payload)
      .eq('id', row.id);

    if (!error) return;

    throw new Error(`Failed updating kelly_video_assets row ${row.id}: ${error.message}`);
  }
}

async function updateRowFailed(row: any, message: string) {
  const now = new Date().toISOString();

  // Attempt A
  {
    const { error } = await supabase
      .from('kelly_video_assets')
      .update({ status: 'archived', updated_at: now, error_message: message } as any)
      .eq('id', row.id);

    if (!error) return;
  }

  // Attempt B
  {
    const { error } = await supabase
      .from('kelly_video_assets')
      .update({ status: 'failed', updated_at: now, error_message: message } as any)
      .eq('id', row.id);

    if (error) throw new Error(`Failed updating row ${row.id} to failed: ${error.message}`);
  }
}

async function main() {
  const dayFilter = getArg('day');

  // Pull pending rows (schema-tolerant by selecting '*')
  let rows: any[] = [];

  // Attempt A: day_number exists
  {
    let q = supabase
      .from('kelly_video_assets')
      .select('*')
      .in('status', ['generating', 'pending'] as any);

    if (dayFilter) {
      q = (q as any).eq('day_number', parseInt(dayFilter, 10));
    }

    const { data, error } = await q.limit(200);
    if (!error && data) rows = data as any[];
  }

  // Attempt B: lesson_day exists (only if A yielded nothing)
  if (rows.length === 0) {
    let q = supabase
      .from('kelly_video_assets')
      .select('*')
      .in('status', ['generating', 'pending'] as any);

    if (dayFilter) {
      q = (q as any).eq('lesson_day', parseInt(dayFilter, 10));
    }

    const { data, error } = await q.limit(200);
    if (!error && data) rows = data as any[];
  }

  // Keep only rows that look like HeyGen jobs
  const heygenRows = rows.filter(r => extractHeygenVideoId(r));

  if (heygenRows.length === 0) {
    console.log('No pending HeyGen videos found');
    return;
  }

  console.log(`Checking ${heygenRows.length} pending HeyGen videos...`);

  for (const row of heygenRows) {
    const videoId = extractHeygenVideoId(row);
    if (!videoId) continue;

    const label = `${row.day_number ?? row.lesson_day ?? '?'}-${row.phase ?? '?'}-${row.age_bucket ?? '?'}-${row.language ?? 'en'}`;

    try {
      const st = await heygenStatus(videoId);
      console.log(`[${label}] ${videoId}: ${st.status}`);

      if (st.status === 'completed' && st.videoUrl) {
        // Download
        const v = await fetch(st.videoUrl);
        if (!v.ok) throw new Error(`Download failed: ${v.status} ${v.statusText}`);
        const buf = Buffer.from(await v.arrayBuffer());

        // Upload to Supabase
        const dayStr = String(row.day_number ?? row.lesson_day ?? 0).padStart(3, '0');
        const storagePath = `heygen/day_${dayStr}/${row.phase || 'phase'}/${row.age_bucket || 'age'}/${row.language || 'en'}/${videoId}.mp4`;

        const { error: upErr } = await supabase.storage
          .from(VIDEO_BUCKET)
          .upload(storagePath, buf, { upsert: true, contentType: 'video/mp4' });

        if (upErr) throw new Error(`Supabase upload failed: ${upErr.message}`);

        const { data: urlData } = supabase.storage.from(VIDEO_BUCKET).getPublicUrl(storagePath);
        const publicUrl = urlData?.publicUrl;
        if (!publicUrl) throw new Error('Supabase getPublicUrl failed');

        await updateRowCompleted(row, { publicUrl, storagePath, durationSeconds: st.duration });
        console.log(`  ✅ Stored: ${publicUrl}`);
      }

      if (st.status === 'failed') {
        await updateRowFailed(row, 'HeyGen reported failed');
        console.log('  ❌ Marked failed');
      }

      await new Promise(r => setTimeout(r, 400));
    } catch (e: any) {
      console.log(`  ⚠️ Error: ${e?.message || String(e)}`);
    }
  }
}

main().catch(err => {
  console.error('❌ Fatal error:', err);
  process.exit(1);
});
