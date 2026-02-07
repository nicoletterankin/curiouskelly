/**
 * SPRINT 2.3 - VERIFY PLAYER CONNECTION
 * Tests the production API to see if the player gets working video URLs.
 * Tests both covered days (should have video) and uncovered days (should have fallback).
 */

import { config } from 'dotenv';
config();

import { neon } from '@neondatabase/serverless';

const sql = neon(process.env.DATABASE_URL!);

// The production API base URL
const PROD_URL = 'https://thedailylesson.com';
const API_PATH = '/api/video/url';

interface TestResult {
  day: number;
  phase: string;
  apiStatus: number | 'ERROR';
  source?: string;
  hasVideoUrl: boolean;
  videoUrl?: string;
  videoUrlLive?: boolean;
  hasAudioUrl: boolean;
  audioUrl?: string;
  hasScript: boolean;
  status?: string;
}

async function testEndpoint(day: number, phase: string, age: number = 30): Promise<TestResult> {
  const url = `${PROD_URL}${API_PATH}?day=${day}&phase=${phase}&age=${age}&archetype=storyteller`;
  
  try {
    const resp = await fetch(url, { 
      signal: AbortSignal.timeout(15000),
      headers: { 'User-Agent': 'Sprint2-PlayerVerification/1.0' }
    });
    
    if (!resp.ok) {
      return {
        day, phase,
        apiStatus: resp.status,
        hasVideoUrl: false,
        hasAudioUrl: false,
        hasScript: false,
      };
    }

    const data = await resp.json() as Record<string, unknown>;
    
    const result: TestResult = {
      day, phase,
      apiStatus: 200,
      source: data.source as string,
      hasVideoUrl: !!data.url,
      videoUrl: (data.url as string)?.substring(0, 80),
      hasAudioUrl: !!data.audioUrl,
      audioUrl: (data.audioUrl as string)?.substring(0, 80),
      hasScript: !!data.script && (data.script as string).length > 10,
      status: data.status as string,
    };

    // Check if video URL is live
    if (data.url) {
      try {
        const videoResp = await fetch(data.url as string, { 
          method: 'HEAD', 
          signal: AbortSignal.timeout(10000) 
        });
        result.videoUrlLive = videoResp.status === 200;
      } catch {
        result.videoUrlLive = false;
      }
    }

    return result;
  } catch (err) {
    return {
      day, phase,
      apiStatus: 'ERROR',
      hasVideoUrl: false,
      hasAudioUrl: false,
      hasScript: false,
    };
  }
}

async function main() {
  console.log('=== SPRINT 2.3: PLAYER CONNECTION VERIFICATION ===\n');
  
  // Test days across the spectrum
  const testDays = [
    // Days with known full heygen coverage
    { day: 1, label: 'Day 1 (heygen full)' },
    { day: 10, label: 'Day 10 (heygen full)' },
    { day: 30, label: 'Day 30 (heygen full)' },
    // Day with known partial coverage
    { day: 14, label: 'Day 14 (heygen partial - missing action)' },
    // Days with known ZERO video
    { day: 100, label: 'Day 100 (NO video in DB)' },
    { day: 200, label: 'Day 200 (NO video in DB)' },
    { day: 365, label: 'Day 365 (NO video in DB)' },
    // Edge cases
    { day: 50, label: 'Day 50 (check coverage)' },
  ];

  const phases = ['hook', 'story', 'wonder', 'action', 'wisdom'];
  
  for (const testDay of testDays) {
    console.log(`\n── ${testDay.label} ──`);
    
    for (const phase of phases) {
      const result = await testEndpoint(testDay.day, phase);
      
      const videoStatus = result.hasVideoUrl 
        ? (result.videoUrlLive ? '✅ LIVE' : '❌ DEAD URL') 
        : '⬜ none';
      const audioStatus = result.hasAudioUrl ? '✅' : '⬜';
      const scriptStatus = result.hasScript ? '✅' : '⬜';
      
      console.log(
        `  ${phase.padEnd(8)} | API: ${result.apiStatus} | Video: ${videoStatus} | Audio: ${audioStatus} | Script: ${scriptStatus} | Source: ${result.source || 'N/A'}`
      );
      
      if (result.hasVideoUrl && result.videoUrl) {
        // Show URL host
        try {
          const host = new URL(result.videoUrl.includes('...') ? 'https://unknown' : result.videoUrl).hostname;
          console.log(`           └─ Host: ${host}`);
        } catch {
          console.log(`           └─ URL: ${result.videoUrl}`);
        }
      }
    }
  }

  // Direct DB check: verify what the API SHOULD return for Day 1
  console.log('\n\n── DIRECT DB VERIFICATION (Day 1 hook) ──');
  const dbResult = await sql`
    SELECT day_of_year, phase, status, video_url, audio_url, age_category, archetype
    FROM heygen_videos 
    WHERE day_of_year = 1 AND phase = 'hook' AND video_url IS NOT NULL
    ORDER BY CASE WHEN age_category = 'adult' THEN 0 ELSE 1 END
    LIMIT 1
  `;
  if (dbResult.length > 0) {
    console.log('  DB row:', JSON.stringify(dbResult[0], null, 2));
  } else {
    console.log('  NO row found in heygen_videos for Day 1 hook!');
  }

  console.log('\n=== VERIFICATION COMPLETE ===');
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
