/**
 * SPRINT 2 - CRITICAL: Check HeyGen URL expiry dates
 * HeyGen uses CloudFront signed URLs that EXPIRE.
 * If they expire, all 513 videos become 403 errors.
 */

import { config } from 'dotenv';
config();

import { neon } from '@neondatabase/serverless';

const sql = neon(process.env.DATABASE_URL!);

async function main() {
  console.log('=== CRITICAL: HeyGen URL EXPIRY CHECK ===\n');

  // Get all unique Expires values from heygen_videos URLs
  const urls = await sql`
    SELECT video_url, day_of_year, phase
    FROM heygen_videos 
    WHERE video_url IS NOT NULL
    ORDER BY day_of_year
  `;

  const expiryDates: number[] = [];
  const expiryMap = new Map<string, number>(); // date string -> count

  for (const row of urls) {
    const url = row.video_url as string;
    const match = url.match(/Expires=(\d+)/);
    if (match) {
      const expires = parseInt(match[1], 10);
      expiryDates.push(expires);
      
      const date = new Date(expires * 1000);
      const dateStr = date.toISOString().split('T')[0];
      expiryMap.set(dateStr, (expiryMap.get(dateStr) || 0) + 1);
    }
  }

  console.log(`Total URLs checked: ${urls.length}`);
  console.log(`URLs with Expires parameter: ${expiryDates.length}`);
  
  if (expiryDates.length > 0) {
    const now = Date.now() / 1000;
    const earliest = Math.min(...expiryDates);
    const latest = Math.max(...expiryDates);
    const expired = expiryDates.filter(e => e < now).length;
    
    console.log(`\nEarliest expiry: ${new Date(earliest * 1000).toISOString()}`);
    console.log(`Latest expiry:   ${new Date(latest * 1000).toISOString()}`);
    console.log(`Current time:    ${new Date().toISOString()}`);
    console.log(`Already expired: ${expired} / ${expiryDates.length}`);
    
    const daysUntilEarliest = (earliest - now) / 86400;
    console.log(`\n⚠️  Days until earliest expiry: ${daysUntilEarliest.toFixed(1)} days`);
    
    if (daysUntilEarliest < 0) {
      console.log('🚨 SOME URLs HAVE ALREADY EXPIRED!');
    } else if (daysUntilEarliest < 7) {
      console.log('🚨 URLs EXPIRE IN LESS THAN 7 DAYS! MUST DOWNLOAD AND RE-HOST!');
    } else if (daysUntilEarliest < 30) {
      console.log('⚠️  URLs expire within 30 days. Plan to download and re-host.');
    } else {
      console.log('✅ URLs are valid for more than 30 days.');
    }
    
    console.log('\nExpiry date distribution:');
    const sortedDates = [...expiryMap.entries()].sort();
    for (const [date, count] of sortedDates) {
      const d = new Date(date);
      const daysUntil = (d.getTime() / 1000 - now) / 86400;
      const status = daysUntil < 0 ? '❌ EXPIRED' : daysUntil < 7 ? '🚨 <7 DAYS' : daysUntil < 30 ? '⚠️' : '✅';
      console.log(`  ${date}: ${count} videos ${status} (${daysUntil.toFixed(0)} days)`);
    }
  } else {
    console.log('No Expires parameters found in URLs.');
  }

  // Quick liveness check on a few URLs with different expiry dates
  console.log('\n\nLiveness check across different expiry dates...');
  const sortedByExpiry = [...urls].sort((a, b) => {
    const ea = parseInt(((a.video_url as string).match(/Expires=(\d+)/) || ['0','0'])[1], 10);
    const eb = parseInt(((b.video_url as string).match(/Expires=(\d+)/) || ['0','0'])[1], 10);
    return ea - eb;
  });
  
  // Check first (earliest expiry), middle, and last (latest expiry)
  const samples = [
    { ...sortedByExpiry[0], label: 'Earliest expiry' },
    { ...sortedByExpiry[Math.floor(sortedByExpiry.length / 2)], label: 'Middle expiry' },
    { ...sortedByExpiry[sortedByExpiry.length - 1], label: 'Latest expiry' },
  ];
  
  for (const sample of samples) {
    const url = sample.video_url as string;
    const match = url.match(/Expires=(\d+)/);
    const expiryDate = match ? new Date(parseInt(match[1], 10) * 1000).toISOString() : 'unknown';
    
    try {
      const resp = await fetch(url, { method: 'HEAD', signal: AbortSignal.timeout(10000) });
      console.log(`  ${sample.label} (${expiryDate}): ${resp.status} - Day ${sample.day_of_year} ${sample.phase}`);
    } catch (err) {
      console.log(`  ${sample.label} (${expiryDate}): ERROR - ${(err as Error).message}`);
    }
  }
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
