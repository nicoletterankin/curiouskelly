#!/usr/bin/env npx tsx
import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

const videoIds = [
  // Earliest submitted (should have most time to process)
  { day: 1, id: '82dece7c6d9c4324b97137f0db00cb05' },
  { day: 2, id: 'e5cb8a4336ac496b914c5ac54fe2aa6e' },
  { day: 10, id: 'a34513ee427142fd829113c6f5fe519d' },
  { day: 31, id: '1075208388d24b66b99e5fa5abc9d687' },
  // Middle batch
  { day: 100, id: 'pending' },
  // Recent
  { day: 351, id: '439e5ccdd22c472fae621d7e00b5e77f' },
  { day: 365, id: '371304b06e754d158a4d78892c0a9139' },
];

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🔍 HEYGEN VIDEO STATUS CHECK                                  ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');

  if (!HEYGEN_API_KEY) {
    console.log('❌ HEYGEN_API_KEY not found in environment');
    return;
  }

  let completed = 0;
  let pending = 0;
  let failed = 0;

  for (const v of videoIds.slice(0, 5)) {
    try {
      const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${v.id}`, {
        headers: { 'X-Api-Key': HEYGEN_API_KEY }
      });
      const data = await res.json();
      const status = data.data?.status || 'unknown';
      const videoUrl = data.data?.video_url;
      
      const icon = status === 'completed' ? '✅' : 
                   status === 'failed' ? '❌' : 
                   status === 'processing' ? '⏳' : '⏸️';
      
      if (status === 'completed') completed++;
      else if (status === 'failed') failed++;
      else pending++;
      
      console.log(`${icon} Day ${v.day}: ${status}`);
      if (videoUrl) {
        console.log(`   → ${videoUrl.substring(0, 70)}...`);
      }
    } catch (err) {
      console.log(`❌ Day ${v.day}: Error - ${err}`);
    }
  }

  console.log('');
  console.log('════════════════════════════════════════════════════════════════');
  console.log(`📊 SUMMARY: ${completed} completed, ${pending} pending, ${failed} failed`);
  console.log('════════════════════════════════════════════════════════════════');
}

main();
