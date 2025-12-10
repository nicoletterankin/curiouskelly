#!/usr/bin/env npx tsx
import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const VIDEO_ID = process.argv[2] || 'd0d8ef89de4b4c399ad033b894f2c3fd';

async function main() {
  console.log(`🔍 Checking video status: ${VIDEO_ID}\n`);
  
  const response = await fetch(
    `https://api.heygen.com/v1/video_status.get?video_id=${VIDEO_ID}`,
    { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
  );
  
  const result = await response.json();
  console.log('Status:', result.data?.status);
  
  if (result.data?.status === 'completed') {
    console.log('\n✅ VIDEO COMPLETE!');
    console.log('Video URL:', result.data.video_url);
    console.log('Thumbnail:', result.data.thumbnail_url);
    console.log('Duration:', result.data.duration, 'seconds');
  } else if (result.data?.status === 'failed') {
    console.log('\n❌ Video failed:', result.data.error);
  } else {
    console.log('\n⏳ Still processing...');
  }
  
  console.log('\nFull response:', JSON.stringify(result, null, 2));
}

main().catch(console.error);

