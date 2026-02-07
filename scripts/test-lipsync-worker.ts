/**
 * Test Lip-Sync Worker Script
 * 
 * Tests the kelly-lipsync Cloudflare worker with a single asset
 * before batch processing all 365 days.
 */

const LIPSYNC_WORKER_URL = 'https://kelly-lipsync.nicoletterankin.workers.dev/lipsync';
const BASE_VIDEO_URL = 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly-base-videos/uncategorized/064dbfab193c461fbb2869f27d663c7b.mp4';

interface LipsyncRequest {
  audio_url: string;
  video_url: string;
}

interface LipsyncResponse {
  success?: boolean;
  video_url?: string;
  error?: string;
  request_id?: string;
}

async function testAudioUrl(url: string): Promise<boolean> {
  console.log(`\n🔍 Testing audio URL...`);
  console.log(`   URL: ${url}`);
  
  try {
    const response = await fetch(url, { method: 'HEAD' });
    console.log(`   Status: ${response.status}`);
    console.log(`   Content-Type: ${response.headers.get('content-type')}`);
    console.log(`   Content-Length: ${response.headers.get('content-length')}`);
    
    return response.ok;
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return false;
  }
}

async function testBaseVideo(url: string): Promise<boolean> {
  console.log(`\n🔍 Testing base video URL...`);
  console.log(`   URL: ${url}`);
  
  try {
    const response = await fetch(url, { method: 'HEAD' });
    console.log(`   Status: ${response.status}`);
    console.log(`   Content-Type: ${response.headers.get('content-type')}`);
    console.log(`   Content-Length: ${response.headers.get('content-length')}`);
    
    return response.ok;
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return false;
  }
}

async function testLipsyncWorker(audioUrl: string, videoUrl: string): Promise<LipsyncResponse> {
  console.log(`\n🎬 Testing lip-sync worker...`);
  console.log(`   Worker URL: ${LIPSYNC_WORKER_URL}`);
  console.log(`   Audio URL: ${audioUrl}`);
  console.log(`   Video URL: ${videoUrl}`);
  
  const payload: LipsyncRequest = {
    audio_url: audioUrl,
    video_url: videoUrl,
  };
  
  console.log(`\n   Sending POST request...`);
  
  try {
    const response = await fetch(LIPSYNC_WORKER_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    
    console.log(`   Response status: ${response.status}`);
    console.log(`   Content-Type: ${response.headers.get('content-type')}`);
    
    // Check if response is binary (video) or JSON
    const contentType = response.headers.get('content-type') || '';
    
    if (contentType.includes('application/json')) {
      const data = await response.json();
      console.log(`   Response (JSON):`, JSON.stringify(data, null, 2));
      return data;
    } else if (contentType.includes('video/')) {
      // Response is binary video
      const buffer = await response.arrayBuffer();
      console.log(`   Response: Binary video data (${buffer.byteLength} bytes)`);
      return {
        success: true,
        video_url: `[Binary data: ${buffer.byteLength} bytes]`,
      };
    } else {
      const text = await response.text();
      console.log(`   Response (text): ${text.substring(0, 500)}`);
      
      // Try parsing as JSON anyway
      try {
        return JSON.parse(text);
      } catch {
        return { error: text };
      }
    }
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return { success: false, error: error.message };
  }
}

async function main() {
  console.log('='.repeat(60));
  console.log('  Kelly Lip-Sync Worker Test');
  console.log('='.repeat(60));
  
  // Test day 34, hook, age35
  const audioUrl = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/audio/2026/en/day-034/hook-age35.mp3';
  
  // Step 1: Verify audio URL
  const audioOk = await testAudioUrl(audioUrl);
  if (!audioOk) {
    console.error('\n❌ Audio URL test failed. Aborting.');
    process.exit(1);
  }
  console.log('   ✅ Audio URL accessible');
  
  // Step 2: Verify base video URL
  const videoOk = await testBaseVideo(BASE_VIDEO_URL);
  if (!videoOk) {
    console.error('\n❌ Base video URL test failed. Aborting.');
    process.exit(1);
  }
  console.log('   ✅ Base video URL accessible');
  
  // Step 3: Test lip-sync worker
  const result = await testLipsyncWorker(audioUrl, BASE_VIDEO_URL);
  
  console.log('\n' + '='.repeat(60));
  console.log('  RESULT');
  console.log('='.repeat(60));
  
  if (result.success) {
    console.log('✅ Lip-sync worker test PASSED');
    if (result.video_url) {
      console.log(`   Output URL: ${result.video_url}`);
    }
  } else {
    console.log('❌ Lip-sync worker test FAILED');
    console.log(`   Error: ${result.error || 'Unknown error'}`);
  }
  
  console.log('\nDone.');
}

main().catch(console.error);
