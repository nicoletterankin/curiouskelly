#!/usr/bin/env npx tsx
/**
 * 🔍 HEYGEN RECONNAISSANCE v2
 * 
 * FOCUSED on:
 * 1. Finding existing talking photos in the account
 * 2. Understanding how to create new ones via API
 * 3. Getting the exact video generation endpoint
 */

import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

async function tryEndpoint(method: string, path: string, body?: object): Promise<any> {
  const url = `https://api.heygen.com${path}`;
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`📡 ${method} ${url}`);
  
  try {
    const options: RequestInit = {
      method,
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
    };
    
    if (body) {
      options.body = JSON.stringify(body);
    }
    
    const response = await fetch(url, options);
    const contentType = response.headers.get('content-type') || '';
    
    console.log(`   Status: ${response.status}`);
    
    if (contentType.includes('application/json')) {
      const data = await response.json();
      return { status: response.status, data };
    } else {
      const text = await response.text();
      return { status: response.status, html: true };
    }
  } catch (error: any) {
    return { status: 'error', message: error.message };
  }
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🔍 HEYGEN DEEP RECON - Finding Talking Photos             ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log(`\nAPI Key: ${HEYGEN_API_KEY.substring(0, 20)}...`);

  // 1. Check quota first
  console.log('\n\n=== 1. QUOTA CHECK ===');
  const quota = await tryEndpoint('GET', '/v1/user/remaining_quota');
  if (quota.data) {
    console.log('   Quota:', JSON.stringify(quota.data, null, 2));
  }

  // 2. List avatars (this is where talking photos should appear)
  console.log('\n\n=== 2. LIST AVATARS (v1) ===');
  const avatarsV1 = await tryEndpoint('GET', '/v1/avatars');
  if (avatarsV1.data?.data?.avatars) {
    console.log(`   Found ${avatarsV1.data.data.avatars.length} avatars!`);
    
    // Look for talking_photo type
    const talkingPhotos = avatarsV1.data.data.avatars.filter((a: any) => 
      a.avatar_type === 'talking_photo' || a.type === 'talking_photo'
    );
    console.log(`   Talking photos: ${talkingPhotos.length}`);
    
    if (talkingPhotos.length > 0) {
      console.log('\n   📸 FOUND TALKING PHOTOS:');
      talkingPhotos.forEach((tp: any) => {
        console.log(`      ID: ${tp.avatar_id || tp.talking_photo_id}`);
        console.log(`      Name: ${tp.avatar_name || tp.name}`);
        console.log(`      ---`);
      });
    }
    
    // Show all avatar types
    const types = [...new Set(avatarsV1.data.data.avatars.map((a: any) => a.avatar_type || a.type))];
    console.log(`\n   Avatar types in account: ${types.join(', ')}`);
  } else if (avatarsV1.data) {
    console.log('   Response:', JSON.stringify(avatarsV1.data, null, 2).substring(0, 500));
  }

  // 3. Try v2 avatars
  console.log('\n\n=== 3. LIST AVATARS (v2) ===');
  const avatarsV2 = await tryEndpoint('GET', '/v2/avatars');
  if (avatarsV2.data?.data?.avatars) {
    console.log(`   Found ${avatarsV2.data.data.avatars.length} avatars (v2)`);
  } else if (avatarsV2.data) {
    console.log('   Response:', JSON.stringify(avatarsV2.data, null, 2).substring(0, 500));
  }

  // 4. Try talking_photo specific endpoints
  console.log('\n\n=== 4. TALKING PHOTO ENDPOINTS ===');
  const tpList = await tryEndpoint('GET', '/v1/talking_photo.list');
  if (tpList.data?.data) {
    console.log('   v1/talking_photo.list:', JSON.stringify(tpList.data.data, null, 2).substring(0, 500));
  }

  // 5. Check if there's a way to create talking photo via API
  console.log('\n\n=== 5. CREATE TALKING PHOTO (test with fake URL) ===');
  // First, check if endpoint exists by trying with minimal payload
  const createTp = await tryEndpoint('POST', '/v1/talking_photo.add', {
    image_url: 'https://example.com/test.png'  // Fake URL just to check if endpoint exists
  });
  console.log('   v1/talking_photo.add:', createTp.data ? JSON.stringify(createTp.data) : `status ${createTp.status}`);

  // 6. Check v2 video generation endpoint
  console.log('\n\n=== 6. VIDEO GENERATION INFO ===');
  const videoGen = await tryEndpoint('POST', '/v2/video/generate', {
    test: true,  // Some APIs support this for validation
    video_inputs: [{
      character: {
        type: 'talking_photo',
        talking_photo_id: 'test'
      },
      voice: {
        type: 'text',
        input_text: 'Hello'
      }
    }]
  });
  console.log('   v2/video/generate response:', videoGen.data ? JSON.stringify(videoGen.data) : `status ${videoGen.status}`);
  
  // 7. Check previously created videos for clues
  console.log('\n\n=== 7. RECENT VIDEO DETAILS ===');
  const videos = await tryEndpoint('GET', '/v1/video.list');
  if (videos.data?.data?.videos?.length > 0) {
    const recentVideo = videos.data.data.videos[0];
    console.log('   Most recent video:', JSON.stringify(recentVideo, null, 2));
    
    // Get full details of a completed video
    if (recentVideo.status === 'completed') {
      const details = await tryEndpoint('GET', `/v1/video_status.get?video_id=${recentVideo.video_id}`);
      console.log('   Full details:', JSON.stringify(details.data, null, 2));
    }
  }

  console.log('\n\n' + '═'.repeat(60));
  console.log('🎯 SUMMARY');
  console.log('═'.repeat(60));
}

main().catch(console.error);

