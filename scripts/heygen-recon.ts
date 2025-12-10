#!/usr/bin/env npx tsx
/**
 * 🔍 HEYGEN RECONNAISSANCE
 * 
 * This script explores the HeyGen API to understand:
 * 1. What avatars/talking photos already exist in the account
 * 2. What endpoints are available
 * 3. How to properly create and use photo avatars
 */

import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

async function tryEndpoint(method: string, path: string, body?: object): Promise<void> {
  const url = `https://api.heygen.com${path}`;
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`📡 ${method} ${path}`);
  
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
      console.log(`   Body: ${JSON.stringify(body, null, 2).substring(0, 200)}...`);
    }
    
    const response = await fetch(url, options);
    const contentType = response.headers.get('content-type') || '';
    
    console.log(`   Status: ${response.status} ${response.statusText}`);
    
    if (contentType.includes('application/json')) {
      const data = await response.json();
      console.log(`   Response:`);
      console.log(JSON.stringify(data, null, 2).split('\n').map(l => `      ${l}`).join('\n'));
    } else {
      const text = await response.text();
      if (text.length < 500) {
        console.log(`   Response: ${text}`);
      } else {
        console.log(`   Response: [HTML or large response - ${text.length} chars]`);
      }
    }
  } catch (error: any) {
    console.log(`   Error: ${error.message}`);
  }
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🔍 HEYGEN API RECONNAISSANCE                              ║');
  console.log('║  Exploring available endpoints and resources               ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log(`\nAPI Key: ${HEYGEN_API_KEY.substring(0, 15)}...`);

  // 1. Check account/quota
  await tryEndpoint('GET', '/v1/user/remaining_quota');
  await tryEndpoint('GET', '/v2/user/info');
  
  // 2. List existing avatars
  await tryEndpoint('GET', '/v1/avatars');
  await tryEndpoint('GET', '/v2/avatars');
  
  // 3. List talking photos specifically
  await tryEndpoint('GET', '/v1/talking_photo.list');
  await tryEndpoint('GET', '/v2/talking_photo.list');
  await tryEndpoint('GET', '/v1/photo_avatars');
  await tryEndpoint('GET', '/v2/photo_avatars');
  
  // 4. List voices
  await tryEndpoint('GET', '/v1/voices');
  await tryEndpoint('GET', '/v2/voices');
  
  // 5. Check video generation endpoint structure
  await tryEndpoint('GET', '/v2/video/list');
  await tryEndpoint('GET', '/v1/video.list');
  
  console.log('\n\n' + '═'.repeat(60));
  console.log('🎯 NEXT STEPS based on findings above:');
  console.log('═'.repeat(60));
}

main().catch(console.error);

