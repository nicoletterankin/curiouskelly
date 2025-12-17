#!/usr/bin/env npx tsx
/**
 * HEYGEN KELLY AVATAR FINDER
 * 
 * Tries multiple endpoints to find Kelly avatars/talking photos.
 */

import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

async function tryEndpoint(url: string, name: string): Promise<any> {
  console.log(`\n🔍 Trying: ${name}`);
  console.log(`   ${url}`);
  
  try {
    const response = await fetch(url, {
      headers: { 
        'X-Api-Key': HEYGEN_API_KEY,
        'Accept': 'application/json'
      }
    });
    
    const text = await response.text();
    
    if (!response.ok) {
      console.log(`   ❌ ${response.status}: ${text.slice(0, 100)}`);
      return null;
    }
    
    const data = JSON.parse(text);
    console.log(`   ✅ Success!`);
    return data;
  } catch (e: any) {
    console.log(`   ❌ Error: ${e.message}`);
    return null;
  }
}

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🔎 HEYGEN KELLY AVATAR FINDER                                 ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');

  // Try various endpoints
  const endpoints = [
    ['https://api.heygen.com/v2/avatars', 'List Avatars (v2)'],
    ['https://api.heygen.com/v1/avatar.list', 'List Avatars (v1)'],
    ['https://api.heygen.com/v2/user/talking_photo', 'User Talking Photos'],
    ['https://api.heygen.com/v1/talking_photo.list', 'Talking Photo List (v1)'],
    ['https://api.heygen.com/v2/photo_avatars', 'Photo Avatars (v2)'],
    ['https://api.heygen.com/v1/photo_avatar.list', 'Photo Avatar List (v1)'],
    ['https://api.heygen.com/v2/assets', 'Assets'],
  ];

  let foundData: any = null;
  let talkingPhotos: any[] = [];

  for (const [url, name] of endpoints) {
    const result = await tryEndpoint(url, name);
    if (result?.data) {
      foundData = result;
      
      // Look for talking photos or photo avatars
      if (result.data.talking_photos) {
        talkingPhotos = result.data.talking_photos;
        console.log(`   📸 Found ${talkingPhotos.length} talking photos`);
      }
      if (result.data.photo_avatars) {
        talkingPhotos = result.data.photo_avatars;
        console.log(`   📸 Found ${talkingPhotos.length} photo avatars`);
      }
      if (result.data.avatars) {
        // Filter for custom/photo avatars
        const photoTypes = result.data.avatars.filter((a: any) => 
          a.avatar_type === 'talking_photo' || 
          a.avatar_type === 'photo' ||
          a.type === 'talking_photo'
        );
        if (photoTypes.length > 0) {
          console.log(`   📸 Found ${photoTypes.length} photo-type avatars`);
          talkingPhotos = photoTypes;
        }
      }
    }
  }

  // Also check the existing kelly-talking-photos.json
  console.log('\n📁 Checking existing kelly-talking-photos.json...');
  try {
    const existing = await import('../generated-images/kelly-talking-photos.json');
    if (Array.isArray(existing.default)) {
      console.log(`   Found ${existing.default.length} existing entries`);
      
      // Show some recent ones
      console.log('\n   Recent entries:');
      existing.default.slice(-10).forEach((item: any) => {
        console.log(`   - ${item.id}`);
      });
    }
  } catch (e) {
    console.log('   No existing file found');
  }

  // Try to get avatar groups
  console.log('\n🔍 Trying avatar groups...');
  const groupsResult = await tryEndpoint('https://api.heygen.com/v2/avatar_group.list', 'Avatar Groups');
  if (groupsResult?.data?.avatar_groups) {
    console.log(`   Found ${groupsResult.data.avatar_groups.length} groups:`);
    for (const group of groupsResult.data.avatar_groups.slice(0, 10)) {
      console.log(`   - ${group.name || group.id}: ${group.avatar_group_id || group.id}`);
    }
  }

  // Summary
  console.log('\n' + '═'.repeat(60));
  console.log('📊 FINDINGS:');
  console.log(`   Talking Photos Found: ${talkingPhotos.length}`);
  
  if (talkingPhotos.length > 0) {
    console.log('\n   Latest 20 talking photos:');
    for (const photo of talkingPhotos.slice(-20)) {
      const id = photo.talking_photo_id || photo.id || photo.avatar_id;
      const name = photo.talking_photo_name || photo.name || photo.avatar_name || 'Unnamed';
      console.log(`   ${name}: ${id}`);
    }
  }
}

main().catch(console.error);
