#!/usr/bin/env npx tsx
/**
 * HEYGEN KELLY LIBRARY FETCHER
 * 
 * Fetches all talking photos and filters for Kelly motion library.
 */

import 'dotenv/config';
import * as fs from 'fs';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  📚 HEYGEN KELLY LIBRARY                                       ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');

  // Fetch talking photos using v1 endpoint
  console.log('🔍 Fetching talking photos (v1)...');
  const tpResponse = await fetch('https://api.heygen.com/v1/talking_photo.list', {
    headers: { 
      'X-Api-Key': HEYGEN_API_KEY,
      'Accept': 'application/json'
    }
  });
  
  const tpData = await tpResponse.json();
  const talkingPhotos = tpData.data?.talking_photos || [];
  console.log(`   Found ${talkingPhotos.length} talking photos`);

  // Fetch avatars using v2 endpoint
  console.log('🔍 Fetching avatars (v2)...');
  const avResponse = await fetch('https://api.heygen.com/v2/avatars', {
    headers: { 
      'X-Api-Key': HEYGEN_API_KEY,
      'Accept': 'application/json'
    }
  });
  
  const avData = await avResponse.json();
  const allAvatars = avData.data?.avatars || [];
  console.log(`   Found ${allAvatars.length} avatars`);

  // Filter for Kelly-related items
  const kellyKeywords = [
    'kelly', 'scientist', 'explorer', 'rebel', 'architect', 
    'diplomat', 'empath', 'macgyver', 'mystic', 'provider',
    'storyteller', 'strategist', 'survivor',
    'hypothesis', 'discovery', 'conclusion', 'scanning', 
    'campfire', 'challenge', 'provocation', 'truth',
    'blueprint', 'foundation', 'cornerstone', 'bridge',
    'invitation', 'accord', 'attunement', 'connection',
    'embrace', 'assembly', 'eureka', 'toolkit', 'revelation',
    'vision', 'blessing', 'guidance', 'shelter', 'weaving',
    'hook', 'moral', 'analysis', 'gambit', 'victory',
    'assessment', 'alert', 'resilience', 'motion_a', 'motion_b', 'motion_c'
  ];

  // Check talking photos
  const kellyTalkingPhotos = talkingPhotos.filter((tp: any) => {
    const name = (tp.talking_photo_name || '').toLowerCase();
    return kellyKeywords.some(kw => name.includes(kw));
  });

  // Check avatars
  const kellyAvatars = allAvatars.filter((av: any) => {
    const name = (av.avatar_name || '').toLowerCase();
    return kellyKeywords.some(kw => name.includes(kw));
  });

  console.log('');
  console.log('═'.repeat(60));
  console.log('🎯 KELLY MOTION LIBRARY:');
  console.log('═'.repeat(60));

  if (kellyTalkingPhotos.length > 0) {
    console.log(`\n📸 Kelly Talking Photos (${kellyTalkingPhotos.length}):`);
    console.log('─'.repeat(60));
    for (const tp of kellyTalkingPhotos) {
      console.log(`  ${tp.talking_photo_name || 'Unnamed'}`);
      console.log(`    ID: ${tp.talking_photo_id}`);
    }
  }

  if (kellyAvatars.length > 0) {
    console.log(`\n🎭 Kelly Avatars (${kellyAvatars.length}):`);
    console.log('─'.repeat(60));
    for (const av of kellyAvatars) {
      console.log(`  ${av.avatar_name || 'Unnamed'}`);
      console.log(`    ID: ${av.avatar_id}`);
    }
  }

  // Also show the most recently created talking photos (likely the new uploads)
  console.log('\n📅 RECENTLY CREATED TALKING PHOTOS (last 20):');
  console.log('─'.repeat(60));
  
  // Sort by created date if available, otherwise just show last 20
  const recentPhotos = talkingPhotos.slice(-20);
  
  for (const tp of recentPhotos) {
    console.log(`  ${tp.talking_photo_name || 'Unnamed'}`);
    console.log(`    ID: ${tp.talking_photo_id}`);
    if (tp.created_at) {
      console.log(`    Created: ${tp.created_at}`);
    }
    console.log('');
  }

  // Save all Kelly items to a manifest
  const manifest = {
    updated: new Date().toISOString(),
    talking_photos: kellyTalkingPhotos.map((tp: any) => ({
      id: tp.talking_photo_id,
      name: tp.talking_photo_name,
      preview: tp.preview_image_url,
    })),
    avatars: kellyAvatars.map((av: any) => ({
      id: av.avatar_id,
      name: av.avatar_name,
      preview: av.preview_image_url,
    })),
    recent: recentPhotos.map((tp: any) => ({
      id: tp.talking_photo_id,
      name: tp.talking_photo_name,
    })),
  };

  const outputPath = 'generated-images/kelly-motion-library.json';
  fs.writeFileSync(outputPath, JSON.stringify(manifest, null, 2));
  console.log(`\n💾 Saved manifest to: ${outputPath}`);

  // Summary
  console.log('\n' + '═'.repeat(60));
  console.log('📊 SUMMARY:');
  console.log(`   Kelly Talking Photos: ${kellyTalkingPhotos.length}`);
  console.log(`   Kelly Avatars: ${kellyAvatars.length}`);
  console.log(`   Total Kelly Items: ${kellyTalkingPhotos.length + kellyAvatars.length}`);
  
  if (kellyTalkingPhotos.length + kellyAvatars.length === 0) {
    console.log('\n💡 No Kelly items found yet. They may still be processing.');
    console.log('   Check again in a few minutes, or look for them by a different name.');
  }
}

main().catch(console.error);
