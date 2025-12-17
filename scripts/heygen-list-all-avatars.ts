#!/usr/bin/env npx tsx
/**
 * HEYGEN AVATAR LIBRARY MANAGER
 * 
 * Lists all avatars/talking photos in your HeyGen account.
 * Helps manage the Kelly motion library.
 * 
 * Usage:
 *   npx tsx scripts/heygen-list-all-avatars.ts
 *   npx tsx scripts/heygen-list-all-avatars.ts --json
 *   npx tsx scripts/heygen-list-all-avatars.ts --filter kelly
 */

import 'dotenv/config';
import * as fs from 'fs';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

interface TalkingPhoto {
  talking_photo_id: string;
  talking_photo_name?: string;
  preview_image_url?: string;
  created_at?: string;
}

interface Avatar {
  avatar_id: string;
  avatar_name?: string;
  preview_image_url?: string;
  gender?: string;
}

async function listTalkingPhotos(): Promise<TalkingPhoto[]> {
  const response = await fetch('https://api.heygen.com/v2/talking_photo.list', {
    headers: { 
      'X-Api-Key': HEYGEN_API_KEY,
      'Accept': 'application/json'
    }
  });
  
  const text = await response.text();
  try {
    const data = JSON.parse(text);
    return data.data?.talking_photos || [];
  } catch {
    console.error('Failed to parse talking photos:', text.slice(0, 200));
    return [];
  }
}

async function listAvatars(): Promise<Avatar[]> {
  const response = await fetch('https://api.heygen.com/v2/avatars', {
    headers: { 
      'X-Api-Key': HEYGEN_API_KEY,
      'Accept': 'application/json'
    }
  });
  
  const text = await response.text();
  try {
    const data = JSON.parse(text);
    return data.data?.avatars || [];
  } catch {
    console.error('Failed to parse avatars:', text.slice(0, 200));
    return [];
  }
}

async function listPhotoAvatars(): Promise<any[]> {
  // Try the photo avatar endpoint
  const response = await fetch('https://api.heygen.com/v2/photo_avatar.list', {
    headers: { 
      'X-Api-Key': HEYGEN_API_KEY,
      'Accept': 'application/json'
    }
  });
  
  const text = await response.text();
  try {
    const data = JSON.parse(text);
    return data.data?.photo_avatars || data.data || [];
  } catch {
    return [];
  }
}

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  📚 HEYGEN AVATAR LIBRARY                                      ║');
  console.log('║  Listing all avatars and talking photos                        ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');

  const filter = process.argv.find(a => a.startsWith('--filter='))?.split('=')[1]?.toLowerCase();
  const jsonOutput = process.argv.includes('--json');

  // Fetch all types in parallel
  console.log('🔍 Fetching avatars...\n');
  
  const [talkingPhotos, avatars, photoAvatars] = await Promise.all([
    listTalkingPhotos(),
    listAvatars(),
    listPhotoAvatars(),
  ]);

  const allItems: any[] = [];

  // Process talking photos
  if (talkingPhotos.length > 0) {
    console.log(`📸 TALKING PHOTOS (${talkingPhotos.length}):`);
    console.log('─'.repeat(60));
    
    for (const photo of talkingPhotos) {
      const name = photo.talking_photo_name || 'Unnamed';
      if (filter && !name.toLowerCase().includes(filter)) continue;
      
      console.log(`  ID: ${photo.talking_photo_id}`);
      console.log(`  Name: ${name}`);
      if (photo.preview_image_url) {
        console.log(`  Preview: ${photo.preview_image_url.slice(0, 60)}...`);
      }
      console.log('');
      
      allItems.push({
        type: 'talking_photo',
        id: photo.talking_photo_id,
        name: name,
        preview: photo.preview_image_url,
      });
    }
  }

  // Process regular avatars (filter to custom ones if possible)
  const customAvatars = avatars.filter((a: any) => 
    a.avatar_name?.toLowerCase().includes('kelly') || 
    !a.avatar_name?.includes('HeyGen')
  );
  
  if (customAvatars.length > 0) {
    console.log(`\n🎭 CUSTOM AVATARS (${customAvatars.length}):`);
    console.log('─'.repeat(60));
    
    for (const avatar of customAvatars.slice(0, 20)) { // Limit to first 20
      const name = avatar.avatar_name || 'Unnamed';
      if (filter && !name.toLowerCase().includes(filter)) continue;
      
      console.log(`  ID: ${avatar.avatar_id}`);
      console.log(`  Name: ${name}`);
      console.log('');
      
      allItems.push({
        type: 'avatar',
        id: avatar.avatar_id,
        name: name,
        preview: avatar.preview_image_url,
      });
    }
  }

  // Process photo avatars
  if (photoAvatars.length > 0) {
    console.log(`\n🖼️ PHOTO AVATARS (${photoAvatars.length}):`);
    console.log('─'.repeat(60));
    
    for (const avatar of photoAvatars) {
      const id = avatar.photo_avatar_id || avatar.id || avatar.talking_photo_id;
      const name = avatar.name || avatar.photo_avatar_name || 'Unnamed';
      if (filter && !name.toLowerCase().includes(filter)) continue;
      
      console.log(`  ID: ${id}`);
      console.log(`  Name: ${name}`);
      console.log('');
      
      allItems.push({
        type: 'photo_avatar',
        id: id,
        name: name,
      });
    }
  }

  // Summary
  console.log('═'.repeat(60));
  console.log(`📊 SUMMARY:`);
  console.log(`   Talking Photos: ${talkingPhotos.length}`);
  console.log(`   Custom Avatars: ${customAvatars.length}`);
  console.log(`   Photo Avatars: ${photoAvatars.length}`);
  console.log(`   Total Items: ${allItems.length}`);

  // Save to JSON if requested
  if (jsonOutput || allItems.length > 0) {
    const outputPath = 'generated-images/heygen-avatar-library.json';
    fs.writeFileSync(outputPath, JSON.stringify(allItems, null, 2));
    console.log(`\n💾 Saved to: ${outputPath}`);
  }

  // Look for Kelly-named items specifically
  const kellyItems = allItems.filter(item => 
    item.name?.toLowerCase().includes('kelly') ||
    item.name?.toLowerCase().includes('scientist') ||
    item.name?.toLowerCase().includes('explorer') ||
    item.name?.toLowerCase().includes('rebel') ||
    item.name?.toLowerCase().includes('hypothesis') ||
    item.name?.toLowerCase().includes('discovery')
  );

  if (kellyItems.length > 0) {
    console.log(`\n🎯 KELLY MOTION LIBRARY CANDIDATES (${kellyItems.length}):`);
    console.log('─'.repeat(60));
    for (const item of kellyItems) {
      console.log(`  ${item.name}: ${item.id}`);
    }
  }
}

main().catch(console.error);
