#!/usr/bin/env npx tsx
/**
 * Map Kelly talking photos to archetypes
 * Gets details including preview images
 */
import 'dotenv/config';
import * as fs from 'fs';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

async function main() {
  console.log('🔍 Fetching all Kelly talking photos...\n');

  const response = await fetch(
    'https://api.heygen.com/v1/talking_photo.list',
    { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
  );
  
  const data = await response.json();
  const photos = data.data || [];
  
  console.log(`Total talking photos: ${photos.length}\n`);
  
  // Get first 70 (your Kelly avatars are likely in here)
  const kellyPhotos = photos.slice(0, 70);
  
  console.log('Recent 70 photos with details:\n');
  console.log('NUM  ID                                       IMAGE_URL_PREVIEW');
  console.log('─'.repeat(100));
  
  const mapping: any[] = [];
  
  kellyPhotos.forEach((p: any, i: number) => {
    const id = p.talking_photo_id || p.id;
    const imageUrl = p.image_url || p.circle_image || '';
    const preview = imageUrl.substring(0, 60);
    
    console.log(`${String(i+1).padStart(2)}   ${id}   ${preview}...`);
    
    mapping.push({
      index: i + 1,
      id: id,
      image_url: imageUrl,
    });
  });

  // Save mapping
  fs.writeFileSync(
    'generated-images/kelly-talking-photos.json',
    JSON.stringify(mapping, null, 2)
  );
  
  console.log('\n💾 Saved to generated-images/kelly-talking-photos.json');
  console.log('\n📋 NEXT: Tell me which IDs map to which archetype!');
  console.log('   e.g., "1-12 are the 12 archetypes in order: Scientist, Explorer, Rebel..."');
}

main().catch(console.error);

















