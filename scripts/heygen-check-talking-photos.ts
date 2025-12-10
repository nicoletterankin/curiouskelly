#!/usr/bin/env npx tsx
import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

async function main() {
  console.log('🔍 Checking all talking photos in account...\n');
  
  const response = await fetch('https://api.heygen.com/v1/talking_photo.list', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  
  const result = await response.json();
  console.log('Full response:');
  console.log(JSON.stringify(result, null, 2));
  
  if (result.data?.length > 0) {
    console.log('\n\n📸 TALKING PHOTOS FOUND:');
    result.data.forEach((tp: any, i: number) => {
      console.log(`\n${i + 1}. ID: ${tp.id}`);
      console.log(`   Image: ${tp.image_url?.substring(0, 80)}...`);
      console.log(`   Circle: ${tp.circle_image || 'none'}`);
    });
  }
  
  // Also check v2 avatars for photo_avatars
  console.log('\n\n🔍 Checking v2 avatars for photo types...');
  const avatarsResponse = await fetch('https://api.heygen.com/v2/avatars', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const avatarsResult = await avatarsResponse.json();
  
  if (avatarsResult.data?.avatars) {
    const photoAvatars = avatarsResult.data.avatars.filter((a: any) => 
      a.avatar_type === 'photo_avatar' || 
      a.avatar_type === 'talking_photo' ||
      a.preview_image_url?.includes('talking_photo')
    );
    
    console.log(`Found ${photoAvatars.length} photo-type avatars out of ${avatarsResult.data.avatars.length} total`);
    
    if (photoAvatars.length > 0) {
      console.log('\n📸 PHOTO AVATARS:');
      photoAvatars.slice(0, 10).forEach((a: any) => {
        console.log(`   ID: ${a.avatar_id}`);
        console.log(`   Name: ${a.avatar_name}`);
        console.log(`   Type: ${a.avatar_type}`);
        console.log(`   ---`);
      });
    }
    
    // Show first few avatars of any type to understand structure
    console.log('\n\n📋 Sample avatar structure (first 3):');
    avatarsResult.data.avatars.slice(0, 3).forEach((a: any) => {
      console.log(JSON.stringify(a, null, 2));
      console.log('---');
    });
  }
}

main().catch(console.error);

