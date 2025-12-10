#!/usr/bin/env npx tsx
import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

async function main() {
  console.log('🔍 Finding Kelly in your HeyGen account...\n');
  
  // Check v2 avatars
  const response = await fetch('https://api.heygen.com/v2/avatars', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const result = await response.json();
  
  if (result.data?.avatars) {
    // Find avatars with "kelly" in the name
    const kellyAvatars = result.data.avatars.filter((a: any) => 
      a.avatar_name?.toLowerCase().includes('kelly')
    );
    
    console.log(`Found ${kellyAvatars.length} avatars with "kelly" in name:\n`);
    
    kellyAvatars.forEach((a: any) => {
      console.log(`📸 ${a.avatar_name}`);
      console.log(`   ID: ${a.avatar_id}`);
      console.log(`   Type: ${a.avatar_type}`);
      console.log(`   Status: ${a.status || 'unknown'}`);
      console.log(`   Preview: ${a.preview_image_url?.substring(0, 60)}...`);
      console.log('');
    });
    
    // Also show any photo_avatar types
    const photoAvatars = result.data.avatars.filter((a: any) => 
      a.avatar_type === 'photo_avatar'
    );
    
    if (photoAvatars.length > 0) {
      console.log(`\n\n📷 Found ${photoAvatars.length} photo avatars:\n`);
      photoAvatars.forEach((a: any) => {
        console.log(`   ${a.avatar_id}: ${a.avatar_name} (${a.status || 'unknown'})`);
      });
    }
  }
  
  // Also check talking photos
  console.log('\n\n🎤 Checking talking photos...');
  const tpResponse = await fetch('https://api.heygen.com/v1/talking_photo.list', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const tpResult = await tpResponse.json();
  
  if (tpResult.data?.length > 0) {
    // Show first few that might be custom (not stock)
    console.log(`Found ${tpResult.data.length} talking photos total`);
    console.log('\nFirst 10 (most likely custom):');
    tpResult.data.slice(0, 10).forEach((tp: any) => {
      console.log(`   ID: ${tp.id}`);
      console.log(`   Image: ${tp.image_url?.substring(0, 50)}...`);
      console.log('');
    });
  }
}

main().catch(console.error);

