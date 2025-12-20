#!/usr/bin/env npx tsx
/**
 * Get all Kelly avatar IDs from the group
 */
import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const KELLY_GROUP_ID = 'a762125d3107477aba43d1bd79f90d6e';

async function main() {
  console.log('🔍 Fetching Kelly avatar group...\n');

  // Try to get avatars from the group
  const response = await fetch(
    `https://api.heygen.com/v2/avatars?limit=100`,
    { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
  );

  const data = await response.json();
  
  if (data.data?.avatars) {
    // Filter for Kelly group
    const kellyAvatars = data.data.avatars.filter((a: any) => 
      a.avatar_group_id === KELLY_GROUP_ID || 
      a.group_id === KELLY_GROUP_ID ||
      a.avatar_name?.toLowerCase().includes('kelly')
    );

    console.log(`Found ${kellyAvatars.length} Kelly avatars:\n`);
    console.log('ID'.padEnd(40) + 'NAME');
    console.log('─'.repeat(70));
    
    kellyAvatars.forEach((a: any) => {
      console.log(`${(a.avatar_id || a.id).padEnd(40)} ${a.avatar_name || a.name || 'unnamed'}`);
    });

    // Save to JSON
    const fs = await import('fs');
    fs.writeFileSync(
      'generated-images/kelly-avatar-ids.json',
      JSON.stringify(kellyAvatars, null, 2)
    );
    console.log('\n💾 Saved to generated-images/kelly-avatar-ids.json');
  }

  // Also try talking photos
  console.log('\n\n🔍 Checking talking photos...\n');
  const tpResponse = await fetch(
    'https://api.heygen.com/v1/talking_photo.list',
    { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
  );
  
  const tpData = await tpResponse.json();
  
  // Get recent ones (first 20)
  const recentPhotos = tpData.data?.slice(0, 30) || [];
  console.log(`Recent ${recentPhotos.length} talking photos:\n`);
  
  recentPhotos.forEach((p: any, i: number) => {
    console.log(`${i+1}. ${p.talking_photo_id || p.id}`);
  });
}

main().catch(console.error);
















