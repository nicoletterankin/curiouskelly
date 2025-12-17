#!/usr/bin/env npx tsx
/**
 * Check for recently created avatars/talking photos
 */

import 'dotenv/config';
import * as fs from 'fs';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

async function main() {
  console.log('🔍 Checking for recent/photo-type avatars...\n');

  const response = await fetch('https://api.heygen.com/v2/avatars', {
    headers: { 
      'X-Api-Key': HEYGEN_API_KEY,
      'Accept': 'application/json'
    }
  });
  
  const data = await response.json();
  const avatars = data.data?.avatars || [];
  
  // Look for photo-type avatars (not the stock video ones)
  const photoAvatars = avatars.filter((av: any) => {
    // Check for indicators it's a photo avatar
    return av.avatar_type === 'talking_photo' ||
           av.avatar_type === 'photo' ||
           av.type === 'talking_photo' ||
           (av.preview_image_url && av.preview_image_url.includes('talking_photo'));
  });
  
  console.log(`Total avatars: ${avatars.length}`);
  console.log(`Photo-type avatars: ${photoAvatars.length}`);
  
  // Show sample of avatar types
  const types = new Set<string>();
  avatars.forEach((av: any) => {
    if (av.avatar_type) types.add(av.avatar_type);
    if (av.type) types.add(av.type);
  });
  console.log(`\nAvatar types found: ${[...types].join(', ')}`);
  
  // Show first 5 avatars with all their fields to understand structure
  console.log('\n📋 Sample avatar structure (first 3):');
  console.log('─'.repeat(60));
  for (const av of avatars.slice(0, 3)) {
    console.log(JSON.stringify(av, null, 2));
    console.log('─'.repeat(60));
  }

  // Check if there are any with 'talking_photo' in preview URL
  const talkingPhotoAvatars = avatars.filter((av: any) => 
    av.preview_image_url?.includes('talking_photo') ||
    av.avatar_id?.includes('talking_photo')
  );
  
  if (talkingPhotoAvatars.length > 0) {
    console.log(`\n🎯 Avatars with 'talking_photo' reference (${talkingPhotoAvatars.length}):`);
    for (const av of talkingPhotoAvatars.slice(0, 10)) {
      console.log(`  ${av.avatar_name}: ${av.avatar_id}`);
    }
  }

  // Also try fetching avatar groups to see if there's a Kelly group
  console.log('\n🔍 Checking avatar groups...');
  const groupsResponse = await fetch('https://api.heygen.com/v2/avatar_group.list', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const groupsData = await groupsResponse.json();
  
  if (groupsData.data?.avatar_groups) {
    console.log(`Found ${groupsData.data.avatar_groups.length} groups`);
    
    // Look for Kelly groups
    const kellyGroups = groupsData.data.avatar_groups.filter((g: any) =>
      (g.name || '').toLowerCase().includes('kelly') ||
      (g.group_name || '').toLowerCase().includes('kelly')
    );
    
    if (kellyGroups.length > 0) {
      console.log(`\n🎯 Kelly Groups (${kellyGroups.length}):`);
      for (const g of kellyGroups) {
        console.log(`  ${g.name || g.group_name}: ${g.avatar_group_id || g.id}`);
        
        // Try to get avatars in this group
        const groupDetailRes = await fetch(
          `https://api.heygen.com/v2/avatar_group/${g.avatar_group_id || g.id}`,
          { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
        );
        const groupDetail = await groupDetailRes.json();
        if (groupDetail.data?.avatars) {
          console.log(`    Avatars in group: ${groupDetail.data.avatars.length}`);
          for (const av of groupDetail.data.avatars.slice(0, 5)) {
            console.log(`      - ${av.avatar_name || av.id}: ${av.avatar_id || av.id}`);
          }
        }
      }
    }
  }
}

main().catch(console.error);
