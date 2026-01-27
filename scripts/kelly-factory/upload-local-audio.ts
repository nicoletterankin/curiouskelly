#!/usr/bin/env npx tsx
/**
 * Upload local audio files to Supabase and update registry
 * Converts file:// URLs to public https:// URLs
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import fs from 'fs';
import path from 'path';

const CONFIG = {
  supabaseUrl: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '',
  supabaseKey: process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY || '',
};

const supabase = createClient(CONFIG.supabaseUrl, CONFIG.supabaseKey);

async function main() {
  console.log('📤 Uploading local audio files to Supabase...\n');

  // Get all assets with file:// URLs
  const { data: assets, error } = await supabase
    .from('kelly_lesson_assets')
    .select('*')
    .like('audio_url', 'file://%');

  if (error) {
    console.error('Error fetching assets:', error);
    return;
  }

  if (!assets || assets.length === 0) {
    console.log('No local audio files found.');
    return;
  }

  console.log(`Found ${assets.length} assets with local audio files\n`);

  let success = 0;
  let failed = 0;

  for (const asset of assets) {
    // Convert file:// URL to local path
    // file:///mnt/c/Users/... → C:\Users\...
    let localPath = asset.audio_url
      .replace('file://', '')
      .replace('/mnt/c/', 'C:/')
      .replace('/mnt/d/', 'D:/')
      .replace(/\//g, '\\');

    console.log(`📦 Day ${asset.day_number} | ${asset.phase} | Age ${asset.age_group}`);
    console.log(`   Local: ${localPath}`);

    // Check if file exists
    if (!fs.existsSync(localPath)) {
      console.log(`   ❌ File not found`);
      failed++;
      continue;
    }

    try {
      // Read file
      const buffer = fs.readFileSync(localPath);
      
      // Upload to Supabase
      const storagePath = `factory/day-${asset.day_number}/${asset.phase}-age${asset.age_group}-${asset.language}.mp3`;
      
      const { error: uploadError } = await supabase.storage
        .from('kelly-templates')
        .upload(storagePath, buffer, { 
          contentType: 'audio/mpeg', 
          upsert: true 
        });

      if (uploadError) {
        console.log(`   ❌ Upload error: ${uploadError.message}`);
        failed++;
        continue;
      }

      // Get public URL
      const { data: urlData } = supabase.storage
        .from('kelly-templates')
        .getPublicUrl(storagePath);

      const publicUrl = urlData.publicUrl;

      // Update registry
      await supabase
        .from('kelly_lesson_assets')
        .update({ 
          audio_url: publicUrl,
          updated_at: new Date().toISOString() 
        })
        .eq('id', asset.id);

      console.log(`   ✅ Uploaded: ${publicUrl.substring(0, 70)}...`);
      success++;

    } catch (err) {
      console.log(`   ❌ Error: ${(err as Error).message}`);
      failed++;
    }
  }

  console.log(`
╔══════════════════════════════════════════════════════════════════════════╗
║                              📊 SUMMARY                                  ║
╚══════════════════════════════════════════════════════════════════════════╝

Total: ${assets.length}
  ✅ Uploaded: ${success}
  ❌ Failed: ${failed}
`);
}

main().catch(console.error);
