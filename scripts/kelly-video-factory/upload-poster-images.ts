/**
 * Upload Poster Images to Supabase
 * 
 * Downloads poster images from Replicate and uploads to Supabase
 * for permanent storage (replicate URLs are temporary).
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const BUCKET = 'kelly-templates';
const FOLDER = 'base-templates-v2';

const REGENERATED_DIR = path.join(process.cwd(), 'template-forge', 'regenerated-templates');

interface PosterResult {
  templateId: string;
  originalUrl: string;
  supabaseUrl: string;
}

async function uploadPosterImages(): Promise<PosterResult[]> {
  console.log('\n');
  console.log('╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║  🖼️  UPLOAD POSTER IMAGES TO SUPABASE                                 ║');
  console.log('╚══════════════════════════════════════════════════════════════════════╝');
  console.log('');
  
  const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
  
  // Read the batch report
  const reports = fs.readdirSync(REGENERATED_DIR)
    .filter(f => f.startsWith('batch_report_'))
    .sort()
    .reverse();
  
  if (reports.length === 0) {
    throw new Error('No batch report found');
  }
  
  const latestReport = JSON.parse(
    fs.readFileSync(path.join(REGENERATED_DIR, reports[0]), 'utf-8')
  );
  
  console.log(`  📄 Using report: ${reports[0]}`);
  console.log(`  🪣 Bucket: ${BUCKET}/${FOLDER}`);
  console.log('');
  
  const results: PosterResult[] = [];
  
  for (const result of latestReport.results) {
    if (!result.success || !result.imageUrl) continue;
    
    const templateId = result.templateId;
    const imageUrl = result.imageUrl;
    
    console.log(`  📤 Uploading poster for ${templateId}...`);
    
    try {
      // Download the image
      const response = await fetch(imageUrl);
      if (!response.ok) {
        console.log(`     ⚠️ Failed to download: ${response.status}`);
        continue;
      }
      
      const buffer = Buffer.from(await response.arrayBuffer());
      const filename = `${templateId.toLowerCase()}_poster.png`;
      const storagePath = `${FOLDER}/${templateId}/${filename}`;
      
      // Upload to Supabase
      const { error } = await supabase.storage
        .from(BUCKET)
        .upload(storagePath, buffer, {
          contentType: 'image/png',
          upsert: true,
        });
      
      if (error) {
        console.log(`     ❌ Error: ${error.message}`);
        continue;
      }
      
      const { data: urlData } = supabase.storage
        .from(BUCKET)
        .getPublicUrl(storagePath);
      
      console.log(`     ✅ ${urlData.publicUrl}`);
      
      results.push({
        templateId,
        originalUrl: imageUrl,
        supabaseUrl: urlData.publicUrl,
      });
      
    } catch (err: any) {
      console.log(`     ❌ Error: ${err.message}`);
    }
  }
  
  // Save poster manifest
  const manifestPath = path.join(REGENERATED_DIR, 'poster_urls.json');
  fs.writeFileSync(manifestPath, JSON.stringify({
    generated: new Date().toISOString(),
    posters: results,
  }, null, 2));
  
  console.log('\n');
  console.log('═'.repeat(70));
  console.log('📋 POSTER URL MANIFEST');
  console.log('═'.repeat(70));
  console.log('');
  
  for (const r of results) {
    console.log(`  "${r.templateId}": "${r.supabaseUrl}",`);
  }
  
  console.log('');
  console.log(`  📄 Manifest saved: ${manifestPath}`);
  console.log('═'.repeat(70));
  
  return results;
}

uploadPosterImages().catch(console.error);


