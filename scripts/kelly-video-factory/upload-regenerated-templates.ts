/**
 * Upload Regenerated Templates to Supabase
 * 
 * Uploads all regenerated base video templates to Supabase storage
 * and outputs the public URLs for updating the artist portal.
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

interface UploadResult {
  templateId: string;
  templateName: string;
  localPath: string;
  publicUrl: string;
  imageUrl: string;
}

async function uploadTemplates(): Promise<UploadResult[]> {
  console.log('\n');
  console.log('╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║  ☁️  UPLOAD REGENERATED TEMPLATES TO SUPABASE                         ║');
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
  console.log(`  📊 Templates: ${latestReport.summary.successful}/${latestReport.summary.total}`);
  console.log(`  🪣 Bucket: ${BUCKET}/${FOLDER}`);
  console.log('');
  
  const results: UploadResult[] = [];
  
  for (const result of latestReport.results) {
    if (!result.success || !result.outputPath) continue;
    
    const localPath = result.outputPath;
    const templateId = result.templateId;
    const templateName = result.templateName;
    
    if (!fs.existsSync(localPath)) {
      console.log(`  ⚠️ File not found: ${localPath}`);
      continue;
    }
    
    const filename = path.basename(localPath);
    const storagePath = `${FOLDER}/${templateId}/${filename}`;
    
    console.log(`  📤 Uploading ${templateId}: ${templateName}...`);
    
    const fileBuffer = fs.readFileSync(localPath);
    
    const { error } = await supabase.storage
      .from(BUCKET)
      .upload(storagePath, fileBuffer, {
        contentType: 'video/mp4',
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
      templateName,
      localPath,
      publicUrl: urlData.publicUrl,
      imageUrl: result.imageUrl,
    });
  }
  
  // Save URL manifest
  const manifest = {
    generated: new Date().toISOString(),
    bucket: BUCKET,
    folder: FOLDER,
    templates: results,
  };
  
  const manifestPath = path.join(REGENERATED_DIR, 'supabase_urls.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log('\n');
  console.log('═'.repeat(70));
  console.log('📋 URL MANIFEST');
  console.log('═'.repeat(70));
  console.log('');
  console.log('// Copy these URLs to update artist-portal.html:');
  console.log('');
  console.log('const TEMPLATE_URLS = {');
  for (const r of results) {
    console.log(`  "${r.templateId}": {`);
    console.log(`    video: "${r.publicUrl}",`);
    console.log(`    poster: "${r.imageUrl}",`);
    console.log(`  },`);
  }
  console.log('};');
  console.log('');
  console.log(`  📄 Manifest saved: ${manifestPath}`);
  console.log('═'.repeat(70));
  
  return results;
}

uploadTemplates().catch(console.error);

