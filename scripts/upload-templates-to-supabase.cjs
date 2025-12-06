/**
 * 🚀 Upload Templates to Supabase Storage
 * 
 * Uploads the generated template videos to permanent storage
 * so they don't expire like Replicate CDN URLs.
 */

const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

require('dotenv').config();

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

const TEMPLATES_DIR = path.join(__dirname, '..', 'template-forge', 'templates');
const BUCKET = 'kelly-templates';

async function ensureBucket() {
  // Check if bucket exists
  const { data: buckets, error: listError } = await supabase.storage.listBuckets();
  
  if (listError) {
    console.log('⚠️  Could not list buckets:', listError.message);
    return false;
  }
  
  const bucketExists = buckets.some(b => b.name === BUCKET);
  
  if (!bucketExists) {
    console.log(`📦 Creating bucket: ${BUCKET}`);
    const { error: createError } = await supabase.storage.createBucket(BUCKET, {
      public: true,
      fileSizeLimit: 52428800, // 50MB
    });
    
    if (createError) {
      console.log('❌ Could not create bucket:', createError.message);
      return false;
    }
    console.log('✅ Bucket created');
  } else {
    console.log(`✅ Bucket exists: ${BUCKET}`);
  }
  
  return true;
}

async function uploadTemplate(filepath) {
  const filename = path.basename(filepath);
  const fileBuffer = fs.readFileSync(filepath);
  
  console.log(`📤 Uploading: ${filename} (${(fileBuffer.length / 1024 / 1024).toFixed(2)}MB)`);
  
  const { data, error } = await supabase.storage
    .from(BUCKET)
    .upload(`v1/${filename}`, fileBuffer, {
      contentType: 'video/mp4',
      upsert: true,
    });
  
  if (error) {
    console.log(`   ❌ Error: ${error.message}`);
    return null;
  }
  
  const { data: urlData } = supabase.storage
    .from(BUCKET)
    .getPublicUrl(`v1/${filename}`);
  
  console.log(`   ✅ Uploaded: ${urlData.publicUrl}`);
  return urlData.publicUrl;
}

async function main() {
  console.log('═'.repeat(70));
  console.log('🚀 UPLOAD TEMPLATES TO SUPABASE');
  console.log('═'.repeat(70));
  
  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    console.error('❌ Missing Supabase credentials');
    process.exit(1);
  }
  
  // Ensure bucket exists
  const bucketReady = await ensureBucket();
  if (!bucketReady) {
    console.log('⚠️  Proceeding anyway, bucket may already exist...');
  }
  
  // Get all template files
  const files = fs.readdirSync(TEMPLATES_DIR).filter(f => f.endsWith('.mp4'));
  console.log(`\n📁 Found ${files.length} template files\n`);
  
  const results = [];
  
  for (const file of files) {
    const filepath = path.join(TEMPLATES_DIR, file);
    const url = await uploadTemplate(filepath);
    
    if (url) {
      // Parse template info from filename
      const match = file.match(/^(T\d+)_([a-z_]+)_minimax/);
      if (match) {
        results.push({
          id: match[1],
          name: match[2].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
          filename: file,
          url,
        });
      }
    }
  }
  
  console.log('\n' + '═'.repeat(70));
  console.log('📊 UPLOAD SUMMARY');
  console.log('═'.repeat(70));
  console.log(`   Uploaded: ${results.length}/${files.length}`);
  
  if (results.length > 0) {
    console.log('\n   URLs for templates.html:');
    console.log('   ' + '-'.repeat(65));
    for (const r of results) {
      console.log(`   ${r.id}: ${r.url}`);
    }
  }
  
  // Save results
  const resultsFile = path.join(__dirname, '..', 'template-forge', 'supabase_urls.json');
  fs.writeFileSync(resultsFile, JSON.stringify(results, null, 2));
  console.log(`\n   📄 Saved URLs to: ${resultsFile}`);
}

main().catch(console.error);

