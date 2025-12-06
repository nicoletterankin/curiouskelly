#!/usr/bin/env node
/**
 * Register existing images to DB and generate animations for Days 6-17
 */

require('dotenv').config();
const Replicate = require('replicate');
const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

const supabase = createClient(
    process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
);

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

const SVD_MODEL = 'stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438';

const PHASES = ['hook', 'q1', 'q2', 'q3', 'wisdom'];

async function main() {
    console.log('══════════════════════════════════════════════════════════════════════');
    console.log('🎬 REGISTER & ANIMATE');
    console.log('   Register images, generate animations for Days 6-17');
    console.log('══════════════════════════════════════════════════════════════════════\n');

    const imagesDir = path.join(__dirname, '../../template-forge/production-images');
    const animsDir = path.join(__dirname, '../../template-forge/production-anims');
    
    if (!fs.existsSync(animsDir)) {
        fs.mkdirSync(animsDir, { recursive: true });
    }

    // Find all days that need animations (check by both DB and local files)
    const { data: existing } = await supabase
        .from('kelly_video_assets')
        .select('day_number, phase')
        .eq('asset_type', 'animation');
    
    const existingDays = new Set((existing || []).map(r => r.day_number));
    console.log(`  DB has animations for days: ${[...existingDays].sort((a,b)=>a-b).join(', ') || 'none'}`);
    
    // Get all image files
    const imageFiles = fs.readdirSync(imagesDir)
        .filter(f => f.endsWith('.png') && f.startsWith('day_'));
    
    console.log(`  Found ${imageFiles.length} image files`);
    console.log(`  Days with animations: ${[...existingDays].join(', ') || 'none'}`);
    
    // Process each day that needs animations (skip days 1-5 which we already have)
    const daysToProcess = [];
    for (const file of imageFiles) {
        const match = file.match(/day_(\d+)_(\w+)\.png/);
        if (!match) continue;
        
        const dayNum = parseInt(match[1]);
        const phase = match[2];
        
        // Skip days 1-5 (already have animations) or any day in DB
        if (dayNum <= 5 || existingDays.has(dayNum)) {
            continue;
        }
        daysToProcess.push({ dayNum, phase, file });
    }
    
    // Group by day
    const dayGroups = {};
    for (const item of daysToProcess) {
        if (!dayGroups[item.dayNum]) dayGroups[item.dayNum] = [];
        dayGroups[item.dayNum].push(item);
    }
    
    const daysNeedingAnims = Object.keys(dayGroups).sort((a,b) => a-b);
    console.log(`  Days needing animations: ${daysNeedingAnims.join(', ')}`);
    console.log(`  Total images to animate: ${daysToProcess.length}\n`);
    
    if (daysToProcess.length === 0) {
        console.log('  ✅ All days have animations!');
        return;
    }
    
    let generated = 0;
    let failed = 0;
    
    // Process day 6 first as a test
    const firstDay = daysNeedingAnims[0];
    console.log(`\n📸 Processing Day ${firstDay}...`);
    
    for (const item of dayGroups[firstDay]) {
        const imagePath = path.join(imagesDir, item.file);
        const animFile = `day_${String(item.dayNum).padStart(3, '0')}_${item.phase}.mp4`;
        const animPath = path.join(animsDir, animFile);
        
        // Check if animation exists locally
        if (fs.existsSync(animPath)) {
            console.log(`  ⏭️ ${animFile} (exists)`);
            continue;
        }
        
        process.stdout.write(`  🎬 ${animFile}...`);
        
        try {
            // Read image and upload to Replicate
            const imageData = fs.readFileSync(imagePath);
            const base64 = imageData.toString('base64');
            const dataUrl = `data:image/png;base64,${base64}`;
            
            // Generate animation
            const output = await replicate.run(SVD_MODEL, {
                input: {
                    input_image: dataUrl,
                    video_length: '14_frames_with_svd',
                    sizing_strategy: 'maintain_aspect_ratio',
                    motion_bucket_id: 40,
                    cond_aug: 0.02,
                    fps: 6
                }
            });
            
            // Download animation
            const response = await fetch(output);
            const buffer = Buffer.from(await response.arrayBuffer());
            fs.writeFileSync(animPath, buffer);
            
            // Upload to Supabase Storage
            const { error: uploadError } = await supabase.storage
                .from('kelly-templates')
                .upload(`production/animations/${animFile}`, buffer, {
                    contentType: 'video/mp4',
                    upsert: true
                });
            
            if (uploadError) throw uploadError;
            
            const publicUrl = `${process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL}/storage/v1/object/public/kelly-templates/production/animations/${animFile}`;
            
            // Register in DB
            await supabase.from('kelly_video_assets').upsert({
                day_number: item.dayNum,
                phase: item.phase,
                asset_type: 'animation',
                storage_path: `production/animations/${animFile}`,
                public_url: publicUrl,
                status: 'completed'
            }, {
                onConflict: 'day_number,phase,age_bucket,language,asset_type'
            });
            
            console.log(' ✅');
            generated++;
            
        } catch (err) {
            console.log(` ❌ ${err.message}`);
            failed++;
        }
    }
    
    console.log('\n══════════════════════════════════════════════════════════════════════');
    console.log('📊 COMPLETE');
    console.log('══════════════════════════════════════════════════════════════════════');
    console.log(`  Generated: ${generated}`);
    console.log(`  Failed: ${failed}`);
    console.log(`  Remaining days: ${daysNeedingAnims.slice(1).join(', ')}`);
}

main().catch(console.error);

