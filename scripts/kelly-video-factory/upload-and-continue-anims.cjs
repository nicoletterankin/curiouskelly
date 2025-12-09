#!/usr/bin/env node
/**
 * Upload existing animations to Supabase and continue generating
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');
const https = require('https');

const supabase = createClient(
    process.env.PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
);

const ANIMS_DIR = path.join(__dirname, '../../template-forge/production-animations');
const IMAGES_DIR = path.join(__dirname, '../../template-forge/production-images');

// Replicate animation
const REPLICATE_TOKEN = process.env.REPLICATE_API_TOKEN;
const SVD_VERSION = '3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438';

async function replicateRequest(method, urlPath, data = null) {
    return new Promise((resolve, reject) => {
        const options = {
            hostname: 'api.replicate.com',
            path: `/v1${urlPath}`,
            method,
            headers: {
                'Authorization': `Bearer ${REPLICATE_TOKEN}`,
                'Content-Type': 'application/json',
            },
        };
        
        const req = https.request(options, (res) => {
            let body = [];
            res.on('data', chunk => body.push(chunk));
            res.on('end', () => {
                try {
                    resolve(JSON.parse(Buffer.concat(body).toString()));
                } catch (e) {
                    reject(e);
                }
            });
        });
        req.on('error', reject);
        if (data) req.write(JSON.stringify(data));
        req.end();
    });
}

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function downloadFile(url, filepath) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(filepath);
        https.get(url, (res) => {
            if (res.statusCode === 302 || res.statusCode === 301) {
                https.get(res.headers.location, (res2) => {
                    res2.pipe(file);
                    file.on('finish', () => { file.close(); resolve(); });
                }).on('error', reject);
            } else {
                res.pipe(file);
                file.on('finish', () => { file.close(); resolve(); });
            }
        }).on('error', reject);
    });
}

async function main() {
    console.log('══════════════════════════════════════════════════════════════════════');
    console.log('🎬 UPLOAD & CONTINUE ANIMATIONS');
    console.log('══════════════════════════════════════════════════════════════════════\n');

    // Step 1: Upload existing animations
    console.log('📤 Step 1: Uploading existing animations...\n');
    
    const localAnims = fs.readdirSync(ANIMS_DIR).filter(f => f.endsWith('.mp4'));
    console.log(`  Found ${localAnims.length} local animations`);

    let uploaded = 0;
    for (const animFile of localAnims) {
        const match = animFile.match(/day_(\d+)_(\w+)\.mp4/);
        if (!match) continue;
        
        const dayNum = parseInt(match[1]);
        const phase = match[2];
        
        // Check if already in DB
        const { data: existing } = await supabase
            .from('kelly_video_assets')
            .select('id')
            .eq('day_number', dayNum)
            .eq('phase', phase)
            .eq('asset_type', 'animation')
            .single();
        
        if (existing) {
            process.stdout.write('.');
            continue;
        }
        
        // Upload to Supabase Storage
        const filePath = path.join(ANIMS_DIR, animFile);
        const fileBuffer = fs.readFileSync(filePath);
        const storagePath = `production/animations/${animFile}`;
        
        const { error: uploadError } = await supabase.storage
            .from('kelly-templates')
            .upload(storagePath, fileBuffer, {
                contentType: 'video/mp4',
                upsert: true
            });
        
        if (uploadError) {
            console.log(`  ❌ Upload failed: ${animFile} - ${uploadError.message}`);
            continue;
        }
        
        const publicUrl = `${process.env.PUBLIC_SUPABASE_URL}/storage/v1/object/public/kelly-templates/${storagePath}`;
        
        // Register in DB
        await supabase.from('kelly_video_assets').insert({
            day_number: dayNum,
            phase: phase,
            asset_type: 'animation',
            storage_path: storagePath,
            public_url: publicUrl,
            status: 'completed'
        });
        
        console.log(`  ✅ ${animFile}`);
        uploaded++;
    }
    
    console.log(`\n  Uploaded: ${uploaded} new animations\n`);

    // Step 2: Find images that need animations
    console.log('📸 Step 2: Finding images needing animations...\n');
    
    const localImages = fs.readdirSync(IMAGES_DIR).filter(f => f.endsWith('.png'));
    const animatedDays = new Set(localAnims.map(f => {
        const m = f.match(/day_(\d+)/);
        return m ? parseInt(m[1]) : 0;
    }));
    
    const needAnimation = localImages.filter(f => {
        const m = f.match(/day_(\d+)/);
        const dayNum = m ? parseInt(m[1]) : 0;
        return !animatedDays.has(dayNum);
    });
    
    // Group by day
    const dayGroups = {};
    for (const img of needAnimation) {
        const m = img.match(/day_(\d+)/);
        const dayNum = m ? parseInt(m[1]) : 0;
        if (!dayGroups[dayNum]) dayGroups[dayNum] = [];
        dayGroups[dayNum].push(img);
    }
    
    const daysToAnimate = Object.keys(dayGroups).sort((a,b) => a-b);
    console.log(`  Need animations for days: ${daysToAnimate.join(', ') || 'none'}`);
    console.log(`  Total images: ${needAnimation.length}\n`);
    
    if (daysToAnimate.length === 0) {
        console.log('  ✅ All images have animations!\n');
        return;
    }
    
    // Step 3: Generate animations for first 2 days (in parallel)
    const targetDays = daysToAnimate.slice(0, 2);
    console.log(`🎬 Step 3: Generating animations for days ${targetDays.join(', ')}...\n`);
    
    let generated = 0;
    for (const day of targetDays) {
        console.log(`  Day ${day}:`);
        
        for (const imgFile of dayGroups[day]) {
            const match = imgFile.match(/day_(\d+)_(\w+)\.png/);
            if (!match) continue;
            
            const phase = match[2];
            const animFile = `day_${String(day).padStart(3, '0')}_${phase}.mp4`;
            const animPath = path.join(ANIMS_DIR, animFile);
            
            if (fs.existsSync(animPath)) {
                process.stdout.write('.');
                continue;
            }
            
            process.stdout.write(`    ${animFile}...`);
            
            try {
                // Read image and convert to base64
                const imgPath = path.join(IMAGES_DIR, imgFile);
                const imgData = fs.readFileSync(imgPath);
                const base64 = imgData.toString('base64');
                const dataUrl = `data:image/png;base64,${base64}`;
                
                // Create prediction
                const prediction = await replicateRequest('POST', '/predictions', {
                    version: SVD_VERSION,
                    input: {
                        input_image: dataUrl,
                        video_length: '14_frames_with_svd',
                        fps: 8,
                        motion_bucket_id: 80,
                        cond_aug: 0.02,
                        decoding_t: 7,
                    }
                });
                
                // Poll for completion
                while (true) {
                    await sleep(5000);
                    const status = await replicateRequest('GET', `/predictions/${prediction.id}`);
                    
                    if (status.status === 'succeeded') {
                        const videoUrl = Array.isArray(status.output) ? status.output[0] : status.output;
                        
                        // Download video
                        await downloadFile(videoUrl, animPath);
                        
                        // Upload to Supabase
                        const fileBuffer = fs.readFileSync(animPath);
                        const storagePath = `production/animations/${animFile}`;
                        
                        await supabase.storage.from('kelly-templates').upload(storagePath, fileBuffer, {
                            contentType: 'video/mp4',
                            upsert: true
                        });
                        
                        const publicUrl = `${process.env.PUBLIC_SUPABASE_URL}/storage/v1/object/public/kelly-templates/${storagePath}`;
                        
                        await supabase.from('kelly_video_assets').insert({
                            day_number: parseInt(day),
                            phase: phase,
                            asset_type: 'animation',
                            storage_path: storagePath,
                            public_url: publicUrl,
                            status: 'completed'
                        });
                        
                        console.log(' ✅');
                        generated++;
                        break;
                    }
                    
                    if (status.status === 'failed') {
                        console.log(` ❌ ${status.error}`);
                        break;
                    }
                    
                    if (status.status === 'canceled') {
                        console.log(' ❌ Canceled');
                        break;
                    }
                    
                    process.stdout.write('.');
                }
                
            } catch (err) {
                console.log(` ❌ ${err.message}`);
            }
        }
        console.log();
    }
    
    console.log('══════════════════════════════════════════════════════════════════════');
    console.log('📊 COMPLETE');
    console.log('══════════════════════════════════════════════════════════════════════');
    console.log(`  Uploaded: ${uploaded}`);
    console.log(`  Generated: ${generated}`);
    console.log(`  Remaining days: ${daysToAnimate.slice(2).join(', ')}`);
}

main().catch(console.error);


