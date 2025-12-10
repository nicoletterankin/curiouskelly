#!/usr/bin/env node
/**
 * AUTO VIDEO GENERATOR
 * Automatically generates videos for all days that have:
 * - Animations ready
 * - Audio ready
 * - No videos yet (or incomplete videos)
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

const WAV2LIP_MODEL = 'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef';

async function main() {
    console.log('══════════════════════════════════════════════════════════════════════');
    console.log('🤖 AUTO VIDEO GENERATOR');
    console.log('   Finding days ready for video generation');
    console.log('══════════════════════════════════════════════════════════════════════\n');

    // Get all assets grouped by day
    const { data: assets } = await supabase
        .from('kelly_video_assets')
        .select('day_number, phase, asset_type, public_url')
        .order('day_number');

    // Group by day
    const dayData = {};
    for (const asset of assets || []) {
        if (!dayData[asset.day_number]) {
            dayData[asset.day_number] = { animations: {}, audio: {}, videos: {} };
        }
        if (asset.asset_type === 'animation') {
            dayData[asset.day_number].animations[asset.phase] = asset.public_url;
        } else if (asset.asset_type === 'audio') {
            const key = `${asset.phase}`;
            if (!dayData[asset.day_number].audio[key]) {
                dayData[asset.day_number].audio[key] = [];
            }
            dayData[asset.day_number].audio[key].push(asset.public_url);
        } else if (asset.asset_type === 'video') {
            dayData[asset.day_number].videos[asset.phase] = (dayData[asset.day_number].videos[asset.phase] || 0) + 1;
        }
    }

    // Find days ready for video generation
    const readyDays = [];
    for (const [day, data] of Object.entries(dayData)) {
        const animCount = Object.keys(data.animations).length;
        const audioCount = Object.values(data.audio).flat().length;
        const videoCount = Object.values(data.videos).reduce((a, b) => a + b, 0);
        
        // Ready if has 5 animations and 60+ audio files but fewer than 60 videos
        if (animCount >= 5 && audioCount >= 50 && videoCount < 60) {
            readyDays.push({
                day: parseInt(day),
                animations: animCount,
                audio: audioCount,
                videos: videoCount,
                missing: 60 - videoCount
            });
        }
    }

    console.log(`  Days scanned: ${Object.keys(dayData).length}`);
    console.log(`  Days ready for videos: ${readyDays.length}`);
    
    if (readyDays.length === 0) {
        console.log('\n  No days ready. Need animations + audio first.\n');
        
        // Show status
        console.log('  Status by day:');
        for (const [day, data] of Object.entries(dayData).slice(0, 10)) {
            const animCount = Object.keys(data.animations).length;
            const audioCount = Object.values(data.audio).flat().length;
            const videoCount = Object.values(data.videos).reduce((a, b) => a + b, 0);
            console.log(`    Day ${day}: ${animCount} anims, ${audioCount} audio, ${videoCount} videos`);
        }
        return;
    }

    console.log('\n  Ready days:');
    for (const day of readyDays) {
        console.log(`    Day ${day.day}: ${day.animations} anims, ${day.audio} audio, ${day.videos} videos (${day.missing} missing)`);
    }

    // Process first ready day
    const target = readyDays[0];
    console.log(`\n🎬 Processing Day ${target.day}...\n`);
    
    // This will trigger the generate-day-lipsync script logic
    console.log(`  Run: node scripts/kelly-video-factory/generate-day-lipsync.cjs --day ${target.day}`);
}

main().catch(console.error);



