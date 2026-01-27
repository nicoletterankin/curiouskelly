/**
 * Re-submit HeyGen jobs with CORRECT Kelly avatar
 * 
 * Previous submissions used talking_photo with wrong ID.
 * This resets those and uses the custom Kelly avatar: 1bd001e7e50f421d891986aad5158bc8
 */

const { createClient } = require('@supabase/supabase-js');
const fetch = require('node-fetch');
require('dotenv').config({ path: 'C:/Users/user/ANTIGRAVITY/media_keys.env' });

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const HEYGEN_KEY = process.env.HEYGEN_API_KEY;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// CORRECT Kelly Avatar - Kelly in Blue Shirt (Front)
// This is the HeyGen public Kelly avatar matching our brand
const KELLY_AVATAR_ID = 'Kelly_Blue_Shirt_Front';

async function submitToHeyGen(audioUrl) {
  const payload = {
    video_inputs: [{
      character: {
        type: 'avatar',
        avatar_id: KELLY_AVATAR_ID,
        avatar_style: 'normal'
      },
      voice: {
        type: 'audio',
        audio_url: audioUrl
      }
    }],
    dimension: {
      width: 1080,
      height: 1920  // Portrait for mobile
    }
  };
  
  const resp = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  
  const data = await resp.json();
  
  if (data.error) {
    throw new Error(`HeyGen error: ${JSON.stringify(data.error)}`);
  }
  
  return data.data?.video_id;
}

async function resubmitAll() {
  console.log('='.repeat(60));
  console.log('RE-SUBMITTING WITH CORRECT KELLY AVATAR');
  console.log('Avatar ID:', KELLY_AVATAR_ID);
  console.log('='.repeat(60));
  
  // Get all HeyGen jobs for Day 20 that used wrong avatar
  const { data: jobs, error } = await supabase
    .from('video_jobs')
    .select('*')
    .eq('engine', 'heygen')
    .eq('day_of_year', 20)
    .in('status', ['completed', 'submitted', 'queued']);
  
  if (error) {
    console.error('DB error:', error);
    return;
  }
  
  console.log(`Found ${jobs.length} HeyGen jobs to check`);
  
  // Filter jobs that used talking_photo or wrong avatar
  const wrongJobs = jobs.filter(j => {
    const payload = j.input_payload;
    // Wrong if: has talking_photo_id, OR avatar_id is not Kelly_Blue_Shirt_Front
    return payload?.talking_photo_id || 
           (payload?.avatar_id && payload.avatar_id !== 'Kelly_Blue_Shirt_Front') ||
           !payload?.avatar_id;
  });
  
  console.log(`Jobs using wrong avatar: ${wrongJobs.length}`);
  
  if (wrongJobs.length === 0) {
    console.log('All jobs already using correct avatar!');
    return;
  }
  
  let resubmitted = 0;
  let errors = [];
  
  for (const job of wrongJobs) {
    try {
      const audioUrl = job.input_payload?.audio_url;
      if (!audioUrl) {
        console.log(`  Skip ${job.id} - no audio_url`);
        continue;
      }
      
      console.log(`  Resubmitting: ${job.language}/${job.age_category}/${job.phase}`);
      
      // Submit with correct avatar
      const videoId = await submitToHeyGen(audioUrl);
      console.log(`    HeyGen video ID: ${videoId}`);
      
      // Update job
      await supabase
        .from('video_jobs')
        .update({
          external_id: videoId,
          status: 'submitted',
          output_url: null,  // Clear old output
          submitted_at: new Date().toISOString(),
          input_payload: {
            ...job.input_payload,
            avatar_id: KELLY_AVATAR_ID,
            avatar_type: 'avatar'
          },
          notes: 'Resubmitted with correct Kelly avatar'
        })
        .eq('id', job.id);
      
      resubmitted++;
      
      // Rate limit
      await new Promise(r => setTimeout(r, 1000));
      
    } catch (err) {
      console.log(`    ERROR: ${err.message}`);
      errors.push({ job: `${job.language}/${job.age_category}/${job.phase}`, error: err.message });
    }
  }
  
  console.log('\n' + '='.repeat(60));
  console.log('RESUBMIT SUMMARY');
  console.log('='.repeat(60));
  console.log(`Resubmitted: ${resubmitted}`);
  console.log(`Errors: ${errors.length}`);
  
  if (errors.length > 0) {
    console.log('\nErrors:');
    errors.slice(0, 10).forEach(e => console.log(`  ${e.job}: ${e.error}`));
  }
}

async function checkAvatar() {
  console.log('Verifying Kelly Avatar...');
  console.log('Avatar ID:', KELLY_AVATAR_ID);
  
  // Check avatar exists
  const resp = await fetch(`https://api.heygen.com/v2/avatars/${KELLY_AVATAR_ID}`, {
    headers: { 'X-Api-Key': HEYGEN_KEY }
  });
  
  const data = await resp.json();
  console.log('Response:', JSON.stringify(data, null, 2).slice(0, 500));
}

// Parse command
const cmd = process.argv[2] || 'check';
if (cmd === 'run') {
  resubmitAll().catch(console.error);
} else {
  checkAvatar().catch(console.error);
}
