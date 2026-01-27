/**
 * Re-submit HeyGen jobs with CORRECT Kelly avatar
 * Using talking_photo_id with look IDs
 */

const { createClient } = require('@supabase/supabase-js');
const fetch = require('node-fetch');
const { getKellyAvatar } = require('./kelly-avatars');
require('dotenv').config({ path: 'C:/Users/user/ANTIGRAVITY/media_keys.env' });

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const HEYGEN_KEY = process.env.HEYGEN_API_KEY;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

async function submitToHeyGen(audioUrl, age, archetype = 'storyteller') {
  const kelly = getKellyAvatar(age, archetype);
  
  // Use talking_photo_id with look_id (this works!)
  const payload = {
    video_inputs: [{
      character: {
        type: 'talking_photo',
        talking_photo_id: kelly.look_id
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
  
  return {
    video_id: data.data?.video_id,
    look_id: kelly.look_id,
    age_group: kelly.age_group
  };
}

async function resubmitAll() {
  console.log('='.repeat(60));
  console.log('RE-SUBMITTING WITH CORRECT KELLY AVATARS');
  console.log('Using talking_photo_id with age-specific look IDs');
  console.log('='.repeat(60));
  
  // Get all HeyGen jobs for Day 20
  const { data: jobs, error } = await supabase
    .from('video_jobs')
    .select('*')
    .eq('engine', 'heygen')
    .eq('day_of_year', 20)
    .in('status', ['completed', 'submitted', 'queued', 'failed']);
  
  if (error) {
    console.error('DB error:', error);
    return;
  }
  
  console.log(`Found ${jobs.length} HeyGen jobs`);
  
  // Filter jobs that need correct Kelly avatar
  const needsResubmit = jobs.filter(j => {
    const payload = j.input_payload;
    // Check if using wrong avatar (public Kelly or talking_photo with wrong ID)
    const kelly = getKellyAvatar(j.age_category);
    return payload?.avatar_id === 'Kelly_Blue_Shirt_Front' ||
           (payload?.talking_photo_id && payload.talking_photo_id !== kelly.look_id);
  });
  
  console.log(`Jobs needing correct Kelly: ${needsResubmit.length}`);
  
  if (needsResubmit.length === 0) {
    console.log('All jobs already using correct Kelly avatar!');
    return;
  }
  
  let resubmitted = 0;
  let errors = [];
  
  for (const job of needsResubmit) {
    try {
      const audioUrl = job.input_payload?.audio_url;
      if (!audioUrl) {
        console.log(`  Skip ${job.id} - no audio_url`);
        continue;
      }
      
      const kelly = getKellyAvatar(job.age_category);
      console.log(`  Resubmitting: ${job.language}/${job.age_category}/${job.phase}`);
      console.log(`    -> Look ID: ${kelly.look_id} (${kelly.age_group})`);
      
      // Submit with correct avatar
      const result = await submitToHeyGen(audioUrl, job.age_category);
      console.log(`    HeyGen video ID: ${result.video_id}`);
      
      // Update job
      await supabase
        .from('video_jobs')
        .update({
          external_id: result.video_id,
          status: 'submitted',
          output_url: null,
          submitted_at: new Date().toISOString(),
          input_payload: {
            ...job.input_payload,
            talking_photo_id: result.look_id,
            kelly_age_group: result.age_group,
            correct_kelly: true
          },
          notes: `Resubmitted with correct Kelly (${result.age_group} storyteller)`
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

async function showStatus() {
  const { data, error } = await supabase
    .from('video_jobs')
    .select('*')
    .eq('engine', 'heygen')
    .eq('day_of_year', 20);
  
  if (error) {
    console.error('Error:', error);
    return;
  }
  
  console.log('='.repeat(60));
  console.log('KELLY AVATAR STATUS - Day 20');
  console.log('='.repeat(60));
  
  // Count by status
  const byStatus = {};
  data.forEach(j => {
    byStatus[j.status] = (byStatus[j.status] || 0) + 1;
  });
  console.log('\nBy status:', byStatus);
  
  // Count correct vs wrong avatar
  let correctKelly = 0;
  let wrongKelly = 0;
  
  data.forEach(j => {
    const kelly = getKellyAvatar(j.age_category);
    if (j.input_payload?.talking_photo_id === kelly.look_id ||
        j.input_payload?.correct_kelly) {
      correctKelly++;
    } else {
      wrongKelly++;
    }
  });
  
  console.log(`\nCorrect Kelly avatar: ${correctKelly}`);
  console.log(`Wrong Kelly avatar: ${wrongKelly}`);
  
  // By age
  const byAge = {};
  data.forEach(j => {
    byAge[j.age_category] = (byAge[j.age_category] || 0) + 1;
  });
  console.log('\nBy age:', byAge);
}

// Parse command
const cmd = process.argv[2] || 'status';
if (cmd === 'run') {
  resubmitAll().catch(console.error);
} else {
  showStatus().catch(console.error);
}
