/**
 * Poll HeyGen for video completion status - Generic day poller
 */

const { createClient } = require('@supabase/supabase-js');
const fetch = require('node-fetch');
require('dotenv').config({ path: 'C:/Users/user/ANTIGRAVITY/media_keys.env' });

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const HEYGEN_KEY = process.env.HEYGEN_API_KEY;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

async function checkHeyGenStatus(videoId) {
  const resp = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
    headers: { 'X-Api-Key': HEYGEN_KEY }
  });
  return resp.json();
}

async function pollDay(dayNum) {
  console.log(`Polling HeyGen for Day ${dayNum}...`);
  
  const { data: jobs, error } = await supabase
    .from('video_jobs')
    .select('*')
    .eq('engine', 'heygen')
    .eq('status', 'submitted')
    .eq('day_of_year', dayNum);
  
  if (error) {
    console.error('DB error:', error);
    return;
  }
  
  console.log(`Found ${jobs.length} submitted jobs`);
  
  let completed = 0;
  let failed = 0;
  let processing = 0;
  
  for (const job of jobs) {
    if (!job.external_id) {
      console.log(`  Job ${job.id} missing external_id`);
      continue;
    }
    
    try {
      const status = await checkHeyGenStatus(job.external_id);
      
      if (status.data?.status === 'completed') {
        console.log(`  COMPLETED: ${job.language}/${job.age_category}/${job.phase}`);
        
        await supabase
          .from('video_jobs')
          .update({
            status: 'completed',
            output_url: status.data.video_url,
            completed_at: new Date().toISOString()
          })
          .eq('id', job.id);
        
        completed++;
      } else if (status.data?.status === 'failed') {
        console.log(`  FAILED: ${job.language}/${job.age_category}/${job.phase} - ${status.data.error}`);
        
        await supabase
          .from('video_jobs')
          .update({
            status: 'failed',
            error_message: status.data.error || 'Unknown error'
          })
          .eq('id', job.id);
        
        failed++;
      } else {
        processing++;
      }
      
      await new Promise(r => setTimeout(r, 200));
      
    } catch (err) {
      console.log(`  Error checking ${job.id}: ${err.message}`);
    }
  }
  
  console.log('\n--- POLL SUMMARY ---');
  console.log(`Completed: ${completed}`);
  console.log(`Failed: ${failed}`);
  console.log(`Still processing: ${processing}`);
  
  return { completed, failed, processing };
}

async function showStatus(dayNum) {
  const { data, error } = await supabase
    .from('video_jobs')
    .select('status, language, age_category')
    .eq('engine', 'heygen')
    .eq('day_of_year', dayNum);
  
  if (error) {
    console.error('Error:', error);
    return;
  }
  
  console.log(`\nDay ${dayNum} HeyGen Status:`);
  
  const byStatus = {};
  data.forEach(j => {
    byStatus[j.status] = (byStatus[j.status] || 0) + 1;
  });
  console.log('By status:', byStatus);
  
  const byLang = {};
  data.filter(j => j.status === 'completed').forEach(j => {
    byLang[j.language] = (byLang[j.language] || 0) + 1;
  });
  console.log('Completed by language:', byLang);
}

// Main
const dayNum = parseInt(process.argv[2]) || 21;
const cmd = process.argv[3] || 'status';

if (cmd === 'poll') {
  pollDay(dayNum).catch(console.error);
} else {
  showStatus(dayNum).catch(console.error);
}
