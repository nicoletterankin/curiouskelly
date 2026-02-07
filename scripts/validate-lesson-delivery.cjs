/**
 * Sprint F: Validate Lesson Delivery
 * Checks all 365 days for deliverability
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

function checkUrl(url) {
  return new Promise((resolve) => {
    if (!url) return resolve(false);
    try {
      const mod = url.startsWith('https') ? https : http;
      const req = mod.request(url, { method: 'HEAD', timeout: 5000 }, (res) => {
        resolve(res.statusCode >= 200 && res.statusCode < 400);
      });
      req.on('error', () => resolve(false));
      req.on('timeout', () => { req.destroy(); resolve(false); });
      req.end();
    } catch (e) {
      resolve(false);
    }
  });
}

async function validateDelivery() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  console.log('=== Lesson Delivery Validation ===\n');
  
  const report = { days: {}, summary: {} };
  
  // Get all data
  const data = await client.query(`
    SELECT cl.day_number, cl.title, la.phase, la.variant, la.audio_url, la.video_url, la.status,
           COUNT(ls.id) as script_count
    FROM core_lessons_v2 cl
    LEFT JOIN lesson_atoms la ON la.lesson_id = cl.id AND la.age_group = 'adult' AND la.language = 'en'
    LEFT JOIN lesson_scripts ls ON ls.atom_id = la.id
    GROUP BY cl.day_number, cl.title, la.phase, la.variant, la.audio_url, la.video_url, la.status
    ORDER BY cl.day_number, la.phase
  `);
  
  // Group by day
  const byDay = {};
  for (const row of data.rows) {
    if (!byDay[row.day_number]) {
      byDay[row.day_number] = { title: row.title, phases: [] };
    }
    if (row.phase) {
      byDay[row.day_number].phases.push(row);
    }
  }
  
  let fullyDeliverable = 0;
  let partiallyDeliverable = 0;
  let emptyDays = 0;
  
  // Check sample URLs (10 random days)
  const sampleDays = [1, 15, 30, 61, 75, 100, 150, 200, 250, 365];
  const urlResults = {};
  
  console.log('Checking sample audio URLs...');
  for (const day of sampleDays) {
    const dayData = byDay[day];
    if (!dayData) continue;
    for (const phase of dayData.phases) {
      if (phase.audio_url) {
        const valid = await checkUrl(phase.audio_url);
        urlResults[`day${day}_phase${phase.phase}`] = { url: phase.audio_url.substring(0, 60), valid };
        process.stdout.write(`  Day ${day} Phase ${phase.phase}: ${valid ? 'OK' : 'FAILED'}\n`);
        break; // Just check one per day
      }
    }
  }
  
  for (let day = 1; day <= 365; day++) {
    const dayData = byDay[day];
    if (!dayData || dayData.phases.length === 0) {
      emptyDays++;
      report.days[day] = { status: 'empty', title: dayData?.title || 'Unknown' };
      continue;
    }
    
    const phasesWithScripts = dayData.phases.filter(p => parseInt(p.script_count) > 0).length;
    const phasesWithAudio = dayData.phases.filter(p => p.audio_url).length;
    const phasesWithVideo = dayData.phases.filter(p => p.video_url).length;
    
    const status = phasesWithScripts >= 5 ? 'full' : phasesWithScripts > 0 ? 'partial' : 'empty';
    
    if (status === 'full') fullyDeliverable++;
    else if (status === 'partial') partiallyDeliverable++;
    else emptyDays++;
    
    report.days[day] = {
      title: dayData.title,
      status,
      scripts: phasesWithScripts,
      audio: phasesWithAudio,
      video: phasesWithVideo,
      total_phases: dayData.phases.length,
    };
  }
  
  report.summary = {
    total: 365,
    fully_deliverable: fullyDeliverable,
    partially_deliverable: partiallyDeliverable,
    empty: emptyDays,
    sample_url_checks: urlResults,
    generated_at: new Date().toISOString(),
  };
  
  const reportPath = path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'delivery-readiness-report.json');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  
  console.log(`\n=== Delivery Readiness ===`);
  console.log(`${fullyDeliverable} of 365 days fully deliverable`);
  console.log(`${partiallyDeliverable} partially deliverable`);
  console.log(`${emptyDays} empty`);
  console.log(`\nSaved to: ${reportPath}`);
  
  await client.end();
}

validateDelivery().catch(e => { console.error(e); process.exit(1); });
