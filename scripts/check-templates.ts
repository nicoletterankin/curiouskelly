#!/usr/bin/env npx tsx
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';

const envContent = fs.readFileSync('.env', 'utf-8');
let key = '';
for (const line of envContent.split('\n')) {
  if (line.startsWith('SUPABASE_SERVICE_ROLE_KEY=') && !key) { 
    key = line.split('=')[1].trim(); 
    break; 
  }
}
const supabase = createClient(process.env.PUBLIC_SUPABASE_URL!, key);

async function main() {
  console.log('Checking available templates by phase for Day 1...\n');
  
  const phases = ['hook', 'q1', 'q2', 'q3', 'wisdom'];
  
  for (const phase of phases) {
    const { data, error } = await supabase
      .from('kelly_video_assets')
      .select('template')
      .eq('day_number', 1)
      .eq('phase', phase)
      .eq('asset_type', 'video')
      .eq('age_bucket', 'adult');
    
    if (error) {
      console.log(`${phase}: ERROR - ${error.message}`);
      continue;
    }
    
    const templates = [...new Set((data || []).map(r => r.template))].sort();
    console.log(`${phase}: ${templates.join(', ')}`);
  }
  
  // Find templates available in ALL phases
  const allPhases: Record<string, Set<string>> = {};
  for (const phase of phases) {
    const { data } = await supabase
      .from('kelly_video_assets')
      .select('template')
      .eq('day_number', 1)
      .eq('phase', phase)
      .eq('asset_type', 'video')
      .eq('age_bucket', 'adult');
    
    allPhases[phase] = new Set((data || []).map(r => r.template));
  }
  
  // Find common templates
  const hookTemplates = allPhases['hook'];
  const commonTemplates = [...hookTemplates].filter(t => 
    allPhases['q1'].has(t) && 
    allPhases['q2'].has(t) && 
    allPhases['q3'].has(t) && 
    allPhases['wisdom'].has(t)
  );
  
  console.log('\n=== TEMPLATES AVAILABLE FOR ALL PHASES ===');
  console.log(commonTemplates.join(', ') || 'None found');
  
  console.log('\n=== RECOMMENDATION ===');
  if (commonTemplates.length >= 2) {
    console.log(`CLIFF: Use "${commonTemplates[0]}" (has video in all phases)`);
    console.log(`OUTRO: Use "${commonTemplates[1]}" (has video in all phases)`);
  } else {
    console.log('Need to pick from hook templates that have q1/q2/q3/wisdom coverage');
  }
  
  // Verify strategist and storyteller
  console.log('\n=== CURRENT ISSUE ANALYSIS ===');
  
  const { data: storyteller } = await supabase
    .from('kelly_video_assets')
    .select('phase, template')
    .eq('day_number', 1)
    .eq('template', 'storyteller')
    .eq('asset_type', 'video')
    .eq('age_bucket', 'adult');
  console.log(`storyteller phases: ${(storyteller || []).map(v => v.phase).join(', ') || 'NONE'}`);
  
  const { data: strategist } = await supabase
    .from('kelly_video_assets')
    .select('phase, template')
    .eq('day_number', 1)
    .eq('template', 'strategist')
    .eq('asset_type', 'video')
    .eq('age_bucket', 'adult');
  console.log(`strategist phases: ${(strategist || []).map(v => v.phase).join(', ') || 'NONE'}`);
  
  // The issue: cliff and outro phases don't have videos - only hook and q1/q2/q3/wisdom
  // We need to either:
  // 1. Change PHASE_ARCHETYPES for cliff/outro to use templates that exist in hook
  // 2. Or realize cliff/outro don't have their own phase videos, they reuse hook
  console.log('\n=== SOLUTION ===');
  console.log('The player queries phase="cliff" or phase="outro" but we only have:');
  console.log('- hook phase videos');
  console.log('- q1/q2/q3/wisdom phase videos');
  console.log('Option 1: Map cliff dbName to hook, outro dbName to wisdom');
  console.log('Option 2: Generate cliff/outro specific videos');
}

main().catch(console.error);
