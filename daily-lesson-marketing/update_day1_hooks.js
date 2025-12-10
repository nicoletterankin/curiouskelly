import fs from 'fs';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function updateDay1() {
    // Update local JSON
    const data = JSON.parse(fs.readFileSync('age_hooks.json', 'utf8'));
    
    const day1 = data.hooks.find(h => h.day === 1);
    if (day1) {
        day1.topic = 'Starting Fresh';
        day1.hooks = {
            '5-7': 'Every day is a brand new adventure!',
            '8-12': 'Today you get to become whoever you want',
            '13-17': 'Plot twist: you can reinvent yourself anytime',
            '18-35': 'The science of why fresh starts actually work',
            '36-60': 'It is never too late to begin again',
            '61+': 'Each morning brings wisdom and new possibilities'
        };
        
        fs.writeFileSync('age_hooks.json', JSON.stringify(data, null, 2));
        console.log('✅ Local age_hooks.json updated');
    }
    
    // Update Supabase
    const newHooks = [
        { day_number: 1, topic: 'Starting Fresh', age_bucket: '5-7', hook: 'Every day is a brand new adventure!' },
        { day_number: 1, topic: 'Starting Fresh', age_bucket: '8-12', hook: 'Today you get to become whoever you want' },
        { day_number: 1, topic: 'Starting Fresh', age_bucket: '13-17', hook: 'Plot twist: you can reinvent yourself anytime' },
        { day_number: 1, topic: 'Starting Fresh', age_bucket: '18-35', hook: 'The science of why fresh starts actually work' },
        { day_number: 1, topic: 'Starting Fresh', age_bucket: '36-60', hook: 'It is never too late to begin again' },
        { day_number: 1, topic: 'Starting Fresh', age_bucket: '61+', hook: 'Each morning brings wisdom and new possibilities' }
    ];
    
    // Delete old Day 1 hooks
    const { error: deleteError } = await supabase
        .from('lesson_age_hooks')
        .delete()
        .eq('day_number', 1);
    
    if (deleteError) {
        console.log('Delete error:', deleteError.message);
    }
    
    // Insert new hooks
    const { error: insertError } = await supabase
        .from('lesson_age_hooks')
        .insert(newHooks);
    
    if (insertError) {
        console.log('Insert error:', insertError.message);
    } else {
        console.log('✅ Supabase lesson_age_hooks updated for Day 1');
    }
    
    // Verify
    const { data: verify } = await supabase
        .from('lesson_age_hooks')
        .select('*')
        .eq('day_number', 1);
    
    console.log('\n=== Day 1 Hooks in Supabase ===');
    verify.forEach(h => {
        console.log(`  ${h.age_bucket}: "${h.hook}"`);
    });
}

updateDay1().catch(console.error);










