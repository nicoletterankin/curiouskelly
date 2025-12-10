
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';

dotenv.config();

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI'; // Anon key from public/index.html

// if (!SUPABASE_URL || !SUPABASE_KEY) {
//   console.error('Missing Supabase credentials');
//   process.exit(1);
// }

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

async function syncVisualPlans() {
  const phasesDir = path.join(process.cwd(), 'public', 'kelly', 'phases');
  
  if (!fs.existsSync(phasesDir)) {
    console.error('Phases directory not found:', phasesDir);
    return;
  }

  const days = fs.readdirSync(phasesDir);
  
  for (const dayDir of days) {
    const dayPath = path.join(phasesDir, dayDir);
    const visualPlanPath = path.join(dayPath, 'visual-plan.json');
    
    if (!fs.existsSync(visualPlanPath)) continue;
    
    const dayNumber = parseInt(dayDir);
    if (isNaN(dayNumber)) continue;

    console.log(`Processing Day ${dayNumber} (Type: ${typeof dayNumber})...`);
    console.log(`Supabase Key Start: ${SUPABASE_KEY ? SUPABASE_KEY.slice(0, 5) : 'UNDEFINED'}`);
    
    try {
      const plan = JSON.parse(fs.readFileSync(visualPlanPath, 'utf-8'));
      
      // Get core_lesson_id
      const { data: lesson } = await supabase
        .from('core_lessons')
        .select('id')
        .eq('day_number', dayNumber)
        .single();
        
      if (!lesson) {
        console.warn(`  No core lesson found for Day ${dayNumber}`);
        continue;
      }

      // Sync each phase
      for (const item of plan) {
        // phases in DB are lowercase: hook, q1, q2, q3, wisdom
        const phase = item.phase.toLowerCase();
        
        // Check for image
        const imagePath = path.join(dayPath, `${phase}.png`);
        let visualUrl = null;
        
        if (fs.existsSync(imagePath)) {
            console.log(`    Uploading ${phase}.png...`);
            const fileContent = fs.readFileSync(imagePath);
            const storagePath = `phases/${dayNumber}/${phase}.png`;
            
            const { data: uploadData, error: uploadError } = await supabase
                .storage
                .from('lesson-visuals') // Correct bucket name
                .upload(storagePath, fileContent, {
                    contentType: 'image/png',
                    upsert: true
                });
                
            if (uploadError) {
                console.error(`    Upload failed: ${uploadError.message}`);
            } else {
                const { data: publicUrlData } = supabase
                    .storage
                    .from('lesson-visuals')
                    .getPublicUrl(storagePath);
                visualUrl = publicUrlData.publicUrl;
                console.log(`    URL: ${visualUrl}`);
            }
        }

        // Update lesson_atoms
        const updatePayload: any = { 
            content: item
        };
        if (visualUrl) {
            updatePayload.visual_url = visualUrl;
        }

        const { error } = await supabase
          .from('lesson_atoms')
          .update(updatePayload)
          .eq('core_lesson_id', lesson.id)
          .eq('phase', phase)
          //.eq('archetype', 'The Explorer'); // Remove archetype constraint to update ALL archetypes for this phase?
          // Actually, lesson_atoms are per archetype. If we want this visual for ALL archetypes, we should update all.
          // Let's try updating without archetype constraint first.
          ;
          
        if (error) {
            console.error(`  Failed to update ${phase}:`, error.message);
        } else {
            console.log(`  Updated ${phase} atoms`);
        }
      }
      
    } catch (e) {
      console.error(`  Error processing Day ${dayNumber}:`, e);
    }
  }
}

syncVisualPlans().catch(console.error);

