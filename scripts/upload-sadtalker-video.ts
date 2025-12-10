import 'dotenv/config';
import * as fs from 'fs';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function main() {
  console.log('📤 Uploading SadTalker video...');
  
  const videoPath = 'generated-videos/photorealistic-test/kelly_sadtalker.mp4';
  const videoBuffer = fs.readFileSync(videoPath);
  
  const remotePath = 'production/videos/sadtalker/day_001_Hook_sadtalker.mp4';
  
  await supabase.storage.from('kelly-videos').upload(remotePath, videoBuffer, {
    upsert: true,
    contentType: 'video/mp4'
  });
  
  const { data } = supabase.storage.from('kelly-videos').getPublicUrl(remotePath);
  console.log('✅ Uploaded:', data.publicUrl);
  
  // Update database
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', 1)
    .single();
  
  if (lesson) {
    await supabase
      .from('lesson_atoms')
      .update({ hd_video_url: data.publicUrl })
      .eq('core_lesson_id', lesson.id)
      .eq('phase', 'Hook')
      .eq('archetype', 'The Scientist');
    
    console.log('✅ Database updated!');
    console.log('\n🎬 TEST NOW: http://localhost:3000/learn?day=1&clearcache=1');
  }
}

main().catch(console.error);

