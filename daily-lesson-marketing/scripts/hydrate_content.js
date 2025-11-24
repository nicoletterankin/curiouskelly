import { createClient } from '@supabase/supabase-js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Explicitly load .env from the project root
dotenv.config({ path: path.join(__dirname, '../.env') });

// Configuration - Prioritize Env Vars, Fallback to Public Anon Key
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.PUBLIC_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY) {
    console.error('❌ Missing Supabase Environment Variables in .env file');
    // Don't crash, just warn so build can proceed if offline
}

const supabase = createClient(
    SUPABASE_URL || 'https://placeholder.supabase.co', 
    SUPABASE_KEY || 'placeholder', 
    {
        auth: {
            persistSession: false,
            autoRefreshToken: false,
        }
});

const OUTPUT_DIR = path.join(__dirname, '../public/lessons');

async function hydrate() {
  console.log('🧠 Hydrating content from Supabase...');
  console.log(`Target URL: ${SUPABASE_URL}`);
  // console.log(`Using Key: ${SUPABASE_KEY.substring(0, 10)}...`); 

  const { data: lessons, error } = await supabase
    .from('lessons')
    .select('*')
    .eq('is_published', true);

  if (error) {
    console.error('Error fetching lessons:', error);
    // Don't fail build if DB connection fails, just warn (allows offline builds)
    console.warn('⚠️ Could not fetch from Supabase. Skipping hydration.');
    return;
  }

  console.log(`Found ${lessons.length} published lessons.`);

  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Read existing calendar to preserve data for days not in DB yet
  const calendarPath = path.join(OUTPUT_DIR, '365_day_calendar.json');
  let calendar = { lessons: [] };
  
  if (fs.existsSync(calendarPath)) {
      try {
          calendar = JSON.parse(fs.readFileSync(calendarPath, 'utf8'));
      } catch (e) {
          console.warn("Could not parse existing calendar, starting fresh.");
      }
  }

  const calendarUpdates = [];

  for (const lesson of lessons) {
    // 1. Generate Slug / Filename
    // If content has an ID, use it. Otherwise slugify title.
    let slug = lesson.content.id;
    if (!slug) {
        slug = lesson.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    }
    
    // Ensure suffix convention if needed, though player handles both.
    // Stick to <slug>-dna.json for consistency
    const filename = `${slug}-dna.json`;
    const filepath = path.join(OUTPUT_DIR, filename);

    // 2. Write DNA File
    // We trust the DB content is the "Source of Truth"
    fs.writeFileSync(filepath, JSON.stringify(lesson.content, null, 2));
    console.log(`Saved: ${filename}`);

    // 3. Prepare Calendar Entry Update
    calendarUpdates.push({
        day: lesson.day_number,
        date: getDateForDay(lesson.day_number),
        title: lesson.title,
        subtitle: lesson.subtitle,
        lesson_id: slug,
        dna_file: slug,
        has_dna: true,
        category: lesson.tags && lesson.tags.length > 0 ? lesson.tags[0] : 'general',
        tags: lesson.tags || []
    });
  }

  // 4. Merge Calendar Updates
  calendarUpdates.forEach(dbLesson => {
      const index = calendar.lessons.findIndex(l => l.day === dbLesson.day);
      if (index !== -1) {
          // Merge, prioritizing DB values
          calendar.lessons[index] = { ...calendar.lessons[index], ...dbLesson };
      } else {
          calendar.lessons.push(dbLesson);
      }
  });

  // Sort by day
  calendar.lessons.sort((a, b) => a.day - b.day);

  fs.writeFileSync(calendarPath, JSON.stringify(calendar, null, 2));
  console.log(`Updated 365_day_calendar.json with ${calendarUpdates.length} lessons from DB.`);
}

function getDateForDay(day) {
    const date = new Date(2025, 0); // Jan 1, 2025
    date.setDate(day);
    return date.toLocaleDateString('en-US', { month: 'long', day: 'numeric' });
}

hydrate();
