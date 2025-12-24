/**
 * SYNC THUMBNAILS TO SUPABASE
 * 
 * Syncs existing file-based thumbnails to Supabase core_lessons.thumbnail_url
 * This ensures the database has URLs for all available thumbnails.
 * 
 * Usage: npx tsx scripts/sync-thumbnails-to-supabase.ts
 */

import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });
dotenv.config();

import { createClient } from "@supabase/supabase-js";
import * as fs from "fs";
import * as path from "path";

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error("❌ Missing Supabase credentials!");
  console.error("Set PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Base URL for production thumbnails (update to your actual domain)
const BASE_URL = "https://curiouskelly.com";

// Thumbnail directory structure
const THUMBNAILS_DIR = path.join(process.cwd(), "public", "assets", "kelly", "production", "thumbnails");

interface ThumbnailMapping {
  dayNumber: number;
  filePath: string;
  publicUrl: string;
}

/**
 * Scan for existing thumbnail files
 */
function scanThumbnails(): ThumbnailMapping[] {
  const mappings: ThumbnailMapping[] = [];
  
  // Scan month directories
  const months = ['january', 'february', 'march', 'april', 'may', 'june', 
                  'july', 'august', 'september', 'october', 'november', 'december'];
  
  const daysPerMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  
  let dayOffset = 0;
  
  for (let m = 0; m < months.length; m++) {
    const monthDir = path.join(THUMBNAILS_DIR, months[m]);
    
    if (!fs.existsSync(monthDir)) {
      dayOffset += daysPerMonth[m];
      continue;
    }
    
    const files = fs.readdirSync(monthDir).filter(f => f.endsWith('.webp') || f.endsWith('.png'));
    
    for (const file of files) {
      // Extract day number from filename (lesson-1.webp -> 1)
      const match = file.match(/lesson-(\d+)\.(webp|png)/);
      if (match) {
        const dayInMonth = parseInt(match[1]);
        const dayNumber = dayOffset + dayInMonth;
        const ext = match[2];
        
        mappings.push({
          dayNumber,
          filePath: path.join(monthDir, file),
          publicUrl: `${BASE_URL}/assets/kelly/production/thumbnails/${months[m]}/lesson-${dayInMonth}.${ext}`
        });
      }
    }
    
    dayOffset += daysPerMonth[m];
  }
  
  // Also scan raw directory for PNG thumbnails
  const rawDir = path.join(process.cwd(), "public", "kelly", "thumbnails", "raw");
  if (fs.existsSync(rawDir)) {
    const rawFiles = fs.readdirSync(rawDir).filter(f => f.endsWith('.png'));
    
    for (const file of rawFiles) {
      // Extract day from filename (lesson-001-*.png -> 1)
      const match = file.match(/lesson-(\d+)-/);
      if (match) {
        const dayNumber = parseInt(match[1]);
        
        // Only add if not already in production
        if (!mappings.find(m => m.dayNumber === dayNumber)) {
          mappings.push({
            dayNumber,
            filePath: path.join(rawDir, file),
            publicUrl: `${BASE_URL}/kelly/thumbnails/raw/${file}`
          });
        }
      }
    }
  }
  
  return mappings.sort((a, b) => a.dayNumber - b.dayNumber);
}

/**
 * Sync thumbnails to Supabase
 */
async function syncToSupabase(mappings: ThumbnailMapping[]) {
  console.log(`\n📤 Syncing ${mappings.length} thumbnails to Supabase...\n`);
  
  let updated = 0;
  let errors = 0;
  
  for (const mapping of mappings) {
    try {
      const { error } = await supabase
        .from('core_lessons')
        .update({ 
          thumbnail_url: mapping.publicUrl,
          updated_at: new Date().toISOString()
        })
        .eq('day_number', mapping.dayNumber);
      
      if (error) {
        console.error(`❌ Day ${mapping.dayNumber}: ${error.message}`);
        errors++;
      } else {
        console.log(`✅ Day ${mapping.dayNumber}: ${mapping.publicUrl}`);
        updated++;
      }
    } catch (err: any) {
      console.error(`❌ Day ${mapping.dayNumber}: ${err.message}`);
      errors++;
    }
  }
  
  return { updated, errors };
}

/**
 * Generate a report of current state
 */
async function generateReport() {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('day_number, topic, thumbnail_url, hero_image_url')
    .order('day_number');
  
  if (error) {
    console.error("❌ Failed to fetch lessons:", error.message);
    return;
  }
  
  const withThumbnail = data?.filter(d => d.thumbnail_url) || [];
  const withHero = data?.filter(d => d.hero_image_url) || [];
  const withEither = data?.filter(d => d.thumbnail_url || d.hero_image_url) || [];
  const withNeither = data?.filter(d => !d.thumbnail_url && !d.hero_image_url) || [];
  
  console.log("\n" + "═".repeat(60));
  console.log("SUPABASE THUMBNAIL COVERAGE REPORT");
  console.log("═".repeat(60));
  console.log(`Total lessons: ${data?.length || 0}`);
  console.log(`With thumbnail_url: ${withThumbnail.length}`);
  console.log(`With hero_image_url: ${withHero.length}`);
  console.log(`With either: ${withEither.length}`);
  console.log(`Missing both: ${withNeither.length}`);
  
  if (withNeither.length > 0 && withNeither.length <= 20) {
    console.log("\nMissing days:");
    withNeither.forEach(d => console.log(`  Day ${d.day_number}: ${d.topic}`));
  }
  
  return { withThumbnail, withHero, withEither, withNeither };
}

async function main() {
  console.log("═".repeat(60));
  console.log("🖼️  THUMBNAIL SYNC TO SUPABASE");
  console.log("═".repeat(60));
  
  // Step 1: Report current state
  console.log("\n📊 Current Supabase state:");
  await generateReport();
  
  // Step 2: Scan local files
  console.log("\n📁 Scanning local thumbnail files...");
  const mappings = scanThumbnails();
  console.log(`Found ${mappings.length} thumbnail files`);
  
  if (mappings.length === 0) {
    console.log("\n⚠️  No thumbnails found locally. Generate some first:");
    console.log("npx tsx scripts/kelly-visual-identity/generate-all-365-thumbnails.ts");
    return;
  }
  
  // Show sample
  console.log("\nSample mappings:");
  mappings.slice(0, 5).forEach(m => {
    console.log(`  Day ${m.dayNumber}: ${m.publicUrl}`);
  });
  if (mappings.length > 5) {
    console.log(`  ... and ${mappings.length - 5} more`);
  }
  
  // Step 3: Sync to Supabase
  const { updated, errors } = await syncToSupabase(mappings);
  
  // Step 4: Final report
  console.log("\n" + "═".repeat(60));
  console.log("SYNC COMPLETE");
  console.log("═".repeat(60));
  console.log(`Updated: ${updated}`);
  console.log(`Errors: ${errors}`);
  
  // Step 5: Show new state
  console.log("\n📊 New Supabase state:");
  await generateReport();
}

main().catch(err => {
  console.error("Fatal error:", err);
  process.exit(1);
});




