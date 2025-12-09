/**
 * OPTIMIZER - Production Thumbnail Deployment
 * 
 * 1. Reads raw hero images from public/kelly/lessons/[day]/lesson-[day]-hero.png
 * 2. Optimizes them (resize to 1280x720, quality 85, WebP)
 * 3. Saves to public/assets/kelly/production/thumbnails/january/lesson-[day].webp
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";

const SOURCE_ROOT = path.join(process.cwd(), "public", "kelly", "lessons");
const DEST_ROOT = path.join(process.cwd(), "public", "assets", "kelly", "production", "thumbnails", "january");

// Ensure destination exists
fs.mkdirSync(DEST_ROOT, { recursive: true });

// ImageMagick Path
const MAGICK = "C:\\Program Files\\ImageMagick-7.1.2-Q16-HDRI\\magick.exe";

async function processThumbnails() {
  console.log("🚀 DEPLOYING JANUARY THUMBNAILS");
  console.log("=".repeat(50));
  
  // 31 Days of January
  for (let day = 1; day <= 31; day++) {
    // Source folder name can be "001" or "1" depending on how it was generated
    // The factory used formatted day numbers in folders? Let's check.
    // The factory script used: String(lessonDay).padStart(3, '0') -> "001", "002"
    
    const dayFolder = String(day).padStart(3, '0');
    const srcFile = path.join(SOURCE_ROOT, dayFolder, `lesson-${day}-hero.png`);
    const destFile = path.join(DEST_ROOT, `lesson-${day}.webp`);
    
    if (fs.existsSync(srcFile)) {
      try {
        // Convert & Optimize
        // -resize 1280x720^ : Resize to fill 1280x720 (16:9)
        // -gravity center -extent 1280x720 : Crop to center if aspect ratio differs slightly
        // -quality 85 : Good balance
        const cmd = `"${MAGICK}" "${srcFile}" -resize 1280x720^ -gravity center -extent 1280x720 -quality 85 "${destFile}"`;
        execSync(cmd, { stdio: 'pipe' });
        
        console.log(`✅ Deployed: lesson-${day}.webp`);
      } catch (e) {
        console.error(`❌ Failed Day ${day}: ${e.message}`);
      }
    } else {
      console.warn(`⚠️ Missing source: ${srcFile}`);
    }
  }
  
  console.log("\n✨ Deployment Complete");
}

processThumbnails().catch(console.error);






