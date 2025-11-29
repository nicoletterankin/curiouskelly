#!/usr/bin/env node
/**
 * Kelly Asset Production Pipeline
 * 
 * Processes raw Kelly images into optimized web assets.
 * Run: node scripts/process-kelly-assets.js
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Check for Sharp installation
let sharp;
try {
  sharp = (await import('sharp')).default;
} catch (e) {
  console.error('❌ Sharp not installed. Run: npm install sharp');
  console.log('\nTo install Sharp:');
  console.log('  cd daily-lesson-marketing');
  console.log('  npm install sharp');
  process.exit(1);
}

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
  sourceDir: path.join(__dirname, '../public/lessons/images'),
  outputDir: path.join(__dirname, '../public/assets/kelly'),
  
  // Hero image variants
  heroVariants: [
    { name: '4k', width: 3840, height: 2160 },
    { name: 'desktop', width: 1920, height: 1080 },
    { name: 'tablet', width: 1280, height: 720 },
    { name: 'mobile', width: 640, height: 360 },
  ],
  
  // Avatar size variants
  avatarSizes: [512, 256, 128, 64],
  
  // Expression states to generate
  expressions: ['curious', 'explaining', 'celebrating', 'listening', 'wisdom'],
  
  // Quality settings
  quality: {
    webp: 85,
    jpeg: 90,
    avif: 80,
  },
  
  // Source file mapping (raw file → expression)
  sourceMapping: {
    // Best source for each expression state
    'curious': 'hero/neutral.jpeg',
    'explaining': 'hero/looking-at-us.jpeg', 
    'celebrating': 'hero/big-smile.jpeg',
    'listening': 'hero/neutral.jpeg',
    'wisdom': 'hero/neutral.jpeg',
    // Hero images - use highest quality portrait
    'hero-primary': '1.jpg', // 3072x5504 portrait
    'hero-landscape': 'kelly2-directors-chair.jpeg', // 6000x3375 landscape
  },
  
  // Social media sizes
  socialSizes: {
    'og-image': { width: 1200, height: 630 },
    'twitter-card': { width: 1200, height: 600 },
    'linkedin-banner': { width: 1584, height: 396 },
    'instagram-square': { width: 1080, height: 1080 },
    'instagram-story': { width: 1080, height: 1920 },
  }
};

// ============================================
// UTILITY FUNCTIONS
// ============================================

async function ensureDir(dir) {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (e) {
    if (e.code !== 'EEXIST') throw e;
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

async function getFileSize(filePath) {
  const stats = await fs.stat(filePath);
  return stats.size;
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

// ============================================
// IMAGE PROCESSING FUNCTIONS
// ============================================

async function processHeroImages() {
  console.log('\n📸 Processing Hero Images...\n');
  
  const heroDir = path.join(CONFIG.outputDir, 'production/hero');
  await ensureDir(heroDir);
  
  // Use the best portrait source
  let sourcePath = path.join(CONFIG.sourceDir, CONFIG.sourceMapping['hero-primary']);
  
  // Check if source exists, try fallbacks
  if (!await fileExists(sourcePath)) {
    console.log(`  ⚠️ Primary source not found: ${CONFIG.sourceMapping['hero-primary']}`);
    // Try landscape
    sourcePath = path.join(CONFIG.sourceDir, CONFIG.sourceMapping['hero-landscape']);
    if (!await fileExists(sourcePath)) {
      // Try hero folder
      sourcePath = path.join(CONFIG.sourceDir, 'hero/neutral.jpeg');
      if (!await fileExists(sourcePath)) {
        console.error(`  ❌ No suitable hero source found`);
        return;
      }
    }
    console.log(`  Using fallback: ${sourcePath}`);
  }
  
  const results = [];
  
  for (const variant of CONFIG.heroVariants) {
    const baseName = `kelly-hero-${variant.name}`;
    
    try {
      // WebP (primary)
      const webpPath = path.join(heroDir, `${baseName}.webp`);
      await sharp(sourcePath)
        .resize(variant.width, variant.height, { 
          fit: 'cover',
          position: 'top' // Keep face in frame
        })
        .webp({ quality: CONFIG.quality.webp })
        .toFile(webpPath);
      
      const webpSize = await getFileSize(webpPath);
      
      // JPEG fallback
      const jpegPath = path.join(heroDir, `${baseName}.jpg`);
      await sharp(sourcePath)
        .resize(variant.width, variant.height, { 
          fit: 'cover',
          position: 'top'
        })
        .jpeg({ quality: CONFIG.quality.jpeg, progressive: true })
        .toFile(jpegPath);
      
      const jpegSize = await getFileSize(jpegPath);
      
      results.push({
        variant: variant.name,
        dimensions: `${variant.width}×${variant.height}`,
        webp: formatBytes(webpSize),
        jpeg: formatBytes(jpegSize),
        savings: `${Math.round((1 - webpSize/jpegSize) * 100)}%`
      });
      
      console.log(`  ✅ ${baseName}: WebP ${formatBytes(webpSize)}, JPEG ${formatBytes(jpegSize)}`);
    } catch (err) {
      console.error(`  ❌ Failed to process ${baseName}: ${err.message}`);
    }
  }
  
  return results;
}

async function processAvatars() {
  console.log('\n🎭 Processing Avatar States...\n');
  
  const avatarsDir = path.join(CONFIG.outputDir, 'production/avatars');
  
  for (const expression of CONFIG.expressions) {
    const expressionDir = path.join(avatarsDir, expression);
    await ensureDir(expressionDir);
    
    const sourceFile = CONFIG.sourceMapping[expression];
    const sourcePath = path.join(CONFIG.sourceDir, sourceFile);
    
    if (!await fileExists(sourcePath)) {
      console.error(`  ❌ Source not found for ${expression}: ${sourceFile}`);
      continue;
    }
    
    console.log(`  Processing: ${expression}`);
    
    for (const size of CONFIG.avatarSizes) {
      const baseName = `kelly-${expression}-${size}`;
      
      try {
        // WebP
        const webpPath = path.join(expressionDir, `${baseName}.webp`);
        await sharp(sourcePath)
          .resize(size, size, { fit: 'cover', position: 'center' })
          .webp({ quality: CONFIG.quality.webp })
          .toFile(webpPath);
        
        // JPEG fallback
        const jpegPath = path.join(expressionDir, `${baseName}.jpg`);
        await sharp(sourcePath)
          .resize(size, size, { fit: 'cover', position: 'center' })
          .jpeg({ quality: CONFIG.quality.jpeg })
          .toFile(jpegPath);
        
        const webpSize = await getFileSize(webpPath);
        console.log(`    ✅ ${size}px: ${formatBytes(webpSize)}`);
      } catch (err) {
        console.error(`    ❌ Failed ${size}px: ${err.message}`);
      }
    }
  }
}

async function processSocialImages() {
  console.log('\n📱 Processing Social Media Assets...\n');
  
  const socialDir = path.join(CONFIG.outputDir, 'production/social');
  await ensureDir(socialDir);
  
  // Use landscape source for social images
  let sourcePath = path.join(CONFIG.sourceDir, CONFIG.sourceMapping['hero-landscape']);
  
  if (!await fileExists(sourcePath)) {
    // Fallback to portrait or hero
    sourcePath = path.join(CONFIG.sourceDir, CONFIG.sourceMapping['hero-primary']);
    if (!await fileExists(sourcePath)) {
      sourcePath = path.join(CONFIG.sourceDir, 'hero/neutral.jpeg');
      if (!await fileExists(sourcePath)) {
        console.error('  ❌ No suitable source for social images');
        return;
      }
    }
    console.log(`  Using fallback source: ${path.basename(sourcePath)}`);
  }
  
  for (const [name, dims] of Object.entries(CONFIG.socialSizes)) {
    try {
      const outputPath = path.join(socialDir, `${name}.jpg`);
      
      await sharp(sourcePath)
        .resize(dims.width, dims.height, { 
          fit: 'cover',
          position: 'center'
        })
        .jpeg({ quality: 90, progressive: true })
        .toFile(outputPath);
      
      const size = await getFileSize(outputPath);
      console.log(`  ✅ ${name}: ${dims.width}×${dims.height} (${formatBytes(size)})`);
    } catch (err) {
      console.error(`  ❌ Failed ${name}: ${err.message}`);
    }
  }
}

async function generateManifest() {
  console.log('\n📋 Generating Asset Manifest...\n');
  
  const manifest = {
    version: '1.0.0',
    generated: new Date().toISOString(),
    hero: {},
    avatars: {},
    social: {}
  };
  
  // Hero images
  for (const variant of CONFIG.heroVariants) {
    manifest.hero[variant.name] = {
      webp: `/assets/kelly/production/hero/kelly-hero-${variant.name}.webp`,
      jpeg: `/assets/kelly/production/hero/kelly-hero-${variant.name}.jpg`,
      width: variant.width,
      height: variant.height
    };
  }
  
  // Avatars
  for (const expression of CONFIG.expressions) {
    manifest.avatars[expression] = {};
    for (const size of CONFIG.avatarSizes) {
      manifest.avatars[expression][size] = {
        webp: `/assets/kelly/production/avatars/${expression}/kelly-${expression}-${size}.webp`,
        jpeg: `/assets/kelly/production/avatars/${expression}/kelly-${expression}-${size}.jpg`
      };
    }
  }
  
  // Social
  for (const [name, dims] of Object.entries(CONFIG.socialSizes)) {
    manifest.social[name] = {
      path: `/assets/kelly/production/social/${name}.jpg`,
      width: dims.width,
      height: dims.height
    };
  }
  
  const manifestPath = path.join(CONFIG.outputDir, 'manifest.json');
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log(`  ✅ Manifest saved to: ${manifestPath}`);
  
  return manifest;
}

async function generateSrcsetHelper() {
  console.log('\n🔧 Generating Srcset Helper...\n');
  
  const helperCode = `/**
 * Kelly Image Helper
 * Auto-generated - do not edit manually
 * Generated: ${new Date().toISOString()}
 */

export const kellyImages = {
  hero: {
    srcset: {
      webp: \`
        /assets/kelly/production/hero/kelly-hero-mobile.webp 640w,
        /assets/kelly/production/hero/kelly-hero-tablet.webp 1280w,
        /assets/kelly/production/hero/kelly-hero-desktop.webp 1920w,
        /assets/kelly/production/hero/kelly-hero-4k.webp 3840w
      \`.trim(),
      jpeg: \`
        /assets/kelly/production/hero/kelly-hero-mobile.jpg 640w,
        /assets/kelly/production/hero/kelly-hero-tablet.jpg 1280w,
        /assets/kelly/production/hero/kelly-hero-desktop.jpg 1920w,
        /assets/kelly/production/hero/kelly-hero-4k.jpg 3840w
      \`.trim()
    },
    fallback: '/assets/kelly/production/hero/kelly-hero-desktop.jpg'
  },
  
  getAvatar(expression = 'curious', size = 256, format = 'webp') {
    const validExpressions = ${JSON.stringify(CONFIG.expressions)};
    const validSizes = ${JSON.stringify(CONFIG.avatarSizes)};
    
    const exp = validExpressions.includes(expression) ? expression : 'curious';
    const sz = validSizes.includes(size) ? size : 256;
    const fmt = format === 'jpeg' ? 'jpg' : 'webp';
    
    return \`/assets/kelly/production/avatars/\${exp}/kelly-\${exp}-\${sz}.\${fmt}\`;
  },
  
  social: {
    og: '/assets/kelly/production/social/og-image.jpg',
    twitter: '/assets/kelly/production/social/twitter-card.jpg',
    linkedin: '/assets/kelly/production/social/linkedin-banner.jpg',
    instagram: '/assets/kelly/production/social/instagram-square.jpg',
    instagramStory: '/assets/kelly/production/social/instagram-story.jpg'
  }
};

export default kellyImages;
`;

  const helperPath = path.join(CONFIG.outputDir, 'kelly-images.js');
  await fs.writeFile(helperPath, helperCode);
  
  console.log(`  ✅ Helper saved to: ${helperPath}`);
}

// ============================================
// MAIN EXECUTION
// ============================================

async function main() {
  console.log('═══════════════════════════════════════════════════════');
  console.log('  🎨 KELLY ASSET PRODUCTION PIPELINE');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`\nSource: ${CONFIG.sourceDir}`);
  console.log(`Output: ${CONFIG.outputDir}`);
  
  // Create output directory structure
  await ensureDir(CONFIG.outputDir);
  await ensureDir(path.join(CONFIG.outputDir, 'production'));
  await ensureDir(path.join(CONFIG.outputDir, 'source'));
  
  try {
    // Process all asset types
    await processHeroImages();
    await processAvatars();
    await processSocialImages();
    
    // Generate manifest and helpers
    await generateManifest();
    await generateSrcsetHelper();
    
    console.log('\n═══════════════════════════════════════════════════════');
    console.log('  ✅ ASSET PRODUCTION COMPLETE');
    console.log('═══════════════════════════════════════════════════════');
    console.log('\nNext steps:');
    console.log('  1. Review generated assets in /public/assets/kelly/');
    console.log('  2. Update code to use new asset paths');
    console.log('  3. Test responsive images on all devices');
    console.log('  4. Deploy and verify CDN caching');
    
  } catch (error) {
    console.error('\n❌ Pipeline failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
