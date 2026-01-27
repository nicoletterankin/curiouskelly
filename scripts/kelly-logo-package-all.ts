#!/usr/bin/env npx tsx
/**
 * 📦 KELLY LOGO COMPLETE PACKAGE
 * 
 * Takes the winning Kelly logo and generates ALL sizes for ALL platforms:
 * - Social media (Twitter, Instagram, LinkedIn, TikTok, YouTube, Discord)
 * - Favicons (ICO, PNG all sizes)
 * - App icons (iOS, Android)
 * - OG/Meta images
 * - Email signatures
 * - Print/Press (high-res)
 * - Square and circle crops
 * 
 * Outputs a complete folder ready for distribution with index HTML.
 */

import * as fs from 'fs';
import * as path from 'path';
import sharp from 'sharp';
import archiver from 'archiver';

const SOURCE_IMAGE = path.join(process.cwd(), 'generated-images', 'kelly-logo-polish', 'kelly-curious-v3-seed33333333.png');
const OUTPUT_DIR = path.join(process.cwd(), 'generated-images', 'kelly-logo-complete-package');

interface AssetSpec {
  name: string;
  filename: string;
  width: number;
  height: number;
  format: 'png' | 'jpeg' | 'webp';
  quality?: number;
  circle?: boolean;
  category: string;
}

const ASSET_SPECS: AssetSpec[] = [
  // ============================================
  // SOCIAL MEDIA PROFILE PICTURES
  // ============================================
  // Twitter/X
  { name: 'Twitter Profile', filename: 'social/twitter-profile-400.png', width: 400, height: 400, format: 'png', category: 'Social Media' },
  { name: 'Twitter Header', filename: 'social/twitter-header-1500x500.png', width: 1500, height: 500, format: 'png', category: 'Social Media' },
  
  // Instagram
  { name: 'Instagram Profile', filename: 'social/instagram-profile-320.png', width: 320, height: 320, format: 'png', category: 'Social Media' },
  { name: 'Instagram Post Square', filename: 'social/instagram-post-1080.png', width: 1080, height: 1080, format: 'png', category: 'Social Media' },
  { name: 'Instagram Story', filename: 'social/instagram-story-1080x1920.png', width: 1080, height: 1920, format: 'png', category: 'Social Media' },
  
  // LinkedIn
  { name: 'LinkedIn Profile', filename: 'social/linkedin-profile-400.png', width: 400, height: 400, format: 'png', category: 'Social Media' },
  { name: 'LinkedIn Cover', filename: 'social/linkedin-cover-1584x396.png', width: 1584, height: 396, format: 'png', category: 'Social Media' },
  
  // YouTube
  { name: 'YouTube Profile', filename: 'social/youtube-profile-800.png', width: 800, height: 800, format: 'png', category: 'Social Media' },
  { name: 'YouTube Channel Art', filename: 'social/youtube-banner-2560x1440.png', width: 2560, height: 1440, format: 'png', category: 'Social Media' },
  
  // TikTok
  { name: 'TikTok Profile', filename: 'social/tiktok-profile-200.png', width: 200, height: 200, format: 'png', category: 'Social Media' },
  
  // Discord
  { name: 'Discord Avatar', filename: 'social/discord-avatar-512.png', width: 512, height: 512, format: 'png', category: 'Social Media' },
  { name: 'Discord Server Icon', filename: 'social/discord-server-512.png', width: 512, height: 512, format: 'png', category: 'Social Media' },
  
  // Facebook
  { name: 'Facebook Profile', filename: 'social/facebook-profile-180.png', width: 180, height: 180, format: 'png', category: 'Social Media' },
  { name: 'Facebook Cover', filename: 'social/facebook-cover-820x312.png', width: 820, height: 312, format: 'png', category: 'Social Media' },
  
  // ============================================
  // FAVICONS
  // ============================================
  { name: 'Favicon 16', filename: 'favicon/favicon-16.png', width: 16, height: 16, format: 'png', category: 'Favicons' },
  { name: 'Favicon 32', filename: 'favicon/favicon-32.png', width: 32, height: 32, format: 'png', category: 'Favicons' },
  { name: 'Favicon 48', filename: 'favicon/favicon-48.png', width: 48, height: 48, format: 'png', category: 'Favicons' },
  { name: 'Favicon 64', filename: 'favicon/favicon-64.png', width: 64, height: 64, format: 'png', category: 'Favicons' },
  { name: 'Favicon 96', filename: 'favicon/favicon-96.png', width: 96, height: 96, format: 'png', category: 'Favicons' },
  { name: 'Favicon 128', filename: 'favicon/favicon-128.png', width: 128, height: 128, format: 'png', category: 'Favicons' },
  { name: 'Favicon 192', filename: 'favicon/favicon-192.png', width: 192, height: 192, format: 'png', category: 'Favicons' },
  { name: 'Favicon 256', filename: 'favicon/favicon-256.png', width: 256, height: 256, format: 'png', category: 'Favicons' },
  { name: 'Favicon 512', filename: 'favicon/favicon-512.png', width: 512, height: 512, format: 'png', category: 'Favicons' },
  
  // ============================================
  // APPLE TOUCH ICONS
  // ============================================
  { name: 'Apple Touch 57', filename: 'apple/apple-touch-icon-57.png', width: 57, height: 57, format: 'png', category: 'Apple Icons' },
  { name: 'Apple Touch 60', filename: 'apple/apple-touch-icon-60.png', width: 60, height: 60, format: 'png', category: 'Apple Icons' },
  { name: 'Apple Touch 72', filename: 'apple/apple-touch-icon-72.png', width: 72, height: 72, format: 'png', category: 'Apple Icons' },
  { name: 'Apple Touch 76', filename: 'apple/apple-touch-icon-76.png', width: 76, height: 76, format: 'png', category: 'Apple Icons' },
  { name: 'Apple Touch 114', filename: 'apple/apple-touch-icon-114.png', width: 114, height: 114, format: 'png', category: 'Apple Icons' },
  { name: 'Apple Touch 120', filename: 'apple/apple-touch-icon-120.png', width: 120, height: 120, format: 'png', category: 'Apple Icons' },
  { name: 'Apple Touch 144', filename: 'apple/apple-touch-icon-144.png', width: 144, height: 144, format: 'png', category: 'Apple Icons' },
  { name: 'Apple Touch 152', filename: 'apple/apple-touch-icon-152.png', width: 152, height: 152, format: 'png', category: 'Apple Icons' },
  { name: 'Apple Touch 180', filename: 'apple/apple-touch-icon-180.png', width: 180, height: 180, format: 'png', category: 'Apple Icons' },
  { name: 'Apple Touch 1024', filename: 'apple/apple-touch-icon-1024.png', width: 1024, height: 1024, format: 'png', category: 'Apple Icons' },
  
  // ============================================
  // ANDROID ICONS
  // ============================================
  { name: 'Android 36 LDPI', filename: 'android/android-icon-36.png', width: 36, height: 36, format: 'png', category: 'Android Icons' },
  { name: 'Android 48 MDPI', filename: 'android/android-icon-48.png', width: 48, height: 48, format: 'png', category: 'Android Icons' },
  { name: 'Android 72 HDPI', filename: 'android/android-icon-72.png', width: 72, height: 72, format: 'png', category: 'Android Icons' },
  { name: 'Android 96 XHDPI', filename: 'android/android-icon-96.png', width: 96, height: 96, format: 'png', category: 'Android Icons' },
  { name: 'Android 144 XXHDPI', filename: 'android/android-icon-144.png', width: 144, height: 144, format: 'png', category: 'Android Icons' },
  { name: 'Android 192 XXXHDPI', filename: 'android/android-icon-192.png', width: 192, height: 192, format: 'png', category: 'Android Icons' },
  { name: 'Android 512 Play Store', filename: 'android/android-icon-512.png', width: 512, height: 512, format: 'png', category: 'Android Icons' },
  
  // ============================================
  // META / OG IMAGES
  // ============================================
  { name: 'OG Image', filename: 'meta/og-image-1200x630.png', width: 1200, height: 630, format: 'png', category: 'Meta/OG' },
  { name: 'OG Image Square', filename: 'meta/og-image-1200x1200.png', width: 1200, height: 1200, format: 'png', category: 'Meta/OG' },
  { name: 'Twitter Card', filename: 'meta/twitter-card-1200x600.png', width: 1200, height: 600, format: 'png', category: 'Meta/OG' },
  { name: 'Twitter Card Large', filename: 'meta/twitter-card-1500x750.png', width: 1500, height: 750, format: 'png', category: 'Meta/OG' },
  
  // ============================================
  // EMAIL SIGNATURES
  // ============================================
  { name: 'Email Sig Small', filename: 'email/email-sig-100.png', width: 100, height: 100, format: 'png', category: 'Email' },
  { name: 'Email Sig Medium', filename: 'email/email-sig-150.png', width: 150, height: 150, format: 'png', category: 'Email' },
  { name: 'Email Sig Large', filename: 'email/email-sig-200.png', width: 200, height: 200, format: 'png', category: 'Email' },
  
  // ============================================
  // PRINT / HIGH-RES
  // ============================================
  { name: 'Print 4K', filename: 'print/kelly-logo-4k-2048.png', width: 2048, height: 2048, format: 'png', category: 'Print' },
  { name: 'Print 2K', filename: 'print/kelly-logo-2k-1024.png', width: 1024, height: 1024, format: 'png', category: 'Print' },
  { name: 'Print 1K', filename: 'print/kelly-logo-1k-512.png', width: 512, height: 512, format: 'png', category: 'Print' },
  
  // ============================================
  // WEB OPTIMIZED (WebP)
  // ============================================
  { name: 'Web 2048 WebP', filename: 'web/kelly-logo-2048.webp', width: 2048, height: 2048, format: 'webp', quality: 95, category: 'Web Optimized' },
  { name: 'Web 1024 WebP', filename: 'web/kelly-logo-1024.webp', width: 1024, height: 1024, format: 'webp', quality: 95, category: 'Web Optimized' },
  { name: 'Web 512 WebP', filename: 'web/kelly-logo-512.webp', width: 512, height: 512, format: 'webp', quality: 95, category: 'Web Optimized' },
  { name: 'Web 256 WebP', filename: 'web/kelly-logo-256.webp', width: 256, height: 256, format: 'webp', quality: 95, category: 'Web Optimized' },
  
  // ============================================
  // CIRCLE CROPS
  // ============================================
  { name: 'Circle 512', filename: 'circle/kelly-circle-512.png', width: 512, height: 512, format: 'png', circle: true, category: 'Circle Crops' },
  { name: 'Circle 256', filename: 'circle/kelly-circle-256.png', width: 256, height: 256, format: 'png', circle: true, category: 'Circle Crops' },
  { name: 'Circle 128', filename: 'circle/kelly-circle-128.png', width: 128, height: 128, format: 'png', circle: true, category: 'Circle Crops' },
  { name: 'Circle 64', filename: 'circle/kelly-circle-64.png', width: 64, height: 64, format: 'png', circle: true, category: 'Circle Crops' },
];

async function generateAsset(sourceBuffer: Buffer, spec: AssetSpec): Promise<void> {
  const outputPath = path.join(OUTPUT_DIR, spec.filename);
  const outputDir = path.dirname(outputPath);
  
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  let pipeline = sharp(sourceBuffer);
  
  // For non-square outputs, we need to handle the crop/fit differently
  if (spec.width !== spec.height) {
    // Create a canvas with the target dimensions and place Kelly centered
    pipeline = pipeline
      .resize(spec.width, spec.height, {
        fit: 'contain',
        background: { r: 255, g: 255, b: 255, alpha: 1 }
      });
  } else {
    // Square - simple resize
    pipeline = pipeline.resize(spec.width, spec.height, { fit: 'cover' });
  }
  
  // Apply circle mask if needed
  if (spec.circle) {
    const circleSize = Math.min(spec.width, spec.height);
    const circleSvg = `<svg width="${circleSize}" height="${circleSize}">
      <circle cx="${circleSize/2}" cy="${circleSize/2}" r="${circleSize/2}" fill="white"/>
    </svg>`;
    
    pipeline = sharp(sourceBuffer)
      .resize(circleSize, circleSize, { fit: 'cover' })
      .composite([{
        input: Buffer.from(circleSvg),
        blend: 'dest-in'
      }]);
  }
  
  // Output format
  if (spec.format === 'png') {
    await pipeline.png({ compressionLevel: 9 }).toFile(outputPath);
  } else if (spec.format === 'jpeg') {
    await pipeline.jpeg({ quality: spec.quality || 90 }).toFile(outputPath);
  } else if (spec.format === 'webp') {
    await pipeline.webp({ quality: spec.quality || 90 }).toFile(outputPath);
  }
}

async function createZipArchive(): Promise<string> {
  const zipPath = path.join(OUTPUT_DIR, 'curious-kelly-logo-complete.zip');
  const output = fs.createWriteStream(zipPath);
  const archive = archiver('zip', { zlib: { level: 9 } });
  
  return new Promise((resolve, reject) => {
    output.on('close', () => resolve(zipPath));
    archive.on('error', reject);
    
    archive.pipe(output);
    
    // Add all subdirectories
    const subdirs = ['social', 'favicon', 'apple', 'android', 'meta', 'email', 'print', 'web', 'circle'];
    for (const subdir of subdirs) {
      const dirPath = path.join(OUTPUT_DIR, subdir);
      if (fs.existsSync(dirPath)) {
        archive.directory(dirPath, subdir);
      }
    }
    
    // Add the index HTML
    const indexPath = path.join(OUTPUT_DIR, 'index.html');
    if (fs.existsSync(indexPath)) {
      archive.file(indexPath, { name: 'index.html' });
    }
    
    archive.finalize();
  });
}

function generateIndexHtml(generatedAssets: { spec: AssetSpec; path: string }[]): void {
  const categories = [...new Set(ASSET_SPECS.map(s => s.category))];
  
  const assetsByCategory = categories.map(cat => ({
    name: cat,
    assets: generatedAssets.filter(a => a.spec.category === cat)
  }));
  
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>✨ Curious Kelly Logo - Complete Asset Package</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --kelly-orange: #d97757;
      --kelly-blue: #7BA7C2;
      --bg: #0a0a0c;
      --card: #141418;
      --border: rgba(255,255,255,0.08);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'DM Sans', sans-serif;
      background: var(--bg);
      color: white;
      line-height: 1.6;
    }
    .hero {
      text-align: center;
      padding: 4rem 2rem;
      background: linear-gradient(180deg, rgba(217,119,87,0.1) 0%, transparent 100%);
      border-bottom: 1px solid var(--border);
    }
    .hero h1 {
      font-size: 2.5rem;
      background: linear-gradient(135deg, var(--kelly-blue), var(--kelly-orange));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    .hero p { color: #888; }
    .hero .stats {
      display: flex;
      justify-content: center;
      gap: 2rem;
      margin-top: 1.5rem;
      flex-wrap: wrap;
    }
    .hero .stat {
      text-align: center;
    }
    .hero .stat-value {
      font-size: 2rem;
      font-weight: 700;
      color: var(--kelly-orange);
    }
    .hero .stat-label {
      font-size: 0.875rem;
      color: #666;
    }
    .container { max-width: 1400px; margin: 0 auto; padding: 3rem 2rem; }
    
    .download-zip {
      display: flex;
      justify-content: center;
      margin-bottom: 3rem;
    }
    .download-zip a {
      display: inline-flex;
      align-items: center;
      gap: 0.75rem;
      padding: 1.25rem 3rem;
      background: linear-gradient(135deg, var(--kelly-blue), var(--kelly-orange));
      color: white;
      text-decoration: none;
      border-radius: 16px;
      font-weight: 600;
      font-size: 1.1rem;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .download-zip a:hover {
      transform: scale(1.02);
      box-shadow: 0 8px 32px rgba(217,119,87,0.3);
    }
    
    .category {
      margin-bottom: 3rem;
    }
    .category h2 {
      font-size: 1.25rem;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .category h2 .count {
      font-size: 0.75rem;
      background: var(--kelly-orange);
      padding: 0.25rem 0.5rem;
      border-radius: 999px;
      font-weight: 600;
    }
    
    .assets-table {
      width: 100%;
      border-collapse: collapse;
    }
    .assets-table th,
    .assets-table td {
      padding: 0.75rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    .assets-table th {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #666;
      font-weight: 500;
    }
    .assets-table tr:hover {
      background: rgba(255,255,255,0.02);
    }
    .assets-table .preview {
      width: 40px;
      height: 40px;
      background: white;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .assets-table .preview img {
      max-width: 100%;
      max-height: 100%;
    }
    .assets-table .filename {
      font-family: monospace;
      font-size: 0.8rem;
      color: #aaa;
    }
    .assets-table .size {
      color: var(--kelly-blue);
      font-weight: 500;
    }
    .assets-table a {
      color: var(--kelly-orange);
      text-decoration: none;
      font-weight: 500;
    }
    .assets-table a:hover {
      text-decoration: underline;
    }
    
    footer {
      text-align: center;
      padding: 2rem;
      border-top: 1px solid var(--border);
      color: #666;
      font-size: 0.875rem;
    }
    footer a { color: var(--kelly-orange); text-decoration: none; }
  </style>
</head>
<body>
  <header class="hero">
    <h1>✨ Curious Kelly Logo</h1>
    <p>Complete Asset Package for All Platforms</p>
    <div class="stats">
      <div class="stat">
        <div class="stat-value">${generatedAssets.length}</div>
        <div class="stat-label">Total Assets</div>
      </div>
      <div class="stat">
        <div class="stat-value">${categories.length}</div>
        <div class="stat-label">Categories</div>
      </div>
      <div class="stat">
        <div class="stat-value">4K</div>
        <div class="stat-label">Max Resolution</div>
      </div>
    </div>
  </header>
  
  <main class="container">
    <div class="download-zip">
      <a href="curious-kelly-logo-complete.zip" download>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Download Complete ZIP (${generatedAssets.length} assets)
      </a>
    </div>
    
    ${assetsByCategory.map(cat => `
    <section class="category">
      <h2>${getCategoryIcon(cat.name)} ${cat.name} <span class="count">${cat.assets.length}</span></h2>
      <table class="assets-table">
        <thead>
          <tr>
            <th style="width:60px">Preview</th>
            <th>Name</th>
            <th>Dimensions</th>
            <th>Format</th>
            <th>File</th>
            <th style="width:100px"></th>
          </tr>
        </thead>
        <tbody>
          ${cat.assets.map(a => `
          <tr>
            <td><div class="preview"><img src="${a.spec.filename}" alt="${a.spec.name}"></div></td>
            <td>${a.spec.name}</td>
            <td class="size">${a.spec.width}×${a.spec.height}</td>
            <td>${a.spec.format.toUpperCase()}</td>
            <td class="filename">${path.basename(a.spec.filename)}</td>
            <td><a href="${a.spec.filename}" download>Download</a></td>
          </tr>`).join('')}
        </tbody>
      </table>
    </section>`).join('')}
  </main>
  
  <footer>
    <p>© 2025 Lesson of the Day PBC. All rights reserved.</p>
    <p><a href="mailto:hello@curiouskelly.com">hello@curiouskelly.com</a> • <a href="https://curiouskelly.com">curiouskelly.com</a></p>
  </footer>
</body>
</html>`;

  fs.writeFileSync(path.join(OUTPUT_DIR, 'index.html'), html);
}

function getCategoryIcon(category: string): string {
  const icons: Record<string, string> = {
    'Social Media': '📱',
    'Favicons': '🌐',
    'Apple Icons': '🍎',
    'Android Icons': '🤖',
    'Meta/OG': '🔗',
    'Email': '📧',
    'Print': '🖨️',
    'Web Optimized': '⚡',
    'Circle Crops': '⭕',
  };
  return icons[category] || '📁';
}

async function main() {
  console.log('═'.repeat(60));
  console.log('📦 CURIOUS KELLY LOGO - COMPLETE PACKAGE GENERATOR');
  console.log('═'.repeat(60));
  
  // Check source exists
  if (!fs.existsSync(SOURCE_IMAGE)) {
    console.error(`❌ Source image not found: ${SOURCE_IMAGE}`);
    process.exit(1);
  }
  
  console.log(`\n📸 Source: ${path.basename(SOURCE_IMAGE)}`);
  console.log(`📁 Output: ${OUTPUT_DIR}`);
  console.log(`🎯 Generating ${ASSET_SPECS.length} assets across ${[...new Set(ASSET_SPECS.map(s => s.category))].length} categories\n`);
  
  // Create output directory
  if (fs.existsSync(OUTPUT_DIR)) {
    fs.rmSync(OUTPUT_DIR, { recursive: true });
  }
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  
  // Load source image
  const sourceBuffer = fs.readFileSync(SOURCE_IMAGE);
  
  // Generate all assets
  const generatedAssets: { spec: AssetSpec; path: string }[] = [];
  
  for (let i = 0; i < ASSET_SPECS.length; i++) {
    const spec = ASSET_SPECS[i];
    process.stdout.write(`\r[${i + 1}/${ASSET_SPECS.length}] Generating ${spec.name}...`);
    
    try {
      await generateAsset(sourceBuffer, spec);
      generatedAssets.push({ spec, path: path.join(OUTPUT_DIR, spec.filename) });
    } catch (error) {
      console.error(`\n❌ Error generating ${spec.name}:`, error);
    }
  }
  
  console.log(`\r✅ Generated ${generatedAssets.length}/${ASSET_SPECS.length} assets                    `);
  
  // Generate index HTML
  console.log('📄 Generating index.html...');
  generateIndexHtml(generatedAssets);
  
  // Create ZIP archive
  console.log('📦 Creating ZIP archive...');
  const zipPath = await createZipArchive();
  const zipSize = (fs.statSync(zipPath).size / 1024 / 1024).toFixed(2);
  
  console.log('\n' + '═'.repeat(60));
  console.log('✅ PACKAGE COMPLETE');
  console.log('═'.repeat(60));
  console.log(`\n📁 Location: ${OUTPUT_DIR}`);
  console.log(`📦 ZIP: curious-kelly-logo-complete.zip (${zipSize} MB)`);
  console.log(`🌐 Open index.html to browse and download individual assets`);
  console.log(`\n✨ Kelly is ready for the world!`);
}

main().catch(console.error);


