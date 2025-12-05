#!/usr/bin/env node
/**
 * Generate Visual Asset Manifest
 * 
 * Scans the /kelly/lessons directory and generates a JSON manifest
 * of all available visual assets, their status, and metadata.
 * 
 * Usage:
 *   node scripts/generate-visual-manifest.js
 * 
 * Output:
 *   public/kelly/lessons/manifest.json
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const LESSONS_DIR = path.join(__dirname, '../public/kelly/lessons');
const MANIFEST_PATH = path.join(LESSONS_DIR, 'manifest.json');
const LESSONS_JSON = path.join(__dirname, '../lessons/365_day_calendar.json');

const ASSET_TYPES = ['bg', 'hero', 'prop', 'guide-point', 'reaction'];

function getFileHash(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    return crypto.createHash('md5').update(buffer).digest('hex').substring(0, 8);
  } catch (e) {
    return null;
  }
}

function getFileSize(filePath) {
  try {
    const stats = fs.statSync(filePath);
    return stats.size;
  } catch (e) {
    return 0;
  }
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function generateManifest() {
  console.log('\n📋 Generating Visual Asset Manifest\n');
  
  // Load lesson data
  const lessonsData = JSON.parse(fs.readFileSync(LESSONS_JSON, 'utf8'));
  const lessonsByDay = {};
  lessonsData.lessons.forEach(l => {
    lessonsByDay[l.day] = l;
  });
  
  const manifest = {
    version: '1.0.0',
    generatedAt: new Date().toISOString(),
    totalLessons: 365,
    assetTypes: ASSET_TYPES,
    stats: {
      totalAssetsExpected: 365 * ASSET_TYPES.length,
      totalAssetsFound: 0,
      totalMissing: 0,
      totalSizeBytes: 0,
      completeLessons: 0,
      partialLessons: 0,
      emptyLessons: 0
    },
    lessons: {}
  };
  
  // Scan each day
  for (let day = 1; day <= 365; day++) {
    const paddedDay = String(day).padStart(3, '0');
    const dayDir = path.join(LESSONS_DIR, paddedDay);
    const lesson = lessonsByDay[day] || { title: 'Unknown', objective: '' };
    
    const lessonEntry = {
      day: day,
      title: lesson.title,
      objective: lesson.objective,
      directory: `/kelly/lessons/${paddedDay}`,
      assets: {},
      assetsFound: 0,
      assetsMissing: 0,
      totalSize: 0,
      status: 'empty' // 'complete', 'partial', 'empty'
    };
    
    // Check each asset type
    for (const assetType of ASSET_TYPES) {
      const fileName = `lesson-${day}-${assetType}.png`;
      const filePath = path.join(dayDir, fileName);
      const exists = fs.existsSync(filePath);
      
      if (exists) {
        const size = getFileSize(filePath);
        const hash = getFileHash(filePath);
        
        lessonEntry.assets[assetType] = {
          exists: true,
          path: `/kelly/lessons/${paddedDay}/${fileName}`,
          size: size,
          sizeFormatted: formatBytes(size),
          hash: hash
        };
        
        lessonEntry.assetsFound++;
        lessonEntry.totalSize += size;
        manifest.stats.totalAssetsFound++;
        manifest.stats.totalSizeBytes += size;
      } else {
        lessonEntry.assets[assetType] = {
          exists: false,
          path: `/kelly/lessons/${paddedDay}/${fileName}`
        };
        
        lessonEntry.assetsMissing++;
        manifest.stats.totalMissing++;
      }
    }
    
    // Determine lesson status
    if (lessonEntry.assetsFound === ASSET_TYPES.length) {
      lessonEntry.status = 'complete';
      manifest.stats.completeLessons++;
    } else if (lessonEntry.assetsFound > 0) {
      lessonEntry.status = 'partial';
      manifest.stats.partialLessons++;
    } else {
      lessonEntry.status = 'empty';
      manifest.stats.emptyLessons++;
    }
    
    lessonEntry.totalSizeFormatted = formatBytes(lessonEntry.totalSize);
    manifest.lessons[day] = lessonEntry;
  }
  
  manifest.stats.totalSizeFormatted = formatBytes(manifest.stats.totalSizeBytes);
  manifest.stats.completionPercentage = 
    ((manifest.stats.totalAssetsFound / manifest.stats.totalAssetsExpected) * 100).toFixed(1);
  
  // Write manifest
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
  
  // Print summary
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('                  VISUAL ASSET MANIFEST SUMMARY                 ');
  console.log('═══════════════════════════════════════════════════════════════\n');
  
  console.log(`📁 Output: ${MANIFEST_PATH}\n`);
  
  console.log('PROGRESS:');
  console.log(`  ✅ Complete lessons (5/5 assets): ${manifest.stats.completeLessons}`);
  console.log(`  🟡 Partial lessons:               ${manifest.stats.partialLessons}`);
  console.log(`  ❌ Empty lessons:                 ${manifest.stats.emptyLessons}`);
  console.log('');
  
  console.log('ASSETS:');
  console.log(`  Total expected: ${manifest.stats.totalAssetsExpected}`);
  console.log(`  Total found:    ${manifest.stats.totalAssetsFound}`);
  console.log(`  Total missing:  ${manifest.stats.totalMissing}`);
  console.log(`  Completion:     ${manifest.stats.completionPercentage}%`);
  console.log('');
  
  console.log('STORAGE:');
  console.log(`  Total size: ${manifest.stats.totalSizeFormatted}`);
  console.log('');
  
  // List complete lessons
  const completeDays = Object.values(manifest.lessons)
    .filter(l => l.status === 'complete')
    .map(l => l.day)
    .sort((a, b) => a - b);
  
  if (completeDays.length > 0) {
    console.log('COMPLETE LESSONS:');
    
    // Group consecutive days
    const groups = [];
    let currentGroup = [completeDays[0]];
    
    for (let i = 1; i < completeDays.length; i++) {
      if (completeDays[i] === completeDays[i-1] + 1) {
        currentGroup.push(completeDays[i]);
      } else {
        groups.push(currentGroup);
        currentGroup = [completeDays[i]];
      }
    }
    groups.push(currentGroup);
    
    const groupStrings = groups.map(g => {
      if (g.length === 1) return String(g[0]);
      return `${g[0]}-${g[g.length-1]}`;
    });
    
    console.log(`  Days: ${groupStrings.join(', ')}`);
  }
  
  console.log('\n═══════════════════════════════════════════════════════════════\n');
  
  return manifest;
}

// Run if called directly
if (require.main === module) {
  generateManifest();
}

module.exports = { generateManifest };


