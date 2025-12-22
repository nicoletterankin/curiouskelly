#!/usr/bin/env node
/**
 * Generate lessons-metadata.json from all 365 lesson files
 * 
 * Extracts: day, topic, headline, category, emoji, duration
 * Output: public/data/lessons-metadata.json (~70KB)
 * 
 * Usage: node scripts/generate-lessons-metadata.js
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const LESSONS_DIR = path.join(__dirname, '../public/lessons');
const OUTPUT_FILE = path.join(__dirname, '../public/data/lessons-metadata.json');

console.log('🔍 Scanning lessons directory...');

const lessons = [];
const errors = [];

// Scan all day-*.json files
for (let day = 1; day <= 365; day++) {
  const filename = `day-${day}.json`;
  const filepath = path.join(LESSONS_DIR, filename);
  
  try {
    if (!fs.existsSync(filepath)) {
      errors.push(`Missing: ${filename}`);
      continue;
    }
    
    const raw = fs.readFileSync(filepath, 'utf8');
    const data = JSON.parse(raw);
    
    // Extract metadata
    const meta = data.meta || {};
    const topic = typeof meta.topic === 'string' ? meta.topic : (meta.topic?.en || 'Unknown');
    const headline = typeof data.headline === 'string' ? data.headline : (data.headline?.en || '');
    const universalTruth = typeof data.universal_truth === 'string' ? data.universal_truth : (data.universal_truth?.en || '');
    
    lessons.push({
      day: day,
      topic: topic,
      hook: headline || universalTruth || `Day ${day} lesson`,
      category: meta.category || 'General',
      icon: meta.emoji || '📚',
      duration: 100, // All lessons are 100 seconds
      age_range: '8-102',
      tags: extractTags(data),
      date: meta.date || getDateForDay(day)
    });
    
  } catch (err) {
    errors.push(`Error parsing ${filename}: ${err.message}`);
  }
}

// Helper: Extract tags from lesson content
function extractTags(data) {
  const tags = [];
  const category = data.meta?.category;
  if (category) tags.push(category.toLowerCase());
  
  // Add topic-based tags
  const topic = typeof data.meta?.topic === 'string' ? data.meta.topic : data.meta?.topic?.en;
  if (topic) {
    const words = topic.toLowerCase().split(/\s+/);
    words.forEach(word => {
      if (word.length > 4 && !['about', 'things', 'where', 'what', 'when'].includes(word)) {
        tags.push(word);
      }
    });
  }
  
  return tags.slice(0, 5); // Max 5 tags
}

// Helper: Get date for day number (2026 calendar)
function getDateForDay(day) {
  const start = new Date('2026-01-01');
  const date = new Date(start);
  date.setDate(date.getDate() + day - 1);
  return date.toISOString().split('T')[0];
}

// Sort by day
lessons.sort((a, b) => a.day - b.day);

// Generate output
const output = {
  generated_at: new Date().toISOString(),
  version: '1.0.0',
  total_lessons: lessons.length,
  lessons: lessons
};

// Write to file
fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
fs.writeFileSync(OUTPUT_FILE, JSON.stringify(output, null, 2));

console.log(`✅ Generated ${lessons.length} lesson metadata entries`);
console.log(`📦 Output: ${OUTPUT_FILE}`);
console.log(`📊 File size: ${(fs.statSync(OUTPUT_FILE).size / 1024).toFixed(1)} KB`);

if (errors.length > 0) {
  console.log(`\n⚠️  ${errors.length} errors:`);
  errors.forEach(err => console.log(`   ${err}`));
}

// Sample output
console.log('\n📝 Sample entries:');
console.log(JSON.stringify(lessons.slice(0, 3), null, 2));
console.log('...');
console.log(JSON.stringify(lessons.slice(-2), null, 2));

