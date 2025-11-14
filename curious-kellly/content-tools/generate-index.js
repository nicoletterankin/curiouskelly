#!/usr/bin/env node

/**
 * Generate Lesson Index
 * Creates/updates the .index.json catalog file
 */

const fs = require('fs');
const path = require('path');
const { detectSchemaVersion, checkMultilingualCompleteness, checkAgeVariantCompleteness } = require('./organize-lessons');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');
const INDEX_FILE = path.join(LESSONS_DIR, '.index.json');

/**
 * Generate index catalog
 */
function generateIndex() {
  const index = {
    "$schema": "../lesson-dna-schema.json",
    "version": "1.0.0",
    "generatedAt": new Date().toISOString(),
    "description": "Catalog of all lessons in the system",
    "lessons": []
  };
  
  if (!fs.existsSync(LESSONS_DIR)) {
    console.error(`Error: Lessons directory not found: ${LESSONS_DIR}`);
    return index;
  }
  
  const files = fs.readdirSync(LESSONS_DIR).filter(f => 
    f.endsWith('.json') && !f.startsWith('.')
  );
  
  console.log(`\n📚 Generating index for ${files.length} lessons...\n`);
  
  files.forEach((file, indexNum) => {
    const filePath = path.join(LESSONS_DIR, file);
    console.log(`[${indexNum + 1}/${files.length}] ${file}`);
    
    try {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const schemaVersion = detectSchemaVersion(data);
      const multilingual = checkMultilingualCompleteness(data);
      const ageVariants = checkAgeVariantCompleteness(data);
      
      const lessonEntry = {
        id: data.id || data.lesson_id || file.replace('.json', ''),
        filename: file,
        title: data.title || data.universal_concept_translations?.en || 'Untitled',
        schemaVersion,
        multilingual: {
          hasEn: multilingual.hasEn,
          hasEs: multilingual.hasEs,
          hasFr: multilingual.hasFr,
          completeness: {
            en: multilingual.hasEn ? '100%' : '0%',
            es: multilingual.hasEs ? '100%' : '0%',
            fr: multilingual.hasFr ? '100%' : '0%'
          }
        },
        ageVariants: {
          present: ageVariants.present.length,
          missing: ageVariants.missing.length,
          completeness: Math.round(ageVariants.completeness * 100) + '%'
        },
        metadata: {
          category: data.metadata?.category || 'unknown',
          difficulty: data.metadata?.difficulty || 'unknown',
          tags: data.metadata?.tags || []
        },
        size: fs.statSync(filePath).size,
        updatedAt: data.updatedAt || data.createdAt || new Date().toISOString()
      };
      
      index.lessons.push(lessonEntry);
    } catch (e) {
      console.log(`  ⚠️  Error reading file: ${e.message}`);
      index.lessons.push({
        filename: file,
        error: e.message
      });
    }
  });
  
  // Sort by ID
  index.lessons.sort((a, b) => {
    if (a.id && b.id) {
      return a.id.localeCompare(b.id);
    }
    return 0;
  });
  
  return index;
}

// CLI
if (require.main === module) {
  const index = generateIndex();
  
  // Write index file
  fs.writeFileSync(INDEX_FILE, JSON.stringify(index, null, 2));
  console.log(`\n✅ Index generated: ${INDEX_FILE}`);
  console.log(`   Total lessons: ${index.lessons.length}\n`);
}

module.exports = { generateIndex };

