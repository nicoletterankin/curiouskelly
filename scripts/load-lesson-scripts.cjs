/**
 * Load Lesson Scripts to Supabase
 * LEARNER CLASS - Isolated from investor systems
 * 
 * Parses public/data/day-*.js files and inserts into lesson_scripts table
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '../public/data');

// Extract JSON from JS window assignment
function extractJSON(content) {
  // Remove window assignment wrapper
  const match = content.match(/window\.CURIOUS_KELLY\.(DAY_\d+_?[A-Z]*|LOCAL_PACKS)\s*=\s*({[\s\S]+?});?\s*(?:\/\/|window\.|$)/);
  if (!match) {
    // Try alternative pattern for unified format
    const altMatch = content.match(/window\.CURIOUS_KELLY\.DAY_\d+_UNIFIED\s*=\s*({[\s\S]+});/);
    if (altMatch) {
      try {
        return JSON.parse(altMatch[1].replace(/\/\/[^\n]*/g, '')); // Remove comments
      } catch (e) {
        return null;
      }
    }
    return null;
  }
  
  try {
    // Clean up JS-style comments and parse
    const jsonStr = match[2].replace(/\/\/[^\n]*/g, '').replace(/,\s*}/g, '}').replace(/,\s*]/g, ']');
    return JSON.parse(jsonStr);
  } catch (e) {
    return null;
  }
}

// Convert day data to script record
function toScriptRecord(dayNumber, data, sourceFile) {
  const record = {
    day_number: dayNumber,
    track: 'learn',
    source_file: sourceFile,
    version: data.meta?.version || 'v1.0',
    status: 'approved' // Mark as approved since these are existing content
  };

  // Handle unified format (day-001-unified.js)
  if (data.learn) {
    record.topic = data.learn.topic;
    record.headline = data.learn.headline;
    record.universal_truth = data.learn.universal_truth;
    record.emoji = data.learn.emoji;
    record.category = data.learn.category || 'general';
    record.thumbnail_url = data.learn.thumbnail_url;
    record.phases = JSON.stringify(data.learn.phases || []);
    record.age_variants = JSON.stringify(data.ageVariants || {});
  }
  // Handle standard format (day-XXX-complete.js)
  else if (data.lesson) {
    record.topic = data.lesson.topic;
    record.headline = data.lesson.headline;
    record.universal_truth = data.lesson.universal_truth;
    record.emoji = data.lesson.emoji;
    record.category = data.lesson.category || 'general';
    record.thumbnail_url = data.lesson.thumbnail_url;
    
    // Convert atoms to phases format
    const phases = (data.atoms || []).map((atom, idx) => ({
      id: atom.id,
      phase_key: atom.phase?.toLowerCase() || `phase_${idx}`,
      phase_index: idx,
      script: atom.content?.script || '',
      kelly_pose: atom.content?.kellyPose || 'neutral',
      kelly_emotion: atom.content?.kellyEmotion || 'curious',
      options: atom.content?.options || null,
      visual_url: atom.visual_url || null
    }));
    record.phases = JSON.stringify(phases);
    record.age_variants = JSON.stringify(data.ageVariants || {});
  }

  return record;
}

// Generate SQL INSERT
function toSQL(record) {
  const escapedTopic = (record.topic || '').replace(/'/g, "''");
  const escapedHeadline = (record.headline || '').replace(/'/g, "''");
  const escapedTruth = (record.universal_truth || '').replace(/'/g, "''");
  
  return `INSERT INTO lesson_scripts (day_number, track, topic, headline, universal_truth, emoji, category, thumbnail_url, phases, age_variants, status, version, source_file)
VALUES (${record.day_number}, '${record.track}', '${escapedTopic}', '${escapedHeadline}', '${escapedTruth}', '${record.emoji || '📚'}', '${record.category || 'general'}', '${record.thumbnail_url || ''}', '${record.phases}'::jsonb, '${record.age_variants}'::jsonb, '${record.status}', '${record.version}', '${record.source_file}')
ON CONFLICT (day_number, track, version) DO UPDATE SET
  topic = EXCLUDED.topic,
  headline = EXCLUDED.headline,
  universal_truth = EXCLUDED.universal_truth,
  phases = EXCLUDED.phases,
  age_variants = EXCLUDED.age_variants,
  updated_at = NOW();`;
}

// Main
async function main() {
  const files = fs.readdirSync(DATA_DIR)
    .filter(f => f.match(/^day-\d{3}.*\.js$/))
    .sort();
  
  console.log(`Found ${files.length} day files\n`);
  
  const records = [];
  const errors = [];
  
  for (const file of files) {
    const dayMatch = file.match(/day-(\d{3})/);
    if (!dayMatch) continue;
    
    const dayNumber = parseInt(dayMatch[1], 10);
    const filePath = path.join(DATA_DIR, file);
    const content = fs.readFileSync(filePath, 'utf-8');
    
    const data = extractJSON(content);
    if (!data) {
      errors.push({ file, error: 'Failed to parse JSON' });
      continue;
    }
    
    const record = toScriptRecord(dayNumber, data, file);
    if (record.topic) {
      records.push(record);
    } else {
      errors.push({ file, error: 'Missing topic' });
    }
  }
  
  console.log(`Successfully parsed: ${records.length}`);
  console.log(`Errors: ${errors.length}`);
  
  if (errors.length > 0) {
    console.log('\nFirst 10 errors:');
    errors.slice(0, 10).forEach(e => console.log(`  - ${e.file}: ${e.error}`));
  }
  
  // Generate SQL file
  const sqlStatements = records.map(toSQL);
  const sqlFile = path.join(__dirname, 'lesson-scripts-insert.sql');
  fs.writeFileSync(sqlFile, sqlStatements.join('\n\n'));
  console.log(`\nGenerated SQL file: ${sqlFile}`);
  console.log(`Total INSERT statements: ${sqlStatements.length}`);
  
  // Also output summary
  const summary = records.slice(0, 30).map(r => ({
    day: r.day_number,
    topic: r.topic,
    phases: JSON.parse(r.phases).length
  }));
  console.log('\nFirst 30 days summary:');
  console.table(summary);
}

main().catch(console.error);
