/**
 * Emergency Database Backup
 * Exports all critical tables to JSON files locally
 */

import { createClient } from '@supabase/supabase-js';
import { writeFileSync, mkdirSync, existsSync } from 'fs';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const BACKUP_DIR = './backups';
const DATE = new Date().toISOString().split('T')[0];

async function backupTable(tableName, selectColumns = '*') {
  console.log(`📦 Backing up ${tableName}...`);
  
  const { data, error, count } = await supabase
    .from(tableName)
    .select(selectColumns, { count: 'exact' });
  
  if (error) {
    console.error(`❌ Error backing up ${tableName}:`, error.message);
    return { table: tableName, success: false, error: error.message };
  }
  
  const filename = `${BACKUP_DIR}/${tableName}_${DATE}.json`;
  writeFileSync(filename, JSON.stringify(data, null, 2));
  
  console.log(`✅ ${tableName}: ${data.length} rows → ${filename}`);
  return { table: tableName, success: true, rows: data.length, file: filename };
}

async function main() {
  console.log('🚀 Starting Emergency Backup...');
  console.log(`📅 Date: ${DATE}`);
  console.log('');
  
  // Create backup directory
  if (!existsSync(BACKUP_DIR)) {
    mkdirSync(BACKUP_DIR, { recursive: true });
  }
  
  const results = [];
  
  // Core content tables
  results.push(await backupTable('core_lessons'));
  results.push(await backupTable('lesson_atoms'));
  results.push(await backupTable('lesson_shards'));
  
  // User tables (if accessible with anon key)
  try {
    results.push(await backupTable('users'));
  } catch (e) {
    console.log('⚠️  users table not accessible with anon key (expected)');
  }
  
  try {
    results.push(await backupTable('user_progress'));
  } catch (e) {
    console.log('⚠️  user_progress table not accessible with anon key (expected)');
  }
  
  // Summary
  console.log('');
  console.log('═══════════════════════════════════════════');
  console.log('📊 BACKUP SUMMARY');
  console.log('═══════════════════════════════════════════');
  
  let totalRows = 0;
  for (const r of results) {
    if (r.success) {
      console.log(`✅ ${r.table}: ${r.rows} rows`);
      totalRows += r.rows;
    } else {
      console.log(`❌ ${r.table}: ${r.error}`);
    }
  }
  
  console.log('───────────────────────────────────────────');
  console.log(`📁 Total: ${totalRows} rows backed up to ${BACKUP_DIR}/`);
  console.log(`📅 Files dated: ${DATE}`);
  console.log('═══════════════════════════════════════════');
}

main().catch(console.error);






