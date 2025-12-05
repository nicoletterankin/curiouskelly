/**
 * FULL Database Backup with Pagination
 * Handles Supabase's 1000-row limit
 */

import { createClient } from '@supabase/supabase-js';
import { writeFileSync, mkdirSync, existsSync } from 'fs';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const BACKUP_DIR = './backups';
const DATE = new Date().toISOString().split('T')[0];
const TIMESTAMP = new Date().toISOString().replace(/[:.]/g, '-');
const PAGE_SIZE = 1000;

async function backupTableWithPagination(tableName) {
  console.log(`📦 Backing up ${tableName}...`);
  
  let allData = [];
  let page = 0;
  let hasMore = true;
  
  while (hasMore) {
    const from = page * PAGE_SIZE;
    const to = from + PAGE_SIZE - 1;
    
    const { data, error } = await supabase
      .from(tableName)
      .select('*')
      .range(from, to);
    
    if (error) {
      console.error(`❌ Error at page ${page}:`, error.message);
      break;
    }
    
    if (data && data.length > 0) {
      allData = allData.concat(data);
      console.log(`   Page ${page + 1}: ${data.length} rows (total: ${allData.length})`);
      
      if (data.length < PAGE_SIZE) {
        hasMore = false;
      } else {
        page++;
      }
    } else {
      hasMore = false;
    }
  }
  
  if (allData.length === 0) {
    console.log(`⚠️  ${tableName}: No data or not accessible`);
    return { table: tableName, success: false, rows: 0 };
  }
  
  const filename = `${BACKUP_DIR}/${tableName}_${TIMESTAMP}.json`;
  writeFileSync(filename, JSON.stringify(allData, null, 2));
  
  console.log(`✅ ${tableName}: ${allData.length} rows → ${filename}`);
  return { table: tableName, success: true, rows: allData.length, file: filename };
}

async function main() {
  console.log('');
  console.log('╔═══════════════════════════════════════════════════════════╗');
  console.log('║        🚀 FULL DATABASE BACKUP WITH PAGINATION            ║');
  console.log('╚═══════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`📅 Date: ${DATE}`);
  console.log(`🕐 Timestamp: ${TIMESTAMP}`);
  console.log('');
  
  // Create backup directory
  if (!existsSync(BACKUP_DIR)) {
    mkdirSync(BACKUP_DIR, { recursive: true });
  }
  
  const results = [];
  
  // Core content tables (THE BIG ONES)
  console.log('═══ CORE CONTENT ═══');
  results.push(await backupTableWithPagination('core_lessons'));
  results.push(await backupTableWithPagination('lesson_atoms'));
  results.push(await backupTableWithPagination('lesson_shards'));
  
  // User tables
  console.log('');
  console.log('═══ USER DATA ═══');
  results.push(await backupTableWithPagination('users'));
  results.push(await backupTableWithPagination('user_progress'));
  
  // Summary
  console.log('');
  console.log('╔═══════════════════════════════════════════════════════════╗');
  console.log('║                    📊 BACKUP SUMMARY                       ║');
  console.log('╠═══════════════════════════════════════════════════════════╣');
  
  let totalRows = 0;
  for (const r of results) {
    const status = r.success ? '✅' : '❌';
    const rowStr = r.rows.toString().padStart(6, ' ');
    console.log(`║ ${status} ${r.table.padEnd(20)} ${rowStr} rows               ║`);
    if (r.success) totalRows += r.rows;
  }
  
  console.log('╠═══════════════════════════════════════════════════════════╣');
  console.log(`║ 📁 TOTAL: ${totalRows.toString().padStart(6, ' ')} rows backed up                       ║`);
  console.log(`║ 📂 Location: ${BACKUP_DIR}/                                   ║`);
  console.log('╚═══════════════════════════════════════════════════════════╝');
  console.log('');
  
  // File sizes
  console.log('📏 File sizes:');
  const files = results.filter(r => r.success).map(r => r.file);
  for (const f of files) {
    try {
      const stats = await import('fs').then(fs => fs.statSync(f));
      const sizeMB = (stats.size / 1024 / 1024).toFixed(2);
      console.log(`   ${f}: ${sizeMB} MB`);
    } catch (e) {
      // ignore
    }
  }
}

main().catch(console.error);








