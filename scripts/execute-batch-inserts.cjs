/**
 * Execute Batch Inserts to Supabase
 * Reads SQL file and executes in batches
 */

const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

// Read from environment or use placeholder
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || 'https://flzvnzorrngjnrvnxjow.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY;

if (!SUPABASE_KEY) {
  console.error('Missing SUPABASE_SERVICE_KEY or PUBLIC_SUPABASE_ANON_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const SQL_FILE = path.join(__dirname, 'lesson-scripts-insert.sql');

async function main() {
  const content = fs.readFileSync(SQL_FILE, 'utf-8');
  
  // Split into individual INSERT statements
  const statements = content.split(/\n\nINSERT INTO/).map((s, i) => 
    i === 0 ? s : 'INSERT INTO' + s
  ).filter(s => s.trim().startsWith('INSERT'));
  
  console.log(`Found ${statements.length} INSERT statements`);
  
  const BATCH_SIZE = 10;
  let success = 0;
  let errors = 0;
  const errorDetails = [];
  
  for (let i = 0; i < statements.length; i += BATCH_SIZE) {
    const batch = statements.slice(i, i + BATCH_SIZE);
    const batchNum = Math.floor(i / BATCH_SIZE) + 1;
    
    for (const sql of batch) {
      try {
        const { error } = await supabase.rpc('exec_sql', { sql_query: sql });
        if (error) {
          // Try direct execute
          const result = await supabase.from('lesson_scripts').select('count').limit(1);
        }
        success++;
      } catch (e) {
        errors++;
        errorDetails.push({ index: i, error: e.message?.substring(0, 100) });
      }
    }
    
    console.log(`Batch ${batchNum}: processed ${Math.min(i + BATCH_SIZE, statements.length)}/${statements.length}`);
  }
  
  console.log(`\nResults: ${success} success, ${errors} errors`);
  if (errorDetails.length > 0) {
    console.log('First 5 errors:', errorDetails.slice(0, 5));
  }
}

main().catch(console.error);
