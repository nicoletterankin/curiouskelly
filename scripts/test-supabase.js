/**
 * Supabase Connection Test Script
 * Tests connectivity to Supabase and queries core_lessons table
 *
 * Run from project root: node scripts/test-supabase.js
 */

import { createRequire } from 'module';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

// Create require to load from daily-lesson-marketing node_modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(resolve(__dirname, '../daily-lesson-marketing/package.json'));

const { createClient } = require('@supabase/supabase-js');
const dotenv = require('dotenv');

// Load environment variables from project root .env
const envPath = resolve(__dirname, '..', '.env');
dotenv.config({ path: envPath });

async function testSupabaseConnection() {
  console.log('🔄 Testing Supabase connection...\n');

  // Check for required environment variables
  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl) {
    console.error('❌ Error: PUBLIC_SUPABASE_URL is not set in .env');
    process.exit(1);
  }

  if (!supabaseKey) {
    console.error('❌ Error: PUBLIC_SUPABASE_ANON_KEY is not set in .env');
    process.exit(1);
  }

  console.log(`📍 Supabase URL: ${supabaseUrl.substring(0, 30)}...`);
  console.log(`🔑 API Key: ${supabaseKey.substring(0, 20)}...[redacted]\n`);

  try {
    // Create Supabase client
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Query core_lessons table count
    const { count, error } = await supabase
      .from('core_lessons')
      .select('*', { count: 'exact', head: true });

    if (error) {
      throw error;
    }

    console.log(`✅ Supabase connected! Found ${count} lessons`);
    console.log('\n📊 Connection test passed successfully!');
  } catch (error) {
    console.error('❌ Connection failed:', error.message);

    if (error.message.includes('relation') && error.message.includes('does not exist')) {
      console.log('\n💡 Hint: The core_lessons table may not exist yet.');
      console.log('   Run migrations or check your database schema.');
    } else if (error.message.includes('Invalid API key')) {
      console.log('\n💡 Hint: Check that PUBLIC_SUPABASE_ANON_KEY is correct.');
    } else if (error.code === 'PGRST301') {
      console.log('\n💡 Hint: Row Level Security may be blocking access.');
      console.log('   Check your RLS policies on the core_lessons table.');
    }

    process.exit(1);
  }
}

testSupabaseConnection();
