#!/usr/bin/env node
/**
 * Environment Variables Verification Script
 * Checks that all required environment variables are set
 */

require('dotenv').config();

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  blue: '\x1b[34m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function checkEnv() {
  log('\n🔍 Curious Kellly - Environment Variables Check', 'cyan');
  log('='.repeat(50), 'cyan');
  
  let hasErrors = false;
  let hasWarnings = false;

  // Required variables
  const required = {
    'OPENAI_API_KEY': 'OpenAI API key for AI services',
    'NODE_ENV': 'Node environment (development/production)',
    'PORT': 'Server port number'
  };

  // Optional but recommended
  const recommended = {
    'REDIS_URL': 'Redis connection (for session persistence)',
    'PINECONE_API_KEY': 'Pinecone API key (for RAG vector DB)',
    'QDRANT_URL': 'Qdrant URL (alternative to Pinecone)',
    'ELEVENLABS_API_KEY': 'ElevenLabs API key (for fallback TTS)'
  };

  log('\n📋 Required Variables:', 'blue');
  for (const [key, description] of Object.entries(required)) {
    const value = process.env[key];
    if (!value || value === '') {
      log(`  ❌ ${key}: MISSING - ${description}`, 'red');
      hasErrors = true;
    } else if (key === 'OPENAI_API_KEY' && (!value.startsWith('sk-') || value.length < 20)) {
      log(`  ⚠️  ${key}: INVALID - Should start with 'sk-' and be at least 20 chars`, 'yellow');
      hasWarnings = true;
    } else {
      const masked = key.includes('KEY') || key.includes('SECRET') || key.includes('PASSWORD')
        ? `${value.substring(0, 8)}...${value.substring(value.length - 4)}`
        : value;
      log(`  ✅ ${key}: ${masked}`, 'green');
    }
  }

  log('\n💡 Recommended Variables:', 'blue');
  for (const [key, description] of Object.entries(recommended)) {
    const value = process.env[key];
    if (!value || value === '') {
      log(`  ⚠️  ${key}: NOT SET - ${description}`, 'yellow');
      hasWarnings = true;
    } else {
      const masked = key.includes('KEY') || key.includes('SECRET') || key.includes('PASSWORD')
        ? `${value.substring(0, 8)}...${value.substring(value.length - 4)}`
        : value;
      log(`  ✅ ${key}: ${masked}`, 'green');
    }
  }

  // Check for conflicting configs
  log('\n🔍 Configuration Checks:', 'blue');
  if (process.env.PINECONE_API_KEY && process.env.QDRANT_URL) {
    log('  ⚠️  Both Pinecone and Qdrant configured - only one will be used', 'yellow');
    hasWarnings = true;
  }

  if (process.env.NODE_ENV === 'production' && !process.env.REDIS_URL) {
    log('  ⚠️  Production mode without Redis - sessions will be in-memory only', 'yellow');
    hasWarnings = true;
  }

  if (process.env.NODE_ENV === 'production' && !process.env.PINECONE_API_KEY && !process.env.QDRANT_URL) {
    log('  ⚠️  Production mode without vector DB - RAG features disabled', 'yellow');
    hasWarnings = true;
  }

  // Summary
  log('\n' + '='.repeat(50), 'cyan');
  if (hasErrors) {
    log('❌ CRITICAL ERRORS FOUND - Fix required variables before starting', 'red');
    process.exit(1);
  } else if (hasWarnings) {
    log('⚠️  WARNINGS FOUND - Some optional features may not work', 'yellow');
    log('✅ Core functionality should work', 'green');
    process.exit(0);
  } else {
    log('✅ All checks passed! Environment is properly configured.', 'green');
    process.exit(0);
  }
}

checkEnv();







