#!/usr/bin/env node
/**
 * Automated Environment Variable Setup
 * Creates .env file with all required variables
 * 
 * Usage: node scripts/setup-env.js
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function question(prompt) {
  return new Promise((resolve) => {
    rl.question(prompt, resolve);
  });
}

async function setupEnv() {
  console.log('🔐 Environment Variable Setup\n');
  console.log('This script will help you create a .env file with all required variables.\n');

  const env = {
    // Site Configuration
    PUBLIC_SITE_URL: await question('Site URL (e.g., https://curiouskelly.com): ') || 'https://curiouskelly.com',
    NODE_ENV: 'production',
    
    // Stripe Configuration
    STRIPE_SECRET_KEY: await question('Stripe Secret Key (sk_test_... or sk_live_...): ') || '',
    STRIPE_PUBLISHABLE_KEY: await question('Stripe Publishable Key (pk_test_... or pk_live_...): ') || '',
    STRIPE_WEBHOOK_SECRET: await question('Stripe Webhook Secret (whsec_...): ') || '',
    STRIPE_PRICE_MONTHLY: await question('Stripe Monthly Price ID (price_...): ') || '',
    STRIPE_PRICE_ANNUAL: await question('Stripe Annual Price ID (price_...): ') || '',
    STRIPE_PRICE_FAMILY: await question('Stripe Family Price ID (price_...): ') || '',
    STRIPE_PRICE_GIFT: await question('Stripe Gift Price ID (price_...): ') || '',
    
    // Email Configuration
    SENDGRID_API_KEY: await question('SendGrid API Key (sg....): ') || '',
    EMAIL_FROM: await question('Email From Address (hello@curiouskelly.com): ') || 'hello@curiouskelly.com',
    
    // OpenAI Configuration
    OPENAI_API_KEY: await question('OpenAI API Key (sk-...): ') || '',
    
    // ElevenLabs Configuration
    ELEVENLABS_API_KEY: await question('ElevenLabs API Key: ') || '',
    
    // Supabase Configuration (if using)
    SUPABASE_URL: await question('Supabase URL (https://xxx.supabase.co): ') || '',
    SUPABASE_ANON_KEY: await question('Supabase Anon Key: ') || '',
  };

  // Create .env file content
  let envContent = '# Environment Variables for Curious Kelly\n';
  envContent += '# Generated automatically - DO NOT COMMIT TO GIT\n\n';
  
  Object.entries(env).forEach(([key, value]) => {
    if (value) {
      envContent += `${key}=${value}\n`;
    } else {
      envContent += `# ${key}=\n`;
    }
  });

  // Write to .env file
  const envPath = path.join(process.cwd(), '.env');
  fs.writeFileSync(envPath, envContent);
  
  console.log('\n✅ .env file created at:', envPath);
  console.log('\n⚠️  IMPORTANT: Add .env to .gitignore if not already there!');
  console.log('\n📋 Next steps:');
  console.log('1. Review the .env file');
  console.log('2. Fill in any missing values');
  console.log('3. Test your configuration');
  
  rl.close();
}

setupEnv().catch(console.error);











