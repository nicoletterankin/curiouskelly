#!/usr/bin/env node
/**
 * Automated Progress Checker
 * Checks completion status of critical tasks
 * 
 * Usage: node scripts/check-progress.js
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const dns = require('dns').promises;

const tasks = {
  domain: {
    name: 'Domain Setup',
    check: async () => {
      try {
        await dns.resolve4('curiouskelly.com');
        return { status: 'complete', message: 'Domain resolves ✅' };
      } catch (err) {
        return { status: 'pending', message: 'Domain not resolving yet ⏳' };
      }
    }
  },
  email: {
    name: 'Email Setup',
    check: async () => {
      // Check if .env has email config
      const envPath = path.join(process.cwd(), '.env');
      if (fs.existsSync(envPath)) {
        const envContent = fs.readFileSync(envPath, 'utf8');
        if (envContent.includes('EMAIL_FROM') || envContent.includes('SENDGRID')) {
          return { status: 'complete', message: 'Email config found ✅' };
        }
      }
      return { status: 'pending', message: 'Email not configured ⏳' };
    }
  },
  stripe: {
    name: 'Stripe Setup',
    check: async () => {
      const envPath = path.join(process.cwd(), '.env');
      if (fs.existsSync(envPath)) {
        const envContent = fs.readFileSync(envPath, 'utf8');
        if (envContent.includes('STRIPE_SECRET_KEY') && envContent.includes('STRIPE_PUBLISHABLE_KEY')) {
          return { status: 'complete', message: 'Stripe keys found ✅' };
        }
      }
      return { status: 'pending', message: 'Stripe not configured ⏳' };
    }
  },
  landingPage: {
    name: 'Landing Page',
    check: async () => {
      const landingPagePath = path.join(process.cwd(), 'public', 'index.html');
      if (fs.existsSync(landingPagePath)) {
        return { status: 'complete', message: 'Landing page exists ✅' };
      }
      return { status: 'pending', message: 'Landing page not found ⏳' };
    }
  },
  envFile: {
    name: 'Environment File',
    check: async () => {
      const envPath = path.join(process.cwd(), '.env');
      if (fs.existsSync(envPath)) {
        return { status: 'complete', message: '.env file exists ✅' };
      }
      return { status: 'pending', message: '.env file not found ⏳' };
    }
  }
};

async function checkProgress() {
  console.log('📊 Progress Check\n');
  console.log('Checking critical tasks...\n');

  const results = [];
  
  for (const [key, task] of Object.entries(tasks)) {
    try {
      const result = await task.check();
      results.push({ ...result, name: task.name });
      const icon = result.status === 'complete' ? '✅' : '⏳';
      console.log(`${icon} ${task.name}: ${result.message}`);
    } catch (err) {
      console.log(`❌ ${task.name}: Error checking - ${err.message}`);
      results.push({ status: 'error', name: task.name, message: err.message });
    }
  }

  const completed = results.filter(r => r.status === 'complete').length;
  const total = results.length;
  const percentage = Math.round((completed / total) * 100);

  console.log(`\n📈 Progress: ${completed}/${total} tasks complete (${percentage}%)`);

  if (percentage === 100) {
    console.log('\n🎉 All critical tasks complete! Ready for launch!');
  } else {
    console.log('\n📋 Next steps:');
    results
      .filter(r => r.status !== 'complete')
      .forEach(r => {
        console.log(`   - ${r.name}: ${r.message}`);
      });
  }
}

checkProgress().catch(console.error);

