/**
 * Browser Test Script for Curious Kelly
 * 
 * Tests lesson loading, popovers, and core functionality
 * 
 * Prerequisites: npm install puppeteer
 * Run: node scripts/browser_test.js
 */

import puppeteer from 'puppeteer';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const BASE_URL = 'http://localhost:8080';
const SCREENSHOT_DIR = resolve(__dirname, '../test-screenshots');

// Test results
const results = {
  passed: 0,
  failed: 0,
  tests: []
};

function logTest(name, passed, details = '') {
  const status = passed ? '✅ PASS' : '❌ FAIL';
  console.log(`${status}: ${name}`);
  if (details) console.log(`   ${details}`);
  
  results.tests.push({ name, passed, details });
  if (passed) results.passed++;
  else results.failed++;
}

async function test1_BasicLessonLoad(browser) {
  console.log('\n📝 TEST 1: Basic Lesson Load (Day 1)');
  console.log('=' .repeat(60));
  
  const page = await browser.newPage();
  
  // Listen for console errors
  const consoleErrors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  
  try {
    // Navigate to Day 1
    await page.goto(`${BASE_URL}/learn.html?day=1`, { 
      waitUntil: 'networkidle2',
      timeout: 10000 
    });
    
    // Wait for Kelly avatar (it's inside kelly-avatar-container)
    const avatarExists = await page.waitForSelector('#kelly-avatar-container', { timeout: 5000 }).catch(() => null);
    logTest('Kelly avatar container visible', !!avatarExists);
    
    // Check speech text
    const speechText = await page.$eval('#speech-text', el => el.textContent);
    const hasRealContent = !speechText.includes('being prepared') && 
                           !speechText.includes('Loading');
    logTest('Speech has real content', hasRealContent, `Text: "${speechText.substring(0, 50)}..."`);
    
    // Check for Supabase errors
    const hasSupabaseError = consoleErrors.some(err => 
      err.includes('ERR_NAME_NOT_RESOLVED') || 
      err.includes('Failed to fetch')
    );
    logTest('No Supabase connection errors', !hasSupabaseError);
    
    // Take screenshot
    await page.screenshot({ path: resolve(SCREENSHOT_DIR, 'test-day1.png') });
    console.log('   📸 Screenshot saved: test-day1.png');
    
  } catch (error) {
    logTest('Basic lesson load', false, error.message);
  } finally {
    await page.close();
  }
}

async function test2_MultipleDays(browser) {
  console.log('\n📝 TEST 2: Multiple Days Load');
  console.log('=' .repeat(60));
  
  const testDays = [1, 5, 10, 15, 20, 25, 30];
  
  for (const day of testDays) {
    const page = await browser.newPage();
    
    try {
      await page.goto(`${BASE_URL}/learn.html?day=${day}`, { 
        waitUntil: 'networkidle2',
        timeout: 8000 
      });
      
      // Wait for content to load
      await page.waitForSelector('#speech-text', { timeout: 3000 });
      
      // Get topic and speech text
      const topic = await page.$eval('#topic-text', el => el.textContent).catch(() => 'Unknown');
      const speechText = await page.$eval('#speech-text', el => el.textContent);
      
      const hasContent = speechText.length > 20 && 
                        !speechText.includes('being prepared');
      
      const status = hasContent ? '✅' : '❌';
      console.log(`   Day ${day} - Topic: ${topic} - ${status} ${hasContent ? 'Loaded' : 'Failed'}`);
      
      logTest(`Day ${day} loads`, hasContent);
      
    } catch (error) {
      console.log(`   Day ${day} - ❌ Failed: ${error.message}`);
      logTest(`Day ${day} loads`, false, error.message);
    } finally {
      await page.close();
    }
  }
}

async function test3_PopoverInteractions(browser) {
  console.log('\n📝 TEST 3: Popover Interactions');
  console.log('=' .repeat(60));
  
  const page = await browser.newPage();
  
  try {
    await page.goto(`${BASE_URL}/learn.html?day=1`, { 
      waitUntil: 'networkidle2',
      timeout: 10000 
    });
    
    // Wait for page to be ready
    await page.waitForSelector('#btn-language', { timeout: 5000 });
    
    // Click language button
    await page.click('#btn-language');
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Check if popover is visible
    const popoverVisible = await page.$eval('#popover-language', el => {
      return el.classList.contains('active');
    });
    logTest('Language popover opens', popoverVisible);
    
    // Click Español option
    await page.click('#popover-language [data-language="es"]');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Check badge changed
    const badgeText = await page.$eval('#badge-language', el => el.textContent);
    logTest('Language badge updates to ES', badgeText === 'ES', `Badge: "${badgeText}"`);
    
    // Take screenshot
    await page.screenshot({ path: resolve(SCREENSHOT_DIR, 'test-popover.png') });
    console.log('   📸 Screenshot saved: test-popover.png');
    
  } catch (error) {
    logTest('Popover interactions', false, error.message);
  } finally {
    await page.close();
  }
}

async function test4_ToneSwitching(browser) {
  console.log('\n📝 TEST 4: Tone Switching (Archetype Reload)');
  console.log('=' .repeat(60));
  
  const page = await browser.newPage();
  
  // Capture console logs
  const consoleLogs = [];
  page.on('console', msg => {
    if (msg.type() === 'log') {
      consoleLogs.push(msg.text());
    }
  });
  
  try {
    await page.goto(`${BASE_URL}/learn.html?day=1`, { 
      waitUntil: 'networkidle2',
      timeout: 10000 
    });
    
    // Wait for page ready
    await page.waitForSelector('#btn-tone', { timeout: 5000 });
    
    // Get initial speech text
    const initialText = await page.$eval('#speech-text', el => el.textContent);
    
    // Click tone button
    await page.click('#btn-tone');
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Click playful option
    await page.click('#popover-tone [data-tone="playful"]');
    
    // Wait for reload
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check if lesson reloaded
    const newText = await page.$eval('#speech-text', el => el.textContent);
    const textChanged = newText !== initialText;
    
    // Check console for archetype log
    const hasArchetypeLog = consoleLogs.some(log => 
      log.includes('archetype') || log.includes('Jester') || log.includes('playful')
    );
    
    logTest('Tone switch triggers reload', textChanged || hasArchetypeLog);
    logTest('Archetype logged in console', hasArchetypeLog, 
      hasArchetypeLog ? 'Found archetype reference' : 'No archetype log found');
    
  } catch (error) {
    logTest('Tone switching', false, error.message);
  } finally {
    await page.close();
  }
}

async function test5_AllPopovers(browser) {
  console.log('\n📝 TEST 5: All Popovers Work');
  console.log('=' .repeat(60));
  
  const page = await browser.newPage();
  
  try {
    await page.goto(`${BASE_URL}/learn.html?day=1`, { 
      waitUntil: 'networkidle2',
      timeout: 10000 
    });
    
    const popovers = [
      { btn: '#btn-age', popover: '#popover-age', name: 'Age' },
      { btn: '#btn-language', popover: '#popover-language', name: 'Language' },
      { btn: '#btn-tone', popover: '#popover-tone', name: 'Tone' },
      { btn: '#btn-difficulty', popover: '#popover-difficulty', name: 'Difficulty' }
    ];
    
    for (const { btn, popover, name } of popovers) {
      try {
        // Click button
        await page.click(btn);
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Check if popover is visible
        const isVisible = await page.$eval(popover, el => {
          return el.classList.contains('active');
        });
        
        logTest(`${name} popover opens`, isVisible);
        
        // Close by clicking overlay
        await page.click('#popover-overlay');
        await new Promise(resolve => setTimeout(resolve, 200));
        
      } catch (error) {
        logTest(`${name} popover opens`, false, error.message);
      }
    }
    
  } catch (error) {
    logTest('All popovers test', false, error.message);
  } finally {
    await page.close();
  }
}

async function test6_SettingsPage(browser) {
  console.log('\n📝 TEST 6: Settings Page');
  console.log('=' .repeat(60));
  
  const page = await browser.newPage();
  
  try {
    await page.goto(`${BASE_URL}/settings.html`, { 
      waitUntil: 'networkidle2',
      timeout: 8000 
    });
    
    // Check if page loaded
    const title = await page.$eval('.settings-title', el => el.textContent);
    logTest('Settings page loads', title === 'Settings', `Title: "${title}"`);
    
    // Check for settings sections
    const hasSections = await page.$$eval('.settings-section', sections => sections.length >= 3);
    logTest('Settings sections present', hasSections);
    
  } catch (error) {
    logTest('Settings page', false, error.message);
  } finally {
    await page.close();
  }
}

async function runAllTests() {
  console.log('🧪 CURIOUS KELLY BROWSER TESTS');
  console.log('=' .repeat(60));
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`Screenshots: ${SCREENSHOT_DIR}`);
  console.log('=' .repeat(60));
  
  // Create screenshot directory
  const fs = await import('fs');
  if (!fs.existsSync(SCREENSHOT_DIR)) {
    fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  }
  
  let browser;
  
  try {
    // Launch browser
    console.log('\n🚀 Launching browser...');
    browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    console.log('✅ Browser launched');
    
    // Run tests
    await test1_BasicLessonLoad(browser);
    await test2_MultipleDays(browser);
    await test3_PopoverInteractions(browser);
    await test4_ToneSwitching(browser);
    await test5_AllPopovers(browser);
    await test6_SettingsPage(browser);
    
  } catch (error) {
    console.error('\n❌ Test suite failed:', error);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
  
  // Print summary
  console.log('\n' + '=' .repeat(60));
  console.log('📊 TEST SUMMARY');
  console.log('=' .repeat(60));
  console.log(`Total Tests: ${results.passed + results.failed}`);
  console.log(`✅ Passed: ${results.passed}`);
  console.log(`❌ Failed: ${results.failed}`);
  console.log(`Success Rate: ${Math.round(results.passed / (results.passed + results.failed) * 100)}%`);
  console.log('=' .repeat(60));
  
  // Exit with error code if tests failed
  process.exit(results.failed > 0 ? 1 : 0);
}

// Run tests
runAllTests().catch(console.error);

