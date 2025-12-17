/**
 * UI Test Script - Puppeteer
 * Simulates a learner clicking through the app and captures screenshots
 * to identify cut-off text, missing elements, and UI issues
 */

import puppeteer from 'puppeteer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SCREENSHOT_DIR = path.join(__dirname, '..', 'test-screenshots');
const BASE_URL = 'http://localhost:3000';

async function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  // Create screenshot directory
  if (!fs.existsSync(SCREENSHOT_DIR)) {
    fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  }

  console.log('\n🎭 Starting Puppeteer UI Test...\n');

  const browser = await puppeteer.launch({
    headless: false, // Show browser so user can watch
    defaultViewport: { width: 1280, height: 800 },
    args: ['--start-maximized']
  });

  const page = await browser.newPage();
  
  // Set viewport to common desktop size
  await page.setViewport({ width: 1280, height: 800 });

  try {
    // ========================================
    // TEST 1: Home/Landing Page
    // ========================================
    console.log('📸 Test 1: Loading Home page...');
    await page.goto(`${BASE_URL}/learn.html?testing=true&bypass=testing`, { 
      waitUntil: 'networkidle2',
      timeout: 30000 
    });
    await delay(2000);
    await page.screenshot({ 
      path: path.join(SCREENSHOT_DIR, '01-home-initial.png'),
      fullPage: false 
    });
    console.log('   ✅ Screenshot: 01-home-initial.png');

    // ========================================
    // TEST 2: Click Play to start lesson
    // ========================================
    console.log('📸 Test 2: Starting lesson (clicking Play)...');
    
    // Try to find and click the play button
    const playBtn = await page.$('#nav-play-btn, .play-btn, [data-action="play"]');
    if (playBtn) {
      await playBtn.click();
      await delay(2000);
    }
    
    await page.screenshot({ 
      path: path.join(SCREENSHOT_DIR, '02-lesson-started.png'),
      fullPage: false 
    });
    console.log('   ✅ Screenshot: 02-lesson-started.png');

    // ========================================
    // TEST 3: Check Phase Bar visibility
    // ========================================
    console.log('📸 Test 3: Checking phase bar and dots...');
    
    const phaseBar = await page.$('#phase-bar');
    if (phaseBar) {
      const phaseBounds = await phaseBar.boundingBox();
      console.log(`   Phase bar position: ${JSON.stringify(phaseBounds)}`);
      
      if (phaseBounds) {
        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, '03-phase-bar-closeup.png'),
          clip: {
            x: Math.max(0, phaseBounds.x - 20),
            y: Math.max(0, phaseBounds.y - 20),
            width: phaseBounds.width + 40,
            height: phaseBounds.height + 40
          }
        });
        console.log('   ✅ Screenshot: 03-phase-bar-closeup.png');
      }
    } else {
      console.log('   ⚠️ Phase bar not found!');
    }

    // ========================================
    // TEST 4: Check for choice options
    // ========================================
    console.log('📸 Test 4: Looking for choice options panel...');
    
    // Wait longer for the phase to complete and choices to appear
    console.log('   Waiting 8s for lesson to progress to choice prompt...');
    await delay(8000);
    
    // Check if cliff-container is visible
    const cliffVisible = await page.evaluate(() => {
      const el = document.getElementById('cliff-container');
      if (!el) return { exists: false };
      const style = window.getComputedStyle(el);
      return { 
        exists: true, 
        hidden: el.hidden,
        display: style.display,
        visibility: style.visibility,
        classList: Array.from(el.classList)
      };
    });
    console.log(`   Cliff container state: ${JSON.stringify(cliffVisible)}`);
    
    const choicePanel = await page.$('#cliff-container:not(.hidden):not([hidden])');
    if (choicePanel) {
      const choiceBounds = await choicePanel.boundingBox();
      console.log(`   Choice panel found: ${JSON.stringify(choiceBounds)}`);
      
      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, '04-choice-panel.png'),
        fullPage: false
      });
      console.log('   ✅ Screenshot: 04-choice-panel.png');
    } else {
      console.log('   ⚠️ Choice panel not visible (may need to advance phases)');
      
      // Try clicking next to advance to a phase with choices
      console.log('   Trying to advance phases...');
      const nextBtn = await page.$('#phase-next, [data-action="next"]');
      if (nextBtn) {
        await nextBtn.click();
        await delay(2000);
        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, '04a-after-next.png'),
          fullPage: false
        });
        console.log('   ✅ Screenshot: 04a-after-next.png');
      }
    }

    // ========================================
    // TEST 5: Full page screenshot to see cut-offs
    // ========================================
    console.log('📸 Test 5: Full lesson view...');
    await page.screenshot({ 
      path: path.join(SCREENSHOT_DIR, '05-lesson-full.png'),
      fullPage: true 
    });
    console.log('   ✅ Screenshot: 05-lesson-full.png');

    // ========================================
    // TEST 6: Navigate to Journey
    // ========================================
    console.log('📸 Test 6: Opening Journey panel...');
    
    const journeyBtn = await page.$('#nav-journey-btn, [data-nav="journey"]');
    if (journeyBtn) {
      await journeyBtn.click();
      await delay(1500);
    }
    
    await page.screenshot({ 
      path: path.join(SCREENSHOT_DIR, '06-journey-panel.png'),
      fullPage: false 
    });
    console.log('   ✅ Screenshot: 06-journey-panel.png');

    // ========================================
    // TEST 7: Navigate to Settings
    // ========================================
    console.log('📸 Test 7: Opening Settings panel...');
    
    const settingsBtn = await page.$('#nav-settings-btn, [data-nav="settings"]');
    if (settingsBtn) {
      await settingsBtn.click();
      await delay(1500);
    }
    
    await page.screenshot({ 
      path: path.join(SCREENSHOT_DIR, '07-settings-panel.png'),
      fullPage: false 
    });
    console.log('   ✅ Screenshot: 07-settings-panel.png');

    // ========================================
    // TEST 8: Check bottom navigation visibility
    // ========================================
    console.log('📸 Test 8: Bottom navigation check...');
    
    const bottomNav = await page.$('.bottom-nav, #bottom-nav, nav');
    if (bottomNav) {
      const navBounds = await bottomNav.boundingBox();
      console.log(`   Bottom nav position: ${JSON.stringify(navBounds)}`);
      
      // Check if it's cut off
      const viewport = page.viewport();
      if (navBounds && navBounds.y + navBounds.height > viewport.height) {
        console.log('   ⚠️ ISSUE: Bottom nav is cut off!');
      }
    }

    // ========================================
    // TEST 9: Mobile viewport test
    // ========================================
    console.log('📸 Test 9: Testing mobile viewport (375x667)...');
    await page.setViewport({ width: 375, height: 667 });
    await delay(1000);
    
    // Go back to lesson
    await page.goto(`${BASE_URL}/learn.html?testing=true&bypass=testing`, { 
      waitUntil: 'networkidle2' 
    });
    await delay(2000);
    
    await page.screenshot({ 
      path: path.join(SCREENSHOT_DIR, '08-mobile-view.png'),
      fullPage: false 
    });
    console.log('   ✅ Screenshot: 08-mobile-view.png');

    // ========================================
    // TEST 10: Check text overflow/clipping
    // ========================================
    console.log('📸 Test 10: Checking for text clipping...');
    
    // Find all text elements and check for overflow
    const textIssues = await page.evaluate(() => {
      const issues = [];
      const textElements = document.querySelectorAll('p, span, h1, h2, h3, h4, .caption, .title, .label');
      
      textElements.forEach(el => {
        const styles = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        
        // Check if text is clipped
        if (el.scrollWidth > el.clientWidth || el.scrollHeight > el.clientHeight) {
          issues.push({
            tag: el.tagName,
            class: el.className,
            text: el.textContent?.slice(0, 50),
            scrollWidth: el.scrollWidth,
            clientWidth: el.clientWidth,
            overflow: styles.overflow
          });
        }
        
        // Check if element is outside viewport
        if (rect.right > window.innerWidth || rect.bottom > window.innerHeight) {
          issues.push({
            tag: el.tagName,
            class: el.className,
            text: el.textContent?.slice(0, 50),
            issue: 'outside viewport',
            rect: { top: rect.top, right: rect.right, bottom: rect.bottom }
          });
        }
      });
      
      return issues;
    });
    
    if (textIssues.length > 0) {
      console.log('   ⚠️ Found text clipping issues:');
      textIssues.slice(0, 5).forEach(issue => {
        console.log(`      - ${issue.tag}.${issue.class}: "${issue.text}..."`);
      });
    } else {
      console.log('   ✅ No obvious text clipping detected');
    }

    // ========================================
    // SUMMARY
    // ========================================
    console.log('\n' + '='.repeat(50));
    console.log('📊 UI TEST COMPLETE');
    console.log('='.repeat(50));
    console.log(`Screenshots saved to: ${SCREENSHOT_DIR}`);
    console.log('\nReview the screenshots to identify:');
    console.log('  - Cut-off text');
    console.log('  - Missing phase dots');
    console.log('  - Hidden choice options');
    console.log('  - Navigation issues');
    console.log('='.repeat(50) + '\n');

    // Keep browser open for manual inspection
    console.log('Browser left open for manual inspection. Press Ctrl+C to close.');
    
    // Wait indefinitely (user can inspect)
    await new Promise(() => {});
    
  } catch (error) {
    console.error('❌ Error during test:', error);
    await page.screenshot({ 
      path: path.join(SCREENSHOT_DIR, 'error-state.png'),
      fullPage: true 
    });
  }
}

main().catch(console.error);
