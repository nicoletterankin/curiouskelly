/**
 * COMPREHENSIVE PRODUCT TEST - curiouskelly.com
 * 
 * Tests the live production site like a real user would.
 * Run with: node tests/full-product-test.js
 */

import { chromium } from 'playwright';

async function runTests() {
  console.log('===========================================');
  console.log('CURIOUS KELLY - FULL PRODUCT TEST');
  console.log('Testing: https://www.curiouskelly.com');
  console.log('Time: ' + new Date().toISOString());
  console.log('===========================================\n');
  
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
  });
  const page = await context.newPage();
  
  const results = {
    passed: 0,
    failed: 0,
    errors: [],
    warnings: []
  };
  
  // Capture console messages
  const consoleMessages = [];
  page.on('console', msg => {
    consoleMessages.push({ type: msg.type(), text: msg.text() });
  });
  
  // Capture errors
  page.on('pageerror', error => {
    results.errors.push(`Page Error: ${error.message}`);
  });

  // Capture failed network requests
  const failedRequests = [];
  page.on('requestfailed', request => {
    failedRequests.push({
      url: request.url(),
      error: request.failure()?.errorText || 'Unknown error'
    });
  });

  try {
    // ============================================
    // TEST 1: HOMEPAGE LOADS
    // ============================================
    console.log('TEST 1: Homepage Load');
    await page.goto('https://www.curiouskelly.com/', { waitUntil: 'networkidle', timeout: 30000 });
    
    const title = await page.title();
    if (title && title.length > 0) {
      console.log('  ✅ Homepage loads with title:', title);
      results.passed++;
    } else {
      console.log('  ❌ Homepage title missing');
      results.failed++;
    }
    
    // Check for Kelly visibility (looking for various selectors)
    const kellyVisible = await page.$('img[src*="kelly"], img[src*="Kelly"], video, .kelly-image, .kelly-video, #kelly-video, img[alt*="Kelly"]');
    if (kellyVisible) {
      console.log('  ✅ Kelly avatar/image element found');
      results.passed++;
    } else {
      console.log('  ⚠️ Kelly avatar element NOT found (may use different selector)');
      results.warnings.push('Kelly avatar element not found with expected selectors');
    }

    // Check page has actual content
    const pageContent = await page.content();
    if (pageContent.length > 5000) {
      console.log('  ✅ Homepage has substantial content (' + pageContent.length + ' chars)');
      results.passed++;
    } else {
      console.log('  ❌ Homepage content too small');
      results.failed++;
    }
    
    // ============================================
    // TEST 2: LEARN PAGE LOADS
    // ============================================
    console.log('\nTEST 2: Learn Page Load');
    consoleMessages.length = 0; // Clear console messages
    
    await page.goto('https://www.curiouskelly.com/learn.html?debug=true', { 
      waitUntil: 'networkidle', 
      timeout: 30000 
    });
    
    // Wait for app initialization
    await page.waitForTimeout(5000);
    
    // Check if page loaded
    const learnPageTitle = await page.title();
    console.log('  Page title:', learnPageTitle);
    
    // Check for learn page content
    const learnContent = await page.content();
    if (learnContent.includes('learn') || learnContent.includes('lesson') || learnContent.includes('Kelly')) {
      console.log('  ✅ Learn page has relevant content');
      results.passed++;
    } else {
      console.log('  ❌ Learn page missing expected content');
      results.failed++;
    }
    
    // Check console for Supabase status
    const emergencyFallback = consoleMessages.some(m => 
      m.text.toLowerCase().includes('emergency') || 
      m.text.toLowerCase().includes('fallback') ||
      m.text.includes('Using local')
    );
    
    if (emergencyFallback) {
      console.log('  ⚠️ May be using fallback/emergency content');
      results.warnings.push('Possible emergency fallback active - check console');
    } else {
      console.log('  ✅ No emergency fallback messages detected');
      results.passed++;
    }
    
    // Log console messages for debugging
    const relevantConsoleMessages = consoleMessages.filter(m => 
      m.text.toLowerCase().includes('supabase') ||
      m.text.toLowerCase().includes('error') ||
      m.text.toLowerCase().includes('lesson') ||
      m.text.toLowerCase().includes('init')
    );
    
    if (relevantConsoleMessages.length > 0) {
      console.log('  Console messages:');
      relevantConsoleMessages.slice(0, 5).forEach(m => {
        console.log('    [' + m.type + '] ' + m.text.substring(0, 100));
      });
    }
    
    // ============================================
    // TEST 3: LESSON CONTENT LOADS (NOT "Loading...")
    // ============================================
    console.log('\nTEST 3: Lesson Content');
    
    // Look for content elements with various selectors
    const contentSelectors = [
      '.caption-text',
      '.lesson-content', 
      '#caption-text',
      '.lesson-text',
      '[class*="caption"]',
      '[class*="lesson"]',
      '.content',
      'main'
    ];
    
    let hasContent = null;
    for (const selector of contentSelectors) {
      hasContent = await page.$(selector);
      if (hasContent) {
        const contentText = await hasContent.textContent();
        if (contentText && contentText.trim().length > 10) {
          console.log('  ✅ Found content in:', selector);
          console.log('    Preview:', contentText.substring(0, 80).replace(/\n/g, ' ') + '...');
          results.passed++;
          break;
        }
      }
    }
    
    if (!hasContent) {
      console.log('  ⚠️ Could not find lesson content container');
      results.warnings.push('Lesson content container not found');
    }
    
    // Check for loading state
    const pageText = await page.textContent('body');
    if (pageText.includes('Loading...') && !pageText.includes('lesson')) {
      console.log('  ❌ Page stuck in Loading... state');
      results.failed++;
    }
    
    // ============================================
    // TEST 4: NO PAYWALL BLOCKING (testing mode)
    // ============================================
    console.log('\nTEST 4: Paywall Status');
    
    const paywallSelectors = ['.paywall', '#paywall', '[class*="paywall"]', '.subscription-wall', '.payment-wall'];
    let paywallBlocking = false;
    
    for (const selector of paywallSelectors) {
      const paywall = await page.$(selector);
      if (paywall) {
        const isVisible = await paywall.isVisible();
        if (isVisible) {
          console.log('  ❌ PAYWALL IS VISIBLE at:', selector);
          results.failed++;
          results.errors.push('Paywall blocking content');
          paywallBlocking = true;
          break;
        }
      }
    }
    
    if (!paywallBlocking) {
      console.log('  ✅ No paywall blocking content');
      results.passed++;
    }
    
    // ============================================
    // TEST 5: KELLY VIDEO/ANIMATION
    // ============================================
    console.log('\nTEST 5: Kelly Video');
    
    const video = await page.$('video');
    if (video) {
      const videoSrc = await video.getAttribute('src');
      const posterSrc = await video.getAttribute('poster');
      if (videoSrc && videoSrc.length > 0) {
        console.log('  ✅ Video element has source:', videoSrc.substring(0, 60) + '...');
        results.passed++;
      } else if (posterSrc) {
        console.log('  ✅ Video element has poster (video may lazy load)');
        results.passed++;
      } else {
        // Check for source children
        const sourceElements = await page.$$('video source');
        if (sourceElements.length > 0) {
          console.log('  ✅ Video has source elements');
          results.passed++;
        } else {
          console.log('  ⚠️ Video element exists but no source detected');
          results.warnings.push('Video element has no visible source');
        }
      }
    } else {
      // Check for image fallback
      const kellyImage = await page.$('img[src*="kelly"], img[alt*="Kelly"]');
      if (kellyImage) {
        console.log('  ✅ Using Kelly image (no video, which is acceptable)');
        results.passed++;
      } else {
        console.log('  ⚠️ No video or Kelly image found');
        results.warnings.push('No video element - may be using alternative display');
      }
    }
    
    // ============================================
    // TEST 6: UI COMPONENTS
    // ============================================
    console.log('\nTEST 6: UI Components');
    
    // Check for phase indicators
    const phaseIndicators = await page.$('[class*="phase"], [class*="dot"], [class*="progress"], .phase-bar, .phase-dots, #phase-dots');
    if (phaseIndicators) {
      console.log('  ✅ Phase/progress indicators found');
      results.passed++;
    } else {
      console.log('  ⚠️ Phase indicators not found');
      results.warnings.push('Phase indicators not visible');
    }
    
    // Check for navigation elements
    const navigation = await page.$('nav, [class*="nav"], .bottom-nav, #bottom-nav, footer');
    if (navigation) {
      console.log('  ✅ Navigation elements found');
      results.passed++;
    } else {
      console.log('  ⚠️ Navigation not found');
      results.warnings.push('Navigation elements not visible');
    }
    
    // ============================================
    // TEST 7: CONSOLE ERRORS
    // ============================================
    console.log('\nTEST 7: Console Errors');
    
    const jsErrors = consoleMessages.filter(m => m.type === 'error');
    if (jsErrors.length === 0) {
      console.log('  ✅ No JavaScript errors in console');
      results.passed++;
    } else {
      console.log('  ❌ JavaScript errors found (' + jsErrors.length + '):');
      jsErrors.slice(0, 5).forEach(e => console.log('    - ' + e.text.substring(0, 100)));
      results.failed++;
      results.errors.push(jsErrors.length + ' JS errors in console');
    }
    
    // ============================================
    // TEST 8: FAILED NETWORK REQUESTS
    // ============================================
    console.log('\nTEST 8: Network Requests');
    
    if (failedRequests.length === 0) {
      console.log('  ✅ No failed network requests');
      results.passed++;
    } else {
      console.log('  ⚠️ Some network requests failed (' + failedRequests.length + '):');
      failedRequests.slice(0, 5).forEach(r => {
        console.log('    - ' + r.url.substring(0, 80) + ' : ' + r.error);
      });
      results.warnings.push(failedRequests.length + ' failed network requests');
    }
    
    // ============================================
    // TEST 9: API ENDPOINTS
    // ============================================
    console.log('\nTEST 9: API Health');
    
    // Test motion-progress API
    try {
      const apiResponse = await page.evaluate(async () => {
        try {
          const res = await fetch('/api/motion-progress');
          if (!res.ok) return { ok: false, status: res.status };
          const data = await res.json();
          return { ok: true, data };
        } catch (e) {
          return { ok: false, error: e.message };
        }
      });
      
      if (apiResponse.ok) {
        console.log('  ✅ Motion API responding:', JSON.stringify(apiResponse.data?.stats || apiResponse.data).substring(0, 60));
        results.passed++;
      } else {
        console.log('  ⚠️ Motion API issue:', apiResponse.error || 'Status ' + apiResponse.status);
        results.warnings.push('Motion API not responding as expected');
      }
    } catch (e) {
      console.log('  ⚠️ Could not test Motion API:', e.message);
      results.warnings.push('Motion API test failed');
    }
    
    // Test lessons API
    try {
      const lessonsResponse = await page.evaluate(async () => {
        try {
          const res = await fetch('/api/lessons/1');
          if (res.ok) {
            const data = await res.json();
            return { ok: true, hasContent: !!data.topic || !!data.headline || !!data.id };
          }
          return { ok: false, status: res.status };
        } catch (e) {
          return { ok: false, error: e.message };
        }
      });
      
      if (lessonsResponse.ok && lessonsResponse.hasContent) {
        console.log('  ✅ Lessons API returning content');
        results.passed++;
      } else {
        console.log('  ⚠️ Lessons API issue:', lessonsResponse.error || 'Status ' + lessonsResponse.status);
        results.warnings.push('Lessons API not returning expected content');
      }
    } catch (e) {
      console.log('  ⚠️ Could not test Lessons API:', e.message);
      results.warnings.push('Lessons API test failed');
    }
    
    // ============================================
    // TEST 10: USER FLOW - Start Lesson Button
    // ============================================
    console.log('\nTEST 10: User Flow - Start Lesson');
    
    // Go back to homepage
    await page.goto('https://www.curiouskelly.com/', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2000);
    
    // Find clickable elements that might start a lesson
    const possibleButtons = [
      'text="Start today\'s lesson"',
      'text="Start lesson"',
      'text="Start"',
      'text="Begin"',
      'text="Learn"',
      '[class*="start"]',
      '[class*="cta"]',
      'button',
      'a[href*="learn"]'
    ];
    
    let buttonFound = false;
    for (const selector of possibleButtons) {
      try {
        const button = await page.$(selector);
        if (button) {
          const isVisible = await button.isVisible();
          if (isVisible) {
            const buttonText = await button.textContent();
            console.log('  ✅ Found button:', (buttonText || selector).substring(0, 50));
            buttonFound = true;
            
            // Try clicking it
            await button.click();
            await page.waitForTimeout(3000);
            
            const currentUrl = page.url();
            if (currentUrl.includes('learn')) {
              console.log('  ✅ Button navigates to learn page');
              results.passed++;
            } else {
              console.log('  ⚠️ Button clicked but URL is:', currentUrl);
            }
            break;
          }
        }
      } catch (e) {
        // Continue to next selector
      }
    }
    
    if (!buttonFound) {
      console.log('  ⚠️ Could not find start lesson button');
      results.warnings.push('Start button not found or not visible');
    }
    
    // ============================================
    // TEST 11: MOBILE RESPONSIVENESS
    // ============================================
    console.log('\nTEST 11: Mobile Responsiveness');
    
    await page.setViewportSize({ width: 375, height: 667 }); // iPhone SE
    await page.goto('https://www.curiouskelly.com/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    
    const mobileContent = await page.content();
    if (mobileContent.length > 3000) {
      console.log('  ✅ Mobile viewport renders content');
      results.passed++;
    } else {
      console.log('  ❌ Mobile viewport has issues');
      results.failed++;
    }
    
    // Take mobile screenshot
    await page.screenshot({ path: 'tests/screenshot-mobile.png', fullPage: true });
    console.log('  📸 Mobile screenshot: tests/screenshot-mobile.png');
    
    // Reset viewport
    await page.setViewportSize({ width: 1280, height: 720 });
    
    // ============================================
    // TAKE DESKTOP SCREENSHOT FOR EVIDENCE
    // ============================================
    console.log('\nCapturing desktop screenshot...');
    await page.goto('https://www.curiouskelly.com/learn.html?debug=true', { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);
    await page.screenshot({ path: 'tests/screenshot-learn.png', fullPage: true });
    console.log('  📸 Desktop screenshot: tests/screenshot-learn.png');
    
  } catch (error) {
    console.log('\n❌ TEST CRASHED:', error.message);
    results.errors.push('Test crash: ' + error.message);
  } finally {
    await browser.close();
  }
  
  // ============================================
  // FINAL REPORT
  // ============================================
  console.log('\n===========================================');
  console.log('TEST RESULTS SUMMARY');
  console.log('===========================================');
  console.log('✅ Passed: ' + results.passed);
  console.log('❌ Failed: ' + results.failed);
  console.log('⚠️ Warnings: ' + results.warnings.length);
  
  if (results.errors.length > 0) {
    console.log('\n🚨 CRITICAL ERRORS:');
    results.errors.forEach(e => console.log('  - ' + e));
  }
  
  if (results.warnings.length > 0) {
    console.log('\n⚠️ WARNINGS (may need attention):');
    results.warnings.forEach(w => console.log('  - ' + w));
  }
  
  console.log('\n===========================================');
  const successRate = Math.round((results.passed / (results.passed + results.failed)) * 100);
  console.log('SUCCESS RATE: ' + successRate + '%');
  
  if (results.failed === 0 && results.errors.length === 0) {
    console.log('✅ ALL TESTS PASSED - READY FOR LAUNCH');
  } else if (results.failed <= 2 && results.errors.length === 0) {
    console.log('⚠️ MOSTLY PASSING - Review warnings before launch');
  } else {
    console.log('❌ TESTS FAILED - DO NOT LAUNCH');
    console.log('Fix the critical issues above before proceeding.');
  }
  console.log('===========================================\n');
  
  // Return results for programmatic access
  return results;
}

// Run if executed directly
runTests()
  .then(results => {
    process.exit(results.failed > 0 ? 1 : 0);
  })
  .catch(err => {
    console.error('Test runner failed:', err);
    process.exit(1);
  });
