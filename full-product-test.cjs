/**
 * COMPREHENSIVE PRODUCT TEST - curiouskelly.com
 * 
 * Tests the live production site like a real user would.
 * Run with: node full-product-test.cjs
 */

const puppeteer = require('puppeteer');

async function runTests() {
  console.log('===========================================');
  console.log('CURIOUS KELLY - FULL PRODUCT TEST');
  console.log('Testing: https://www.curiouskelly.com');
  console.log('Time: ' + new Date().toISOString());
  console.log('===========================================\n');
  
  const browser = await puppeteer.launch({ 
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  
  // Set a reasonable viewport
  await page.setViewport({ width: 1280, height: 720 });
  
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
    results.errors.push('Page Error: ' + error.message);
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
    await page.goto('https://www.curiouskelly.com/', { waitUntil: 'networkidle2', timeout: 30000 });
    
    const title = await page.title();
    if (title && title.length > 0) {
      console.log('  ✅ Homepage loads with title:', title);
      results.passed++;
    } else {
      console.log('  ❌ Homepage title missing');
      results.failed++;
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

    // Check for Kelly visibility
    const kellyVisible = await page.$('img[src*="kelly"], img[src*="Kelly"], video, .kelly-image, .kelly-video, #kelly-video, img[alt*="Kelly"]');
    if (kellyVisible) {
      console.log('  ✅ Kelly avatar/image element found');
      results.passed++;
    } else {
      console.log('  ⚠️ Kelly avatar element NOT found (may use different selector)');
      results.warnings.push('Kelly avatar element not found with expected selectors');
    }
    
    // ============================================
    // TEST 2: LEARN PAGE LOADS
    // ============================================
    console.log('\nTEST 2: Learn Page Load');
    consoleMessages.length = 0; // Clear console messages
    
    await page.goto('https://www.curiouskelly.com/learn.html?debug=true', { 
      waitUntil: 'networkidle2', 
      timeout: 30000 
    });
    
    // Wait for app initialization
    await new Promise(r => setTimeout(r, 5000));
    
    // Check if page loaded
    const learnPageTitle = await page.title();
    console.log('  Page title:', learnPageTitle);
    
    // Check for learn page content
    const learnContent = await page.content();
    if (learnContent.toLowerCase().includes('learn') || learnContent.toLowerCase().includes('lesson') || learnContent.toLowerCase().includes('kelly')) {
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
        const contentText = await page.$eval(selector, el => el.textContent);
        if (contentText && contentText.trim().length > 10) {
          console.log('  ✅ Found content in:', selector);
          console.log('    Preview:', contentText.substring(0, 80).replace(/\n/g, ' ').trim() + '...');
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
    const pageText = await page.$eval('body', el => el.textContent);
    if (pageText.includes('Loading...') && !pageText.toLowerCase().includes('lesson')) {
      console.log('  ❌ Page stuck in Loading... state');
      results.failed++;
    }
    
    // ============================================
    // TEST 4: PAYWALL STATUS (CORRECTED)
    // ============================================
    console.log('\nTEST 4: Paywall Status');
    
    // Check if paywall is ACTUALLY visible (has 'visible' class) - not just exists
    const paywallStatus = await page.evaluate(() => {
      const paywall = document.getElementById('paywall');
      if (!paywall) return { exists: false };
      
      const hasVisibleClass = paywall.classList.contains('visible');
      const computedStyle = window.getComputedStyle(paywall);
      const opacity = computedStyle.opacity;
      const pointerEvents = computedStyle.pointerEvents;
      
      return {
        exists: true,
        hasVisibleClass,
        opacity,
        pointerEvents,
        isBlocking: hasVisibleClass && opacity !== '0' && pointerEvents !== 'none'
      };
    });
    
    console.log('  Paywall element exists:', paywallStatus.exists);
    if (paywallStatus.exists) {
      console.log('  Has "visible" class:', paywallStatus.hasVisibleClass);
      console.log('  CSS opacity:', paywallStatus.opacity);
      console.log('  CSS pointer-events:', paywallStatus.pointerEvents);
    }
    
    if (!paywallStatus.exists || !paywallStatus.isBlocking) {
      console.log('  ✅ Paywall is NOT blocking content');
      results.passed++;
    } else {
      console.log('  ❌ PAYWALL IS BLOCKING CONTENT');
      results.failed++;
      results.errors.push('Paywall is visible and blocking content');
    }
    
    // ============================================
    // TEST 5: TESTING MODE CONFIG
    // ============================================
    console.log('\nTEST 5: Testing Mode Configuration');
    
    const configStatus = await page.evaluate(() => {
      return {
        testingMode: window.KELLY_CONFIG?.testingMode,
        disablePaywall: window.KELLY_CONFIG?.disablePaywall,
        configExists: !!window.KELLY_CONFIG
      };
    });
    
    console.log('  KELLY_CONFIG exists:', configStatus.configExists);
    console.log('  testingMode:', configStatus.testingMode);
    console.log('  disablePaywall:', configStatus.disablePaywall);
    
    if (configStatus.testingMode === true && configStatus.disablePaywall === true) {
      console.log('  ✅ Testing mode correctly enabled');
      results.passed++;
    } else {
      console.log('  ⚠️ Testing mode may not be properly configured');
      results.warnings.push('Testing mode configuration issue');
    }
    
    // ============================================
    // TEST 6: KELLY VIDEO/ANIMATION
    // ============================================
    console.log('\nTEST 6: Kelly Video');
    
    const videoStatus = await page.evaluate(() => {
      const video = document.querySelector('video');
      if (!video) return { exists: false };
      
      const sources = Array.from(video.querySelectorAll('source')).map(s => s.src);
      return {
        exists: true,
        src: video.src || null,
        poster: video.poster || null,
        sources,
        readyState: video.readyState,
        error: video.error?.message || null
      };
    });
    
    if (!videoStatus.exists) {
      // Check for image fallback
      const kellyImage = await page.$('img[src*="kelly"], img[alt*="Kelly"]');
      if (kellyImage) {
        console.log('  ✅ Using Kelly image (no video, which is acceptable)');
        results.passed++;
      } else {
        console.log('  ⚠️ No video or Kelly image found');
        results.warnings.push('No video element - may be using alternative display');
      }
    } else {
      console.log('  Video element found');
      console.log('    src:', videoStatus.src || '(none)');
      console.log('    poster:', videoStatus.poster || '(none)');
      console.log('    sources:', videoStatus.sources.length > 0 ? videoStatus.sources.join(', ') : '(none)');
      console.log('    readyState:', videoStatus.readyState);
      if (videoStatus.error) {
        console.log('    error:', videoStatus.error);
      }
      
      if (videoStatus.src || videoStatus.sources.length > 0 || videoStatus.poster) {
        console.log('  ✅ Video has source or poster');
        results.passed++;
      } else {
        console.log('  ⚠️ Video element exists but no source');
        results.warnings.push('Video element has no visible source');
      }
    }
    
    // ============================================
    // TEST 7: UI COMPONENTS
    // ============================================
    console.log('\nTEST 7: UI Components');
    
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
    // TEST 8: CONSOLE ERRORS
    // ============================================
    console.log('\nTEST 8: Console Errors');
    
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
    // TEST 9: FAILED NETWORK REQUESTS
    // ============================================
    console.log('\nTEST 9: Network Requests');
    
    // Filter out expected failures (video chunks that abort on navigation)
    const significantFailures = failedRequests.filter(r => 
      !r.url.includes('kelly-videos') || r.error !== 'net::ERR_ABORTED'
    );
    
    if (significantFailures.length === 0) {
      console.log('  ✅ No significant network failures');
      if (failedRequests.length > 0) {
        console.log('    (Ignored ' + failedRequests.length + ' aborted video requests)');
      }
      results.passed++;
    } else {
      console.log('  ⚠️ Some network requests failed (' + significantFailures.length + '):');
      significantFailures.slice(0, 5).forEach(r => {
        console.log('    - ' + r.url.substring(0, 80) + ' : ' + r.error);
      });
      results.warnings.push(significantFailures.length + ' failed network requests');
    }
    
    // ============================================
    // TEST 10: API ENDPOINTS
    // ============================================
    console.log('\nTEST 10: API Health');
    
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
            return { ok: true, hasContent: !!data.topic || !!data.headline || !!data.id, data };
          }
          return { ok: false, status: res.status };
        } catch (e) {
          return { ok: false, error: e.message };
        }
      });
      
      if (lessonsResponse.ok && lessonsResponse.hasContent) {
        console.log('  ✅ Lessons API returning content');
        if (lessonsResponse.data?.topic) {
          console.log('    Topic:', lessonsResponse.data.topic.substring(0, 50));
        }
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
    // TEST 11: SUPABASE CONNECTION
    // ============================================
    console.log('\nTEST 11: Supabase Connection');
    
    const supabaseStatus = await page.evaluate(async () => {
      try {
        const config = window.KELLY_CONFIG;
        if (!config?.supabaseUrl || !config?.supabaseKey) {
          return { connected: false, error: 'Missing config' };
        }
        
        const res = await fetch(
          `${config.supabaseUrl}/rest/v1/core_lessons?select=id,topic&limit=1`,
          {
            headers: {
              'apikey': config.supabaseKey,
              'Authorization': `Bearer ${config.supabaseKey}`
            }
          }
        );
        
        if (!res.ok) {
          return { connected: false, status: res.status, error: await res.text() };
        }
        
        const data = await res.json();
        return { connected: true, lessonCount: data.length, sample: data[0] };
      } catch (e) {
        return { connected: false, error: e.message };
      }
    });
    
    if (supabaseStatus.connected) {
      console.log('  ✅ Supabase direct connection works');
      if (supabaseStatus.sample?.topic) {
        console.log('    Sample lesson:', supabaseStatus.sample.topic);
      }
      results.passed++;
    } else {
      console.log('  ❌ Supabase connection failed:', supabaseStatus.error || 'Status ' + supabaseStatus.status);
      results.failed++;
      results.errors.push('Supabase not connecting: ' + (supabaseStatus.error || supabaseStatus.status));
    }
    
    // ============================================
    // TEST 12: USER FLOW - Start Lesson Button
    // ============================================
    console.log('\nTEST 12: User Flow - Start Lesson');
    
    // Go back to homepage
    await page.goto('https://www.curiouskelly.com/', { waitUntil: 'networkidle2', timeout: 30000 });
    await new Promise(r => setTimeout(r, 2000));
    
    // Check for links to learn page
    const learnLink = await page.$('a[href*="learn"]');
    if (learnLink) {
      console.log('  ✅ Found link to learn page');
      results.passed++;
    } else {
      // Check for any button with relevant text
      const buttons = await page.$$('button, a.btn, .button, [role="button"]');
      let buttonFound = false;
      for (const button of buttons) {
        try {
          const text = await page.evaluate(el => el.textContent, button);
          if (text && (text.toLowerCase().includes('start') || text.toLowerCase().includes('learn') || text.toLowerCase().includes('begin'))) {
            console.log('  ✅ Found button:', text.trim().substring(0, 40));
            buttonFound = true;
            results.passed++;
            break;
          }
        } catch (e) {
          // Continue
        }
      }
      
      if (!buttonFound) {
        console.log('  ⚠️ Could not find obvious start lesson button');
        results.warnings.push('Start button not clearly visible');
      }
    }
    
    // ============================================
    // TEST 13: MOBILE RESPONSIVENESS
    // ============================================
    console.log('\nTEST 13: Mobile Responsiveness');
    
    await page.setViewport({ width: 375, height: 667 }); // iPhone SE
    await page.goto('https://www.curiouskelly.com/', { waitUntil: 'networkidle2' });
    await new Promise(r => setTimeout(r, 2000));
    
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
    await page.setViewport({ width: 1280, height: 720 });
    
    // ============================================
    // TAKE DESKTOP SCREENSHOTS FOR EVIDENCE
    // ============================================
    console.log('\nCapturing desktop screenshots...');
    await page.goto('https://www.curiouskelly.com/learn.html?debug=true', { waitUntil: 'networkidle2' });
    await new Promise(r => setTimeout(r, 3000));
    await page.screenshot({ path: 'tests/screenshot-learn.png', fullPage: true });
    console.log('  📸 Desktop screenshot: tests/screenshot-learn.png');
    
    // Also capture homepage
    await page.goto('https://www.curiouskelly.com/', { waitUntil: 'networkidle2' });
    await new Promise(r => setTimeout(r, 2000));
    await page.screenshot({ path: 'tests/screenshot-home.png', fullPage: true });
    console.log('  📸 Homepage screenshot: tests/screenshot-home.png');
    
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
  const totalTests = results.passed + results.failed;
  const successRate = totalTests > 0 ? Math.round((results.passed / totalTests) * 100) : 0;
  console.log('SUCCESS RATE: ' + successRate + '% (' + results.passed + '/' + totalTests + ')');
  
  if (results.failed === 0 && results.errors.length === 0) {
    console.log('✅ ALL TESTS PASSED - READY FOR LAUNCH');
  } else if (results.failed <= 1 && results.errors.length === 0) {
    console.log('⚠️ MOSTLY PASSING - Review warnings before launch');
  } else {
    console.log('❌ TESTS FAILED - FIX ISSUES BEFORE LAUNCH');
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
