/**
 * Hybrid Compositor System - Puppeteer Test Suite
 * 
 * Tests the actual deployed site to verify:
 * - PixiJS compositor initialization
 * - Mouth overlay rendering
 * - Expression system
 * - iOS autoplay handling
 * - Performance optimizations
 * 
 * Run: node tests/hybrid-compositor-test.js
 */

import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';

const TEST_URL = process.env.TEST_URL || 'https://curiouskelly.com/learn.html?talkingPhoto=1&pixiDebug=1&day=1';
const TIMEOUT = 30000;

async function testHybridCompositor() {
  const browser = await puppeteer.launch({
    headless: false, // Show browser for debugging
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  const logs = [];
  const errors = [];
  const screenshots = [];

  // Capture console logs
  page.on('console', msg => {
    const text = msg.text();
    logs.push({
      type: msg.type(),
      text: text,
      timestamp: Date.now()
    });
    console.log(`[${msg.type()}] ${text}`);
  });

  // Capture errors
  page.on('pageerror', error => {
    errors.push({
      message: error.message,
      stack: error.stack,
      timestamp: Date.now()
    });
    console.error('Page error:', error.message);
  });

  // Capture network failures
  page.on('requestfailed', request => {
    errors.push({
      type: 'network',
      url: request.url(),
      failureText: request.failure()?.errorText,
      timestamp: Date.now()
    });
    console.error('Request failed:', request.url());
  });

  try {
    console.log(`\n🧪 Testing: ${TEST_URL}\n`);
    
    // Navigate to test page
    await page.goto(TEST_URL, {
      waitUntil: 'networkidle2',
      timeout: TIMEOUT
    });

    // Wait for page to stabilize
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Take initial screenshot
    const screenshot1 = await page.screenshot({ path: 'test-screenshot-1-initial.png', fullPage: true });
    screenshots.push('test-screenshot-1-initial.png');
    console.log('✅ Screenshot 1: Initial page load');

    // Test 1: Check if PixiJS compositor script loaded
    console.log('\n📋 Test 1: PixiJS Compositor Script Loading');
    const compositorVersion = await page.evaluate(() => {
      const scripts = Array.from(document.querySelectorAll('script[src*="kelly-pixi-compositor"]'));
      return scripts.length > 0 ? scripts[0].src : null;
    });
    console.log(`   Compositor script: ${compositorVersion || 'NOT FOUND'}`);

    // Test 2: Check console logs for compositor initialization
    console.log('\n📋 Test 2: Compositor Initialization Logs');
    const initLogs = logs.filter(log => 
      log.text.includes('kelly-pixi-compositor.js LOADED') ||
      log.text.includes('Compositor READY') ||
      log.text.includes('Pixi init')
    );
    console.log(`   Found ${initLogs.length} initialization logs:`);
    initLogs.forEach(log => console.log(`   - [${log.type}] ${log.text}`));

    // Test 3: Check if PixiJS is available
    console.log('\n📋 Test 3: PixiJS Library Availability');
    const pixiAvailable = await page.evaluate(() => {
      return typeof window.PIXI !== 'undefined';
    });
    console.log(`   PIXI available: ${pixiAvailable}`);

    // Test 4: Check if compositor initialized (wait for it)
    console.log('\n📋 Test 4: Compositor Initialization Status');
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for init
    const compositorStatus = await page.evaluate(() => {
      if (!window.KellyPixiCompositor) {
        return { initialized: false, error: 'KellyPixiCompositor not found' };
      }
      const state = {
        initialized: window.KellyPixiCompositor.isInitialized || false,
        enabled: window.KellyPixiCompositor.isEnabled || false,
        hasApp: !!window.KellyPixiCompositor.app,
        hasCanvas: !!window.KellyPixiCompositor.app?.canvas,
        mode: window.KellyPixiCompositor.mode || 'none',
        containerEl: !!window.KellyPixiCompositor.containerEl,
        _initPromise: !!window.KellyPixiCompositor._initPromise
      };
      
      // Check for global state marker
      state.__KELLY_PIXI_READY = window.__KELLY_PIXI_READY || false;
      state.__KELLY_PIXI_STATE = window.__KELLY_PIXI_STATE || null;
      
      return state;
    });
    console.log(`   Status:`, JSON.stringify(compositorStatus, null, 2));
    
    // If not initialized, check why
    if (!compositorStatus.initialized) {
      console.log(`   ⚠️  Compositor NOT initialized! Checking logs...`);
      const initLogs = logs.filter(log => 
        log.text.includes('init') || 
        log.text.includes('FAILED') ||
        log.text.includes('error')
      );
      initLogs.forEach(log => console.log(`      [${log.type}] ${log.text}`));
    }

    // Test 5: Check for canvas element
    console.log('\n📋 Test 5: Canvas Element');
    const canvasInfo = await page.evaluate(() => {
      const container = document.getElementById('kelly-stage');
      if (!container) return { found: false, error: 'kelly-stage container not found' };
      
      const canvas = container.querySelector('canvas');
      return {
        found: !!canvas,
        visible: canvas ? window.getComputedStyle(canvas).display !== 'none' : false,
        width: canvas?.width || 0,
        height: canvas?.height || 0,
        zIndex: canvas ? window.getComputedStyle(canvas).zIndex : null
      };
    });
    console.log(`   Canvas:`, JSON.stringify(canvasInfo, null, 2));

    // Test 6: Check for debug marker (red dot)
    console.log('\n📋 Test 6: Debug Marker');
    const debugMarker = await page.evaluate(() => {
      if (!window.KellyPixiCompositor || !window.KellyPixiCompositor.app) {
        return { found: false, error: 'Compositor not initialized' };
      }
      const stage = window.KellyPixiCompositor.app.stage;
      const marker = stage.children.find(child => child.name === 'debugMarker');
      return {
        found: !!marker,
        visible: marker ? marker.visible : false,
        x: marker?.x || 0,
        y: marker?.y || 0
      };
    });
    console.log(`   Debug marker:`, JSON.stringify(debugMarker, null, 2));

    // Test 7: Check for mouth overlay
    console.log('\n📋 Test 7: Mouth Overlay');
    const mouthOverlay = await page.evaluate(() => {
      if (!window.KellyPixiCompositor || !window.KellyPixiCompositor.app) {
        return { found: false, error: 'Compositor not initialized' };
      }
      const stage = window.KellyPixiCompositor.app.stage;
      const mouth = stage.children.find(child => child.name === 'mouth');
      return {
        found: !!mouth,
        visible: mouth ? mouth.visible : false,
        x: mouth?.x || 0,
        y: mouth?.y || 0,
        children: mouth ? mouth.children.length : 0
      };
    });
    console.log(`   Mouth overlay:`, JSON.stringify(mouthOverlay, null, 2));

    // Test 8: Check for eyebrow overlays
    console.log('\n📋 Test 8: Eyebrow Overlays');
    const eyebrows = await page.evaluate(() => {
      if (!window.KellyPixiCompositor || !window.KellyPixiCompositor.app) {
        return { found: false, error: 'Compositor not initialized' };
      }
      const stage = window.KellyPixiCompositor.app.stage;
      const left = stage.children.find(child => child.name === 'eyebrowLeft');
      const right = stage.children.find(child => child.name === 'eyebrowRight');
      return {
        leftFound: !!left,
        rightFound: !!right,
        leftVisible: left ? left.visible : false,
        rightVisible: right ? right.visible : false
      };
    });
    console.log(`   Eyebrows:`, JSON.stringify(eyebrows, null, 2));

    // Test 9: Check expression bridge
    console.log('\n📋 Test 9: Expression Bridge');
    const expressionBridge = await page.evaluate(() => {
      return {
        available: typeof window.KellyExpressionBridge !== 'undefined',
        initialized: window.KellyExpressionBridge?.isInitialized || false,
        currentExpression: window.KellyExpressionBridge?.currentExpression || null
      };
    });
    console.log(`   Expression bridge:`, JSON.stringify(expressionBridge, null, 2));

    // Test 10: Check autoplay handler
    console.log('\n📋 Test 10: Autoplay Handler');
    const autoplayHandler = await page.evaluate(() => {
      return {
        available: typeof window.KellyAutoplayHandler !== 'undefined',
        initialized: window.KellyAutoplayHandler?.isInitialized || false,
        unlocked: window.KellyAutoplayHandler?.audioUnlocked || false,
        isIOS: window.KellyAutoplayHandler?.isIOS || false
      };
    });
    console.log(`   Autoplay handler:`, JSON.stringify(autoplayHandler, null, 2));

    // Test 11: Check for blendshapes
    console.log('\n📋 Test 11: Blendshapes');
    const blendshapes = await page.evaluate(() => {
      if (!window.KellyPixiCompositor) {
        return { found: false, error: 'Compositor not found' };
      }
      return {
        hasBlendshapes: Object.keys(window.KellyPixiCompositor.lastBlendshapes || {}).length > 0,
        blendshapeKeys: Object.keys(window.KellyPixiCompositor.lastBlendshapes || {}),
        sampleValues: Object.fromEntries(
          Object.entries(window.KellyPixiCompositor.lastBlendshapes || {}).slice(0, 5)
        )
      };
    });
    console.log(`   Blendshapes:`, JSON.stringify(blendshapes, null, 2));

    // Test 12: Check TTS audio
    console.log('\n📋 Test 12: TTS Audio');
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for audio to potentially start
    const audioStatus = await page.evaluate(() => {
      const audioEl = document.querySelector('audio');
      return {
        found: !!audioEl,
        playing: audioEl ? !audioEl.paused : false,
        src: audioEl?.src || null,
        muted: audioEl?.muted || false
      };
    });
    console.log(`   Audio:`, JSON.stringify(audioStatus, null, 2));

    // Test 13: Check for errors
    console.log('\n📋 Test 13: Errors');
    console.log(`   Total errors: ${errors.length}`);
    errors.forEach((err, i) => {
      console.log(`   Error ${i + 1}:`, err.type || 'unknown', err.message || err.url);
    });

    // Test 14: Check script versions
    console.log('\n📋 Test 14: Script Versions');
    const scriptVersions = await page.evaluate(() => {
      const scripts = Array.from(document.querySelectorAll('script[src]'));
      const compositorScript = scripts.find(s => s.src.includes('kelly-pixi-compositor'));
      return {
        compositorVersion: compositorScript ? new URL(compositorScript.src).searchParams.get('v') : null,
        pixiVersion: typeof window.PIXI !== 'undefined' ? window.PIXI.VERSION : null
      };
    });
    console.log(`   Versions:`, JSON.stringify(scriptVersions, null, 2));
    
    // Test 15: Check if playPhaseMedia was called
    console.log('\n📋 Test 15: playPhaseMedia Execution');
    const playPhaseLogs = logs.filter(log => 
      log.text.includes('TALKING PHOTO') ||
      log.text.includes('playPhaseMedia') ||
      log.text.includes('📸') ||
      log.text.includes('[DEBUG]')
    );
    console.log(`   Found ${playPhaseLogs.length} relevant logs:`);
    playPhaseLogs.forEach(log => console.log(`   - [${log.type}] ${log.text}`));
    
    // Test 16: Check if kelly-stage container exists
    console.log('\n📋 Test 16: Container Element');
    const containerInfo = await page.evaluate(() => {
      const container = document.getElementById('kelly-stage');
      return {
        found: !!container,
        visible: container ? window.getComputedStyle(container).display !== 'none' : false,
        hasChildren: container ? container.children.length : 0,
        innerHTML: container ? container.innerHTML.substring(0, 200) : null
      };
    });
    console.log(`   Container:`, JSON.stringify(containerInfo, null, 2));

    // Take final screenshot
    await new Promise(resolve => setTimeout(resolve, 2000));
    const screenshot2 = await page.screenshot({ path: 'test-screenshot-2-final.png', fullPage: true });
    screenshots.push('test-screenshot-2-final.png');
    console.log('✅ Screenshot 2: Final state');

    // Generate test report
    const report = {
      timestamp: new Date().toISOString(),
      url: TEST_URL,
      tests: {
        compositorScriptLoaded: !!compositorVersion,
        pixiAvailable: pixiAvailable,
        compositorInitialized: compositorStatus.initialized,
        canvasFound: canvasInfo.found,
        debugMarkerFound: debugMarker.found,
        mouthOverlayFound: mouthOverlay.found,
        eyebrowsFound: eyebrows.leftFound && eyebrows.rightFound,
        expressionBridgeAvailable: expressionBridge.available,
        autoplayHandlerAvailable: autoplayHandler.available,
        hasBlendshapes: blendshapes.hasBlendshapes,
        audioFound: audioStatus.found
      },
      errors: errors,
      logs: logs.filter(log => 
        log.text.includes('Pixi') || 
        log.text.includes('Compositor') ||
        log.text.includes('Expression') ||
        log.text.includes('Autoplay')
      ),
      screenshots: screenshots,
      scriptVersions: scriptVersions
    };

    // Save report
    const reportPath = 'test-report.json';
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📊 Test report saved to: ${reportPath}`);

    // Print summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 TEST SUMMARY');
    console.log('='.repeat(60));
    console.log(`✅ Compositor Script Loaded: ${report.tests.compositorScriptLoaded}`);
    console.log(`✅ PixiJS Available: ${report.tests.pixiAvailable}`);
    console.log(`✅ Compositor Initialized: ${report.tests.compositorInitialized}`);
    console.log(`✅ Canvas Found: ${report.tests.canvasFound}`);
    console.log(`✅ Debug Marker: ${report.tests.debugMarkerFound}`);
    console.log(`✅ Mouth Overlay: ${report.tests.mouthOverlayFound}`);
    console.log(`✅ Eyebrows: ${report.tests.eyebrowsFound}`);
    console.log(`✅ Expression Bridge: ${report.tests.expressionBridgeAvailable}`);
    console.log(`✅ Autoplay Handler: ${report.tests.autoplayHandlerAvailable}`);
    console.log(`✅ Has Blendshapes: ${report.tests.hasBlendshapes}`);
    console.log(`✅ Audio Found: ${report.tests.audioFound}`);
    console.log(`❌ Errors: ${errors.length}`);
    console.log('='.repeat(60));

    if (errors.length > 0) {
      console.log('\n⚠️  ERRORS DETECTED:');
      errors.forEach((err, i) => {
        console.log(`\n${i + 1}. ${err.type || 'Error'}:`);
        console.log(`   ${err.message || err.url || 'Unknown error'}`);
        if (err.stack) console.log(`   Stack: ${err.stack.split('\n')[0]}`);
      });
    }

  } catch (error) {
    console.error('Test failed:', error);
    await page.screenshot({ path: 'test-screenshot-error.png', fullPage: true });
    throw error;
  } finally {
    await browser.close();
  }
}

// Run tests
testHybridCompositor().catch(error => {
  console.error('Test suite failed:', error);
  process.exit(1);
});

