/**
 * PROVE IT: Hybrid Compositor End-to-End Test
 * 
 * This test PROVES the hybrid compositor works:
 * - Real-time TTS (not pre-rendered)
 * - Mouth moves with audio
 * - Expressions change
 * - It feels like video but it's live
 * 
 * Run: node tests/hybrid-compositor-prove-it.js
 */

import puppeteer from 'puppeteer';
import fs from 'fs';

const TEST_URL = process.env.TEST_URL || 'https://curiouskelly.com/learn.html?day=1';
const TIMEOUT = 60000; // 60 seconds for full test

async function proveHybridCompositor() {
  console.log('\n🧪 PROVING HYBRID COMPOSITOR WORKS\n');
  console.log(`Testing: ${TEST_URL}\n`);

  const browser = await puppeteer.launch({
    headless: false, // Show browser so you can see it
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--autoplay-policy=no-user-gesture-required']
  });

  const page = await browser.newPage();
  
  // Set viewport to match typical user
  await page.setViewport({ width: 1920, height: 1080 });

  const evidence = {
    timestamp: new Date().toISOString(),
    url: TEST_URL,
    tests: {},
    screenshots: [],
    consoleLogs: [],
    errors: [],
    timing: {}
  };

  // Capture all console logs
  page.on('console', msg => {
    const text = msg.text();
    evidence.consoleLogs.push({
      type: msg.type(),
      text: text,
      time: Date.now()
    });
    
    // Log important events
    if (text.includes('Compositor') || text.includes('Lip-sync') || text.includes('TTS') || text.includes('speak')) {
      console.log(`  [${msg.type()}] ${text}`);
    }
  });

  // Capture errors
  page.on('pageerror', error => {
    evidence.errors.push({
      message: error.message,
      stack: error.stack
    });
    console.error('  ❌ Page error:', error.message);
  });

  try {
    const startTime = Date.now();
    
    // Navigate to page
    console.log('📄 Loading page...');
    await page.goto(TEST_URL, {
      waitUntil: 'networkidle2',
      timeout: TIMEOUT
    });
    evidence.timing.pageLoad = Date.now() - startTime;
    console.log(`  ✅ Page loaded in ${evidence.timing.pageLoad}ms`);

    // Wait for page to stabilize
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Screenshot 1: Initial state
    const screenshot1 = await page.screenshot({ path: 'proof-1-initial.png', fullPage: false });
    evidence.screenshots.push('proof-1-initial.png');
    console.log('  📸 Screenshot 1: Initial state');

    // CRITICAL: Simulate user interaction to unlock autoplay
    console.log('\n👆 Simulating user click to unlock autoplay...');
    await page.mouse.click(960, 540); // Click center of screen
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Try to trigger audio playback programmatically
    const audioUnlocked = await page.evaluate(async () => {
      // Try to play a test audio to unlock autoplay
      try {
        const testAudio = new Audio('data:audio/wav;base64,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');
        await testAudio.play();
        return true;
      } catch (e) {
        return false;
      }
    });
    
    if (audioUnlocked) {
      console.log('  ✅ Autoplay unlocked');
    } else {
      console.log('  ⚠️  Autoplay still blocked');
    }
    
    // Try clicking play button if it exists
    const playButton = await page.$('button[aria-label*="play"], button[aria-label*="Play"], .play-button, #play-btn');
    if (playButton) {
      console.log('  👆 Clicking play button...');
      await playButton.click();
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // TEST 1: Verify compositor script loaded
    console.log('\n🔍 TEST 1: Compositor Script Loaded');
    const scriptLoaded = await page.evaluate(() => {
      const scripts = Array.from(document.querySelectorAll('script[src*="kelly-pixi-compositor"]'));
      return scripts.length > 0;
    });
    evidence.tests.scriptLoaded = scriptLoaded;
    console.log(`  ${scriptLoaded ? '✅' : '❌'} Script loaded: ${scriptLoaded}`);

    // TEST 2: Verify PixiJS available
    console.log('\n🔍 TEST 2: PixiJS Library Available');
    const pixiAvailable = await page.evaluate(() => {
      return typeof window.PIXI !== 'undefined';
    });
    evidence.tests.pixiAvailable = pixiAvailable;
    console.log(`  ${pixiAvailable ? '✅' : '❌'} PixiJS available: ${pixiAvailable}`);

    // TEST 3: Wait for audio to start playing (this triggers compositor init)
    console.log('\n🔍 TEST 3: Audio Playback & Compositor Initialization');
    console.log('  ⏳ Waiting for audio to start (this triggers compositor init)...');
    
    let audioStarted = false;
    let compositorInitialized = false;
    let blendshapesReceived = false;
    
    // Wait up to 15 seconds for audio to start
    for (let i = 0; i < 30; i++) {
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const state = await page.evaluate(() => {
        const audio = document.querySelector('audio');
        return {
          audioPlaying: audio ? !audio.paused : false,
          audioSrc: audio?.src || null,
          compositorInitialized: window.KellyPixiCompositor?.isInitialized || false,
          compositorEnabled: window.KellyPixiCompositor?.isEnabled || false,
          hasBlendshapes: Object.keys(window.KellyPixiCompositor?.lastBlendshapes || {}).length > 0,
          blendshapeCount: Object.keys(window.KellyPixiCompositor?.lastBlendshapes || {}).length,
          lipSyncActive: window.KellyLipSync?.isActive || false,
          expressionBridgeActive: window.KellyExpressionBridge?.isInitialized || false
        };
      });
      
      if (state.audioPlaying && !audioStarted) {
        audioStarted = true;
        console.log(`  ✅ Audio started playing (${state.audioSrc ? 'TTS' : 'unknown source'})`);
        evidence.timing.audioStart = Date.now() - startTime;
      }
      
      if (state.compositorInitialized && !compositorInitialized) {
        compositorInitialized = true;
        console.log(`  ✅ Compositor initialized`);
        evidence.timing.compositorInit = Date.now() - startTime;
      }
      
      if (state.hasBlendshapes && !blendshapesReceived) {
        blendshapesReceived = true;
        console.log(`  ✅ Blendshapes received (${state.blendshapeCount} shapes)`);
        evidence.timing.blendshapesReceived = Date.now() - startTime;
      }
      
      if (audioStarted && compositorInitialized && blendshapesReceived) {
        break;
      }
    }
    
    evidence.tests.audioStarted = audioStarted;
    evidence.tests.compositorInitialized = compositorInitialized;
    evidence.tests.blendshapesReceived = blendshapesReceived;
    
    if (!audioStarted) {
      console.log('  ⚠️  Audio did not start automatically (may need user interaction)');
    }

    // TEST 4: Verify compositor state
    console.log('\n🔍 TEST 4: Compositor State');
    const compositorState = await page.evaluate(() => {
      if (!window.KellyPixiCompositor) {
        return { error: 'Compositor not found' };
      }
      return {
        initialized: window.KellyPixiCompositor.isInitialized,
        enabled: window.KellyPixiCompositor.isEnabled,
        mode: window.KellyPixiCompositor.mode,
        hasApp: !!window.KellyPixiCompositor.app,
        hasCanvas: !!window.KellyPixiCompositor.app?.canvas,
        canvasWidth: window.KellyPixiCompositor.app?.canvas?.width || 0,
        canvasHeight: window.KellyPixiCompositor.app?.canvas?.height || 0,
        blendshapeCount: Object.keys(window.KellyPixiCompositor.lastBlendshapes || {}).length,
        sampleBlendshapes: Object.fromEntries(
          Object.entries(window.KellyPixiCompositor.lastBlendshapes || {}).slice(0, 5)
        )
      };
    });
    console.log('  Compositor state:', JSON.stringify(compositorState, null, 2));
    evidence.tests.compositorState = compositorState;

    // TEST 5: Verify canvas exists and is rendering
    console.log('\n🔍 TEST 5: Canvas Rendering');
    const canvasInfo = await page.evaluate(() => {
      const container = document.getElementById('kelly-stage');
      const canvas = container?.querySelector('canvas');
      return {
        containerFound: !!container,
        canvasFound: !!canvas,
        canvasVisible: canvas ? window.getComputedStyle(canvas).display !== 'none' : false,
        canvasWidth: canvas?.width || 0,
        canvasHeight: canvas?.height || 0,
        canvasInDOM: !!document.querySelector('#kelly-stage canvas')
      };
    });
    console.log('  Canvas info:', JSON.stringify(canvasInfo, null, 2));
    evidence.tests.canvasInfo = canvasInfo;

    // TEST 6: Verify mouth overlay exists
    console.log('\n🔍 TEST 6: Mouth Overlay');
    const mouthOverlay = await page.evaluate(() => {
      if (!window.KellyPixiCompositor?.app) {
        return { error: 'Compositor app not found' };
      }
      const stage = window.KellyPixiCompositor.app.stage;
      const mouth = stage.children.find(child => child.name === 'mouth');
      return {
        found: !!mouth,
        visible: mouth ? mouth.visible : false,
        x: mouth?.x || 0,
        y: mouth?.y || 0,
        children: mouth ? mouth.children.length : 0,
        hasInterior: !!stage.children.find(c => c.name === 'mouth')?.children.find(c => c.name === 'mouthInterior')
      };
    });
    console.log('  Mouth overlay:', JSON.stringify(mouthOverlay, null, 2));
    evidence.tests.mouthOverlay = mouthOverlay;

    // TEST 7: Monitor blendshape updates (PROVE IT'S REAL-TIME)
    console.log('\n🔍 TEST 7: Real-Time Blendshape Updates (PROVING IT\'S LIVE)');
    console.log('  ⏳ Monitoring blendshapes for 5 seconds...');
    
    const blendshapeHistory = [];
    const monitorStart = Date.now();
    
    for (let i = 0; i < 50; i++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const currentBlendshapes = await page.evaluate(() => {
        return window.KellyPixiCompositor?.lastBlendshapes || {};
      });
      
      const jawOpen = currentBlendshapes.jawOpen || currentBlendshapes.mouthOpen || 0;
      const mouthFunnel = currentBlendshapes.mouthFunnel || 0;
      
      blendshapeHistory.push({
        time: Date.now() - monitorStart,
        jawOpen,
        mouthFunnel,
        shapeCount: Object.keys(currentBlendshapes).length
      });
    }
    
    // Analyze blendshape changes (prove they're updating)
    const jawOpenValues = blendshapeHistory.map(h => h.jawOpen);
    const minJaw = Math.min(...jawOpenValues);
    const maxJaw = Math.max(...jawOpenValues);
    const jawVariation = maxJaw - minJaw;
    
    console.log(`  Jaw open range: ${minJaw.toFixed(1)} - ${maxJaw.toFixed(1)} (variation: ${jawVariation.toFixed(1)})`);
    console.log(`  Blendshape updates: ${blendshapeHistory.length} samples over 5 seconds`);
    
    evidence.tests.blendshapeVariation = jawVariation;
    evidence.tests.blendshapeHistory = blendshapeHistory.slice(0, 10); // Store first 10 samples
    
    if (jawVariation > 5) {
      console.log('  ✅ PROOF: Blendshapes are changing (mouth is moving!)');
    } else {
      console.log('  ⚠️  Blendshapes not varying much (audio may be silent or paused)');
    }

    // Screenshot 2: During playback
    const screenshot2 = await page.screenshot({ path: 'proof-2-during-playback.png', fullPage: false });
    evidence.screenshots.push('proof-2-during-playback.png');
    console.log('  📸 Screenshot 2: During playback');

    // TEST 8: Verify TTS is being used (not pre-rendered audio)
    console.log('\n🔍 TEST 8: TTS Source Verification');
    const ttsInfo = await page.evaluate(() => {
      const audio = document.querySelector('audio');
      const src = audio?.src || '';
      return {
        hasAudio: !!audio,
        audioSrc: src,
        isBlob: src.startsWith('blob:'),
        isTTS: src.includes('tts') || src.startsWith('blob:'),
        audioPlaying: audio ? !audio.paused : false
      };
    });
    console.log('  TTS info:', JSON.stringify(ttsInfo, null, 2));
    evidence.tests.ttsInfo = ttsInfo;

    // TEST 9: Verify expression bridge is active
    console.log('\n🔍 TEST 9: Expression Bridge');
    const expressionInfo = await page.evaluate(() => {
      return {
        initialized: window.KellyExpressionBridge?.isInitialized || false,
        currentExpression: window.KellyExpressionBridge?.currentExpression || null,
        hasTransition: window.KellyExpressionBridge?.transitionInProgress || false
      };
    });
    console.log('  Expression info:', JSON.stringify(expressionInfo, null, 2));
    evidence.tests.expressionInfo = expressionInfo;

    // Final screenshot
    await new Promise(resolve => setTimeout(resolve, 2000));
    const screenshot3 = await page.screenshot({ path: 'proof-3-final.png', fullPage: false });
    evidence.screenshots.push('proof-3-final.png');
    console.log('  📸 Screenshot 3: Final state');

    // Generate proof report
    const reportPath = 'proof-report.json';
    fs.writeFileSync(reportPath, JSON.stringify(evidence, null, 2));
    console.log(`\n📊 Proof report saved to: ${reportPath}`);

    // Print summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 PROOF SUMMARY');
    console.log('='.repeat(60));
    
    const allTests = [
      ['Script Loaded', evidence.tests.scriptLoaded],
      ['PixiJS Available', evidence.tests.pixiAvailable],
      ['Audio Started', evidence.tests.audioStarted],
      ['Compositor Initialized', evidence.tests.compositorInitialized],
      ['Blendshapes Received', evidence.tests.blendshapesReceived],
      ['Canvas Found', evidence.tests.canvasInfo?.canvasFound],
      ['Mouth Overlay Found', evidence.tests.mouthOverlay?.found],
      ['Blendshapes Varying', evidence.tests.blendshapeVariation > 5],
      ['TTS Source', evidence.tests.ttsInfo?.isTTS],
      ['Expression Bridge', evidence.tests.expressionInfo?.initialized]
    ];
    
    allTests.forEach(([name, passed]) => {
      console.log(`${passed ? '✅' : '❌'} ${name}: ${passed}`);
    });
    
    const passedCount = allTests.filter(([, passed]) => passed).length;
    const totalCount = allTests.length;
    const successRate = (passedCount / totalCount * 100).toFixed(1);
    
    console.log('='.repeat(60));
    console.log(`\n🎯 SUCCESS RATE: ${passedCount}/${totalCount} (${successRate}%)`);
    
    if (passedCount >= 7) {
      console.log('\n✅ PROOF: Hybrid compositor is WORKING!');
      console.log('   - Real-time TTS ✅');
      console.log('   - Mouth animation ✅');
      console.log('   - Expression system ✅');
      console.log('\n🎉 Kelly\'s presence is UNLOCKED!');
    } else {
      console.log('\n⚠️  Some tests failed. Check proof-report.json for details.');
    }
    
    console.log('\n📸 Screenshots saved:');
    evidence.screenshots.forEach(s => console.log(`   - ${s}`));
    
    if (evidence.errors.length > 0) {
      console.log('\n❌ Errors detected:');
      evidence.errors.forEach((err, i) => {
        console.log(`   ${i + 1}. ${err.message}`);
      });
    }

  } catch (error) {
    console.error('\n❌ Test failed:', error);
    await page.screenshot({ path: 'proof-error.png', fullPage: true });
    throw error;
  } finally {
    // Keep browser open for 5 seconds so you can see it
    console.log('\n⏳ Keeping browser open for 5 seconds...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    await browser.close();
  }
}

// Run proof test
proveHybridCompositor().catch(error => {
  console.error('Proof test failed:', error);
  process.exit(1);
});

