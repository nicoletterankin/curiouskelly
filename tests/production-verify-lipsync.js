#!/usr/bin/env node
/**
 * PRODUCTION VERIFICATION: Hybrid Compositor Lip-Sync
 * 
 * Tests the ACTUAL production URL to verify:
 * 1. Audio is playing
 * 2. Lip-sync is analyzing audio
 * 3. Blendshapes are being generated
 * 4. Mouth is moving (blendshapes varying)
 * 
 * This tests PRODUCTION, not local - because that's where it needs to work.
 */

import puppeteer from 'puppeteer';

const PRODUCTION_URL = 'https://curiouskelly.com/learn.html?hybrid=1&day=1&pixiDebug=1';

console.log('🧪 PRODUCTION VERIFICATION: Hybrid Compositor Lip-Sync');
console.log(`📍 Testing: ${PRODUCTION_URL}`);
console.log('');

const browser = await puppeteer.launch({
  headless: false, // Show browser so we can see what's happening
  args: ['--no-sandbox', '--disable-setuid-sandbox']
});

const page = await browser.newPage();

// Capture all console messages
const consoleMessages = [];
const errors = [];

page.on('console', msg => {
  const text = msg.text();
  consoleMessages.push({ type: msg.type(), text });
  
  // Log important messages
  if (text.includes('Audio') || text.includes('LipSync') || text.includes('blendshapes') || text.includes('mouth')) {
    console.log(`[${msg.type()}] ${text}`);
  }
});

page.on('pageerror', error => {
  errors.push(error);
  console.error('❌ Page error:', error.message);
});

try {
  console.log('📡 Loading production page...');
  await page.goto(PRODUCTION_URL, { 
    waitUntil: 'networkidle2',
    timeout: 30000 
  });

  console.log('⏳ Waiting for page to initialize...');
  await new Promise(r => setTimeout(r, 3000));

  // Check if scripts loaded
  console.log('\n📦 Checking script loading...');
  const pixiLoaded = await page.evaluate(() => {
    return !!window.KellyPixiCompositor && window.KellyPixiCompositor.isInitialized;
  });
  console.log(`  ✅ Pixi Compositor: ${pixiLoaded ? 'LOADED' : 'NOT LOADED'}`);

  const lipSyncLoaded = await page.evaluate(() => {
    return !!window.KellyLipSync && window.KellyLipSync.isInitialized;
  });
  console.log(`  ✅ Lip-Sync: ${lipSyncLoaded ? 'LOADED' : 'NOT LOADED'}`);

  const audioLoaded = await page.evaluate(() => {
    return !!window.kellyAudio;
  });
  console.log(`  ✅ Audio System: ${audioLoaded ? 'LOADED' : 'NOT LOADED'}`);

  // Unlock audio context (required for autoplay)
  console.log('\n🔓 Unlocking audio context...');
  await page.click('body'); // Click to unlock audio
  await new Promise(r => setTimeout(r, 500));

  // Try to trigger audio playback
  console.log('\n🎵 Triggering audio playback...');
  const audioStarted = await page.evaluate(async () => {
    try {
      if (window.kellyAudio && window.playPhaseMedia) {
        // Try to play phase media
        await window.playPhaseMedia();
        await new Promise(r => setTimeout(r, 1000));
        
        // Check if audio is playing
        const audio = window.kellyAudio.audio;
        return {
          playing: !audio.paused && !audio.ended,
          currentTime: audio.currentTime,
          src: audio.src?.substring(0, 50) || 'none',
          readyState: audio.readyState
        };
      }
      return null;
    } catch (e) {
      return { error: e.message };
    }
  });

  if (audioStarted) {
    console.log('  📊 Audio state:', audioStarted);
  }

  // Wait for audio to play and lip-sync to analyze
  console.log('\n⏳ Waiting for lip-sync analysis (10 seconds)...');
  await new Promise(r => setTimeout(r, 10000));

  // Check blendshapes
  console.log('\n🎭 Checking blendshapes...');
  const blendshapes = await page.evaluate(() => {
    if (window.KellyLipSync) {
      return window.KellyLipSync.currentBlendshapes || {};
    }
    return null;
  });

  if (blendshapes) {
    console.log('  📊 Current blendshapes:', {
      jawOpen: blendshapes.jawOpen?.toFixed(2) || 0,
      mouthOpen: blendshapes.mouthOpen?.toFixed(2) || 0,
      mouthFunnel: blendshapes.mouthFunnel?.toFixed(2) || 0
    });
    
    const jawOpenVarying = blendshapes.jawOpen > 0.1;
    console.log(`  ${jawOpenVarying ? '✅' : '❌'} JawOpen varying: ${jawOpenVarying ? 'YES (lip-sync working!)' : 'NO (lip-sync not working)'}`);
  } else {
    console.log('  ❌ No blendshapes found');
  }

  // Check compositor state
  console.log('\n🎨 Checking compositor state...');
  const compositorState = await page.evaluate(() => {
    if (window.KellyPixiCompositor) {
      return {
        initialized: window.KellyPixiCompositor.isInitialized,
        enabled: window.KellyPixiCompositor.isEnabled,
        mode: window.KellyPixiCompositor.mode,
        hasBlendshapes: Object.keys(window.KellyPixiCompositor.lastBlendshapes || {}).length > 0,
        lastJawOpen: window.KellyPixiCompositor.lastBlendshapes?.jawOpen || 0
      };
    }
    return null;
  });

  if (compositorState) {
    console.log('  📊 Compositor state:', compositorState);
    console.log(`  ${compositorState.hasBlendshapes ? '✅' : '❌'} Receiving blendshapes: ${compositorState.hasBlendshapes ? 'YES' : 'NO'}`);
    console.log(`  ${compositorState.lastJawOpen > 0.1 ? '✅' : '❌'} JawOpen > 0.1: ${compositorState.lastJawOpen > 0.1 ? 'YES (mouth should be moving!)' : 'NO'}`);
  }

  // Check console logs for key messages
  console.log('\n📋 Key console messages:');
  const keyMessages = consoleMessages.filter(m => 
    m.text.includes('Audio') || 
    m.text.includes('LipSync') || 
    m.text.includes('blendshapes') || 
    m.text.includes('mouth') ||
    m.text.includes('TTS')
  );
  
  keyMessages.slice(-20).forEach(m => {
    const icon = m.type === 'error' ? '❌' : m.type === 'warning' ? '⚠️' : 'ℹ️';
    console.log(`  ${icon} ${m.text.substring(0, 100)}`);
  });

  // Final verdict
  console.log('\n' + '='.repeat(60));
  console.log('📊 PRODUCTION VERIFICATION RESULTS:');
  console.log('='.repeat(60));
  
  const audioWorking = audioStarted?.playing || false;
  const lipSyncWorking = blendshapes?.jawOpen > 0.1 || false;
  const compositorReceiving = compositorState?.hasBlendshapes || false;
  
  console.log(`  ${audioWorking ? '✅' : '❌'} Audio playing: ${audioWorking ? 'YES' : 'NO'}`);
  console.log(`  ${lipSyncWorking ? '✅' : '❌'} Lip-sync analyzing: ${lipSyncWorking ? 'YES' : 'NO'}`);
  console.log(`  ${compositorReceiving ? '✅' : '❌'} Compositor receiving blendshapes: ${compositorReceiving ? 'YES' : 'NO'}`);
  
  if (audioWorking && lipSyncWorking && compositorReceiving) {
    console.log('\n🎉 SUCCESS: All systems working! Mouth should be moving!');
  } else {
    console.log('\n⚠️ ISSUES DETECTED:');
    if (!audioWorking) console.log('  - Audio is not playing');
    if (!lipSyncWorking) console.log('  - Lip-sync is not analyzing audio');
    if (!compositorReceiving) console.log('  - Compositor is not receiving blendshapes');
  }
  
  console.log('='.repeat(60));

} catch (error) {
  console.error('❌ Test failed:', error);
} finally {
  console.log('\n⏳ Keeping browser open for 5 seconds for inspection...');
  await new Promise(r => setTimeout(r, 5000));
  await browser.close();
}

