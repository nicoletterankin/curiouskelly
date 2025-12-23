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
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
  protocolTimeout: 60000 // Increase timeout
});

const page = await browser.newPage();

// Capture all console messages
const consoleMessages = [];
const errors = [];

page.on('console', msg => {
  const text = msg.text();
  consoleMessages.push({ type: msg.type(), text });
  
  // Log important messages
  if (text.includes('Audio') || text.includes('LipSync') || text.includes('blendshapes') || text.includes('mouth') || text.includes('TTS')) {
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
  await new Promise(r => setTimeout(r, 5000));

  // Check if scripts loaded
  console.log('\n📦 Checking script loading...');
  const scriptsLoaded = await page.evaluate(() => {
    return {
      pixiCompositor: !!window.KellyPixiCompositor && window.KellyPixiCompositor.isInitialized,
      lipSync: !!window.KellyLipSync && window.KellyLipSync.isInitialized,
      audioSystem: !!window.kellyAudio,
      pixi: !!window.PIXI
    };
  });
  
  console.log(`  ${scriptsLoaded.pixiCompositor ? '✅' : '❌'} Pixi Compositor: ${scriptsLoaded.pixiCompositor ? 'LOADED' : 'NOT LOADED'}`);
  console.log(`  ${scriptsLoaded.lipSync ? '✅' : '❌'} Lip-Sync: ${scriptsLoaded.lipSync ? 'LOADED' : 'NOT LOADED'}`);
  console.log(`  ${scriptsLoaded.audioSystem ? '✅' : '❌'} Audio System: ${scriptsLoaded.audioSystem ? 'LOADED' : 'NOT LOADED'}`);
  console.log(`  ${scriptsLoaded.pixi ? '✅' : '❌'} PixiJS: ${scriptsLoaded.pixi ? 'LOADED' : 'NOT LOADED'}`);

  // Unlock audio context (required for autoplay)
  console.log('\n🔓 Unlocking audio context...');
  await page.click('body'); // Click to unlock audio
  await new Promise(r => setTimeout(r, 1000));

  // Monitor for 15 seconds - check if audio starts playing naturally
  console.log('\n⏳ Monitoring production behavior (15 seconds)...');
  console.log('   (Waiting for natural audio playback to start...)');
  
  let audioDetected = false;
  let blendshapesDetected = false;
  let maxJawOpen = 0;
  
  for (let i = 0; i < 30; i++) {
    await new Promise(r => setTimeout(r, 500));
    
    const state = await page.evaluate(() => {
      const audio = document.querySelector('audio');
      return {
        audioPlaying: audio ? !audio.paused && !audio.ended && audio.currentTime > 0 : false,
        audioSrc: audio?.src || '',
        audioCurrentTime: audio?.currentTime || 0,
        kellyAudioPlaying: window.kellyAudio?.isPlaying || false,
        lipSyncActive: window.KellyLipSync?.isActive || false,
        lipSyncInitialized: window.KellyLipSync?.isInitialized || false,
        blendshapes: window.KellyLipSync?.currentBlendshapes || {},
        compositorBlendshapes: window.KellyPixiCompositor?.lastBlendshapes || {},
        compositorEnabled: window.KellyPixiCompositor?.isEnabled || false
      };
    });
    
    if ((state.audioPlaying || state.kellyAudioPlaying) && !audioDetected) {
      audioDetected = true;
      console.log(`\n  ✅ Audio detected at ${i * 0.5}s:`);
      console.log(`     - Audio element: ${state.audioPlaying ? 'playing' : 'not playing'}`);
      console.log(`     - KellyAudio: ${state.kellyAudioPlaying ? 'playing' : 'not playing'}`);
      console.log(`     - Audio src: ${state.audioSrc.substring(0, 60)}...`);
      console.log(`     - Current time: ${state.audioCurrentTime.toFixed(2)}s`);
    }
    
    if (state.blendshapes.jawOpen > 0.1 || state.compositorBlendshapes.jawOpen > 0.1) {
      const jawOpen = state.blendshapes.jawOpen || state.compositorBlendshapes.jawOpen || 0;
      if (jawOpen > maxJawOpen) maxJawOpen = jawOpen;
      
      if (!blendshapesDetected) {
        blendshapesDetected = true;
        console.log(`\n  ✅ Blendshapes detected at ${i * 0.5}s:`);
        console.log(`     - jawOpen: ${jawOpen.toFixed(2)}`);
        console.log(`     - Lip-sync active: ${state.lipSyncActive}`);
        console.log(`     - Compositor enabled: ${state.compositorEnabled}`);
      }
    }
    
    // Show progress every 5 seconds
    if (i % 10 === 0 && i > 0) {
      console.log(`  ⏳ ${i * 0.5}s elapsed...`);
    }
  }

  // Final state check
  console.log('\n🔍 Final state check...');
  const finalState = await page.evaluate(() => {
    const audio = document.querySelector('audio');
    return {
      audio: {
        found: !!audio,
        playing: audio ? !audio.paused && !audio.ended : false,
        currentTime: audio?.currentTime || 0,
        duration: audio?.duration || 0,
        readyState: audio?.readyState || 0
      },
      kellyAudio: {
        playing: window.kellyAudio?.isPlaying || false,
        currentText: window.kellyAudio?.currentText || ''
      },
      lipSync: {
        initialized: window.KellyLipSync?.isInitialized || false,
        active: window.KellyLipSync?.isActive || false,
        blendshapes: window.KellyLipSync?.currentBlendshapes || {}
      },
      compositor: {
        initialized: window.KellyPixiCompositor?.isInitialized || false,
        enabled: window.KellyPixiCompositor?.isEnabled || false,
        mode: window.KellyPixiCompositor?.mode || 'none',
        blendshapes: window.KellyPixiCompositor?.lastBlendshapes || {}
      }
    };
  });
  
  console.log('\n📊 Final State:');
  console.log('  Audio:', JSON.stringify(finalState.audio, null, 2));
  console.log('  KellyAudio:', JSON.stringify(finalState.kellyAudio, null, 2));
  console.log('  LipSync:', {
    initialized: finalState.lipSync.initialized,
    active: finalState.lipSync.active,
    jawOpen: finalState.lipSync.blendshapes.jawOpen?.toFixed(2) || 0
  });
  console.log('  Compositor:', {
    initialized: finalState.compositor.initialized,
    enabled: finalState.compositor.enabled,
    mode: finalState.compositor.mode,
    jawOpen: finalState.compositor.blendshapes.jawOpen?.toFixed(2) || 0
  });

  // Check console logs for key messages
  console.log('\n📋 Key console messages (last 30):');
  const keyMessages = consoleMessages.filter(m => 
    m.text.includes('Audio') || 
    m.text.includes('LipSync') || 
    m.text.includes('blendshapes') || 
    m.text.includes('mouth') ||
    m.text.includes('TTS') ||
    m.text.includes('Compositor')
  );
  
  keyMessages.slice(-30).forEach(m => {
    const icon = m.type === 'error' ? '❌' : m.type === 'warning' ? '⚠️' : 'ℹ️';
    console.log(`  ${icon} ${m.text.substring(0, 120)}`);
  });

  // Final verdict
  console.log('\n' + '='.repeat(60));
  console.log('📊 PRODUCTION VERIFICATION RESULTS:');
  console.log('='.repeat(60));
  
  const audioWorking = audioDetected || finalState.audio.playing || finalState.kellyAudio.playing;
  const lipSyncWorking = blendshapesDetected || maxJawOpen > 0.1;
  const compositorReceiving = finalState.compositor.blendshapes.jawOpen > 0.1;
  
  console.log(`  ${audioWorking ? '✅' : '❌'} Audio playing: ${audioWorking ? 'YES' : 'NO'}`);
  console.log(`  ${lipSyncWorking ? '✅' : '❌'} Lip-sync analyzing: ${lipSyncWorking ? `YES (max jawOpen: ${maxJawOpen.toFixed(2)})` : 'NO'}`);
  console.log(`  ${compositorReceiving ? '✅' : '❌'} Compositor receiving blendshapes: ${compositorReceiving ? 'YES' : 'NO'}`);
  
  if (audioWorking && lipSyncWorking && compositorReceiving) {
    console.log('\n🎉 SUCCESS: All systems working! Mouth should be moving!');
  } else {
    console.log('\n⚠️ ISSUES DETECTED:');
    if (!audioWorking) console.log('  - Audio is not playing (check TTS endpoint, autoplay, or audio loading)');
    if (!lipSyncWorking) console.log('  - Lip-sync is not analyzing audio (check audio connection to analyser)');
    if (!compositorReceiving) console.log('  - Compositor is not receiving blendshapes (check onBlendshapesUpdate callback)');
  }
  
  console.log('='.repeat(60));

} catch (error) {
  console.error('❌ Test failed:', error);
} finally {
  console.log('\n⏳ Keeping browser open for 10 seconds for inspection...');
  await new Promise(r => setTimeout(r, 10000));
  await browser.close();
}
