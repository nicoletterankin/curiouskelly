/**
 * Debug Compositor - Enhanced Logging
 * 
 * Adds comprehensive debugging to the compositor system.
 * Injects into page to capture all state changes.
 */

const puppeteer = require('puppeteer');

async function debugCompositor() {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();

  // Inject debug script
  await page.evaluateOnNewDocument(() => {
    // Override console methods to capture everything
    const originalLog = console.log;
    const originalError = console.error;
    const originalWarn = console.warn;
    
    window.__KELLY_DEBUG_LOGS = [];
    
    console.log = function(...args) {
      window.__KELLY_DEBUG_LOGS.push({ type: 'log', args, time: Date.now() });
      originalLog.apply(console, args);
    };
    
    console.error = function(...args) {
      window.__KELLY_DEBUG_LOGS.push({ type: 'error', args, time: Date.now() });
      originalError.apply(console, args);
    };
    
    console.warn = function(...args) {
      window.__KELLY_DEBUG_LOGS.push({ type: 'warn', args, time: Date.now() });
      originalWarn.apply(console, args);
    };

    // Monitor PixiJS initialization
    const originalPixiInit = window.PIXI?.Application?.prototype?.init;
    if (originalPixiInit) {
      window.PIXI.Application.prototype.init = function(...args) {
        console.log('[DEBUG] PixiJS Application.init() called', args);
        return originalPixiInit.apply(this, args);
      };
    }

    // Monitor compositor methods
    const monitorCompositor = () => {
      if (window.KellyPixiCompositor) {
        const originalInit = window.KellyPixiCompositor.init;
        if (originalInit) {
          window.KellyPixiCompositor.init = function(...args) {
            console.log('[DEBUG] KellyPixiCompositor.init() called', args);
            const result = originalInit.apply(this, args);
            console.log('[DEBUG] KellyPixiCompositor.init() result:', result);
            return result;
          };
        }

        const originalSetBlendshapes = window.KellyPixiCompositor.setBlendshapes;
        if (originalSetBlendshapes) {
          window.KellyPixiCompositor.setBlendshapes = function(...args) {
            console.log('[DEBUG] setBlendshapes() called', args);
            return originalSetBlendshapes.apply(this, args);
          };
        }
      } else {
        setTimeout(monitorCompositor, 100);
      }
    };
    monitorCompositor();
  });

  const url = process.env.TEST_URL || 'https://curiouskelly.com/learn.html?talkingPhoto=1&pixiDebug=1&day=1';
  console.log(`\n🔍 Debugging: ${url}\n`);

  await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
  await page.waitForTimeout(5000);

  // Capture all debug logs
  const debugLogs = await page.evaluate(() => {
    return window.__KELLY_DEBUG_LOGS || [];
  });

  console.log(`\n📋 Captured ${debugLogs.length} debug logs:\n`);
  debugLogs.forEach((log, i) => {
    const args = log.args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg).substring(0, 100) : String(arg)
    ).join(' ');
    console.log(`${i + 1}. [${log.type}] ${args}`);
  });

  // Check compositor state
  const state = await page.evaluate(() => {
    return {
      pixiAvailable: typeof window.PIXI !== 'undefined',
      compositorAvailable: typeof window.KellyPixiCompositor !== 'undefined',
      compositorInitialized: window.KellyPixiCompositor?.isInitialized,
      compositorEnabled: window.KellyPixiCompositor?.isEnabled,
      hasApp: !!window.KellyPixiCompositor?.app,
      hasCanvas: !!window.KellyPixiCompositor?.app?.canvas,
      canvasInDOM: !!document.querySelector('#kelly-stage canvas'),
      scriptVersion: Array.from(document.querySelectorAll('script[src*="kelly-pixi-compositor"]'))
        .map(s => new URL(s.src).searchParams.get('v'))[0] || null
    };
  });

  console.log('\n📊 Compositor State:');
  console.log(JSON.stringify(state, null, 2));

  await page.screenshot({ path: 'debug-screenshot.png', fullPage: true });
  console.log('\n📸 Screenshot saved: debug-screenshot.png');

  await browser.close();
}

debugCompositor().catch(console.error);

