/**
 * Deep Diagnostic Test - Understanding specific issues
 */

const puppeteer = require('puppeteer');

async function runDiagnostic() {
  console.log('===========================================');
  console.log('DEEP DIAGNOSTIC - curiouskelly.com');
  console.log('===========================================\n');
  
  const browser = await puppeteer.launch({ 
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });
  
  // Capture all console messages
  const consoleMessages = [];
  page.on('console', msg => {
    consoleMessages.push({ type: msg.type(), text: msg.text() });
  });

  try {
    console.log('Loading learn page with debug mode...\n');
    await page.goto('https://www.curiouskelly.com/learn.html?debug=true', { 
      waitUntil: 'networkidle2', 
      timeout: 60000 
    });
    
    // Wait for initialization
    await new Promise(r => setTimeout(r, 8000));
    
    // ============================================
    // DIAGNOSTIC 1: LESSON SOURCE
    // ============================================
    console.log('=== DIAGNOSTIC 1: Lesson Source ===');
    
    const lessonDiagnostic = await page.evaluate(async () => {
      const results = {
        kellyLoaderExists: !!window.KellyLessonLoader,
        supabaseClient: !!window.supabaseClient || !!window.getSupabase?.(),
        configExists: !!window.KELLY_CONFIG,
        emergencyFunctionExists: typeof window.getEmergencyLesson === 'function',
      };
      
      // Try to load a lesson directly
      if (window.KellyLessonLoader) {
        try {
          const loader = window.KellyLessonLoader;
          results.loaderInitialized = !!loader.supabase;
          
          // Get today's lesson
          const lesson = await loader.getLesson(1, { archetype: 'The Scientist', region: 'adult' });
          results.lessonLoaded = !!lesson;
          results.lessonSource = lesson?._source || lesson?.source || 'unknown';
          results.lessonTopic = lesson?.topic || lesson?.title || null;
          results.isEmergencyFallback = lesson?.isEmergencyFallback || false;
          
          // Check cache stats
          results.cacheStats = loader.getCacheStats();
        } catch (e) {
          results.lessonError = e.message;
        }
      }
      
      return results;
    });
    
    console.log('  KellyLessonLoader exists:', lessonDiagnostic.kellyLoaderExists);
    console.log('  Supabase client available:', lessonDiagnostic.supabaseClient);
    console.log('  Loader initialized with Supabase:', lessonDiagnostic.loaderInitialized);
    console.log('  Emergency function exists:', lessonDiagnostic.emergencyFunctionExists);
    console.log('  Lesson loaded:', lessonDiagnostic.lessonLoaded);
    console.log('  Lesson source:', lessonDiagnostic.lessonSource);
    console.log('  Lesson topic:', lessonDiagnostic.lessonTopic);
    console.log('  Is emergency fallback:', lessonDiagnostic.isEmergencyFallback);
    if (lessonDiagnostic.lessonError) {
      console.log('  Error:', lessonDiagnostic.lessonError);
    }
    
    // ============================================
    // DIAGNOSTIC 2: VIDEO LOADING
    // ============================================
    console.log('\n=== DIAGNOSTIC 2: Video Loading ===');
    
    const videoDiagnostic = await page.evaluate(() => {
      const video = document.querySelector('video');
      if (!video) return { exists: false };
      
      // Get all sources
      const sources = Array.from(video.querySelectorAll('source')).map(s => ({
        src: s.src,
        type: s.type
      }));
      
      // Check video element attributes
      const result = {
        exists: true,
        id: video.id,
        className: video.className,
        src: video.src,
        currentSrc: video.currentSrc,
        poster: video.poster,
        sources,
        readyState: video.readyState,
        networkState: video.networkState,
        error: video.error ? {
          code: video.error.code,
          message: video.error.message
        } : null,
        duration: video.duration,
        paused: video.paused,
        muted: video.muted,
        autoplay: video.autoplay,
        loop: video.loop
      };
      
      // Check for dynamicSrc or data attributes
      result.dataAttributes = {};
      for (const attr of video.attributes) {
        if (attr.name.startsWith('data-')) {
          result.dataAttributes[attr.name] = attr.value;
        }
      }
      
      return result;
    });
    
    console.log('  Video exists:', videoDiagnostic.exists);
    if (videoDiagnostic.exists) {
      console.log('  Video ID:', videoDiagnostic.id || '(none)');
      console.log('  Video class:', videoDiagnostic.className || '(none)');
      console.log('  src attribute:', videoDiagnostic.src || '(empty)');
      console.log('  currentSrc:', videoDiagnostic.currentSrc || '(none)');
      console.log('  poster:', videoDiagnostic.poster || '(none)');
      console.log('  sources:', videoDiagnostic.sources.length > 0 ? JSON.stringify(videoDiagnostic.sources) : '(none)');
      console.log('  readyState:', videoDiagnostic.readyState);
      console.log('  networkState:', videoDiagnostic.networkState);
      console.log('  error:', videoDiagnostic.error ? JSON.stringify(videoDiagnostic.error) : '(none)');
      console.log('  duration:', videoDiagnostic.duration);
      console.log('  autoplay:', videoDiagnostic.autoplay);
      console.log('  muted:', videoDiagnostic.muted);
      if (Object.keys(videoDiagnostic.dataAttributes).length > 0) {
        console.log('  data attributes:', JSON.stringify(videoDiagnostic.dataAttributes));
      }
    }
    
    // ============================================
    // DIAGNOSTIC 3: SUPABASE DIRECT QUERY
    // ============================================
    console.log('\n=== DIAGNOSTIC 3: Supabase Direct Query ===');
    
    const supabaseDiagnostic = await page.evaluate(async () => {
      const results = {};
      
      try {
        const config = window.KELLY_CONFIG;
        if (!config?.supabaseUrl || !config?.supabaseKey) {
          return { error: 'Missing KELLY_CONFIG' };
        }
        
        results.supabaseUrl = config.supabaseUrl;
        
        // Query core_lessons
        const lessonsRes = await fetch(
          `${config.supabaseUrl}/rest/v1/core_lessons?select=id,day_number,topic&order=day_number&limit=5`,
          {
            headers: {
              'apikey': config.supabaseKey,
              'Authorization': `Bearer ${config.supabaseKey}`
            }
          }
        );
        
        if (lessonsRes.ok) {
          const lessons = await lessonsRes.json();
          results.coreLessonsCount = lessons.length;
          results.sampleLessons = lessons.slice(0, 3).map(l => ({ day: l.day_number, topic: l.topic }));
        } else {
          results.coreLessonsError = `Status ${lessonsRes.status}`;
        }
        
        // Query lesson_atoms count
        const atomsRes = await fetch(
          `${config.supabaseUrl}/rest/v1/lesson_atoms?select=id&limit=1`,
          {
            headers: {
              'apikey': config.supabaseKey,
              'Authorization': `Bearer ${config.supabaseKey}`,
              'Prefer': 'count=exact'
            }
          }
        );
        
        if (atomsRes.ok) {
          const countHeader = atomsRes.headers.get('content-range');
          results.atomsRange = countHeader || 'unknown';
        }
        
        // Query lesson_shards count
        const shardsRes = await fetch(
          `${config.supabaseUrl}/rest/v1/lesson_shards?select=id&limit=1`,
          {
            headers: {
              'apikey': config.supabaseKey,
              'Authorization': `Bearer ${config.supabaseKey}`,
              'Prefer': 'count=exact'
            }
          }
        );
        
        if (shardsRes.ok) {
          const countHeader = shardsRes.headers.get('content-range');
          results.shardsRange = countHeader || 'unknown';
        }
        
      } catch (e) {
        results.error = e.message;
      }
      
      return results;
    });
    
    console.log('  Supabase URL:', supabaseDiagnostic.supabaseUrl || 'unknown');
    if (supabaseDiagnostic.error) {
      console.log('  ❌ Error:', supabaseDiagnostic.error);
    } else {
      console.log('  Core lessons retrieved:', supabaseDiagnostic.coreLessonsCount || 0);
      if (supabaseDiagnostic.sampleLessons) {
        console.log('  Sample lessons:');
        supabaseDiagnostic.sampleLessons.forEach(l => {
          console.log(`    Day ${l.day}: ${l.topic}`);
        });
      }
      console.log('  Atoms range:', supabaseDiagnostic.atomsRange);
      console.log('  Shards range:', supabaseDiagnostic.shardsRange);
    }
    
    // ============================================
    // DIAGNOSTIC 4: Kelly Video URLs in Storage
    // ============================================
    console.log('\n=== DIAGNOSTIC 4: Kelly Video Storage ===');
    
    const storageDiagnostic = await page.evaluate(async () => {
      const config = window.KELLY_CONFIG;
      if (!config?.supabaseUrl) return { error: 'No Supabase URL' };
      
      const results = {};
      
      // Try to list kelly-videos bucket
      try {
        const baseUrl = config.supabaseUrl.replace(/\/$/, '');
        
        // Try a sample video URL
        const testUrls = [
          `${baseUrl}/storage/v1/object/public/kelly-videos/idle/kelly-idle-01.mp4`,
          `${baseUrl}/storage/v1/object/public/kelly-videos/presenters/kelly-presenter-01.mp4`,
          `${baseUrl}/storage/v1/object/public/kelly-videos/default.mp4`
        ];
        
        results.testUrls = [];
        for (const url of testUrls) {
          try {
            const res = await fetch(url, { method: 'HEAD' });
            results.testUrls.push({
              url: url.substring(url.indexOf('/kelly-videos')),
              status: res.status,
              contentType: res.headers.get('content-type'),
              size: res.headers.get('content-length')
            });
          } catch (e) {
            results.testUrls.push({ url: url.substring(url.indexOf('/kelly-videos')), error: e.message });
          }
        }
      } catch (e) {
        results.error = e.message;
      }
      
      return results;
    });
    
    if (storageDiagnostic.error) {
      console.log('  ❌ Error:', storageDiagnostic.error);
    } else if (storageDiagnostic.testUrls) {
      console.log('  Video URL checks:');
      storageDiagnostic.testUrls.forEach(u => {
        if (u.error) {
          console.log(`    ${u.url}: ❌ ${u.error}`);
        } else {
          const status = u.status === 200 ? '✅' : '❌';
          console.log(`    ${u.url}: ${status} ${u.status} (${u.contentType || 'unknown type'})`);
        }
      });
    }
    
    // ============================================
    // DIAGNOSTIC 5: Console Log Analysis
    // ============================================
    console.log('\n=== DIAGNOSTIC 5: Console Messages ===');
    
    const relevantLogs = consoleMessages.filter(m => 
      m.text.includes('lesson') ||
      m.text.includes('Lesson') ||
      m.text.includes('Supabase') ||
      m.text.includes('supabase') ||
      m.text.includes('Emergency') ||
      m.text.includes('fallback') ||
      m.text.includes('Kelly') ||
      m.text.includes('video') ||
      m.text.includes('error') ||
      m.text.includes('Error')
    );
    
    console.log(`  Total messages: ${consoleMessages.length}`);
    console.log(`  Relevant messages: ${relevantLogs.length}`);
    console.log('  Key logs:');
    relevantLogs.slice(0, 15).forEach(m => {
      const prefix = m.type === 'error' ? '  ❌' : '  ';
      console.log(`${prefix}  [${m.type}] ${m.text.substring(0, 120)}`);
    });
    
    // ============================================
    // SUMMARY
    // ============================================
    console.log('\n===========================================');
    console.log('DIAGNOSTIC SUMMARY');
    console.log('===========================================');
    
    const issues = [];
    
    if (!lessonDiagnostic.loaderInitialized) {
      issues.push('KellyLessonLoader not initialized with Supabase');
    }
    
    if (lessonDiagnostic.lessonSource === 'emergency' || lessonDiagnostic.isEmergencyFallback) {
      issues.push('Lessons loading from emergency fallback instead of Supabase');
    }
    
    if (!videoDiagnostic.exists) {
      issues.push('No video element on page');
    } else if (!videoDiagnostic.src && !videoDiagnostic.currentSrc && videoDiagnostic.sources.length === 0) {
      issues.push('Video element has no source');
    }
    
    if (issues.length === 0) {
      console.log('✅ No critical issues found');
    } else {
      console.log('Issues found:');
      issues.forEach(i => console.log('  ❌ ' + i));
    }
    
    console.log('===========================================\n');
    
  } catch (error) {
    console.log('\n❌ Diagnostic crashed:', error.message);
  } finally {
    await browser.close();
  }
}

runDiagnostic().catch(console.error);
