/**
 * Test if video playback works on the learn page
 */

const puppeteer = require('puppeteer');

async function testVideoPlayback() {
  console.log('===========================================');
  console.log('🎬 TESTING VIDEO PLAYBACK');
  console.log('===========================================\n');
  
  const browser = await puppeteer.launch({ 
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });
  
  // Capture console
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('video') || text.includes('Video') || text.includes('motion') || 
        text.includes('Motion') || text.includes('resolv') || text.includes('kelly_motion')) {
      console.log(`[${msg.type()}] ${text.substring(0, 150)}`);
    }
  });

  try {
    // Add timestamp to bust cache
    const timestamp = Date.now();
    console.log('Loading learn page with cache bust...');
    await page.goto(`https://www.curiouskelly.com/learn.html?debug=true&nocache=${timestamp}`, { 
      waitUntil: 'networkidle2', 
      timeout: 60000 
    });
    
    // Wait for initialization
    await new Promise(r => setTimeout(r, 8000));
    
    // Try to manually trigger a motion clip lookup
    console.log('\n📡 Testing direct motion clip lookup...');
    
    const result = await page.evaluate(async () => {
      // Direct Supabase query (bypassing API cache)
      const config = window.KELLY_CONFIG;
      if (!config?.supabaseUrl || !config?.supabaseKey) {
        return { error: 'No config' };
      }
      
      const res = await fetch(
        `${config.supabaseUrl}/rest/v1/kelly_motion_library?avatar_key=eq.scientist_adult&phase=eq.hook&status=eq.completed&limit=1`,
        {
          headers: {
            'apikey': config.supabaseKey,
            'Authorization': `Bearer ${config.supabaseKey}`
          }
        }
      );
      
      if (!res.ok) {
        return { error: `HTTP ${res.status}` };
      }
      
      const data = await res.json();
      return {
        found: data.length > 0,
        video_url: data[0]?.video_url || null,
        isSupabase: data[0]?.video_url?.includes('supabase.co') || false
      };
    });
    
    console.log('Direct query result:', JSON.stringify(result, null, 2));
    
    if (result.found && result.isSupabase) {
      console.log('\n✅ SUCCESS! Videos are now served from Supabase!');
      
      // Try to actually load the video
      console.log('\n🔗 Testing video URL accessibility...');
      const videoAccessible = await page.evaluate(async (url) => {
        try {
          const res = await fetch(url, { method: 'HEAD' });
          return { 
            accessible: res.ok, 
            status: res.status,
            contentType: res.headers.get('content-type')
          };
        } catch (e) {
          return { accessible: false, error: e.message };
        }
      }, result.video_url);
      
      console.log('Video accessibility:', JSON.stringify(videoAccessible, null, 2));
    } else if (result.error) {
      console.log('\n❌ Error:', result.error);
    } else {
      console.log('\n⚠️ Videos still using HeyGen URLs');
    }
    
    // Check video element status after some time
    console.log('\n📺 Checking video element after page load...');
    const videoStatus = await page.evaluate(() => {
      const v = document.getElementById('kelly-video');
      if (!v) return { exists: false };
      return {
        exists: true,
        src: v.src || '(none)',
        hidden: v.hidden,
        className: v.className
      };
    });
    console.log('Video element:', JSON.stringify(videoStatus, null, 2));
    
  } catch (e) {
    console.error('Error:', e.message);
  } finally {
    await browser.close();
  }
  
  console.log('\n===========================================');
  console.log('🎉 TEST COMPLETE');
  console.log('===========================================');
}

testVideoPlayback().catch(console.error);
