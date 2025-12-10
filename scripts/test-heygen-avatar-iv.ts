import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const KELLY_IMAGE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/kelly_photo_1765362415986.png';
const AUDIO_URL = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/day1_hook_1765362415986.mp3';

async function main() {
  console.log('Testing HeyGen Avatar IV API...');
  
  // Try multiple endpoint variations
  const endpoints = [
    'https://api.heygen.com/v1/video.generate',
    'https://api.heygen.com/v2/video.generate', 
    'https://api.heygen.com/v1/avatar/video',
    'https://api.heygen.com/v2/video/instant_avatar',
  ];
  
  let response: Response | null = null;
  
  for (const endpoint of endpoints) {
    console.log(`\nTrying: ${endpoint}`);
    
    response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        // Try instant avatar format
        video_inputs: [{
          character: {
            type: 'talking_photo',
            talking_photo_url: KELLY_IMAGE_URL,
          },
          voice: {
            type: 'audio', 
            audio_url: AUDIO_URL,
          },
        }],
        dimension: { width: 1920, height: 1080 },
      }),
    });
    
    console.log('   Status:', response.status);
    const text = await response.text();
    console.log('   Response:', text.substring(0, 300));
    
    if (response.ok || response.status < 500) {
      break;
    }
  }
  
  console.log('Status:', response.status);
  const text = await response.text();
  console.log('Response:', text);
  
  if (response.ok) {
    const result = JSON.parse(text);
    if (result.data?.video_id) {
      console.log('\n✅ Video job started! ID:', result.data.video_id);
      console.log('Waiting for completion...');
      
      // Poll for completion
      for (let i = 0; i < 30; i++) {
        await new Promise(r => setTimeout(r, 10000));
        
        const statusRes = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${result.data.video_id}`, {
          headers: { 'X-Api-Key': HEYGEN_API_KEY },
        });
        const status = await statusRes.json();
        console.log(`   Status: ${status.data?.status}`);
        
        if (status.data?.status === 'completed') {
          console.log('\n🎬 VIDEO READY:', status.data.video_url);
          break;
        }
        if (status.data?.status === 'failed') {
          console.log('\n❌ Failed:', status.data.error);
          break;
        }
      }
    }
  }
}

main().catch(console.error);

