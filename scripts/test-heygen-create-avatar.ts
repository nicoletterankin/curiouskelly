import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const KELLY_IMAGE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/kelly_photo_1765362415986.png';

async function main() {
  console.log('Testing HeyGen photo avatar creation...');
  console.log('API Key:', HEYGEN_API_KEY.substring(0, 15) + '...');
  
  // Try different endpoints to create photo avatar
  const endpoints = [
    { url: 'https://api.heygen.com/v2/photo_avatar/generate', method: 'POST' },
    { url: 'https://api.heygen.com/v1/photo_avatar', method: 'POST' },
    { url: 'https://api.heygen.com/v2/talking_photo', method: 'POST' },
    { url: 'https://api.heygen.com/v1/talking_photo.add', method: 'POST' },
  ];
  
  for (const endpoint of endpoints) {
    console.log(`\nTrying: ${endpoint.url}`);
    try {
      const response = await fetch(endpoint.url, {
        method: endpoint.method,
        headers: {
          'X-Api-Key': HEYGEN_API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_url: KELLY_IMAGE_URL,
          name: 'Kelly - Curious Kelly Teacher',
        }),
      });
      
      const text = await response.text();
      console.log(`   Status: ${response.status}`);
      console.log(`   Response: ${text.substring(0, 500)}`);
      
      if (response.ok) {
        console.log('   ✅ SUCCESS!');
        break;
      }
    } catch (e: any) {
      console.log(`   ❌ Error: ${e.message}`);
    }
  }
}

main();

