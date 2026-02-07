/**
 * Test HeyGen video generation with correct voice
 */
const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';
const HEYGEN_API_URL = 'https://api.heygen.com/v2/video/generate';

// HEYGEN voice ID - NOT ElevenLabs!
const KELLY_VOICE_ID = 'BbuMXx40WT4ZuAgRXvNx'; // HeyGen English female voice

// Kelly explorer avatar
const AVATAR_ID = '45e5ef8b651846e0b62b7477e552e87b';

async function test() {
  console.log('Testing HeyGen video generation with correct voice...\n');
  
  const payload = {
    video_inputs: [{
      character: {
        type: 'talking_photo',
        talking_photo_id: AVATAR_ID
      },
      voice: {
        type: 'text',
        input_text: 'Hello! I am Kelly, your AI teacher. Today we are learning about magnets and how they work. Are you ready?',
        voice_id: KELLY_VOICE_ID
      }
    }],
    dimension: { width: 1280, height: 720 }
  };
  
  console.log('Payload:', JSON.stringify(payload, null, 2));
  console.log('\nSubmitting to HeyGen...');
  
  const response = await fetch(HEYGEN_API_URL, {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  
  const data = await response.json();
  console.log('\nResponse:', JSON.stringify(data, null, 2));
  
  if (data.data?.video_id) {
    console.log('\n✅ SUCCESS! Video ID:', data.data.video_id);
  } else if (data.error) {
    console.log('\n❌ ERROR:', data.error.message || JSON.stringify(data.error));
  }
}

test().catch(console.error);
