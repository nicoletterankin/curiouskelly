
import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

if (!HEYGEN_API_KEY) {
  console.error('HEYGEN_API_KEY not found in environment');
  process.exit(1);
}

async function listTalkingPhotos() {
  try {
    const response = await fetch('https://api.heygen.com/v2/talking_photos', {
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Failed to list talking photos: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('Talking Photos:', JSON.stringify(data.data.talking_photos, null, 2));
  } catch (error) {
    console.error('Error:', error);
  }
}

listTalkingPhotos();

