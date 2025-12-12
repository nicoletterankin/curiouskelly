
import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

if (!HEYGEN_API_KEY) {
  console.error('HEYGEN_API_KEY not found in environment');
  process.exit(1);
}

async function listTalkingPhotos() {
  console.log('Listing talking photos...');
  try {
    const response = await fetch('https://api.heygen.com/v1/talking_photo.list', {
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Failed to list talking photos: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log(JSON.stringify(data, null, 2));
  } catch (error) {
    console.error('Error:', error);
  }
}

listTalkingPhotos();
