#!/usr/bin/env npx tsx
import 'dotenv/config';

async function main() {
  const response = await fetch('https://api.heygen.com/v1/talking_photo.list', {
    headers: { 'X-Api-Key': process.env.HEYGEN_API_KEY! },
  });

  const data = await response.json();
  
  console.log('📋 YOUR HEYGEN TALKING PHOTOS:\n');
  console.log('ID'.padEnd(40) + 'NAME');
  console.log('─'.repeat(70));
  
  if (data.data && Array.isArray(data.data)) {
    data.data.forEach((photo: any) => {
      const id = photo.talking_photo_id || photo.id || 'unknown';
      const name = photo.talking_photo_name || photo.name || 'unnamed';
      console.log(`${id.padEnd(40)} ${name}`);
    });
    console.log(`\nTotal: ${data.data.length} talking photos`);
  } else {
    console.log('No talking photos found or unexpected response:');
    console.log(JSON.stringify(data, null, 2));
  }
}

main().catch(console.error);

