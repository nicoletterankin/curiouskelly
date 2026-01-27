#!/usr/bin/env npx tsx
import 'dotenv/config';

const AVATAR_IDS = [
  "7bb18cddacd44333813cc90ffa44f766",  // adult scientist (from json files)
  "6edf9b918f674e9dac2faa59d91f547c",  // mature scientist
  "d2a5133b931541e986912a37139a9398",  // elder scientist
];

async function main() {
  console.log("Checking HeyGen talking photos...\n");
  
  // First list all talking photos
  const listRes = await fetch('https://api.heygen.com/v1/talking_photo.list', {
    headers: { 'X-Api-Key': process.env.HEYGEN_API_KEY! }
  });
  const listData = await listRes.json();
  
  console.log(`Total talking photos in account: ${listData.data?.talking_photos?.length || 0}\n`);
  
  if (listData.data?.talking_photos?.length > 0) {
    console.log("First 10 talking photos:");
    listData.data.talking_photos.slice(0, 10).forEach((p: any) => {
      console.log(`  ${p.talking_photo_id}: ${p.talking_photo_name || '(no name)'}`);
    });
  }
  
  console.log("\nChecking IDs we're using:");
  for (const id of AVATAR_IDS) {
    const res = await fetch(`https://api.heygen.com/v1/talking_photo.get?talking_photo_id=${id}`, {
      headers: { 'X-Api-Key': process.env.HEYGEN_API_KEY! }
    });
    const data = await res.json();
    console.log(`  ${id}: ${data.data?.talking_photo_name || data.error?.message || 'NOT FOUND'}`);
  }
}

main().catch(console.error);
