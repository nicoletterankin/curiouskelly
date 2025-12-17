#!/usr/bin/env npx tsx
import 'dotenv/config';

async function main() {
  const res = await fetch('https://api.heygen.com/v2/voices', {
    headers: { 
      'X-Api-Key': process.env.HEYGEN_API_KEY!,
      'Accept': 'application/json'
    }
  });
  
  const data = await res.json();
  
  if (!res.ok) {
    console.error('Error:', data);
    return;
  }
  
  const voices = data.data?.voices || [];
  console.log(`Found ${voices.length} voices\n`);
  
  // Find English female voices
  const englishFemale = voices.filter((v: any) => 
    v.language?.toLowerCase().includes('english') && 
    v.gender?.toLowerCase() === 'female'
  ).slice(0, 10);
  
  console.log('English Female Voices (first 10):');
  englishFemale.forEach((v: any) => {
    console.log(`  ${v.voice_id} - ${v.name}`);
  });
  
  // Also show first few voices regardless of language
  console.log('\nAll voices (first 5):');
  voices.slice(0, 5).forEach((v: any) => {
    console.log(`  ${v.voice_id} - ${v.name} (${v.language}, ${v.gender})`);
  });
}

main().catch(console.error);
