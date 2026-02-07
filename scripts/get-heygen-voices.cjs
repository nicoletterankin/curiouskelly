const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';

async function main() {
  const res = await fetch('https://api.heygen.com/v1/voice.list', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  
  const data = await res.json();
  
  console.log('=== RAW RESPONSE ===\n');
  const voices = data.data?.list || [];
  console.log('Total voices:', voices.length);
  
  if (voices.length > 0) {
    console.log('\nFirst voice structure:');
    console.log(JSON.stringify(voices[0], null, 2));
    
    // Get unique languages and genders
    const languages = [...new Set(voices.map(v => v.language))].slice(0, 20);
    const genders = [...new Set(voices.map(v => v.gender))];
    console.log('\nLanguages:', languages.join(', '));
    console.log('Genders:', genders.join(', '));
    
    // Find English female voices
    const enFemale = voices.filter(v => 
      (v.language?.toLowerCase().includes('en') || v.language?.toLowerCase().includes('english')) &&
      v.gender?.toLowerCase() === 'female'
    ).slice(0, 10);
    
    console.log('\n=== ENGLISH FEMALE VOICES ===');
    enFemale.forEach(v => {
      console.log(`${v.voice_id} - ${v.display_name || v.name}`);
    });
    
    if (enFemale.length === 0) {
      console.log('\nShowing first 5 voices of any kind:');
      voices.slice(0, 5).forEach(v => {
        console.log(`${v.voice_id} - ${v.display_name || v.name} (${v.gender}, ${v.language})`);
      });
    }
  }
}

main().catch(console.error);
