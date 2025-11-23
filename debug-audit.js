
const fs = require('fs');
const path = require('path');

const filePath = path.join('curious-kellly', 'backend', 'config', 'lessons', 'music-human-culture-dna.json');
console.log(`Reading file: ${filePath}`);

try {
  const content = fs.readFileSync(filePath, 'utf8');
  console.log(`Content length: ${content.length}`);
  
  const data = JSON.parse(content);
  console.log('Keys:', Object.keys(data));
  
  if (data.ageVariants) {
    console.log('ageVariants keys:', Object.keys(data.ageVariants));
    if (data.ageVariants['2-5']) {
      console.log('2-5 keys:', Object.keys(data.ageVariants['2-5']));
      if (data.ageVariants['2-5'].language) {
        console.log('2-5 language keys:', Object.keys(data.ageVariants['2-5'].language));
      } else {
        console.log('2-5 has no language property');
      }
    } else {
      console.log('2-5 variant missing');
    }
  } else {
    console.log('No ageVariants');
  }
} catch (e) {
  console.error('Error:', e);
}


