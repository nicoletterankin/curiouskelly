const fs = require('fs');
const path = require('path');

const htmlPath = path.join(__dirname, 'public/learn.html');

let html = fs.readFileSync(htmlPath, 'utf8');

// Find and remove the first style block (main CSS)
// Pattern: <style> ... </style> after the scripts in head
const firstStyleRegex = /\n\s*<style>\s*\n\s*\/\* ═+\s*\n\s*TIKTOK-STYLE LESSON PLAYER[\s\S]*?<\/style>\s*\n/;

if (firstStyleRegex.test(html)) {
  html = html.replace(firstStyleRegex, '\n');
  console.log('✅ Removed first inline style block (main CSS)');
} else {
  console.log('⚠️ First style block not found with expected pattern');
}

// Find and remove the second style block (commons overlay CSS)
// Pattern: <style> ... </style> containing .commons-overlay
const secondStyleRegex = /\n\s*<style>\s*\n\s*\/\*[\s\S]*?\.commons-overlay[\s\S]*?<\/style>\s*\n/;

if (secondStyleRegex.test(html)) {
  html = html.replace(secondStyleRegex, '\n');
  console.log('✅ Removed second inline style block (commons overlay CSS)');
} else {
  console.log('⚠️ Second style block not found with expected pattern');
}

fs.writeFileSync(htmlPath, html);

// Verify
const lines = html.split('\n');
let styleCount = 0;
lines.forEach(line => {
  if (line.includes('<style>')) styleCount++;
});

console.log('\n📊 Result: ' + styleCount + ' inline <style> blocks remaining');
console.log('📄 File size: ' + (html.length / 1024).toFixed(1) + ' KB');



