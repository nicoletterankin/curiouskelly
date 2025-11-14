#!/usr/bin/env node

/**
 * Lesson Preview
 * Shows what a lesson looks like for a specific age
 */

const fs = require('fs');
const path = require('path');

function previewLesson(lessonPath, learnerAge) {
  console.log(`\n📖 Previewing lesson for age ${learnerAge}\n`);
  console.log('═'.repeat(60));
  
  // Load lesson
  if (!fs.existsSync(lessonPath)) {
    console.error(`❌ File not found: ${lessonPath}`);
    return false;
  }
  
  const lessonData = JSON.parse(fs.readFileSync(lessonPath, 'utf8'));
  
  // Determine age group
  let ageGroup;
  if (learnerAge <= 5) ageGroup = '2-5';
  else if (learnerAge <= 12) ageGroup = '6-12';
  else if (learnerAge <= 17) ageGroup = '13-17';
  else if (learnerAge <= 35) ageGroup = '18-35';
  else if (learnerAge <= 60) ageGroup = '36-60';
  else ageGroup = '61-102';
  
  const variant = lessonData.ageVariants[ageGroup];
  if (!variant) {
    console.error(`❌ Age group ${ageGroup} not found in lesson`);
    return false;
  }
  
  const content = variant.language.en;
  const kellyAge = variant.kellyAge;
  
  // Display lesson info
  console.log(`\n📚 LESSON: ${lessonData.title}`);
  console.log(`📝 ID: ${lessonData.id}`);
  console.log(`👤 Learner Age: ${learnerAge}`);
  console.log(`🎭 Kelly Age: ${kellyAge} (${variant.kellyPersona})`);
  console.log(`⏱️  Estimated Duration: ${lessonData.metadata.duration}`);
  console.log(`\n` + '─'.repeat(60));
  
  // Welcome
  console.log(`\n💬 WELCOME (${variant.pacing.welcome}):\n`);
  console.log(wrapText(content.welcome, 60));
  console.log(`\n` + '─'.repeat(60));
  
  // Main Content
  console.log(`\n📖 MAIN CONTENT (${variant.pacing.teaching}):\n`);
  console.log(wrapText(content.mainContent, 60));
  console.log(`\n` + '─'.repeat(60));
  
  // Key Points
  console.log(`\n🔑 KEY POINTS:\n`);
  content.keyPoints.forEach((point, i) => {
    console.log(`  ${i + 1}. ${wrapText(point, 56, '     ')}`);
  });
  console.log(`\n` + '─'.repeat(60));
  
  // Interaction Prompts
  console.log(`\n💡 INTERACTION PROMPTS:\n`);
  content.interactionPrompts.forEach((prompt, i) => {
    console.log(`  ${i + 1}. ${wrapText(prompt, 56, '     ')}`);
  });
  console.log(`\n` + '─'.repeat(60));
  
  // Teaching Moments
  if (variant.teachingMoments && variant.teachingMoments.length > 0) {
    console.log(`\n✨ TEACHING MOMENTS:\n`);
    variant.teachingMoments.forEach((tm, i) => {
      console.log(`  ${i + 1}. [${tm.timing}] ${tm.type.toUpperCase()}`);
      console.log(`     ${wrapText(tm.content, 56, '     ')}`);
    });
    console.log(`\n` + '─'.repeat(60));
  }
  
  // Wisdom Moment
  console.log(`\n🌟 WISDOM (${variant.pacing.wisdom}):\n`);
  console.log(wrapText(content.wisdomMoment, 60));
  console.log(`\n` + '═'.repeat(60));
  
  // Learning Outcomes
  console.log(`\n🎯 LEARNING OUTCOMES:\n`);
  lessonData.metadata.learningOutcomes.forEach((outcome, i) => {
    console.log(`  ${i + 1}. ${wrapText(outcome, 56, '     ')}`);
  });
  
  console.log(`\n` + '═'.repeat(60) + '\n');
  
  return true;
}

function wrapText(text, width, indent = '') {
  if (!text) return '';
  
  const words = text.split(' ');
  const lines = [];
  let currentLine = '';
  
  words.forEach(word => {
    if ((currentLine + ' ' + word).length > width) {
      if (currentLine) {
        lines.push(currentLine);
        currentLine = indent + word;
      } else {
        lines.push(word);
      }
    } else {
      currentLine += (currentLine ? ' ' : '') + word;
    }
  });
  
  if (currentLine) {
    lines.push(currentLine);
  }
  
  return lines.join('\n');
}

// CLI
if (require.main === module) {
  const lessonPath = process.argv[2];
  const ageFlag = process.argv.indexOf('--age');
  const age = ageFlag > -1 ? parseInt(process.argv[ageFlag + 1]) : 35;
  
  if (!lessonPath) {
    console.error('Usage: node preview-lesson.js <lesson.json> [--age <2-102>]');
    process.exit(1);
  }
  
  const success = previewLesson(lessonPath, age);
  process.exit(success ? 0 : 1);
}

module.exports = { previewLesson };















